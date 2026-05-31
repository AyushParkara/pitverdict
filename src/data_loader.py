from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from .models import DataLoadResult, LapRecord


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_SAMPLE_PATH = _default_project_root() / "data" / "sample_race.csv"
DEFAULT_REAL_DATA_PATH = _default_project_root() / "data" / "processed" / "mdce_kaggle_weather_tyre_latest.csv"

# Opt-in gate for datasets that carry NonCommercial restrictions (e.g. CC BY-NC).
# The default prepared dataset currently comes from Kaggle and is recorded as CC BY-NC 4.0.
NONCOMMERCIAL_OPT_IN_ENV = "MDCE_ALLOW_NONCOMMERCIAL_DATA"
NONCOMMERCIAL_GUARD_WARNING_PREFIX = "NonCommercial data guard:"

REQUIRED_COLUMNS = {"lap", "lap_time_s"}
SCHEMA_COLUMNS = [
    "lap",
    "lap_time_s",
    "sector1_s",
    "sector2_s",
    "sector3_s",
    "tyre_compound",
    "tyre_age",
    "track_status",
    "weather",
    "gap_to_car_ahead_s",
    "predicted_lap_time_s",
    "tyre_temp_proxy_c",
    "speed_consistency",
]
SOURCE_SUPPORT_COLUMNS = [
    "rainfall",
    "track_temp_c",
    "air_temp_c",
    "humidity",
    "pressure",
    "wind_speed",
    "source_year",
    "source_round",
    "source_race_id",
    "source_circuit",
    "source_driver_code",
]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized


def validate_columns(frame: pd.DataFrame) -> list[str]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        return [f"Missing required column: {column}" for column in missing]
    return []


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_noncommercial_license(value: object) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    return "by-nc" in s or "noncommercial" in s or "non-commercial" in s


def _default_real_data_requires_opt_in(real_path: Path) -> tuple[bool, str | None]:
    """Return (requires_opt_in, license_hint).

    We try to read a sibling `*.metadata.json` file written by `tools/prepare_mdce_dataset.py`.
    If metadata is missing for the default Kaggle-derived filename, we act conservatively.
    """

    metadata_path = real_path.with_suffix(".metadata.json")
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = None

        if payload is None:
            if real_path.name.startswith("mdce_kaggle_"):
                return True, "NonCommercial (metadata unreadable; default Kaggle-derived dataset treated as NC)"
            return False, None

        license_spdx = payload.get("license_spdx")
        license_note = payload.get("license_note")

        requires = _looks_noncommercial_license(license_spdx) or _looks_noncommercial_license(license_note)
        hint = None
        if license_spdx:
            hint = str(license_spdx)
        elif license_note:
            hint = str(license_note)
        return bool(requires), hint

    # Default: be conservative for the known Kaggle-derived default filename.
    if real_path.name.startswith("mdce_kaggle_"):
        return True, "NonCommercial (metadata missing; default Kaggle-derived dataset treated as NC)"

    return False, None


def _read_sidecar_metadata(csv_path: Path) -> dict:
    meta_path = csv_path.with_suffix(".metadata.json")
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _derive_missing_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """Map a real/Colab CSV into the MDCE schema.

    Required real fields are `lap` and `lap_time_s`. Other fields are either used
    if present or derived with explicit provenance warnings.
    """
    mapped = frame.copy()
    real_columns = [
        column
        for column in [*SCHEMA_COLUMNS, *SOURCE_SUPPORT_COLUMNS]
        if column in mapped.columns
    ]
    derived_columns: list[str] = []
    proxy_columns: list[str] = []
    warnings: list[str] = []

    mapped["lap"] = pd.to_numeric(mapped["lap"], errors="coerce")
    mapped["lap_time_s"] = pd.to_numeric(mapped["lap_time_s"], errors="coerce")
    mapped = mapped.dropna(subset=["lap", "lap_time_s"]).sort_values("lap")

    if mapped.empty:
        raise ValueError("No valid rows remain after parsing lap and lap_time_s.")

    if "sector1_s" not in mapped.columns:
        mapped["sector1_s"] = mapped["lap_time_s"] * 0.32
        derived_columns.append("sector1_s")
        warnings.append("sector1_s derived from lap_time_s because it was not supplied.")
    if "sector2_s" not in mapped.columns:
        mapped["sector2_s"] = mapped["lap_time_s"] * 0.37
        derived_columns.append("sector2_s")
        warnings.append("sector2_s derived from lap_time_s because it was not supplied.")
    if "sector3_s" not in mapped.columns:
        mapped["sector3_s"] = mapped["lap_time_s"] * 0.31
        derived_columns.append("sector3_s")
        warnings.append("sector3_s derived from lap_time_s because it was not supplied.")

    if "tyre_compound" not in mapped.columns:
        mapped["tyre_compound"] = "UNKNOWN"
        derived_columns.append("tyre_compound")
        warnings.append("tyre_compound set to UNKNOWN because it was not supplied.")
    if "tyre_age" not in mapped.columns:
        mapped["tyre_age"] = range(1, len(mapped) + 1)
        derived_columns.append("tyre_age")
        warnings.append("tyre_age derived from row order because it was not supplied.")
    if "track_status" not in mapped.columns:
        mapped["track_status"] = "NORMAL"
        derived_columns.append("track_status")
        warnings.append("track_status set to NORMAL because it was not supplied.")
    if "weather" not in mapped.columns:
        if "rainfall" in mapped.columns:
            rainfall = pd.to_numeric(mapped["rainfall"], errors="coerce").fillna(0.0)
            mapped["weather"] = np.where(rainfall > 0.0, "WET", "DRY")
            derived_columns.append("weather")
            warnings.append("weather derived from source rainfall values.")
        else:
            mapped["weather"] = "DRY"
            derived_columns.append("weather")
            warnings.append("weather set to DRY because it was not supplied.")
    if "gap_to_car_ahead_s" not in mapped.columns:
        mapped["gap_to_car_ahead_s"] = 0.0
        derived_columns.append("gap_to_car_ahead_s")
        warnings.append("gap_to_car_ahead_s set to 0.0 because it was not supplied.")
    if "predicted_lap_time_s" not in mapped.columns:
        rolling = mapped["lap_time_s"].rolling(window=5, min_periods=1).mean().shift(1)
        mapped["predicted_lap_time_s"] = rolling.fillna(mapped["lap_time_s"])
        derived_columns.append("predicted_lap_time_s")
        warnings.append("predicted_lap_time_s derived from prior rolling lap-time average.")
    if "tyre_temp_proxy_c" not in mapped.columns:
        tyre_age = pd.to_numeric(mapped["tyre_age"], errors="coerce").fillna(1)
        if "track_temp_c" in mapped.columns:
            track_temp = pd.to_numeric(mapped["track_temp_c"], errors="coerce").fillna(25.0)
            mapped["tyre_temp_proxy_c"] = 70.0 + (track_temp * 0.7) + (tyre_age * 0.15)
            warnings.append("tyre_temp_proxy_c is a proxy derived from public track temperature and tyre age, not private tyre telemetry.")
        else:
            mapped["tyre_temp_proxy_c"] = 95.0 + (tyre_age * 0.25)
            warnings.append("tyre_temp_proxy_c is a synthetic proxy because public/private tyre temp was not supplied.")
        proxy_columns.append("tyre_temp_proxy_c")
    if "speed_consistency" not in mapped.columns:
        rolling_variation = mapped["lap_time_s"].pct_change().abs().rolling(window=3, min_periods=1).mean()
        mapped["speed_consistency"] = (1.0 - (rolling_variation.fillna(0.0).clip(0.0, 0.20) / 0.20)).clip(0.0, 1.0)
        proxy_columns.append("speed_consistency")
        warnings.append("speed_consistency is a lap-time stability proxy because direct speed trace was not supplied.")

    for column in ["sector1_s", "sector2_s", "sector3_s", "tyre_age", "gap_to_car_ahead_s", "predicted_lap_time_s", "tyre_temp_proxy_c", "speed_consistency"]:
        mapped[column] = pd.to_numeric(mapped[column], errors="coerce")

    mapped["track_status"] = mapped["track_status"].fillna("NORMAL").astype(str).str.upper()
    mapped["weather"] = mapped["weather"].fillna("DRY").astype(str).str.upper()
    mapped["tyre_compound"] = mapped["tyre_compound"].fillna("UNKNOWN").astype(str).str.upper()
    mapped = mapped.fillna(
        {
            "sector1_s": mapped["lap_time_s"] * 0.32,
            "sector2_s": mapped["lap_time_s"] * 0.37,
            "sector3_s": mapped["lap_time_s"] * 0.31,
            "tyre_age": 1,
            "gap_to_car_ahead_s": 0.0,
            "predicted_lap_time_s": mapped["lap_time_s"],
            "tyre_temp_proxy_c": 95.0,
            "speed_consistency": 1.0,
        }
    )

    return mapped, real_columns, derived_columns, proxy_columns + [column for column in real_columns if column.endswith("_proxy_c")], warnings


def _frame_to_records(frame: pd.DataFrame) -> list[LapRecord]:
    records: list[LapRecord] = []
    for _, row in frame.iterrows():
        records.append(
            LapRecord(
                lap=int(round(float(row["lap"]))),
                lap_time_s=float(row["lap_time_s"]),
                sector1_s=float(row["sector1_s"]),
                sector2_s=float(row["sector2_s"]),
                sector3_s=float(row["sector3_s"]),
                tyre_compound=str(row["tyre_compound"]),
                tyre_age=int(round(float(row["tyre_age"]))),
                track_status=str(row["track_status"]),
                weather=str(row["weather"]),
                gap_to_car_ahead_s=float(row["gap_to_car_ahead_s"]),
                predicted_lap_time_s=float(row["predicted_lap_time_s"]),
                tyre_temp_proxy_c=float(row["tyre_temp_proxy_c"]),
                speed_consistency=float(row["speed_consistency"]),
            )
        )
    return records


def load_race_csv(source: str | Path | IO[str] | IO[bytes], source_name: str = "CSV dataset") -> DataLoadResult:
    """Load a Colab/Drive CSV mapped to the MDCE schema."""
    frame = _normalize_columns(pd.read_csv(source))
    validation_errors = validate_columns(frame)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    mapped, real_columns, derived_columns, proxy_columns, derivation_warnings = _derive_missing_columns(frame)
    records = _frame_to_records(mapped)

    dataset_metadata: dict = {}
    if isinstance(source, (str, Path)):
        try:
            p = Path(source)
            if p.suffix.lower() == ".csv" and p.exists():
                dataset_metadata = _read_sidecar_metadata(p)
        except (OSError, RuntimeError):
            dataset_metadata = {}

    return DataLoadResult(
        records=records,
        source_name=source_name,
        real_columns=real_columns,
        derived_columns=derived_columns,
        proxy_columns=proxy_columns,
        dataset_metadata=dataset_metadata,
        warnings=[
            *[f"Real column used: {column}" for column in real_columns],
            *[f"Derived column: {column}" for column in derived_columns],
            *[f"Proxy column: {column}" for column in proxy_columns],
            *derivation_warnings,
        ],
    )


def load_sample_race(path: str | Path = DEFAULT_SAMPLE_PATH) -> list[LapRecord]:
    """Load the offline demo race dataset."""
    records: list[LapRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            records.append(
                LapRecord(
                    lap=int(round(float(row["lap"]))),
                    lap_time_s=float(row["lap_time_s"]),
                    sector1_s=float(row["sector1_s"]),
                    sector2_s=float(row["sector2_s"]),
                    sector3_s=float(row["sector3_s"]),
                    tyre_compound=row["tyre_compound"],
                    tyre_age=int(round(float(row["tyre_age"]))),
                    track_status=row["track_status"],
                    weather=row["weather"],
                    gap_to_car_ahead_s=float(row["gap_to_car_ahead_s"]),
                    predicted_lap_time_s=float(row["predicted_lap_time_s"]),
                    tyre_temp_proxy_c=float(row["tyre_temp_proxy_c"]),
                    speed_consistency=float(row["speed_consistency"]),
                )
            )
    return records


def load_sample_race_result(path: str | Path = DEFAULT_SAMPLE_PATH) -> DataLoadResult:
    records = load_sample_race(path)
    dataset_metadata = _read_sidecar_metadata(Path(path))
    sample_meta_note = str(dataset_metadata.get("license_note") or "").strip() if dataset_metadata else ""
    return DataLoadResult(
        records=records,
        source_name="offline demo fallback",
        real_columns=["lap", "lap_time_s", "sector1_s", "sector2_s", "sector3_s", "tyre_compound", "tyre_age", "track_status", "weather", "gap_to_car_ahead_s"],
        derived_columns=["predicted_lap_time_s"],
        proxy_columns=["tyre_temp_proxy_c", "speed_consistency"],
        dataset_metadata=dataset_metadata,
        warnings=[
            "Using offline fallback demo data, not official F1 telemetry.",
            "tyre_temp_proxy_c and speed_consistency are synthetic proxy fields.",
            *([f"Dataset metadata: {sample_meta_note}"] if sample_meta_note else []),
        ],
    )


def load_default_data_result(root: str | Path | None = None) -> DataLoadResult:
    """Prefer prepared real public data when available, then fall back to demo data.

    If `root` is provided, resolve default data paths relative to that root.
    This is used by production CLI/Colab runs where the repo may live under a
    different base path.
    """

    root_path = Path(root).resolve() if root is not None else None
    real_path = (root_path / "data" / "processed" / "mdce_kaggle_weather_tyre_latest.csv") if root_path else DEFAULT_REAL_DATA_PATH
    sample_path = (root_path / "data" / "sample_race.csv") if root_path else DEFAULT_SAMPLE_PATH

    if real_path.exists():
        requires_opt_in, license_hint = _default_real_data_requires_opt_in(real_path)
        if requires_opt_in and not _env_truthy(NONCOMMERCIAL_OPT_IN_ENV):
            fallback = load_sample_race_result(sample_path)
            hint = f" ({license_hint})" if license_hint else ""
            guard_warning = (
                f"{NONCOMMERCIAL_GUARD_WARNING_PREFIX} Prepared real dataset at {real_path} appears NonCommercial{hint}. "
                f"Set {NONCOMMERCIAL_OPT_IN_ENV}=1 to opt in; otherwise MDCE uses the offline demo fallback dataset."
            )
            return DataLoadResult(
                records=fallback.records,
                source_name=fallback.source_name,
                real_columns=fallback.real_columns,
                derived_columns=fallback.derived_columns,
                proxy_columns=fallback.proxy_columns,
                dataset_metadata=fallback.dataset_metadata,
                warnings=[guard_warning, *list(fallback.warnings or [])],
            )
        return load_race_csv(real_path, source_name=str(real_path))
    return load_sample_race_result(sample_path)


def available_records(records: list[LapRecord]) -> list[LapRecord]:
    return [record for record in records if not record.missing]


def recent_records(records: list[LapRecord], count: int = 5) -> list[LapRecord]:
    return available_records(records)[-count:]


def linear_trend(values: list[float]) -> float:
    """Return simple per-step trend using first and last values."""
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / (len(values) - 1)


def lap_time_trend(records: list[LapRecord], count: int = 5) -> float:
    return linear_trend([record.lap_time_s for record in recent_records(records, count)])


def predicted_lap_trend(records: list[LapRecord], count: int = 5) -> float:
    return linear_trend([record.predicted_lap_time_s for record in recent_records(records, count)])


def tyre_temp_trend(records: list[LapRecord], count: int = 5) -> float:
    return linear_trend([record.tyre_temp_proxy_c for record in recent_records(records, count)])
