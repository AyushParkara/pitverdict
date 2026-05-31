from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median


REQUIRED_COLUMNS = {"lap", "lap_time_s"}


_NULLISH_STRINGS = {"", "na", "nan", "none", "null"}


def _raw_is_present(value: object) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if s == "":
        return False
    return s.lower() not in _NULLISH_STRINGS


@dataclass(frozen=True)
class DatasetValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _as_float(value: str) -> float | None:
    try:
        # Treat empty/null-ish as missing.
        if value is None:
            return None
        s = str(value).strip()
        if s == "" or s.lower() in {"na", "nan", "none", "null"}:
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def _as_int(value: str) -> int | None:
    f = _as_float(value)
    if f is None:
        return None
    try:
        return int(f)
    except (ValueError, TypeError):
        return None


def validate_mdce_csv(path: str | Path) -> DatasetValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    p = Path(path)
    if not p.exists():
        return DatasetValidationResult(ok=False, errors=[f"File not found: {p}"], warnings=[])

    # Sidecar metadata is recommended for prepared datasets (license/provenance).
    meta_path = p.with_suffix(".metadata.json")
    if p.suffix.lower() == ".csv" and "mdce_kaggle_" in p.name and not meta_path.exists():
        warnings.append(
            f"Sidecar metadata missing: expected {meta_path.name}. "
            "Prepared datasets should include license/provenance metadata (e.g. license_spdx, source_url)."
        )
    elif meta_path.exists():
        try:
            import json

            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if not (payload.get("license_spdx") or payload.get("license_note")):
                    warnings.append(f"Sidecar metadata present ({meta_path.name}) but missing license fields (license_spdx/license_note).")
        except (json.JSONDecodeError, OSError):
            warnings.append(f"Sidecar metadata present ({meta_path.name}) but could not be parsed as JSON.")

    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set((reader.fieldnames or []))
        missing = sorted(REQUIRED_COLUMNS - cols)
        if missing:
            return DatasetValidationResult(ok=False, errors=[f"Missing required columns: {missing}"], warnings=[])

        # Collect the minimum signals needed to catch obvious broken inputs.
        laps: list[int] = []
        lap_times: list[float] = []
        gaps: list[float] = []
        sector_ratios: list[tuple[float, float, float]] = []
        tyre_compounds: list[str] = []

        row_count = 0
        for row in reader:
            row_count += 1
            lap = _as_int(row.get("lap"))
            lt = _as_float(row.get("lap_time_s"))
            if lap is None or lt is None:
                errors.append(f"Row {row_count}: invalid lap/lap_time_s")
                continue
            if lap <= 0:
                errors.append(f"Row {row_count}: lap must be > 0")
            if lt <= 0.0:
                errors.append(f"Row {row_count}: lap_time_s must be > 0")
            if lt < 40.0 or lt > 200.0:
                # Loose sanity range; avoids over-fitting to specific series.
                warnings.append(f"Row {row_count}: lap_time_s={lt} outside expected sanity range (40..200)")

            laps.append(lap)
            lap_times.append(lt)

            gap_raw = row.get("gap_to_car_ahead_s")
            gap = _as_float(gap_raw)
            if _raw_is_present(gap_raw) and gap is None:
                warnings.append(f"Row {row_count}: invalid gap_to_car_ahead_s")
            if gap is not None:
                if gap < 0.0:
                    warnings.append(f"Row {row_count}: gap_to_car_ahead_s is negative ({gap})")
                gaps.append(gap)

            if {"sector1_s", "sector2_s", "sector3_s"}.issubset(cols):
                s1_raw = row.get("sector1_s")
                s2_raw = row.get("sector2_s")
                s3_raw = row.get("sector3_s")
                s1 = _as_float(s1_raw)
                s2 = _as_float(s2_raw)
                s3 = _as_float(s3_raw)
                if _raw_is_present(s1_raw) and s1 is None:
                    warnings.append(f"Row {row_count}: invalid sector1_s")
                if _raw_is_present(s2_raw) and s2 is None:
                    warnings.append(f"Row {row_count}: invalid sector2_s")
                if _raw_is_present(s3_raw) and s3 is None:
                    warnings.append(f"Row {row_count}: invalid sector3_s")

                if s1 is not None and s2 is not None and s3 is not None and lt > 0:
                    if s1 <= 0.0 or s2 <= 0.0 or s3 <= 0.0:
                        warnings.append(f"Row {row_count}: sector times must be > 0")
                    else:
                        sector_ratios.append((s1 / lt, s2 / lt, s3 / lt))
                        # Loose check: in processed datasets, sector sums should resemble lap_time_s.
                        if abs((s1 + s2 + s3) - lt) > 5.0:
                            warnings.append(
                                f"Row {row_count}: sector sum ({s1 + s2 + s3}) differs from lap_time_s ({lt})"
                            )

            if "tyre_compound" in cols:
                tyre_compounds.append((row.get("tyre_compound") or "").strip().upper())

        if row_count == 0:
            errors.append("CSV has no data rows")
            return DatasetValidationResult(ok=False, errors=errors, warnings=warnings)

        if len(laps) >= 2:
            if sorted(laps) != laps:
                warnings.append("lap column is not monotonically increasing; loader will sort but provenance may be confusing")
            if len(set(laps)) != len(laps):
                warnings.append("duplicate lap values detected")

        # Placeholder-like patterns (warnings, not hard failures).
        if gaps:
            zero_frac = sum(1 for g in gaps if abs(g) <= 0.01) / max(1, len(gaps))
            if zero_frac >= 0.9:
                warnings.append("gap_to_car_ahead_s is mostly 0.0; likely placeholder coverage gap")

        if tyre_compounds:
            recent = tyre_compounds[-3:]
            if any(c in {"", "UNKNOWN"} for c in recent):
                warnings.append("tyre_compound is UNKNOWN/blank in recent rows; compound-dependent decisions should be downscoped")

        # Detect proportional sector placeholders: ratios nearly constant.
        if len(sector_ratios) >= 6:
            r1 = [r[0] for r in sector_ratios[-10:]]
            r2 = [r[1] for r in sector_ratios[-10:]]
            r3 = [r[2] for r in sector_ratios[-10:]]

            def _range(x: list[float]) -> float:
                return max(x) - min(x) if x else 0.0

                if _range(r1) <= 0.0015 and _range(r2) <= 0.003 and _range(r3) <= 0.004:
                    warnings.append("sector times appear proportional to lap_time_s (synthetic placeholders likely)")

        # Outlier detection (warning): large last-lap spike vs recent median.
        # This helps catch "bad decision snapshots" where the final lap is dominated by
        # missing context (traffic/yellow/incident) and can distort trust outputs.
        if len(lap_times) >= 8:
            window = lap_times[-8:]
            med = median(window)
            abs_devs = [abs(x - med) for x in window]
            mad = median(abs_devs)
            robust_sigma = max(1.4826 * mad, 0.15)
            last = window[-1]
            z = (last - med) / robust_sigma
            if z >= 6.0 and (last - med) >= 1.2:
                warnings.append(
                    f"lap_time_s outlier detected in recent window: last={round(last,3)} vs median={round(med,3)} (z~{round(z,1)})."
                )

    ok = not errors
    return DatasetValidationResult(ok=ok, errors=errors, warnings=warnings)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a processed MDCE CSV (schema + sanity checks).")
    p.add_argument("--input", required=True, help="Path to processed MDCE CSV")
    p.add_argument("--fail-on-warning", action="store_true", help="Treat warnings as errors")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_mdce_csv(args.input)

    for e in result.errors:
        print("ERROR:", e, file=sys.stderr)
    for w in result.warnings:
        print("WARN:", w, file=sys.stderr)

    if not result.ok:
        raise SystemExit(2)
    if args.fail_on_warning and result.warnings:
        raise SystemExit(3)
    print("OK")


if __name__ == "__main__":
    main()
