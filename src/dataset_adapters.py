from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    metadata: dict


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timedelta_seconds(series: pd.Series) -> pd.Series:
    """Convert a pandas timedelta-like series to seconds (float).

    FastF1 exposes lap/sector times as timedeltas. We keep a small helper here so
    adapters stay consistent and don't leak timedeltas into the processed CSV.
    """

    try:
        td = pd.to_timedelta(series, errors="coerce")
    except Exception:
        td = series
    return td.dt.total_seconds()  # type: ignore[union-attr]


def _select_latest_event(frame: pd.DataFrame) -> tuple[int, int]:
    events = frame[["year", "round"]].dropna().drop_duplicates().sort_values(["year", "round"])
    if events.empty:
        raise ValueError("Dataset does not contain usable year/round values.")
    latest = events.iloc[-1]
    return int(latest["year"]), int(latest["round"])


def _select_best_driver(frame: pd.DataFrame) -> str:
    grouped = (
        frame.dropna(subset=["code"])
        .groupby("code")
        .agg(
            valid_laps=("lap", "count"),
            best_finish_position=("position_y", "min"),
        )
        .reset_index()
        .sort_values(["valid_laps", "best_finish_position", "code"], ascending=[False, True, True])
    )
    if grouped.empty:
        raise ValueError("Dataset does not contain usable driver code values.")
    return str(grouped.iloc[0]["code"])


def _select_first_stint_decision_lap(frame: pd.DataFrame) -> int:
    ordered = frame.sort_values("lap").reset_index(drop=True)
    tyre_age = pd.to_numeric(ordered["TyreLife"], errors="coerce")
    compound = ordered["Compound"].fillna("UNKNOWN").astype(str)
    laps = pd.to_numeric(ordered["lap"], errors="coerce")

    for index in range(len(ordered) - 1):
        current_age = tyre_age.iloc[index]
        next_age = tyre_age.iloc[index + 1]
        current_compound = compound.iloc[index]
        next_compound = compound.iloc[index + 1]
        if pd.notna(current_age) and pd.notna(next_age) and next_age < current_age:
            return int(laps.iloc[index])
        if current_compound != next_compound:
            return int(laps.iloc[index])

    return int(laps.max())


def prepare_kaggle_weather_tyre(
    input_path: str | Path,
    *,
    year: int | None = None,
    round_number: int | None = None,
    driver_code: str | None = None,
    decision_lap: int | None = None,
) -> PreparedDataset:
    """Map the Kaggle weather/tyre parquet dataset into an MDCE-ready CSV frame."""
    source_path = Path(input_path)
    frame = pd.read_parquet(source_path)

    required = {
        "raceId",
        "driverId",
        "lap",
        "milliseconds",
        "year",
        "round",
        "name",
        "code",
        "Compound",
        "TyreLife",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "WindSpeed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Kaggle weather/tyre dataset missing columns: {', '.join(missing)}")

    working = frame.copy()
    working = working.dropna(subset=["lap", "milliseconds", "year", "round", "code"])
    working = working[working["milliseconds"] > 0]

    if year is None or round_number is None:
        selected_year, selected_round = _select_latest_event(working)
        year = selected_year if year is None else year
        round_number = selected_round if round_number is None else round_number

    selected = working[(working["year"] == year) & (working["round"] == round_number)].copy()
    if selected.empty:
        raise ValueError(f"No rows found for year={year}, round={round_number}.")

    if driver_code is None:
        driver_code = _select_best_driver(selected)
    driver_code = driver_code.upper()

    selected = selected[selected["code"].astype(str).str.upper() == driver_code].copy()
    if selected.empty:
        raise ValueError(f"No rows found for driver_code={driver_code} in year={year}, round={round_number}.")

    selected = selected.sort_values("lap")
    if decision_lap is None:
        decision_lap = _select_first_stint_decision_lap(selected)
    selected = selected[pd.to_numeric(selected["lap"], errors="coerce") <= decision_lap].copy()
    if selected.empty:
        raise ValueError(f"No rows found at or before decision_lap={decision_lap}.")

    output = pd.DataFrame(
        {
            "lap": pd.to_numeric(selected["lap"], errors="coerce").astype("Int64"),
            "lap_time_s": pd.to_numeric(selected["milliseconds"], errors="coerce") / 1000.0,
            "tyre_compound": selected["Compound"].fillna("UNKNOWN").astype(str).str.upper(),
            "tyre_age": pd.to_numeric(selected["TyreLife"], errors="coerce"),
            "rainfall": pd.to_numeric(selected["Rainfall"], errors="coerce"),
            "track_temp_c": pd.to_numeric(selected["TrackTemp"], errors="coerce"),
            "air_temp_c": pd.to_numeric(selected["AirTemp"], errors="coerce"),
            "humidity": pd.to_numeric(selected["Humidity"], errors="coerce"),
            "pressure": pd.to_numeric(selected["Pressure"], errors="coerce"),
            "wind_speed": pd.to_numeric(selected["WindSpeed"], errors="coerce"),
            "source_year": pd.to_numeric(selected["year"], errors="coerce").astype("Int64"),
            "source_round": pd.to_numeric(selected["round"], errors="coerce").astype("Int64"),
            "source_race_id": pd.to_numeric(selected["raceId"], errors="coerce").astype("Int64"),
            "source_circuit": selected["name"].fillna("UNKNOWN").astype(str),
            "source_driver_code": selected["code"].fillna(driver_code).astype(str).str.upper(),
        }
    )
    output = output.dropna(subset=["lap", "lap_time_s"]).reset_index(drop=True)

    race_name = str(selected["name"].dropna().iloc[0]) if selected["name"].notna().any() else "UNKNOWN"
    race_id = int(selected["raceId"].dropna().iloc[0])
    metadata = {
        "source": "Kaggle Formula 1 Dataset With Weather and Tyre Features",
        "source_url": "https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features",
        "license_spdx": "CC-BY-NC-4.0",
        # Kaggle dataset page exposes its license in embedded schema.org metadata.
        # Still re-check upstream before final submission and ensure Kaggle terms are followed.
        "license_note": "Kaggle dataset license: CC BY-NC 4.0 (per Kaggle dataset page metadata). Confirm Kaggle terms before final submission.",
        "input_path": str(source_path),
        "selected_year": int(year),
        "selected_round": int(round_number),
        "selected_race_id": race_id,
        "selected_circuit": race_name,
        "selected_driver_code": driver_code,
        "selected_decision_lap": int(decision_lap),
        "rows": int(len(output)),
        "real_fields_used": [
            "lap",
            "milliseconds",
            "Compound",
            "TyreLife",
            "Rainfall",
            "TrackTemp",
            "AirTemp",
            "Humidity",
            "Pressure",
            "WindSpeed",
        ],
        "known_limitations": [
            "No sector times in this dataset; MDCE derives sector proxies from lap time.",
            "No race-control track-status feed in this dataset; MDCE defaults track status to NORMAL unless another source is joined.",
            "No direct gap-to-car-ahead field in this dataset; MDCE uses a documented neutral fallback.",
            "No private tyre temperature telemetry in this dataset; MDCE uses a public track-temperature-based proxy.",
            "No high-frequency speed trace in this dataset; MDCE uses lap-time stability as a speed-consistency proxy.",
            "The prepared CSV is cut at a decision lap, so the app simulates a live decision point instead of using future laps after the decision.",
        ],
    }
    return PreparedDataset(frame=output, metadata=metadata)


def prepare_fastf1_race(
    *,
    year: int,
    round_number: int,
    driver_code: str | None = None,
    decision_lap: int | None = None,
    cache_dir: str | Path | None = None,
) -> PreparedDataset:
    """Prepare a FastF1 race session into an MDCE-ready CSV frame.

    Primary goal: bring in real per-lap + sector timing (and tyre stint fields)
    so MDCE doesn't have to synthesize sector placeholders.

    Notes:
    - FastF1 requires internet on first load to populate cache.
    - Gap-to-car-ahead requires multi-car ordering/context; this adapter does not
      compute it yet (MDCE will fall back and flag the coverage gap explicitly).
    """

    import fastf1

    cache_path = Path(cache_dir) if cache_dir is not None else (_default_project_root() / "data" / "raw" / "fastf1_cache")
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))

    session = fastf1.get_session(year, round_number, "R")
    # Ensure laps are loaded; other streams can remain off for speed.
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = session.laps
    if laps is None or len(laps) == 0:
        raise ValueError(f"FastF1 returned no laps for year={year}, round={round_number}.")

    if driver_code is None:
        # Prefer the driver with the most laps (simple, stable default).
        if "Driver" not in laps.columns:
            raise ValueError("FastF1 laps missing 'Driver' column; cannot select a default driver.")
        driver_code = str(laps["Driver"].value_counts().idxmax()).upper()
    driver_code = str(driver_code).upper()

    selected = laps
    # FastF1 offers pick_driver() on its Laps object, but fall back to filtering
    # for robustness if the method isn't available.
    try:
        selected = laps.pick_driver(driver_code)
    except Exception:
        if "Driver" not in selected.columns:
            raise ValueError("FastF1 laps missing 'Driver' column; cannot filter by driver.")
        selected = selected[selected["Driver"].astype(str).str.upper() == driver_code]

    if selected is None or len(selected) == 0:
        raise ValueError(f"No FastF1 laps found for driver_code={driver_code} in year={year}, round={round_number}.")

    # Ensure ordered laps.
    if "LapNumber" not in selected.columns:
        raise ValueError("FastF1 laps missing 'LapNumber' column.")
    selected = selected.sort_values("LapNumber")

    # Choose a decision lap. If we can identify a first-stint boundary (compound change
    # or TyreLife reset), use that. Otherwise, fall back to max lap.
    if decision_lap is None:
        try:
            decision_lap = _select_first_stint_decision_lap(
                pd.DataFrame(
                    {
                        "lap": pd.to_numeric(selected["LapNumber"], errors="coerce"),
                        "TyreLife": pd.to_numeric(selected.get("TyreLife"), errors="coerce"),
                        "Compound": selected.get("Compound"),
                    }
                ).dropna(subset=["lap"])
            )
        except Exception:
            decision_lap = int(pd.to_numeric(selected["LapNumber"], errors="coerce").max())

    decision_lap = int(decision_lap)
    selected = selected[pd.to_numeric(selected["LapNumber"], errors="coerce") <= decision_lap].copy()
    if selected.empty:
        raise ValueError(f"No rows found at or before decision_lap={decision_lap}.")

    # Build MDCE-ready frame using the fields FastF1 provides.
    out = pd.DataFrame(
        {
            "lap": pd.to_numeric(selected["LapNumber"], errors="coerce").astype("Int64"),
            "lap_time_s": _timedelta_seconds(selected.get("LapTime")),
            "sector1_s": _timedelta_seconds(selected.get("Sector1Time")),
            "sector2_s": _timedelta_seconds(selected.get("Sector2Time")),
            "sector3_s": _timedelta_seconds(selected.get("Sector3Time")),
            "tyre_compound": selected.get("Compound").fillna("UNKNOWN").astype(str).str.upper(),
            "tyre_age": pd.to_numeric(selected.get("TyreLife"), errors="coerce"),
            "source_year": int(year),
            "source_round": int(round_number),
            "source_driver_code": driver_code,
        }
    )

    out = out.dropna(subset=["lap", "lap_time_s"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("FastF1 adapter produced no usable rows after parsing lap and lap_time_s.")

    # Best-effort track status mapping. FastF1 track status encoding is not always
    # straightforward, so we avoid strong claims here and keep it optional.
    if "TrackStatus" in selected.columns:
        raw = selected["TrackStatus"].astype(str)
        def _map_status(s: str) -> str:
            ss = str(s)
            if "4" in ss:
                return "SC"
            if "6" in ss:
                return "VSC"
            return "NORMAL"
        out["track_status"] = raw.map(_map_status)

    metadata = {
        "source": "FastF1 public timing API",
        "source_url": "https://docs.fastf1.dev/",
        "license_spdx": "NOASSERTION",
        "license_note": "FastF1 fetches public historical timing data. Verify event/data usage terms before submission.",
        "selected_year": int(year),
        "selected_round": int(round_number),
        "selected_driver_code": driver_code,
        "selected_decision_lap": int(decision_lap),
        "rows": int(len(out)),
        "real_fields_used": [
            "lap",
            "lap_time_s",
            "sector1_s",
            "sector2_s",
            "sector3_s",
            "tyre_compound",
            "tyre_age",
        ],
        "known_limitations": [
            "gap_to_car_ahead_s is not computed by this adapter yet; MDCE will fall back and flag the coverage gap.",
            "track_status is best-effort mapped when available and should not be treated as an official FIA feed.",
            "This dataset is cut at a decision lap to simulate a live decision point.",
        ],
        "cache_dir": str(cache_path),
    }
    return PreparedDataset(frame=out, metadata=metadata)


def prepare_zenodo_stg_laps(
    input_path: str | Path,
    *,
    year: int | None = None,
    round_number: int | None = None,
    session: str | None = None,
    driver_code: str | None = None,
    decision_lap: int | None = None,
) -> PreparedDataset:
    """Prepare a Zenodo-derived lap table (sample output) into an MDCE-ready CSV.

    This adapter is intentionally offline-friendly: the repo already contains a
    small curated extract under `data/raw/extracted/zenodo_2024_selected/.../stg_laps.parquet`.

    What you get:
    - real per-lap sector times (ms) converted to seconds
    - tyre compound (when present)

    What you don't get from this table:
    - gap_to_car_ahead_s (requires race order/context)
    - official track status feed
    """

    p = Path(input_path)
    if not p.exists():
        raise ValueError(f"Zenodo laps input not found: {p}")
    frame = pd.read_parquet(p)

    required = {"race_year", "race_round", "session", "driver", "lap_number", "lap_time_ms"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Zenodo stg_laps missing columns: {', '.join(missing)}")

    working = frame.copy()
    working["race_year"] = pd.to_numeric(working["race_year"], errors="coerce")
    working["race_round"] = pd.to_numeric(working["race_round"], errors="coerce")
    working["lap_number"] = pd.to_numeric(working["lap_number"], errors="coerce")
    working["lap_time_ms"] = pd.to_numeric(working["lap_time_ms"], errors="coerce")
    working = working.dropna(subset=["race_year", "race_round", "lap_number", "lap_time_ms"]).copy()
    working = working[working["lap_time_ms"] > 0].copy()

    if year is None or round_number is None:
        latest = working[["race_year", "race_round"]].drop_duplicates().sort_values(["race_year", "race_round"]).iloc[-1]
        year = int(latest["race_year"]) if year is None else int(year)
        round_number = int(latest["race_round"]) if round_number is None else int(round_number)

    selected = working[(working["race_year"] == year) & (working["race_round"] == round_number)].copy()
    if selected.empty:
        raise ValueError(f"No rows found for year={year}, round={round_number} in {p.name}.")

    if session is not None:
        sess = str(session).strip().upper()
        selected = selected[selected["session"].astype(str).str.upper() == sess].copy()
        if selected.empty:
            raise ValueError(f"No rows found for session={sess} in year={year}, round={round_number}.")
        session = sess
    else:
        # Prefer Race session when present; else pick the latest session by name.
        sessions = sorted({str(s).upper() for s in selected["session"].dropna().astype(str)})
        session = "R" if "R" in sessions else (sessions[-1] if sessions else "UNKNOWN")
        selected = selected[selected["session"].astype(str).str.upper() == session].copy()

    if selected.empty:
        raise ValueError("No usable rows after filtering.")

    if driver_code is None:
        driver_code = str(selected["driver"].value_counts().idxmax()).upper()
    driver_code = str(driver_code).upper()
    selected = selected[selected["driver"].astype(str).str.upper() == driver_code].copy()
    if selected.empty:
        raise ValueError(f"No rows found for driver={driver_code} in year={year}, round={round_number}, session={session}.")

    selected = selected.sort_values("lap_number")

    if decision_lap is None:
        compound = selected.get("tire_compound") if "tire_compound" in selected.columns else selected.get("tyre_compound")
        decision_lap = _select_first_stint_decision_lap(
            pd.DataFrame(
                {
                    "lap": selected["lap_number"],
                    "TyreLife": pd.Series([pd.NA] * len(selected)),
                    "Compound": (compound.fillna("UNKNOWN") if compound is not None else "UNKNOWN"),
                }
            )
        )

    decision_lap = int(decision_lap)
    selected = selected[selected["lap_number"] <= decision_lap].copy()
    if selected.empty:
        raise ValueError(f"No rows found at or before decision_lap={decision_lap}.")

    out = pd.DataFrame(
        {
            "lap": selected["lap_number"].astype("Int64"),
            "lap_time_s": selected["lap_time_ms"] / 1000.0,
            "sector1_s": pd.to_numeric(selected.get("sector1_ms"), errors="coerce") / 1000.0,
            "sector2_s": pd.to_numeric(selected.get("sector2_ms"), errors="coerce") / 1000.0,
            "sector3_s": pd.to_numeric(selected.get("sector3_ms"), errors="coerce") / 1000.0,
            "tyre_compound": (
                selected.get("tire_compound", selected.get("tyre_compound"))
                .fillna("UNKNOWN")
                .astype(str)
                .str.upper()
            ),
            # Not available in this source table; keep explicit placeholder column so
            # the loader doesn't have to inject one.
            "gap_to_car_ahead_s": 0.0,
            "source_year": int(year),
            "source_round": int(round_number),
            "source_driver_code": driver_code,
            "source_session": session,
        }
    )
    out = out.dropna(subset=["lap", "lap_time_s"]).reset_index(drop=True)

    metadata = {
        "source": "Zenodo-derived curated timing tables (sample output)",
        "source_url": "https://zenodo.org/records/20061496",
        "license_spdx": "CC-BY-4.0",
        "license_note": "Derived data tables under CC-BY-4.0 per bundled LICENSE-DATA.md; attribution required.",
        "input_path": str(p),
        "selected_year": int(year),
        "selected_round": int(round_number),
        "selected_session": session,
        "selected_driver_code": driver_code,
        "selected_decision_lap": int(decision_lap),
        "rows": int(len(out)),
        "real_fields_used": [
            "lap_number",
            "lap_time_ms",
            "sector1_ms",
            "sector2_ms",
            "sector3_ms",
            "tire_compound",
        ],
        "known_limitations": [
            "gap_to_car_ahead_s is not present in this table; MDCE will fall back and flag the coverage gap.",
            "track_status feed is not present; MDCE defaults track_status to NORMAL unless joined from another source.",
        ],
    }
    return PreparedDataset(frame=out, metadata=metadata)


def prepare_zenodo_stg_laps_with_gap_proxy(
    input_path: str | Path,
    *,
    year: int | None = None,
    round_number: int | None = None,
    session: str | None = None,
    driver_code: str | None = None,
    decision_lap: int | None = None,
) -> PreparedDataset:
    """Same as `prepare_zenodo_stg_laps`, but adds a clearly-labeled gap proxy.

    Gap proxy method:
    - Build cumulative lap time per driver (sum of lap_time_s).
    - At each lap, rank drivers by cumulative time and compute the gap to the next
      faster cumulative time ("car ahead" in this proxy ordering).

    This is not an official timing feed gap; it is a derived approximation and must be
    treated as such in claims.
    """

    base = prepare_zenodo_stg_laps(
        input_path,
        year=year,
        round_number=round_number,
        session=session,
        driver_code=driver_code,
        decision_lap=decision_lap,
    )

    p = Path(input_path)
    raw = pd.read_parquet(p)
    # Reproduce the same selection logic to compute gaps using multi-driver context.
    working = raw.copy()
    for col in ("race_year", "race_round", "lap_number", "lap_time_ms"):
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=["race_year", "race_round", "lap_number", "lap_time_ms"]).copy()
    working = working[working["lap_time_ms"] > 0].copy()

    sel_year = int(base.metadata.get("selected_year"))
    sel_round = int(base.metadata.get("selected_round"))
    sel_sess = str(base.metadata.get("selected_session"))
    sel_lap = int(base.metadata.get("selected_decision_lap"))
    focus_driver = str(base.metadata.get("selected_driver_code")).upper()

    working = working[(working["race_year"] == sel_year) & (working["race_round"] == sel_round)].copy()
    working = working[working["session"].astype(str).str.upper() == sel_sess].copy()
    working["driver"] = working["driver"].astype(str).str.upper()
    working["lap_time_s"] = working["lap_time_ms"] / 1000.0
    working = working.sort_values(["driver", "lap_number"])
    working = working[working["lap_number"] <= sel_lap].copy()

    # Compute cumulative times through each lap for each driver.
    working["cumulative_time_s"] = working.groupby("driver")["lap_time_s"].cumsum()

    # For each lap_number, rank drivers by cumulative_time_s.
    gaps = {}
    for lap_num, g in working.groupby("lap_number"):
        g2 = g.sort_values("cumulative_time_s")
        # Map driver -> gap to car ahead (next faster in cumulative time ordering).
        times = list(g2["cumulative_time_s"].astype(float))
        drivers = list(g2["driver"].astype(str))
        for idx, d in enumerate(drivers):
            if idx == 0:
                gap = 0.0
            else:
                gap = float(times[idx] - times[idx - 1])
            gaps[(d, int(lap_num))] = max(0.0, gap)

    out = base.frame.copy()
    # Apply the proxy gaps for the selected driver.
    out["gap_to_car_ahead_s"] = [gaps.get((focus_driver, int(l)), 0.0) for l in out["lap"].astype(int).tolist()]

    meta = dict(base.metadata)
    meta["gap_to_car_ahead_method"] = "cumulative_lap_time_proxy"
    meta["known_limitations"] = list(meta.get("known_limitations") or []) + [
        "gap_to_car_ahead_s is a derived proxy computed from cumulative lap times across drivers; it is not an official timing-feed gap.",
    ]
    return PreparedDataset(frame=out, metadata=meta)
