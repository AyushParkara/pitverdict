from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_adapters import (
    prepare_fastf1_race,
    prepare_kaggle_weather_tyre,
    prepare_zenodo_stg_laps,
    prepare_zenodo_stg_laps_with_gap_proxy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a real public F1 dataset for MDCE.")
    parser.add_argument(
        "--source",
        choices=["kaggle-weather-tyre", "fastf1", "zenodo-stg-laps", "zenodo-stg-laps-gap-proxy"],
        default="kaggle-weather-tyre",
        help="Dataset adapter to use.",
    )
    parser.add_argument(
        "--input",
        default="data/raw/extracted/kaggle_naven_weather_tyre/f1_all.parquet",
        help="Input raw dataset path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/mdce_kaggle_weather_tyre_latest.csv",
        help="Output processed MDCE CSV path.",
    )
    parser.add_argument("--year", type=int, default=None, help="Optional F1 season year to select.")
    parser.add_argument("--round", dest="round_number", type=int, default=None, help="Optional race round to select.")
    parser.add_argument("--driver", dest="driver_code", default=None, help="Optional driver code to select.")
    parser.add_argument("--decision-lap", type=int, default=None, help="Optional lap to cut the data at.")
    parser.add_argument(
        "--fastf1-cache-dir",
        default=None,
        help="Optional FastF1 cache directory (only used when --source fastf1).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Optional session code for sources that support it (e.g. R/FP1/FP2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "kaggle-weather-tyre":
        prepared = prepare_kaggle_weather_tyre(
            args.input,
            year=args.year,
            round_number=args.round_number,
            driver_code=args.driver_code,
            decision_lap=args.decision_lap,
        )
    elif args.source == "fastf1":
        if args.year is None or args.round_number is None:
            raise SystemExit("--source fastf1 requires --year and --round")
        prepared = prepare_fastf1_race(
            year=int(args.year),
            round_number=int(args.round_number),
            driver_code=args.driver_code,
            decision_lap=args.decision_lap,
            cache_dir=args.fastf1_cache_dir,
        )
    elif args.source == "zenodo-stg-laps":
        # Offline-friendly curated lap table with sector times.
        prepared = prepare_zenodo_stg_laps(
            args.input,
            year=args.year,
            round_number=args.round_number,
            session=args.session,
            driver_code=args.driver_code,
            decision_lap=args.decision_lap,
        )
    elif args.source == "zenodo-stg-laps-gap-proxy":
        prepared = prepare_zenodo_stg_laps_with_gap_proxy(
            args.input,
            year=args.year,
            round_number=args.round_number,
            session=args.session,
            driver_code=args.driver_code,
            decision_lap=args.decision_lap,
        )
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    prepared.frame.to_csv(output_path, index=False)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(prepared.metadata, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Wrote {metadata_path}")
    print(json.dumps(prepared.metadata, indent=2))


if __name__ == "__main__":
    main()
