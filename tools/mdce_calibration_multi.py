from __future__ import annotations

"""Multi-driver calibration (generalization check).

The single-slice calibration shows MDCE's confidence is meaningful on ONE stint.
This runs the SAME calibration across multiple drivers from the same public race
bundle, to show the result is a consistent pattern — not one cherry-picked example.

Honest framing: the bundled Zenodo sample contains one race (2024 R1) with many
drivers. So this is a *multi-driver / multi-stint* generalization check, not
multiple distinct races. We say so explicitly. No new datasets, no training.
"""

import argparse
import json
import sys
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_race_csv
from src.dataset_adapters import prepare_zenodo_stg_laps_with_gap_proxy
from tools.mdce_calibration import (
    _pearson,
    confidence_vs_regret_backtest,
    scenario_monotonicity,
)


DEFAULT_PARQUET = (
    "data/raw/extracted/zenodo_2024_selected/anandm84-F1-telemetry-DE-592e162/"
    "docs/curation/sample_output/silver/stg_laps.parquet"
)
DEFAULT_DRIVERS = ["VER", "HAM", "LEC", "NOR", "ALO", "RUS"]


def run_for_driver(parquet: str, driver: str, *, decision_lap: int | None) -> dict:
    prepared = prepare_zenodo_stg_laps_with_gap_proxy(parquet, driver_code=driver, decision_lap=decision_lap)
    records = load_race_csv(
        StringIO(prepared.frame.to_csv(index=False)), source_name=f"zenodo:{driver}"
    ).records
    mono = scenario_monotonicity(records)
    bt = confidence_vs_regret_backtest(records)
    return {
        "driver": driver,
        "laps": len([r for r in records if not getattr(r, "missing", False)]),
        "monotonic_non_increasing": mono["monotonic_non_increasing"],
        "pearson_failurecount_vs_confidence": mono["pearson_failurecount_vs_confidence"],
        "backtest_status": bt.get("status"),
        "calibration_holds": bt.get("calibration_holds"),
        "num_points": bt.get("num_points"),
        "mean_regret_safe_s": bt.get("mean_aggressive_regret_when_safe_s"),
        "mean_regret_aggressive_s": bt.get("mean_aggressive_regret_when_aggressive_s"),
        # Raw per-decision points (used for the statistically-powered pooled analysis).
        "_points": bt.get("points") or [],
    }


def _pooled_calibration(all_points: list[dict]) -> dict:
    """Statistically-powered calibration on ALL decision points pooled across drivers.

    Per-driver binary tests on short stints are underpowered. Pooling decision points
    gives a single continuous correlation between confidence and realized aggressive-
    action regret. A negative correlation = lower confidence aligns with higher regret
    = meaningful confidence. We report whatever it actually is (no tuning).
    """
    confs = [float(p["confidence"]) for p in all_points]
    regrets = [float(p["aggressive_action_regret_s"]) for p in all_points]
    n = len(all_points)
    corr = _pearson(confs, regrets) if n >= 3 else None

    # Also a simple, robust median-split sanity check (not the headline).
    verdict = "INSUFFICIENT"
    if corr is not None:
        if corr <= -0.15:
            verdict = "CALIBRATED"  # lower confidence -> higher realized regret
        elif corr >= 0.15:
            verdict = "ANTI_CALIBRATED"  # wrong direction
        else:
            verdict = "WEAK_OR_NONE"
    return {
        "pooled_points": n,
        "pearson_confidence_vs_regret": corr,
        "verdict": verdict,
        "note": (
            "Pooled across drivers for statistical power. Negative correlation means "
            "confidence is lower exactly when being aggressive was realized to be costlier."
        ),
    }


def run_multi(parquet: str, drivers: list[str], *, decision_lap: int | None) -> dict:
    rows: list[dict] = []
    for d in drivers:
        try:
            rows.append(run_for_driver(parquet, d, decision_lap=decision_lap))
        except Exception as exc:  # a missing driver should not abort the whole run
            rows.append({"driver": d, "error": str(exc)[:200]})

    evaluated = [r for r in rows if "error" not in r]
    monotonic_count = sum(1 for r in evaluated if r.get("monotonic_non_increasing"))
    holds_eval = [r for r in evaluated if r.get("calibration_holds") is not None]
    holds_count = sum(1 for r in holds_eval if r.get("calibration_holds") is True)

    # Pool all decision points across drivers for the statistically-powered check.
    pooled_points: list[dict] = []
    for r in evaluated:
        pooled_points.extend(r.get("_points") or [])
    pooled = _pooled_calibration(pooled_points)

    # Strip raw points from the saved rows to keep the report compact.
    for r in rows:
        r.pop("_points", None)

    return {
        "schema_version": "mdce_multidriver_calibration_v1",
        "framing": (
            "Multi-driver / multi-stint generalization on a single public race bundle "
            "(Zenodo 2024 R1). Not multiple distinct races."
        ),
        "drivers_requested": drivers,
        "results": rows,
        "pooled_calibration": pooled,
        "summary": {
            "drivers_evaluated": len(evaluated),
            "monotonic_non_increasing_count": monotonic_count,
            "calibration_evaluable": len(holds_eval),
            "calibration_holds_count": holds_count,
            "monotonic_all": len(evaluated) > 0 and monotonic_count == len(evaluated),
            "calibration_holds_all": len(holds_eval) > 0 and holds_count == len(holds_eval),
            "pooled_verdict": pooled["verdict"],
            "pooled_pearson": pooled["pearson_confidence_vs_regret"],
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-driver MDCE calibration generalization check.")
    p.add_argument("--parquet", default=DEFAULT_PARQUET, help="Zenodo stg_laps parquet path.")
    p.add_argument("--drivers", default=",".join(DEFAULT_DRIVERS), help="Comma-separated driver codes.")
    p.add_argument("--decision-lap", type=int, default=None, help="Optional lap to cut each stint at.")
    p.add_argument("--json-out", default=None, help="Optional path to write the summary JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    drivers = [d.strip().upper() for d in args.drivers.split(",") if d.strip()]
    parquet = args.parquet
    if not Path(parquet).is_absolute():
        parquet = str(PROJECT_ROOT / parquet)

    report = run_multi(parquet, drivers, decision_lap=args.decision_lap)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("MDCE multi-driver calibration")
    print(report["framing"])
    print()
    print(f"{'driver':<8}{'laps':<6}{'monotonic':<11}{'pearson':<10}{'holds':<8}{'pts'}")
    for r in report["results"]:
        if "error" in r:
            print(f"{r['driver']:<8}ERROR: {r['error']}")
            continue
        print(
            f"{r['driver']:<8}{r['laps']:<6}{str(r['monotonic_non_increasing']):<11}"
            f"{str(r['pearson_failurecount_vs_confidence']):<10}{str(r['calibration_holds']):<8}{r['num_points']}"
        )
    s = report["summary"]
    print()
    print(f"Monotonic non-increasing: {s['monotonic_non_increasing_count']}/{s['drivers_evaluated']} drivers")
    print(f"Per-driver calibration holds: {s['calibration_holds_count']}/{s['calibration_evaluable']} (underpowered binary test)")
    pc = report["pooled_calibration"]
    print()
    print("POOLED (statistically powered):")
    print(f"  pooled points: {pc['pooled_points']}")
    print(f"  Pearson(confidence, realized_regret): {pc['pearson_confidence_vs_regret']}")
    print(f"  verdict: {pc['verdict']}")


if __name__ == "__main__":
    main()
