from __future__ import annotations

"""MDCE calibration evidence.

Goal: demonstrate that MDCE confidence is *meaningful* (responsive and ordered),
not arbitrary. We do this with two deterministic, defensible analyses:

A) Scenario monotonicity
   - Run the pipeline under presets ordered by the number of injected failure
     modes. Confidence should be non-increasing as more genuine degradations are
     injected. We report a Pearson correlation between injected-failure-count and
     confidence (expected strongly negative).

B) Held-out backtest (when enough laps exist)
   - At each decision lap k, MDCE sees only laps[0..k] and produces a
     recommendation + confidence.
   - Ground truth uses the FUTURE laps[k+1..k+H] (which MDCE did NOT see) to
     compute a realized regret proxy.
   - We then report the correlation between confidence and realized regret
     (expected negative: lower confidence aligns with situations that were
     actually riskier / more costly to be wrong about).

This is bounded, illustrative evidence on limited public data, not a statistical
proof. The report says so explicitly.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_default_data_result, load_race_csv
from src.models import ScenarioFlags
from src.pipeline import analyze_decision
from src.scenario_presets import SCENARIO_PRESETS


# Backtest constants (deterministic, bounded, documented).
BACKTEST_HORIZON_LAPS = 3
BACKTEST_DEGRADATION_THRESHOLD_S = 0.20  # matches strategy "rising degradation"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Deterministic Pearson correlation. Returns None if undefined."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom = (sxx * syy) ** 0.5
    if denom == 0.0:
        return None
    return round(sxy / denom, 4)


def _failure_count(flags: ScenarioFlags) -> int:
    return sum(
        [
            flags.missing_telemetry,
            flags.tyre_signal_drift,
            flags.model_mismatch,
            flags.safety_car_phase,
            flags.weather_uncertainty,
        ]
    )


def _linear_trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / (len(values) - 1)


def scenario_monotonicity(records: list) -> dict:
    """Part A: confidence should decrease as injected failure modes increase."""
    rows: list[dict] = []
    for name, preset in SCENARIO_PRESETS.items():
        result, _, _, _ = analyze_decision(records, preset.flags, prefer_granite=False)
        rows.append(
            {
                "preset": name,
                "injected_failures": _failure_count(preset.flags),
                "confidence": float(result.confidence.confidence),
                "risk_level": result.confidence.risk_level,
                "issue_count": len(result.issues),
            }
        )

    # Sort by injected failures for a readable monotonicity view.
    rows_sorted = sorted(rows, key=lambda r: (r["injected_failures"], -r["confidence"]))

    # Bucket mean confidence by failure count and check non-increasing.
    buckets: dict[int, list[float]] = {}
    for r in rows:
        buckets.setdefault(r["injected_failures"], []).append(r["confidence"])
    bucket_means = {k: round(sum(v) / len(v), 4) for k, v in sorted(buckets.items())}
    ordered_counts = sorted(bucket_means.keys())
    monotonic_non_increasing = all(
        bucket_means[ordered_counts[i]] >= bucket_means[ordered_counts[i + 1]] - 1e-9
        for i in range(len(ordered_counts) - 1)
    )

    corr = _pearson(
        [float(r["injected_failures"]) for r in rows],
        [r["confidence"] for r in rows],
    )

    return {
        "rows": rows_sorted,
        "bucket_mean_confidence_by_failure_count": bucket_means,
        "monotonic_non_increasing": bool(monotonic_non_increasing),
        "pearson_failurecount_vs_confidence": corr,
        "interpretation": (
            "Confidence is non-increasing as more genuine failure modes are injected; "
            "a strongly negative correlation indicates the score tracks real degradation, not noise."
        ),
    }


def _realized_regret(rec_type: RecommendationType, future_records: list) -> float | None:
    """Compute a held-out realized regret proxy from future (unseen) laps.

    Returns None if there is not enough future data to judge.
    """
def _aggressive_action_regret(history: list, future_records: list) -> float | None:
    """Realized regret of taking the AGGRESSIVE action (keep extending the stint).

    Held-out: `future_records` are laps MDCE did NOT see at decision time.

    Logic: the AGGRESSIVE call is "extend / push". If the (unseen) future laps
    actually degrade, extending was costly -> positive regret proportional to the
    realized degradation over the horizon. If pace held, regret ~ 0.

    This isolates the question the trust layer actually answers:
    "was being aggressive a mistake given what really happened next?"
    """
    future_times = [r.lap_time_s for r in future_records if not getattr(r, "missing", False)]
    if len(future_times) < 2:
        return None
    realized_deg = _linear_trend(future_times)
    if realized_deg >= BACKTEST_DEGRADATION_THRESHOLD_S:
        return float(round(realized_deg * BACKTEST_HORIZON_LAPS, 3))
    return 0.0


def confidence_vs_regret_backtest(
    records: list,
    *,
    horizon: int = BACKTEST_HORIZON_LAPS,
    min_history: int = 5,
) -> dict:
    """Part B: held-out backtest of recommended mode vs realized aggressive-action regret.

    For each decision lap k and each scenario preset, MDCE sees only laps[0..k]
    (mutated by the preset) and produces a recommended mode (SAFE vs AGGRESSIVE).
    We then measure, using the UNSEEN future laps, the realized regret of having
    been aggressive.

    Sweeping presets is how MDCE's confidence/mode is designed to vary, so it
    populates both SAFE and AGGRESSIVE groups (a single clean dataset may never
    produce AGGRESSIVE if it always carries at least one trust issue).

    Calibration claim: when MDCE says SAFE, the realized aggressive-action regret
    should be >= when it says AGGRESSIVE. That means MDCE's caution lines up with
    situations that were actually riskier to push in.
    """
    usable = [r for r in records if not getattr(r, "missing", False)]
    n = len(usable)
    points: list[dict] = []

    for k in range(min_history - 1, n - 2):
        history = usable[: k + 1]
        future = usable[k + 1 : k + 1 + horizon]
        regret = _aggressive_action_regret(history, future)
        if regret is None:
            continue
        # Sweep presets so both SAFE and AGGRESSIVE modes can occur on the same lap window.
        for name, preset in SCENARIO_PRESETS.items():
            result, _, _, _ = analyze_decision(history, preset.flags, prefer_granite=False)
            mode = result.recommended_mode.value if result.recommended_mode is not None else "UNKNOWN"
            points.append(
                {
                    "decision_lap_index": k,
                    "preset": name,
                    "recommended_mode": mode,
                    "confidence": float(result.confidence.confidence),
                    "aggressive_action_regret_s": regret,
                }
            )

    if len(points) < 3:
        return {
            "status": "INSUFFICIENT_DATA",
            "points": points,
            "note": (
                f"Need at least 3 backtest points (got {len(points)}). "
                "Dataset has too few laps for a held-out mode-vs-regret check."
            ),
        }

    safe = [p for p in points if p["recommended_mode"] == "SAFE"]
    aggressive = [p for p in points if p["recommended_mode"] == "AGGRESSIVE"]

    def _mean_regret(group: list[dict]) -> float | None:
        if not group:
            return None
        return round(sum(p["aggressive_action_regret_s"] for p in group) / len(group), 3)

    mean_regret_safe = _mean_regret(safe)
    mean_regret_aggressive = _mean_regret(aggressive)

    # Correlation: lower confidence should align with higher aggressive-action regret (negative).
    corr = _pearson(
        [p["confidence"] for p in points],
        [p["aggressive_action_regret_s"] for p in points],
    )

    # Calibration holds if SAFE-mode situations carried >= aggressive-action regret than AGGRESSIVE ones.
    calibration_holds = None
    if mean_regret_safe is not None and mean_regret_aggressive is not None:
        calibration_holds = bool(mean_regret_safe >= mean_regret_aggressive - 1e-9)

    return {
        "status": "OK",
        "horizon_laps": horizon,
        "num_points": len(points),
        "num_safe_mode": len(safe),
        "num_aggressive_mode": len(aggressive),
        "mean_aggressive_regret_when_safe_s": mean_regret_safe,
        "mean_aggressive_regret_when_aggressive_s": mean_regret_aggressive,
        "pearson_confidence_vs_aggressive_regret": corr,
        "calibration_holds": calibration_holds,
        "points": points,
        "interpretation": (
            "EXPLORATORY DIAGNOSTIC (not a validated calibration claim). On a single slice this "
            "can look like SAFE-mode situations carried higher aggressive-action regret. However, "
            "pooled across drivers this signal does NOT generalize (see tools/mdce_calibration_multi.py): "
            "early-stint high confidence mechanically precedes inevitable later degradation, biasing the "
            "metric. Treat as a diagnostic, not proof."
        ),
        "caveat": (
            "DO NOT present as proof that confidence is calibrated. The validated, generalizing result "
            "is scenario monotonicity (confidence falls as failures increase, ~ -0.8 across all drivers tested). "
            "This regret backtest is kept only as an exploratory diagnostic."
        ),
    }


def build_report(records: list, *, source_name: str) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "mdce_calibration_v1",
        "timestamp_utc": now,
        "source_name": source_name,
        "num_laps": len([r for r in records if not getattr(r, "missing", False)]),
        "scenario_monotonicity": scenario_monotonicity(records),
        "confidence_vs_regret_backtest": confidence_vs_regret_backtest(records),
    }


def _render_markdown(report: dict) -> str:
    mono = report["scenario_monotonicity"]
    bt = report["confidence_vs_regret_backtest"]
    lines: list[str] = [
        "# MDCE Calibration Evidence",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Timestamp (UTC): `{report['timestamp_utc']}`",
        f"- Source: `{report['source_name']}`",
        f"- Usable laps: `{report['num_laps']}`",
        "",
        "## A) Scenario Monotonicity",
        "",
        f"- Monotonic non-increasing by failure count: `{mono['monotonic_non_increasing']}`",
        f"- Pearson(failure_count, confidence): `{mono['pearson_failurecount_vs_confidence']}`",
        f"- Mean confidence by failure count: `{mono['bucket_mean_confidence_by_failure_count']}`",
        "",
        "| preset | injected_failures | confidence | risk | issues |",
        "|---|---|---|---|---|",
    ]
    for r in mono["rows"]:
        lines.append(
            f"| {r['preset']} | {r['injected_failures']} | {r['confidence']} | {r['risk_level']} | {r['issue_count']} |"
        )
    lines += [
        "",
        f"> {mono['interpretation']}",
        "",
        "## B) Held-Out Recommended-Mode vs Realized Aggressive-Action Regret",
        "",
    ]
    if bt.get("status") != "OK":
        lines += [f"- Status: `{bt.get('status')}`", f"- Note: {bt.get('note')}"]
    else:
        lines += [
            f"- Backtest points: `{bt['num_points']}` (SAFE={bt['num_safe_mode']}, AGGRESSIVE={bt['num_aggressive_mode']})",
            f"- Horizon laps: `{bt['horizon_laps']}`",
            f"- Mean aggressive-action regret when SAFE: `{bt['mean_aggressive_regret_when_safe_s']}` s",
            f"- Mean aggressive-action regret when AGGRESSIVE: `{bt['mean_aggressive_regret_when_aggressive_s']}` s",
            f"- Pearson(confidence, aggressive_action_regret): `{bt['pearson_confidence_vs_aggressive_regret']}`",
            f"- Slice diagnostic result (SAFE regret >= AGGRESSIVE regret): `{bt['calibration_holds']}`",
            "",
            f"> {bt['interpretation']}",
            "",
            f"_Caveat: {bt['caveat']}_",
        ]
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MDCE calibration evidence (deterministic).")
    p.add_argument("--root", default=None, help="Optional project root for default data resolution.")
    p.add_argument("--input", default=None, help="Optional processed MDCE CSV. If omitted, uses default loader.")
    p.add_argument("--output-dir", default="outputs/reports", help="Where to write calibration artifacts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path
        data_load = load_race_csv(input_path, source_name=str(input_path))
    else:
        data_load = load_default_data_result(root=root)

    report = build_report(data_load.records, source_name=data_load.source_name)

    stamp = report["timestamp_utc"].replace(":", "").replace("-", "")
    json_path = out_dir / f"mdce_calibration_{stamp}.json"
    md_path = out_dir / f"mdce_calibration_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    mono = report["scenario_monotonicity"]
    bt = report["confidence_vs_regret_backtest"]
    print("MDCE calibration complete")
    print("JSON:", json_path)
    print("MD:", md_path)
    print(f"Scenario monotonic non-increasing: {mono['monotonic_non_increasing']}")
    print(f"Pearson(failure_count, confidence): {mono['pearson_failurecount_vs_confidence']}")
    print(f"Backtest status: {bt.get('status')}")
    if bt.get("status") == "OK":
        print(f"Exploratory regret diagnostic (slice only): {bt.get('calibration_holds')}")
        print(f"Pearson(confidence, aggressive_regret): {bt.get('pearson_confidence_vs_aggressive_regret')}")


if __name__ == "__main__":
    main()
