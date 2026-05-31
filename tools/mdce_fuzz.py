from __future__ import annotations

"""MDCE robustness fuzz harness.

Goal: prove the pipeline degrades gracefully on hostile / malformed / edge-case
inputs. For every generated case we assert ONE of two acceptable outcomes:

  1) The pipeline runs and produces an internally-valid AnalysisResult
     (confidence in [0,1], finite numbers, every issue has a fallback), OR
  2) The loader/pipeline raises a *controlled* exception (ValueError) —
     i.e. it rejects bad input cleanly.

What is NEVER acceptable:
  - An uncaught/unexpected crash (KeyError, ZeroDivisionError, IndexError, etc.)
  - A "successful" run that emits NaN/Infinity or out-of-range confidence.

This harness is deterministic (no randomness) so results are reproducible.
"""

import argparse
import json
import math
import sys
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_race_csv
from src.fallback_engine import FALLBACKS
from src.models import ScenarioFlags
from src.pipeline import analyze_decision


HEADER = (
    "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,"
    "track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency"
)


def _row(lap, lt, comp="MEDIUM", age=1, status="NORMAL", weather="DRY") -> str:
    s1, s2, s3 = (lt * 0.32, lt * 0.37, lt * 0.31) if isinstance(lt, (int, float)) else (lt, lt, lt)
    return f"{lap},{lt},{s1},{s2},{s3},{comp},{age},{status},{weather},1.0,{lt},100.0,0.98"


def edge_case_csvs() -> dict[str, str]:
    """Return a deterministic dict of {case_name: csv_text}."""
    cases: dict[str, str] = {}

    # 1) Only required columns (loader must derive the rest).
    cases["minimal_required_cols"] = "lap,lap_time_s\n1,95.1\n2,95.4\n3,95.9"

    # 2) Single row.
    cases["single_row"] = HEADER + "\n" + _row(1, 95.0)

    # 3) Empty (header only).
    cases["header_only"] = HEADER

    # 4) Missing required column entirely (no lap_time_s).
    cases["missing_required_col"] = "lap,tyre_age\n1,1\n2,2"

    # 5) Non-numeric lap_time_s values.
    cases["non_numeric_lap_time"] = "lap,lap_time_s\n1,foo\n2,bar\n3,baz"

    # 6) Negative + zero lap times.
    cases["negative_and_zero_times"] = HEADER + "\n" + "\n".join(
        [_row(1, -5.0), _row(2, 0.0), _row(3, 95.0)]
    )

    # 7) Duplicate + unsorted laps.
    cases["dup_unsorted_laps"] = HEADER + "\n" + "\n".join(
        [_row(3, 95.9), _row(1, 95.1), _row(1, 95.2), _row(2, 95.4)]
    )

    # 8) Extreme spike (huge value).
    cases["extreme_spike"] = HEADER + "\n" + "\n".join(
        [_row(i, 95.0) for i in range(1, 8)] + [_row(8, 1e9)]
    )

    # 9) Literal inf / nan strings.
    cases["inf_nan_strings"] = "lap,lap_time_s\n1,inf\n2,nan\n3,95.0\n4,96.0"

    # 10) Constant flatline.
    cases["flatline"] = HEADER + "\n" + "\n".join(_row(i, 95.0) for i in range(1, 12))

    # 11) Steeply degrading stint.
    cases["steep_degradation"] = HEADER + "\n" + "\n".join(
        _row(i, 95.0 + 1.5 * (i - 1)) for i in range(1, 12)
    )

    # 12) Unknown tyre compound + SC status (coverage + context).
    cases["unknown_compound_sc"] = HEADER + "\n" + "\n".join(
        _row(i, 95.0, comp="UNKNOWN", status="SC") for i in range(1, 8)
    )

    # 13) Huge but finite values everywhere.
    cases["huge_values"] = HEADER + "\n" + "\n".join(_row(i, 1e6 + i) for i in range(1, 10))

    # 14) Very long stint (stress length).
    cases["long_stint"] = HEADER + "\n" + "\n".join(
        _row(i, 90.0 + 0.05 * i, age=i) for i in range(1, 121)
    )

    return cases


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _assert_result_internally_valid(result) -> list[str]:
    """Return a list of contract violations (empty == valid)."""
    problems: list[str] = []
    conf = result.confidence
    if not _is_finite_number(conf.confidence) or not (0.0 <= conf.confidence <= 1.0):
        problems.append(f"confidence out of range/non-finite: {conf.confidence}")
    for k, v in (conf.breakdown or {}).items():
        if not _is_finite_number(v) or not (0.0 <= v <= 1.0):
            problems.append(f"breakdown[{k}] invalid: {v}")
    for k, v in (conf.decision_confidence or {}).items():
        if not _is_finite_number(v) or not (0.0 <= v <= 1.0):
            problems.append(f"decision_confidence[{k}] invalid: {v}")
    # Every emitted issue must have a fallback mapping.
    for issue in result.issues:
        if issue.issue not in FALLBACKS:
            problems.append(f"issue without fallback mapping: {issue.issue}")
        if not _is_finite_number(issue.penalty):
            problems.append(f"issue penalty non-finite: {issue.issue}={issue.penalty}")
    # Explanation must be a non-empty string.
    if not isinstance(result.explanation, str) or not result.explanation.strip():
        problems.append("explanation empty/invalid")
    # Decision impacts must carry finite losses.
    if result.decision_impact is not None and not _is_finite_number(result.decision_impact.if_wrong_expected_loss_s):
        problems.append("decision_impact.if_wrong_expected_loss_s non-finite")
    return problems


def run_one(name: str, csv_text: str) -> dict:
    """Run a single fuzz case. Returns a structured outcome record."""
    flag_combos = [
        ScenarioFlags(),
        ScenarioFlags(missing_telemetry=True, model_mismatch=True, safety_car_phase=True, weather_uncertainty=True),
    ]

    try:
        loaded = load_race_csv(StringIO(csv_text), source_name=f"fuzz:{name}")
    except ValueError as exc:
        # Controlled rejection of bad input is an acceptable outcome.
        return {"case": name, "outcome": "REJECTED_CLEANLY", "detail": str(exc)[:200]}
    except Exception as exc:  # noqa: BLE001 - we explicitly want to catch unexpected crashes
        return {"case": name, "outcome": "UNEXPECTED_CRASH", "detail": f"{type(exc).__name__}: {exc}"[:200]}

    violations: list[str] = []
    for flags in flag_combos:
        try:
            result, _, _, _ = analyze_decision(loaded.records, flags, prefer_granite=False)
        except Exception as exc:  # noqa: BLE001
            return {
                "case": name,
                "outcome": "UNEXPECTED_CRASH",
                "detail": f"analyze_decision {type(exc).__name__}: {exc}"[:200],
            }
        violations.extend(_assert_result_internally_valid(result))

    if violations:
        return {"case": name, "outcome": "INVALID_OUTPUT", "detail": "; ".join(violations)[:400]}
    return {"case": name, "outcome": "OK", "detail": f"rows={len(loaded.records)}"}


def run_all() -> dict:
    results = [run_one(name, text) for name, text in edge_case_csvs().items()]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    # Hard failures are crashes or invalid output. Clean rejection / OK are both fine.
    hard_failures = [r for r in results if r["outcome"] in {"UNEXPECTED_CRASH", "INVALID_OUTPUT"}]
    return {
        "schema_version": "mdce_fuzz_v1",
        "total_cases": len(results),
        "counts": counts,
        "passed": len(hard_failures) == 0,
        "hard_failures": hard_failures,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDCE robustness fuzz harness (deterministic edge cases).")
    p.add_argument("--json-out", default=None, help="Optional path to write the fuzz report JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_all()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"MDCE fuzz harness: {report['total_cases']} cases")
    for outcome, n in sorted(report["counts"].items()):
        print(f"  {outcome}: {n}")
    if not report["passed"]:
        print("HARD FAILURES:", file=sys.stderr)
        for f in report["hard_failures"]:
            print(f"  - {f['case']}: {f['outcome']} :: {f['detail']}", file=sys.stderr)
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
