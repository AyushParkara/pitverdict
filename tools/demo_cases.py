from __future__ import annotations

"""MDCE fixed demo cases.

Runs a small, curated set of named decision cases and emits a clean, comparable
summary. The point is to show — in one command — that MDCE produces *different*
confidence, risk, mode, and reasoning as conditions worsen.

Cases (deterministic):
  case_1_normal           : clean baseline (no injected failures)
  case_2_bad_data         : sensor failure (missing telemetry + frozen tyre proxy)
  case_3_model_mismatch   : model predictions optimistic vs actual pace
  case_4_extreme          : extreme uncertainty stack (forces SAFE)

Each row reports: recommendation, confidence, risk, mode, conflict, #issues, and
the top trust-issue driver — so the contrast is obvious for a demo/slide.
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_default_data_result, load_race_csv
from src.pipeline import analyze_decision
from src.scenario_presets import resolve_preset


# Ordered (case_name -> preset_name). Presets resolve to deterministic flag stacks.
DEMO_CASES: list[tuple[str, str]] = [
    ("case_1_normal", "custom"),
    ("case_2_bad_data", "sensor_failure"),
    ("case_3_model_mismatch", "model_wrong"),
    ("case_4_extreme", "high_uncertainty_stack"),
]


def run_cases(records: list) -> list[dict]:
    rows: list[dict] = []
    for case_name, preset_name in DEMO_CASES:
        preset = resolve_preset(preset_name)
        result, _, _, conflict = analyze_decision(records, preset.flags, prefer_granite=False)
        top = sorted(result.issues, key=lambda i: (-float(i.penalty), i.issue))[:1]
        top_driver = top[0].issue if top else "none"
        rows.append(
            {
                "case": case_name,
                "preset": preset_name,
                "recommendation": result.recommendation.recommendation_type.value,
                "recommended_lap": result.recommendation.recommended_lap,
                "confidence": float(result.confidence.confidence),
                "risk_level": result.confidence.risk_level,
                "recommended_mode": (
                    result.recommended_mode.value if result.recommended_mode is not None else "UNKNOWN"
                ),
                "conflict_score": conflict[0],
                "conflict_label": conflict[1],
                "issue_count": len(result.issues),
                "top_driver": top_driver,
            }
        )
    return rows


def _print_table(rows: list[dict]) -> None:
    headers = ["case", "rec", "conf", "risk", "mode", "issues", "top_driver"]
    print(f"{headers[0]:<22}{headers[1]:<10}{headers[2]:<7}{headers[3]:<14}{headers[4]:<11}{headers[5]:<8}{headers[6]}")
    for r in rows:
        print(
            f"{r['case']:<22}{r['recommendation']:<10}{r['confidence']:<7}{r['risk_level']:<14}"
            f"{r['recommended_mode']:<11}{r['issue_count']:<8}{r['top_driver']}"
        )


def _assert_cases_differ(rows: list[dict]) -> list[str]:
    """Return problems if the demo cases do not visibly differ.

    A good demo must show contrast: confidence should not be identical across all
    cases, and the extreme case must not be more confident than the normal case.
    """
    problems: list[str] = []
    confs = [r["confidence"] for r in rows]
    if len(set(confs)) == 1:
        problems.append("all cases produced identical confidence (no demo contrast)")
    by_case = {r["case"]: r for r in rows}
    normal = by_case.get("case_1_normal")
    extreme = by_case.get("case_4_extreme")
    if normal and extreme and extreme["confidence"] > normal["confidence"]:
        problems.append("extreme case is MORE confident than normal case (wrong direction)")
    return problems


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MDCE fixed demo cases and summarize.")
    p.add_argument("--root", default=None, help="Optional project root for default data resolution.")
    p.add_argument("--input", default=None, help="Optional processed MDCE CSV. If omitted, uses default loader.")
    p.add_argument("--json-out", default=None, help="Optional path to write the demo-cases summary JSON.")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if cases do not visibly differ.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else PROJECT_ROOT

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path
        data_load = load_race_csv(input_path, source_name=str(input_path))
    else:
        data_load = load_default_data_result(root=root)

    rows = run_cases(data_load.records)
    _print_table(rows)

    problems = _assert_cases_differ(rows)
    summary = {
        "schema_version": "mdce_demo_cases_v1",
        "source_name": data_load.source_name,
        "cases": rows,
        "contrast_ok": len(problems) == 0,
        "problems": problems,
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if problems:
        print("CONTRAST WARNINGS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        if args.strict:
            raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
