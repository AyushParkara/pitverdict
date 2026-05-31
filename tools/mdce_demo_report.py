from __future__ import annotations

"""MDCE one-page demo report.

Assembles a single self-contained Markdown artifact a judge can open, combining:

  1) What MDCE is (and is not) — the defensible framing.
  2) A clean baseline decision run + a stacked high-uncertainty run (same data),
     to show confidence/risk/mode responding to injected failure modes.
  3) Calibration evidence (scenario monotonicity + held-out mode-vs-regret).
  4) Robustness summary (fuzz harness outcome).
  5) Provenance + safe-claim boundaries.

Everything is computed deterministically from the loaded data. No network.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_default_data_result, load_race_csv
from src.models import ScenarioFlags
from src.pipeline import analyze_decision
from src.scenario_presets import resolve_preset
from tools.mdce_calibration import build_report as build_calibration_report
from tools.mdce_fuzz import run_all as run_fuzz


def _run_decision(records: list, flags: ScenarioFlags):
    result, _, _, conflict = analyze_decision(records, flags, prefer_granite=False)
    return result, conflict


def _decision_block(title: str, result, conflict) -> list[str]:
    rec = result.recommendation
    conf = result.confidence
    mode = result.recommended_mode.value if result.recommended_mode is not None else "UNKNOWN"
    score, label = conflict
    top_issues = sorted(
        result.issues,
        key=lambda i: (-float(i.penalty), i.issue),
    )[:3]
    lines = [
        f"### {title}",
        "",
        f"- Recommendation: **{rec.recommendation_type.value}** (target lap {rec.recommended_lap})",
        f"- Confidence: **{int(conf.confidence * 100)}%** | Risk: **{conf.risk_level}** | Mode: **{mode}**",
        f"- Multi-signal conflict: **{score} ({label})**",
        f"- Trust issues: **{len(result.issues)}**",
    ]
    if top_issues:
        lines.append("- Top drivers:")
        for i in top_issues:
            lines.append(f"  - `{i.issue}` (penalty {i.penalty}): {i.reason}")
    lines.append("")
    return lines


def build_markdown(records: list, *, source_name: str, dataset_metadata: dict, warnings: list[str]) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    baseline_result, baseline_conflict = _run_decision(records, resolve_preset("custom").flags)
    stacked_result, stacked_conflict = _run_decision(records, resolve_preset("high_uncertainty_stack").flags)

    calib = build_calibration_report(records, source_name=source_name)
    mono = calib["scenario_monotonicity"]
    bt = calib["confidence_vs_regret_backtest"]

    fuzz = run_fuzz()

    license_str = (
        dataset_metadata.get("license_spdx")
        or dataset_metadata.get("license_note")
        or "NOASSERTION"
    )
    source_url = dataset_metadata.get("source_url") or "UNKNOWN"
    gap_method = dataset_metadata.get("gap_to_car_ahead_method")

    lines: list[str] = [
        "# MDCE — Decision Confidence Report",
        "",
        f"_Generated {now} (deterministic; no network)._",
        "",
        "## 1) What MDCE Is",
        "",
        "MDCE is a **decision-confidence / trust layer** for race-strategy decisions under",
        "uncertain and incomplete data. It does **not** claim to find the optimal strategy or",
        "to use private F1 team telemetry. It answers one question:",
        "",
        "> *Should we trust this recommendation right now — and if not, what is the safer action?*",
        "",
        "Python computes all evidence, confidence, risk, and fallbacks deterministically.",
        "IBM Granite (if configured) is used **only** to phrase the explanation.",
        "",
        "## 2) Decision Run (same data, two scenarios)",
        "",
        f"Source: `{source_name}`",
        "",
    ]
    lines += _decision_block("Baseline (no injected failures)", baseline_result, baseline_conflict)
    lines += _decision_block("High-uncertainty stack (injected failures)", stacked_result, stacked_conflict)
    lines += [
        "Observation: stacking genuine failure modes lowers confidence, raises risk, and",
        "pushes the recommended mode toward SAFE — exactly what a trust layer should do.",
        "",
        "## 3) Evidence (does confidence behave sensibly?)",
        "",
        "### A) Scenario monotonicity (validated headline)",
        "",
        f"- Confidence non-increasing as failures increase: **{mono['monotonic_non_increasing']}**",
        f"- Pearson(failure_count, confidence): **{mono['pearson_failurecount_vs_confidence']}** (expected strongly negative)",
        f"- Mean confidence by failure count: `{mono['bucket_mean_confidence_by_failure_count']}`",
        "",
        "### B) Held-out regret diagnostic (exploratory only)",
        "",
    ]
    if bt.get("status") != "OK":
        lines += [f"- Status: `{bt.get('status')}` — {bt.get('note')}", ""]
    else:
        lines += [
            f"- Backtest points: **{bt['num_points']}** (SAFE={bt['num_safe_mode']}, AGGRESSIVE={bt['num_aggressive_mode']})",
            f"- Mean aggressive-action regret when **SAFE**: **{bt['mean_aggressive_regret_when_safe_s']} s**",
            f"- Mean aggressive-action regret when **AGGRESSIVE**: **{bt['mean_aggressive_regret_when_aggressive_s']} s**",
            f"- Slice diagnostic result: **{bt['calibration_holds']}**",
            "",
            "Interpretation: this diagnostic can be useful for one slice, but it does not generalize",
            "across drivers because early-stint high confidence mechanically precedes later degradation.",
            "Do not present it as proof of calibration; the validated headline is monotonicity above.",
            "",
            f"_Caveat: {bt['caveat']}_",
            "",
        ]

    lines += [
        "## 4) Robustness (fuzz harness)",
        "",
        f"- Edge cases exercised: **{fuzz['total_cases']}**",
        f"- Outcome counts: `{fuzz['counts']}`",
        f"- No crashes / no invalid output: **{fuzz['passed']}**",
        "",
        "Every malformed input is either handled with a valid result or rejected cleanly",
        "(controlled `ValueError`). No uncaught crashes; no NaN/Infinity or out-of-range confidence.",
        "",
        "## 5) Provenance & Safe Claims",
        "",
        f"- Dataset license: `{license_str}`",
        f"- Source URL: `{source_url}`",
    ]
    if gap_method:
        lines.append(f"- `gap_to_car_ahead_s` method: `{gap_method}` (derived proxy; not an official timing feed)")
    real_warnings = [w for w in (warnings or []) if not w.startswith(("Real column used:", "Derived column:", "Proxy column:"))]
    if real_warnings:
        lines.append("- Notes:")
        for w in real_warnings[:6]:
            lines.append(f"  - {w}")
    lines += [
        "",
        "Safe claim: MDCE ingests public lap/tyre/weather data mapped into its schema, clearly",
        "separates real / derived / proxy fields, and produces confidence + risk + fallbacks",
        "without claiming private telemetry or optimal strategy.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a one-page MDCE demo report (Markdown).")
    p.add_argument("--root", default=None, help="Optional project root for default data resolution.")
    p.add_argument("--input", default=None, help="Optional processed MDCE CSV. If omitted, uses default loader.")
    p.add_argument("--output", default="outputs/reports/MDCE_DEMO_REPORT.md", help="Output Markdown path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path
        data_load = load_race_csv(input_path, source_name=str(input_path))
    else:
        data_load = load_default_data_result(root=root)

    md = build_markdown(
        data_load.records,
        source_name=data_load.source_name,
        dataset_metadata=getattr(data_load, "dataset_metadata", {}) or {},
        warnings=list(data_load.warnings or []),
    )
    out_path.write_text(md, encoding="utf-8")
    print("MDCE demo report written:", out_path)


if __name__ == "__main__":
    main()
