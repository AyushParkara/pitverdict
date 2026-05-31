from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import NONCOMMERCIAL_GUARD_WARNING_PREFIX, load_default_data_result, load_race_csv
from src.models import ScenarioFlags
from src.pipeline import analyze_decision
from src.confidence_engine import CONFIDENCE_SCORING_VERSION
from src.scenario_presets import list_preset_names, resolve_preset


DECISION_RUN_SCHEMA_VERSION = "mdce_decision_run_v2"


# Canonical v2 registry schema (column order is part of the contract).
REGISTRY_V2_FIELDNAMES = [
    "run_id",
    "timestamp_utc",
    "source_name",
    "dataset_license",
    "dataset_source_url",
    "preset",
    "confidence_scoring_version",
    "recommendation_type",
    "recommended_lap",
    "confidence",
    "risk_level",
    "pit_timing_confidence",
    "tyre_strategy_confidence",
    "uncertainty_primary",
    "uncertainty_score",
    "recommended_mode",
    "safe_mode_if_wrong_loss_s",
    "aggressive_mode_if_wrong_loss_s",
    "model_validation_status",
    "model_validation_mae_s",
    "impact_if_wrong_loss_s",
    "push_vs_conserve_if_wrong_loss_s",
    "push_vs_conserve_risk_level",
    "conflict_score",
    "conflict_label",
    "issue_count",
    "issues",
    "json_path",
    "md_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic MDCE decision analysis (headless).")
    parser.add_argument(
        "--root",
        default=None,
        help="Optional project root. If set, resolves relative paths like outputs/ and data/ against this root.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional processed MDCE CSV path. If omitted, MDCE uses prepared real data when available, else sample fallback.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Directory to write run artifacts (JSON/MD).",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/reports/MDCE_DECISION_RUN_REGISTRY.csv",
        help="Append-only registry CSV path for decision runs.",
    )
    parser.add_argument("--no-granite", action="store_true", help="Disable Granite even if env vars are set.")

    parser.add_argument(
        "--preset",
        default="custom",
        choices=list_preset_names(),
        help="Scenario preset (shortcut for multiple failure-mode flags).",
    )

    # Scenario flags
    parser.add_argument("--missing-telemetry", action="store_true")
    parser.add_argument("--tyre-signal-drift", action="store_true")
    parser.add_argument("--model-mismatch", action="store_true")
    parser.add_argument("--safety-car-phase", action="store_true")
    parser.add_argument("--weather-uncertainty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_path = Path(args.registry_csv)
    if not registry_path.is_absolute():
        registry_path = root / registry_path
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path
        data_load = load_race_csv(input_path, source_name=str(input_path))
    else:
        data_load = load_default_data_result(root=root)

    # Make guarded fallback obvious in CLI runs (especially for headless demos).
    for warning in data_load.warnings or []:
        if isinstance(warning, str) and warning.startswith(NONCOMMERCIAL_GUARD_WARNING_PREFIX):
            print(f"WARN: {warning}", file=sys.stderr)
            break

    source_sidecar_metadata: dict = getattr(data_load, "dataset_metadata", {}) or {}

    preset = resolve_preset(args.preset)
    # Preset flags are a base; explicit CLI flags can override them.
    flags = ScenarioFlags(
        missing_telemetry=args.missing_telemetry or preset.flags.missing_telemetry,
        tyre_signal_drift=args.tyre_signal_drift or preset.flags.tyre_signal_drift,
        model_mismatch=args.model_mismatch or preset.flags.model_mismatch,
        safety_car_phase=args.safety_car_phase or preset.flags.safety_car_phase,
        weather_uncertainty=args.weather_uncertainty or preset.flags.weather_uncertainty,
    )

    prefer_granite = not args.no_granite
    result, scenario_records, scenario_notes, conflict = analyze_decision(
        data_load.records,
        flags,
        prefer_granite=prefer_granite,
    )

    # Use timezone-aware UTC timestamps (avoids deprecated utcnow()). Keep the
    # human-facing timestamp stable to whole seconds, but generate run_id with
    # microsecond precision to avoid collisions under fast/parallel runs.
    now_dt_full = datetime.now(timezone.utc)
    now_dt = now_dt_full.replace(microsecond=0)
    now = now_dt.isoformat().replace("+00:00", "Z")
    # Second-granularity run_ids collide easily (two runs in the same second will
    # overwrite each other's JSON/MD and can fail validation).
    run_id = now_dt_full.strftime("%Y%m%d_%H%M%S_%f")
    json_path = out_dir / f"mdce_decision_run_{run_id}.json"
    md_path = out_dir / f"mdce_decision_run_{run_id}.md"

    payload = {
        "schema_version": DECISION_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": now,
        "project_root": str(root),
        "confidence_scoring_version": CONFIDENCE_SCORING_VERSION,
        "source": {
            "source_name": data_load.source_name,
            "real_columns": data_load.real_columns,
            "derived_columns": data_load.derived_columns,
            "proxy_columns": data_load.proxy_columns,
            "warnings": data_load.warnings,
            # Optional sidecar metadata from `tools/prepare_mdce_dataset.py` (license/provenance).
            "dataset_metadata": source_sidecar_metadata,
        },
        "scenario": {
            "flags": {
                "preset": preset.name,
                "missing_telemetry": flags.missing_telemetry,
                "tyre_signal_drift": flags.tyre_signal_drift,
                "model_mismatch": flags.model_mismatch,
                "safety_car_phase": flags.safety_car_phase,
                "weather_uncertainty": flags.weather_uncertainty,
            },
            "notes": scenario_notes,
        },
        "decision": {
            "recommendation": {
                "type": result.recommendation.recommendation_type.value,
                "recommended_lap": result.recommendation.recommended_lap,
                "expected_gain_loss_s": result.recommendation.expected_gain_loss_s,
                "base_reason": result.recommendation.base_reason,
            },
            "confidence": {
                "confidence": result.confidence.confidence,
                "risk_level": result.confidence.risk_level,
                "breakdown": result.confidence.breakdown,
                "decision_confidence": result.confidence.decision_confidence,
                "decision_risk_levels": result.confidence.decision_risk_levels,
            },
            # Keep schema stable for downstream consumers: always present with a consistent shape.
            "uncertainty": {
                "primary_uncertainty": "none",
                "uncertainty_score": 0.0,
                "downstream_decisions_at_risk": [],
                "drivers": [],
            },
            "recommended_mode": "UNKNOWN",
            "mode_options": [],
            "conflict": {
                "score": conflict[0],
                "label": conflict[1],
            },
            "issues": [
                {
                    "issue": issue.issue,
                    "severity": issue.severity.value,
                    "penalty": issue.penalty,
                    "affected_decisions": issue.affected_decisions,
                    "reason": issue.reason,
                }
                for issue in result.issues
            ],
            "fallback_actions": result.fallback_actions,
            "explanation": result.explanation,
        },
        "model_validation": {
            "status": "NO_DATA",
            # Keep numeric types stable even in NO_DATA cases.
            "window_laps": 5,
            "deviation_threshold_s": 0.75,
            "mean_abs_error_s": 0.0,
            "max_abs_error_s": 0.0,
            "recommended_confidence_penalty": 0.0,
            "deviations": [],
        },
        "decision_impact": {
            "decision": "pit_timing",
            "horizon_laps": 3,
            "if_right_expected_gain_s": 0.0,
            "if_wrong_expected_loss_s": 0.0,
            "risk_level": "LOW",
            "assumptions": {"horizon_laps": 3.0},
            "notes": ["NO_DATA"],
        },
        "decision_impacts": [],
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
            # Registry paths are persisted so a run can be traced externally.
            "registry_csv": str(registry_path),
            "registry_v2_csv": None,
        },
    }

    # Compatibility view for demos/docs that prefer the earlier "flat" JSON shape.
    # This is additive and does not change the stable v2 schema.
    payload["plan_view"] = {
        "recommendation": f"{payload['decision']['recommendation']['type']} (lap {payload['decision']['recommendation']['recommended_lap']})",
        "confidence": payload["decision"]["confidence"]["confidence"],
        "risk_level": payload["decision"]["confidence"]["risk_level"],
        # Mirrors v2 breakdown keys while allowing plan consumers to use a single top-level key.
        "confidence_breakdown": payload["decision"]["confidence"]["breakdown"],
        # Keep the plan terminology honest: we implement multi-signal disagreement.
        "conflict": {
            "score": payload["decision"]["conflict"]["score"],
            "label": payload["decision"]["conflict"]["label"],
            "notes": [
                "Conflict scoring is multi-signal disagreement (not multi-model ensemble disagreement).",
            ],
        },
        # Primary decision-domain risk summary (decision impacts are computed separately).
        "decision_risk": payload["decision"]["confidence"].get("decision_risk_levels") or {},
        # Best-effort single impact summary for the plan format.
        "impact": {
            "decision": (payload.get("decision_impact") or {}).get("decision"),
            "if_wrong_expected_loss_s": (payload.get("decision_impact") or {}).get("if_wrong_expected_loss_s"),
            "risk_level": (payload.get("decision_impact") or {}).get("risk_level"),
        },
        # Prefer a single fallback string for plan consumers while preserving the full list in v2.
        "fallback": " ".join(payload["decision"].get("fallback_actions") or [])
        or "None",
        "issues": payload["decision"].get("issues") or [],
        "recommended_mode": payload["decision"].get("recommended_mode") or "UNKNOWN",
        "uncertainty": payload["decision"].get("uncertainty") or {
            "primary_uncertainty": "none",
            "uncertainty_score": 0.0,
            "downstream_decisions_at_risk": [],
            "drivers": [],
        },
        "provenance": {
            "source_name": payload["source"].get("source_name"),
            "dataset_license": (
                source_sidecar_metadata.get("license_spdx")
                or source_sidecar_metadata.get("license_note")
                or "NOASSERTION"
            ),
            "dataset_source_url": (source_sidecar_metadata.get("source_url") or "UNKNOWN"),
            "real_columns": payload["source"].get("real_columns") or [],
            "derived_columns": payload["source"].get("derived_columns") or [],
            "proxy_columns": payload["source"].get("proxy_columns") or [],
            "warnings": payload["source"].get("warnings") or [],
        },
    }

    if result.model_validation is not None:
        payload["model_validation"] = {
            "status": result.model_validation.status,
            "window_laps": result.model_validation.window_laps,
            "deviation_threshold_s": result.model_validation.deviation_threshold_s,
            "mean_abs_error_s": result.model_validation.mean_abs_error_s,
            "max_abs_error_s": result.model_validation.max_abs_error_s,
            "recommended_confidence_penalty": result.model_validation.recommended_confidence_penalty,
            "deviations": [
                {
                    "lap": d.lap,
                    "expected_lap_time_s": d.expected_lap_time_s,
                    "actual_lap_time_s": d.actual_lap_time_s,
                    "delta_s": d.delta_s,
                }
                for d in result.model_validation.deviations
            ],
        }

    if result.decision_impact is not None:
        payload["decision_impact"] = {
            "decision": result.decision_impact.decision,
            "horizon_laps": result.decision_impact.horizon_laps,
            "if_right_expected_gain_s": result.decision_impact.if_right_expected_gain_s,
            "if_wrong_expected_loss_s": result.decision_impact.if_wrong_expected_loss_s,
            "risk_level": result.decision_impact.risk_level,
            "assumptions": result.decision_impact.assumptions,
            "notes": result.decision_impact.notes,
        }

    if getattr(result, "decision_impacts", None):
        payload["decision_impacts"] = [
            {
                "decision": di.decision,
                "horizon_laps": di.horizon_laps,
                "if_right_expected_gain_s": di.if_right_expected_gain_s,
                "if_wrong_expected_loss_s": di.if_wrong_expected_loss_s,
                "risk_level": di.risk_level,
                "assumptions": di.assumptions,
                "notes": di.notes,
            }
            for di in result.decision_impacts
        ]

    # Ensure item shapes are stable even for callers that set decision_impacts=[] explicitly.
    payload["decision_impacts"] = [
        {
            "decision": di.get("decision"),
            "horizon_laps": di.get("horizon_laps"),
            "if_right_expected_gain_s": di.get("if_right_expected_gain_s"),
            "if_wrong_expected_loss_s": di.get("if_wrong_expected_loss_s"),
            "risk_level": di.get("risk_level"),
            "assumptions": di.get("assumptions") or {},
            "notes": di.get("notes") or [],
        }
        for di in (payload.get("decision_impacts") or [])
    ]

    if result.uncertainty is not None:
        payload["decision"]["uncertainty"] = {
            "primary_uncertainty": result.uncertainty.primary_uncertainty,
            "uncertainty_score": result.uncertainty.uncertainty_score,
            "downstream_decisions_at_risk": result.uncertainty.downstream_decisions_at_risk,
            "drivers": result.uncertainty.drivers,
        }

    if result.recommended_mode is not None:
        payload["decision"]["recommended_mode"] = result.recommended_mode.value

    payload["decision"]["mode_options"] = [
        {
            "mode": o.mode.value,
            "recommendation": {
                "type": o.recommendation.recommendation_type.value,
                "recommended_lap": o.recommendation.recommended_lap,
                "expected_gain_loss_s": o.recommendation.expected_gain_loss_s,
                "base_reason": o.recommendation.base_reason,
            },
            "decision_impact": {
                "decision": o.decision_impact.decision,
                "horizon_laps": o.decision_impact.horizon_laps,
                "if_right_expected_gain_s": o.decision_impact.if_right_expected_gain_s,
                "if_wrong_expected_loss_s": o.decision_impact.if_wrong_expected_loss_s,
                "risk_level": o.decision_impact.risk_level,
                "assumptions": o.decision_impact.assumptions,
                "notes": o.decision_impact.notes,
            },
        }
        for o in (result.mode_options or [])
    ]

    # NOTE: we write JSON once early for quick inspection, then re-write at the end
    # after all artifact paths (registry v2) are finalized.
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# MDCE Decision Run",
        "",
        f"- Run ID: `{run_id}`",
        f"- Timestamp (UTC): `{now}`",
        f"- Source: `{data_load.source_name}`",
        f"- Dataset license: `{(source_sidecar_metadata.get('license_spdx') or source_sidecar_metadata.get('license_note') or 'NOASSERTION')}`",
        f"- Dataset source URL: `{(source_sidecar_metadata.get('source_url') or 'UNKNOWN')}`",
        f"- Confidence scoring version: `{payload['confidence_scoring_version']}`",
        "",
        "## Recommendation",
        "",
        f"- Type: `{payload['decision']['recommendation']['type']}`",
        f"- Target lap: `{payload['decision']['recommendation']['recommended_lap']}`",
        f"- Expected gain/loss (heuristic): `{payload['decision']['recommendation']['expected_gain_loss_s']}` seconds",
        f"- Base reason: {payload['decision']['recommendation']['base_reason']}",
        "",
        "## Confidence",
        "",
        f"- Confidence: `{payload['decision']['confidence']['confidence']}`",
        f"- Risk: `{payload['decision']['confidence']['risk_level']}`",
        f"- Conflict score: `{payload['decision']['conflict']['score']}` ({payload['decision']['conflict']['label']})",
        f"- Decision confidence: `{payload['decision']['confidence']['decision_confidence']}`",
        "",
        "## Uncertainty",
        "",
        "## Confidence Breakdown",
        "",
        f"- Breakdown: `{payload['decision']['confidence']['breakdown']}`",
        "",
        "## Recommended Mode",
        "",
        f"- Recommended mode: `{payload['decision'].get('recommended_mode')}`",
        "",
        "## Mode Options",
        "",
        "## Scenario",
        "",
        f"- Flags: `{payload['scenario']['flags']}`",
        f"- Notes: {' '.join(scenario_notes) if scenario_notes else 'None' }",
        "",
        "## Issues",
        "",
    ]

    # Insert uncertainty detail under its section.
    u = payload["decision"].get("uncertainty") or {
        "primary_uncertainty": "none",
        "uncertainty_score": 0.0,
        "downstream_decisions_at_risk": [],
        "drivers": [],
    }
    idx = md_lines.index("## Uncertainty")
    md_lines[idx + 2:idx + 2] = [
        f"- Primary: `{u.get('primary_uncertainty')}`",
        f"- Score: `{u.get('uncertainty_score')}`",
        f"- Downstream decisions at risk: `{u.get('downstream_decisions_at_risk')}`",
    ]

    # Insert mode options detail under its section.
    if not payload["decision"].get("mode_options"):
        idx = md_lines.index("## Mode Options")
        md_lines.insert(idx + 2, "- None")
    else:
        idx = md_lines.index("## Mode Options")
        insert_lines = []
        for opt in payload["decision"]["mode_options"]:
            insert_lines.append(
                "- "
                f"{opt['mode']}: {opt['recommendation']['type']} (lap {opt['recommendation']['recommended_lap']}), "
                f"if-wrong loss {opt['decision_impact']['if_wrong_expected_loss_s']}s (risk={opt['decision_impact']['risk_level']})"
            )
        md_lines[idx + 2:idx + 2] = insert_lines
    if not payload["decision"]["issues"]:
        md_lines.append("- No major trust issues detected.")
    else:
        for issue in payload["decision"]["issues"]:
            md_lines.append(f"- `{issue['issue']}` ({issue['severity']}): {issue['reason']} (penalty={issue['penalty']})")
    md_lines += [
        "",
        "## Fallback Actions",
        "",
    ]
    if not payload["decision"]["fallback_actions"]:
        md_lines.append("- None")
    else:
        for action in payload["decision"]["fallback_actions"]:
            md_lines.append(f"- {action}")
    md_lines += [
        "",
        "## Model Validation (Model vs Reality)",
        "",
    ]
    mv = payload.get("model_validation")
    if not mv:
        md_lines.append("- None")
    else:
        md_lines.append(f"- Status: `{mv.get('status')}`")
        if mv.get("window_laps") is not None and mv.get("mean_abs_error_s") is not None:
            md_lines.append(f"- MAE (last {mv['window_laps']} laps): `{mv['mean_abs_error_s']}` s")
        if mv.get("max_abs_error_s") is not None:
            md_lines.append(f"- Max abs error: `{mv['max_abs_error_s']}` s")
        if mv.get("deviations"):
            md_lines.append("- Deviations:")
            for d in mv["deviations"]:
                md_lines.append(
                    f"  - lap {d['lap']}: expected {d['expected_lap_time_s']}, actual {d['actual_lap_time_s']} (delta {d['delta_s']}s)"
                )

    md_lines += [
        "",
        "## Decision Impact Simulation",
        "",
    ]
    di = payload.get("decision_impact") or {}
    md_lines.append(f"- Decision: `{di.get('decision')}`")
    md_lines.append(f"- Horizon laps: `{di.get('horizon_laps')}`")
    md_lines.append(f"- If right (expected gain): `{di.get('if_right_expected_gain_s')}` s")
    md_lines.append(f"- If wrong (expected loss): `{di.get('if_wrong_expected_loss_s')}` s")
    md_lines.append(f"- Risk: `{di.get('risk_level')}`")
    if di.get("notes"):
        md_lines.append(f"- Notes: {' '.join(str(n) for n in di['notes'])}")

    md_lines += [
        "",
        "## Additional Decision Impacts",
        "",
    ]
    if not payload.get("decision_impacts"):
        md_lines.append("- None")
    else:
        for di in payload["decision_impacts"]:
            md_lines.append(
                f"- {di['decision']}: if-wrong loss `{di['if_wrong_expected_loss_s']}` s (risk={di['risk_level']}, horizon={di['horizon_laps']} laps)"
            )

    md_lines += [
        "",
        "## Explanation",
        "",
        payload["decision"]["explanation"],
        "",
        "## Provenance Warnings",
        "",
    ]
    if not data_load.warnings:
        md_lines.append("- None")
    else:
        for w in data_load.warnings:
            md_lines.append(f"- {w}")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Append a compact row to the registry for quick programmatic comparisons.
    push_vs_conserve_if_wrong_loss_s = None
    push_vs_conserve_risk_level = None
    for di in payload.get("decision_impacts") or []:
        if di.get("decision") == "push_vs_conserve":
            push_vs_conserve_if_wrong_loss_s = di.get("if_wrong_expected_loss_s")
            push_vs_conserve_risk_level = di.get("risk_level")
            break

    safe_mode_if_wrong_loss_s = None
    aggressive_mode_if_wrong_loss_s = None
    for opt in payload.get("decision", {}).get("mode_options") or []:
        mode = opt.get("mode")
        loss = (opt.get("decision_impact") or {}).get("if_wrong_expected_loss_s")
        if mode == "SAFE":
            safe_mode_if_wrong_loss_s = loss
        elif mode == "AGGRESSIVE":
            aggressive_mode_if_wrong_loss_s = loss

    row = {
        "run_id": run_id,
        "timestamp_utc": now,
        "source_name": data_load.source_name,
        "dataset_license": (
            source_sidecar_metadata.get("license_spdx")
            or source_sidecar_metadata.get("license_note")
            or "NOASSERTION"
        ),
        "dataset_source_url": (source_sidecar_metadata.get("source_url") or ""),
        "preset": (payload.get("scenario") or {}).get("flags", {}).get("preset"),
        "confidence_scoring_version": payload["confidence_scoring_version"],
        "recommendation_type": payload["decision"]["recommendation"]["type"],
        "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
        "confidence": payload["decision"]["confidence"]["confidence"],
        "risk_level": payload["decision"]["confidence"]["risk_level"],
        "pit_timing_confidence": payload["decision"]["confidence"]["decision_confidence"].get("pit_timing"),
        "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"].get("tyre_strategy"),
        "uncertainty_primary": (payload["decision"].get("uncertainty") or {}).get("primary_uncertainty"),
        "uncertainty_score": (payload["decision"].get("uncertainty") or {}).get("uncertainty_score"),
        "recommended_mode": payload["decision"].get("recommended_mode"),
        "safe_mode_if_wrong_loss_s": safe_mode_if_wrong_loss_s,
        "aggressive_mode_if_wrong_loss_s": aggressive_mode_if_wrong_loss_s,
        "model_validation_status": (payload.get("model_validation") or {}).get("status"),
        "model_validation_mae_s": (payload.get("model_validation") or {}).get("mean_abs_error_s"),
        "impact_if_wrong_loss_s": (payload.get("decision_impact") or {}).get("if_wrong_expected_loss_s"),
        "push_vs_conserve_if_wrong_loss_s": push_vs_conserve_if_wrong_loss_s,
        "push_vs_conserve_risk_level": push_vs_conserve_risk_level,
        "conflict_score": payload["decision"]["conflict"]["score"],
        "conflict_label": payload["decision"]["conflict"]["label"],
        "issue_count": len(payload["decision"]["issues"]),
        "issues": ";".join(sorted({i["issue"] for i in payload["decision"]["issues"]})),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }

    # Keep v2 registry column order stable.
    desired_fieldnames = REGISTRY_V2_FIELDNAMES

    def _append_row(path: Path, fieldnames: list[str]) -> None:
        write_header = not path.exists()
        cleaned_row = {k: row.get(k) for k in fieldnames}
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(cleaned_row)

    def _ensure_header(path: Path, fieldnames: list[str]) -> None:
        """Upgrade CSV header in-place if it lags behind.

        v2 registries are contract-driven. If we add new columns, we migrate the
        existing file once by rewriting rows into the new header, leaving blanks
        for previously-unknown columns.
        """

        if not path.exists():
            return

        with path.open("r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), None) or []

        if existing_header == fieldnames:
            return

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with path.open("r", newline="", encoding="utf-8") as rf, tmp_path.open(
            "w", newline="", encoding="utf-8"
        ) as wf:
            reader = csv.DictReader(rf)
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()
            for r in reader:
                writer.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in fieldnames})
        tmp_path.replace(path)

    # Registry strategy:
    # - Always write a full-schema v2 registry (canonical).
    # - Also write/append to the requested registry path, preserving any legacy header.
    if registry_path.stem.endswith("_v2"):
        v2_path = registry_path
    else:
        v2_path = registry_path.with_name(f"{registry_path.stem}_v2{registry_path.suffix}")

    _ensure_header(v2_path, desired_fieldnames)
    _append_row(v2_path, desired_fieldnames)
    payload["artifacts"]["registry_v2_csv"] = str(v2_path)

    # Only append to the legacy registry path when it is distinct from v2.
    # If the caller already points --registry-csv at the v2 path, avoid
    # double-appending the same row.
    if registry_path.resolve() != v2_path.resolve():
        if registry_path.exists():
            with registry_path.open("r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), None) or []
            legacy_fieldnames = existing_header or desired_fieldnames
            _append_row(registry_path, legacy_fieldnames)
        else:
            _append_row(registry_path, desired_fieldnames)

    print("MDCE decision run complete")
    print("JSON:", json_path)
    print("MD:", md_path)
    print("Registry row appended:", registry_path)
    print("Registry v2 row appended:", v2_path)

    # Finalize persisted JSON so it contains the computed registry v2 path.
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
