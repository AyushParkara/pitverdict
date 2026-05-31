from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
import math


def _latest_run_json(outputs_dir: Path) -> Path:
    json_files = sorted(outputs_dir.glob("mdce_decision_run_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No mdce_decision_run_*.json found in {outputs_dir}")

    # Prefer the newest artifact that already matches the current schema.
    # This avoids surprising failures when older v1 files exist in the folder.
    for path in reversed(json_files):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if payload.get("schema_version") == "mdce_decision_run_v2":
            return path

    # Fall back to the newest file; downstream validation will explain the mismatch.
    return json_files[-1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _assert_has(d: dict, key: str, *, ctx: str) -> None:
    if key not in d:
        _fail(f"Missing key {key!r} in {ctx}")


def _assert_type(val, types, *, ctx: str) -> None:
    if not isinstance(val, types):
        # Use type names for readable errors.
        if isinstance(types, tuple):
            tnames = ", ".join(t.__name__ for t in types)
        else:
            tnames = types.__name__
        _fail(f"{ctx} must be {tnames}, got {type(val).__name__}")


def _assert_in(val, allowed: set, *, ctx: str) -> None:
    if val not in allowed:
        _fail(f"{ctx} must be one of {sorted(allowed)}, got {val!r}")


def _assert_num_range(val, *, lo: float, hi: float, ctx: str) -> None:
    _assert_type(val, (int, float), ctx=ctx)
    if not math.isfinite(float(val)):
        _fail(f"{ctx} must be finite, got {val}")
    if float(val) < lo or float(val) > hi:
        _fail(f"{ctx} must be in [{lo}, {hi}], got {val}")


def _assert_num_min(val, *, lo: float, ctx: str) -> None:
    _assert_type(val, (int, float), ctx=ctx)
    if not math.isfinite(float(val)):
        _fail(f"{ctx} must be finite, got {val}")
    if float(val) < lo:
        _fail(f"{ctx} must be >= {lo}, got {val}")


def _assert_int_min(val, *, lo: int, ctx: str) -> None:
    _assert_type(val, int, ctx=ctx)
    if int(val) < lo:
        _fail(f"{ctx} must be >= {lo}, got {val}")


def validate_payload(payload: dict) -> None:
    _assert_has(payload, "schema_version", ctx="payload")
    if payload["schema_version"] != "mdce_decision_run_v2":
        _fail(f"Unexpected schema_version: {payload['schema_version']!r}")

    for k in ("run_id", "timestamp_utc", "project_root", "confidence_scoring_version"):
        _assert_has(payload, k, ctx="payload")
    _assert_type(payload["run_id"], str, ctx="payload.run_id")
    if not payload["run_id"].strip():
        _fail("payload.run_id must be non-empty")
    _assert_type(payload["timestamp_utc"], str, ctx="payload.timestamp_utc")
    if not payload["timestamp_utc"].endswith("Z"):
        _fail("payload.timestamp_utc must be UTC ISO8601 ending with 'Z'")
    _assert_type(payload["project_root"], str, ctx="payload.project_root")
    if not os.path.isabs(payload["project_root"]):
        _fail("payload.project_root must be an absolute path")
    if not Path(payload["project_root"]).exists():
        _fail(f"payload.project_root does not exist: {payload['project_root']!r}")
    _assert_type(payload["confidence_scoring_version"], str, ctx="payload.confidence_scoring_version")

    _assert_has(payload, "source", ctx="payload")
    _assert_has(payload["source"], "warnings", ctx="source")
    src = payload["source"]
    _assert_type(src, dict, ctx="payload.source")
    for k in ("source_name", "real_columns", "derived_columns", "proxy_columns", "warnings"):
        _assert_has(src, k, ctx="source")
    _assert_type(src["source_name"], str, ctx="source.source_name")
    for k in ("real_columns", "derived_columns", "proxy_columns", "warnings"):
        _assert_type(src[k], list, ctx=f"source.{k}")
        for it in src[k]:
            _assert_type(it, str, ctx=f"source.{k}[*]")
    if "lap" not in src.get("real_columns", []) or "lap_time_s" not in src.get("real_columns", []):
        _fail("source.real_columns must include at least 'lap' and 'lap_time_s'")

    # Optional sidecar dataset metadata (license/provenance). If present, it must be a JSON object.
    if "dataset_metadata" in src:
        _assert_type(src.get("dataset_metadata"), dict, ctx="source.dataset_metadata")

    _assert_has(payload, "scenario", ctx="payload")
    _assert_has(payload["scenario"], "flags", ctx="scenario")
    _assert_has(payload["scenario"]["flags"], "preset", ctx="scenario.flags")
    _assert_has(payload["scenario"], "notes", ctx="scenario")
    _assert_type(payload["scenario"].get("notes"), list, ctx="scenario.notes")
    for n in payload["scenario"].get("notes") or []:
        _assert_type(n, str, ctx="scenario.notes[*]")

    flags = payload["scenario"]["flags"]
    if not isinstance(flags, dict):
        _fail("scenario.flags must be an object")
    for k in ("preset", "missing_telemetry", "tyre_signal_drift", "model_mismatch", "safety_car_phase", "weather_uncertainty"):
        _assert_has(flags, k, ctx="scenario.flags")
    _assert_type(flags["preset"], str, ctx="scenario.flags.preset")
    for k in ("missing_telemetry", "tyre_signal_drift", "model_mismatch", "safety_car_phase", "weather_uncertainty"):
        _assert_type(flags[k], bool, ctx=f"scenario.flags.{k}")

    _assert_has(payload, "decision", ctx="payload")
    _assert_has(payload["decision"], "recommendation", ctx="decision")
    _assert_has(payload["decision"], "confidence", ctx="decision")

    rec = payload["decision"]["recommendation"]
    _assert_type(rec, dict, ctx="decision.recommendation")
    for k in ("type", "recommended_lap", "expected_gain_loss_s", "base_reason"):
        _assert_has(rec, k, ctx="decision.recommendation")
    _assert_in(rec["type"], {"PIT_NOW", "PIT_SOON", "EXTEND"}, ctx="decision.recommendation.type")
    _assert_type(rec["recommended_lap"], int, ctx="decision.recommendation.recommended_lap")
    if rec["recommended_lap"] <= 0:
        _fail("decision.recommendation.recommended_lap must be > 0")
    _assert_type(rec["expected_gain_loss_s"], (int, float), ctx="decision.recommendation.expected_gain_loss_s")
    _assert_type(rec["base_reason"], str, ctx="decision.recommendation.base_reason")

    conf = payload["decision"]["confidence"]
    _assert_type(conf, dict, ctx="decision.confidence")
    for k in ("confidence", "risk_level", "breakdown", "decision_confidence", "decision_risk_levels"):
        _assert_has(conf, k, ctx="decision.confidence")
    _assert_num_range(conf["confidence"], lo=0.0, hi=1.0, ctx="decision.confidence.confidence")
    _assert_in(conf["risk_level"], {"Low", "Medium", "Medium-High", "High"}, ctx="decision.confidence.risk_level")
    _assert_type(conf["breakdown"], dict, ctx="decision.confidence.breakdown")
    _assert_type(conf["decision_confidence"], dict, ctx="decision.confidence.decision_confidence")
    _assert_type(conf["decision_risk_levels"], dict, ctx="decision.confidence.decision_risk_levels")

    # `penalties` is an internal model field; JSON may omit it in favor of decision.issues.
    if "penalties" in conf:
        _assert_type(conf["penalties"], list, ctx="decision.confidence.penalties")

    for k, v in conf["decision_confidence"].items():
        _assert_num_range(v, lo=0.0, hi=1.0, ctx=f"decision.confidence.decision_confidence[{k!r}]")
    for k, v in conf["decision_risk_levels"].items():
        _assert_in(v, {"Low", "Medium", "Medium-High", "High"}, ctx=f"decision.confidence.decision_risk_levels[{k!r}]")

    # Confidence breakdown keys are part of the v2 contract.
    breakdown = conf["breakdown"]
    expected_breakdown_keys = {
        "data_completeness",
        "signal_agreement",
        "model_alignment",
        "context_stability",
        "penalty_score",
    }
    missing_breakdown = sorted(expected_breakdown_keys - set(breakdown.keys()))
    if missing_breakdown:
        _fail(f"decision.confidence.breakdown missing keys: {missing_breakdown}")
    for k in expected_breakdown_keys:
        _assert_num_range(breakdown[k], lo=0.0, hi=1.0, ctx=f"decision.confidence.breakdown.{k}")

    # Per-decision confidence must include at least the domains we surface in artifacts.
    decision_conf = conf["decision_confidence"]
    for req in ("pit_timing", "tyre_strategy"):
        if req not in decision_conf:
            _fail(f"decision.confidence.decision_confidence missing required domain: {req}")
        _assert_num_range(
            decision_conf[req], lo=0.0, hi=1.0, ctx=f"decision.confidence.decision_confidence.{req}"
        )

    # Keep output schema stable: these keys must exist even if values are null/empty.
    for k in (
        "uncertainty",
        "recommended_mode",
        "mode_options",
        "conflict",
        "issues",
        "fallback_actions",
        "explanation",
    ):
        _assert_has(payload["decision"], k, ctx="decision")

    _assert_type(payload["decision"].get("recommended_mode"), str, ctx="decision.recommended_mode")
    _assert_in(
        payload["decision"]["recommended_mode"],
        {"SAFE", "AGGRESSIVE", "UNKNOWN"},
        ctx="decision.recommended_mode",
    )

    issues = payload["decision"].get("issues")
    if not isinstance(issues, list):
        _fail("decision.issues must be a list")
    for it in issues:
        if not isinstance(it, dict):
            _fail("each decision.issues item must be an object")
        for k in ("issue", "severity", "penalty", "affected_decisions", "reason"):
            _assert_has(it, k, ctx="decision.issues[*]")
        _assert_type(it["issue"], str, ctx="decision.issues[*].issue")
        _assert_in(it["severity"], {"low", "medium", "high"}, ctx="decision.issues[*].severity")
        _assert_num_range(it["penalty"], lo=0.0, hi=1.0, ctx="decision.issues[*].penalty")
        _assert_type(it["affected_decisions"], list, ctx="decision.issues[*].affected_decisions")
        for ad in it["affected_decisions"]:
            _assert_type(ad, str, ctx="decision.issues[*].affected_decisions[*]")
        _assert_type(it["reason"], str, ctx="decision.issues[*].reason")

    fallbacks = payload["decision"].get("fallback_actions")
    if not isinstance(fallbacks, list):
        _fail("decision.fallback_actions must be a list")
    for f in fallbacks:
        _assert_type(f, str, ctx="decision.fallback_actions[*]")

    _assert_type(payload["decision"].get("explanation"), str, ctx="decision.explanation")
    if not str(payload["decision"].get("explanation") or "").strip():
        _fail("decision.explanation must be non-empty")

    # Optional blocks should have stable shapes.
    uncertainty = payload["decision"].get("uncertainty")
    if not isinstance(uncertainty, dict):
        _fail("decision.uncertainty must be an object")
    for k in ("primary_uncertainty", "uncertainty_score", "downstream_decisions_at_risk", "drivers"):
        _assert_has(uncertainty, k, ctx="decision.uncertainty")
    _assert_type(uncertainty["primary_uncertainty"], str, ctx="decision.uncertainty.primary_uncertainty")
    _assert_num_range(uncertainty["uncertainty_score"], lo=0.0, hi=1.0, ctx="decision.uncertainty.uncertainty_score")
    _assert_type(uncertainty["downstream_decisions_at_risk"], list, ctx="decision.uncertainty.downstream_decisions_at_risk")
    _assert_type(uncertainty["drivers"], list, ctx="decision.uncertainty.drivers")

    mode_options = payload["decision"].get("mode_options")
    if not isinstance(mode_options, list):
        _fail("decision.mode_options must be a list")
    for opt in mode_options:
        if not isinstance(opt, dict):
            _fail("each decision.mode_options item must be an object")
        for k in ("mode", "recommendation", "decision_impact"):
            _assert_has(opt, k, ctx="decision.mode_options[*]")
        _assert_in(opt["mode"], {"SAFE", "AGGRESSIVE"}, ctx="decision.mode_options[*].mode")
        rec = opt.get("recommendation")
        if not isinstance(rec, dict):
            _fail("mode option recommendation must be an object")
        for k in ("type", "recommended_lap", "expected_gain_loss_s", "base_reason"):
            _assert_has(rec, k, ctx="decision.mode_options[*].recommendation")
        _assert_in(rec["type"], {"PIT_NOW", "PIT_SOON", "EXTEND"}, ctx="decision.mode_options[*].recommendation.type")
        _assert_type(rec["recommended_lap"], int, ctx="decision.mode_options[*].recommendation.recommended_lap")
        if rec["recommended_lap"] <= 0:
            _fail("decision.mode_options[*].recommendation.recommended_lap must be > 0")
        _assert_type(
            rec["expected_gain_loss_s"],
            (int, float),
            ctx="decision.mode_options[*].recommendation.expected_gain_loss_s",
        )
        _assert_type(rec["base_reason"], str, ctx="decision.mode_options[*].recommendation.base_reason")
        di = opt.get("decision_impact")
        if not isinstance(di, dict):
            _fail("mode option decision_impact must be an object")
        for k in (
            "decision",
            "horizon_laps",
            "if_right_expected_gain_s",
            "if_wrong_expected_loss_s",
            "risk_level",
            "assumptions",
            "notes",
        ):
            _assert_has(di, k, ctx="decision.mode_options[*].decision_impact")
        _assert_type(di["decision"], str, ctx="decision.mode_options[*].decision_impact.decision")
        _assert_type(di["horizon_laps"], int, ctx="decision.mode_options[*].decision_impact.horizon_laps")
        if di["horizon_laps"] <= 0:
            _fail("decision.mode_options[*].decision_impact.horizon_laps must be > 0")
        _assert_num_min(
            di["if_right_expected_gain_s"],
            lo=0.0,
            ctx="decision.mode_options[*].decision_impact.if_right_expected_gain_s",
        )
        _assert_num_min(
            di["if_wrong_expected_loss_s"],
            lo=0.0,
            ctx="decision.mode_options[*].decision_impact.if_wrong_expected_loss_s",
        )
        _assert_type(di["risk_level"], str, ctx="decision.mode_options[*].decision_impact.risk_level")
        _assert_type(di["assumptions"], dict, ctx="decision.mode_options[*].decision_impact.assumptions")
        _assert_type(di["notes"], list, ctx="decision.mode_options[*].decision_impact.notes")

    conflict = payload["decision"].get("conflict")
    _assert_type(conflict, dict, ctx="decision.conflict")
    for k in ("score", "label"):
        _assert_has(conflict, k, ctx="decision.conflict")
    _assert_num_range(conflict["score"], lo=0.0, hi=1.0, ctx="decision.conflict.score")
    _assert_in(conflict["label"], {"NONE", "LOW", "MEDIUM", "HIGH"}, ctx="decision.conflict.label")

    _assert_has(payload, "decision_impact", ctx="payload")
    _assert_has(payload, "decision_impacts", ctx="payload")
    if not isinstance(payload["decision_impacts"], list):
        _fail("decision_impacts must be a list")

    decision_impact = payload.get("decision_impact")
    if not isinstance(decision_impact, dict):
        _fail("decision_impact must be an object")
    for k in ("decision", "horizon_laps", "if_right_expected_gain_s", "if_wrong_expected_loss_s", "risk_level"):
        _assert_has(decision_impact, k, ctx="decision_impact")
    _assert_type(decision_impact["decision"], str, ctx="decision_impact.decision")
    _assert_type(decision_impact["horizon_laps"], int, ctx="decision_impact.horizon_laps")
    if decision_impact["horizon_laps"] <= 0:
        _fail("decision_impact.horizon_laps must be > 0")
    _assert_num_min(decision_impact["if_right_expected_gain_s"], lo=0.0, ctx="decision_impact.if_right_expected_gain_s")
    _assert_num_min(decision_impact["if_wrong_expected_loss_s"], lo=0.0, ctx="decision_impact.if_wrong_expected_loss_s")
    _assert_type(decision_impact["risk_level"], str, ctx="decision_impact.risk_level")
    _assert_type(decision_impact.get("assumptions", {}), dict, ctx="decision_impact.assumptions")
    _assert_type(decision_impact.get("notes", []), list, ctx="decision_impact.notes")

    for di in payload["decision_impacts"]:
        if not isinstance(di, dict):
            _fail("each decision_impacts item must be an object")
        _assert_has(di, "decision", ctx="decision_impacts[*]")
        _assert_has(di, "horizon_laps", ctx=f"decision_impacts[{di.get('decision')}]")
        _assert_has(di, "if_right_expected_gain_s", ctx=f"decision_impacts[{di.get('decision')}]")
        _assert_has(di, "if_wrong_expected_loss_s", ctx=f"decision_impacts[{di.get('decision')}]")
        _assert_has(di, "risk_level", ctx=f"decision_impacts[{di.get('decision')}]")
        _assert_has(di, "assumptions", ctx=f"decision_impacts[{di.get('decision')}]")
        _assert_has(di, "notes", ctx=f"decision_impacts[{di.get('decision')}]")
        _assert_type(di["decision"], str, ctx="decision_impacts[*].decision")
        _assert_type(di["horizon_laps"], int, ctx=f"decision_impacts[{di.get('decision')}].horizon_laps")
        if di["horizon_laps"] <= 0:
            _fail(f"decision_impacts[{di.get('decision')}].horizon_laps must be > 0")
        _assert_num_min(
            di["if_right_expected_gain_s"],
            lo=0.0,
            ctx=f"decision_impacts[{di.get('decision')}].if_right_expected_gain_s",
        )
        _assert_num_min(
            di["if_wrong_expected_loss_s"],
            lo=0.0,
            ctx=f"decision_impacts[{di.get('decision')}].if_wrong_expected_loss_s",
        )
        _assert_type(di["risk_level"], str, ctx=f"decision_impacts[{di.get('decision')}].risk_level")
        _assert_type(di["assumptions"], dict, ctx=f"decision_impacts[{di.get('decision')}].assumptions")
        _assert_type(di["notes"], list, ctx=f"decision_impacts[{di.get('decision')}].notes")

    # Enumerate risk levels for impacts. (The confidence risk_level uses different labels.)
    for _ctx, _di in [
        ("decision_impact", decision_impact),
        *[(f"decision_impacts[{d.get('decision')}]", d) for d in payload["decision_impacts"]],
    ]:
        rl = _di.get("risk_level")
        if not isinstance(rl, str):
            _fail(f"{_ctx}.risk_level must be a string")
        _assert_in(rl, {"LOW", "MEDIUM", "HIGH"}, ctx=f"{_ctx}.risk_level")

    _assert_has(payload, "artifacts", ctx="payload")
    _assert_has(payload["artifacts"], "markdown", ctx="artifacts")
    _assert_has(payload["artifacts"], "registry_v2_csv", ctx="artifacts")
    _assert_has(payload["artifacts"], "json", ctx="artifacts")
    _assert_type(payload["artifacts"]["json"], str, ctx="artifacts.json")
    _assert_type(payload["artifacts"]["markdown"], str, ctx="artifacts.markdown")
    if payload["artifacts"]["json"].strip() == "" or payload["artifacts"]["markdown"].strip() == "":
        _fail("artifacts.json and artifacts.markdown must be non-empty strings")

    for key in ("json", "markdown", "registry_csv", "registry_v2_csv"):
        if key not in payload["artifacts"]:
            continue
        v = payload["artifacts"].get(key)
        if v is None:
            continue
        _assert_type(v, str, ctx=f"artifacts.{key}")
        if v.strip() == "":
            _fail(f"artifacts.{key} must be a non-empty string")
        # We persist absolute paths for traceability.
        if not os.path.isabs(v):
            _fail(f"artifacts.{key} must be an absolute path")
        if not Path(v).exists():
            if key == "markdown":
                _fail(f"Markdown artifact not found: {v!r}")
            _fail(f"artifacts.{key} path does not exist: {v!r}")

    _assert_has(payload, "model_validation", ctx="payload")
    mv = payload["model_validation"]
    if not isinstance(mv, dict):
        _fail("model_validation must be an object")
    for k in (
        "status",
        "window_laps",
        "deviation_threshold_s",
        "mean_abs_error_s",
        "max_abs_error_s",
        "recommended_confidence_penalty",
        "deviations",
    ):
        _assert_has(mv, k, ctx="model_validation")
    _assert_in(mv["status"], {"OK", "DEVIATION", "NO_DATA"}, ctx="model_validation.status")
    _assert_int_min(mv["window_laps"], lo=1, ctx="model_validation.window_laps")
    _assert_num_min(mv["deviation_threshold_s"], lo=0.0, ctx="model_validation.deviation_threshold_s")
    _assert_num_min(mv["mean_abs_error_s"], lo=0.0, ctx="model_validation.mean_abs_error_s")
    _assert_num_min(mv["max_abs_error_s"], lo=0.0, ctx="model_validation.max_abs_error_s")
    if float(mv["max_abs_error_s"]) + 1e-9 < float(mv["mean_abs_error_s"]):
        _fail("model_validation.max_abs_error_s must be >= mean_abs_error_s")
    _assert_num_range(
        mv["recommended_confidence_penalty"],
        lo=0.0,
        hi=1.0,
        ctx="model_validation.recommended_confidence_penalty",
    )
    _assert_type(mv["deviations"], list, ctx="model_validation.deviations")
    for d in mv["deviations"]:
        if not isinstance(d, dict):
            _fail("each model_validation.deviations item must be an object")
        for k in ("lap", "expected_lap_time_s", "actual_lap_time_s", "delta_s"):
            _assert_has(d, k, ctx="model_validation.deviations[*]")
        _assert_type(d["lap"], int, ctx="model_validation.deviations[*].lap")
        _assert_num_min(d["expected_lap_time_s"], lo=0.0, ctx="model_validation.deviations[*].expected_lap_time_s")
        _assert_num_min(d["actual_lap_time_s"], lo=0.0, ctx="model_validation.deviations[*].actual_lap_time_s")
        _assert_type(d["delta_s"], (int, float), ctx="model_validation.deviations[*].delta_s")


def validate_markdown(
    md_path: Path,
    *,
    run_id: str,
    expected_timestamp_utc: str | None = None,
    expected_source_name: str | None = None,
    expected_confidence_scoring_version: str | None = None,
    expected_dataset_license: str | None = None,
    expected_dataset_source_url: str | None = None,
    expected_recommendation_type: str | None = None,
    expected_recommended_lap: int | None = None,
    expected_expected_gain_loss_s: float | None = None,
    expected_confidence: float | None = None,
    expected_risk_level: str | None = None,
    expected_conflict_score: float | None = None,
    expected_conflict_label: str | None = None,
    expected_impact_decision: str | None = None,
    expected_impact_horizon_laps: int | None = None,
    expected_impact_if_right_gain_s: float | None = None,
    expected_impact_if_wrong_loss_s: float | None = None,
    expected_impact_risk_level: str | None = None,
    expected_recommended_mode: str | None = None,
    expected_uncertainty_primary: str | None = None,
    expected_uncertainty_score: float | None = None,
    expected_mode_option_lines: list[str] | None = None,
    expected_additional_impact_lines: list[str] | None = None,
    expected_explanation: str | None = None,
) -> None:
    if not md_path.exists():
        _fail(f"Markdown artifact not found: {md_path}")
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception as exc:
        _fail(f"Markdown artifact could not be read: {md_path} ({exc})")
    if f"Run ID: `{run_id}`" not in text:
        _fail("Markdown missing Run ID line")

    lines = text.splitlines()

    def _section_text(section: str) -> str:
        try:
            idx = lines.index(section)
        except ValueError:
            return ""
        out: list[str] = []
        for line in lines[idx + 1 :]:
            if line.startswith("## "):
                break
            out.append(line)
        return "\n".join(out)

    if expected_timestamp_utc is not None:
        exp = str(expected_timestamp_utc).strip()
        if exp and f"- Timestamp (UTC): `{exp}`" not in text:
            _fail("Markdown timestamp does not match JSON timestamp_utc")

    if expected_source_name is not None:
        exp = str(expected_source_name)
        if f"- Source: `{exp}`" not in text:
            _fail("Markdown source does not match JSON source.source_name")

    if expected_confidence_scoring_version is not None:
        exp = str(expected_confidence_scoring_version).strip()
        if exp and f"- Confidence scoring version: `{exp}`" not in text:
            _fail("Markdown confidence scoring version does not match JSON")

    # Recommendation + core confidence fields should match JSON (demo contract).
    rec_text = _section_text("## Recommendation")
    if expected_recommendation_type is not None:
        exp = str(expected_recommendation_type).strip()
        if exp and f"- Type: `{exp}`" not in rec_text:
            _fail("Markdown recommendation type does not match JSON")
    if expected_recommended_lap is not None:
        if f"- Target lap: `{int(expected_recommended_lap)}`" not in rec_text:
            _fail("Markdown recommended lap does not match JSON")
    if expected_expected_gain_loss_s is not None:
        exp = str(expected_expected_gain_loss_s)
        if f"- Expected gain/loss (heuristic): `{exp}` seconds" not in rec_text:
            _fail("Markdown expected gain/loss does not match JSON")

    conf_text = _section_text("## Confidence")
    if expected_confidence is not None:
        # We render the raw float in markdown; compare by exact string to keep artifacts deterministic.
        exp = str(expected_confidence)
        if f"- Confidence: `{exp}`" not in conf_text:
            _fail("Markdown confidence does not match JSON")
    if expected_risk_level is not None:
        exp = str(expected_risk_level).strip()
        if exp and f"- Risk: `{exp}`" not in conf_text:
            _fail("Markdown risk level does not match JSON")
    if expected_conflict_score is not None or expected_conflict_label is not None:
        score = str(expected_conflict_score) if expected_conflict_score is not None else ""
        label = str(expected_conflict_label) if expected_conflict_label is not None else ""
        if score and label:
            expected = f"- Conflict score: `{score}` ({label})"
        elif score:
            expected = f"- Conflict score: `{score}`"
        else:
            expected = f"- Conflict score: ({label})"
        if expected not in conf_text:
            _fail("Markdown conflict score/label does not match JSON")

    # Decision impact snapshot should match JSON.
    impact_text = _section_text("## Decision Impact Simulation")
    if expected_impact_decision is not None:
        exp = str(expected_impact_decision).strip()
        if exp and f"- Decision: `{exp}`" not in impact_text:
            _fail("Markdown decision impact decision does not match JSON")
    if expected_impact_horizon_laps is not None:
        if f"- Horizon laps: `{int(expected_impact_horizon_laps)}`" not in impact_text:
            _fail("Markdown decision impact horizon does not match JSON")
    if expected_impact_if_right_gain_s is not None:
        exp = str(expected_impact_if_right_gain_s)
        if f"- If right (expected gain): `{exp}` s" not in impact_text:
            _fail("Markdown decision impact if-right gain does not match JSON")
    if expected_impact_if_wrong_loss_s is not None:
        exp = str(expected_impact_if_wrong_loss_s)
        if f"- If wrong (expected loss): `{exp}` s" not in impact_text:
            _fail("Markdown decision impact if-wrong loss does not match JSON")
    if expected_impact_risk_level is not None:
        exp = str(expected_impact_risk_level).strip()
        if exp and f"- Risk: `{exp}`" not in impact_text:
            _fail("Markdown decision impact risk does not match JSON")

    # Provenance fields near the top are part of the demo contract.
    # Keep the check simple to avoid brittle ordering assumptions.
    if "- Dataset license:" not in text:
        _fail("Markdown missing dataset license line")
    if "- Dataset source URL:" not in text:
        _fail("Markdown missing dataset source URL line")

    if expected_dataset_license is not None:
        exp = str(expected_dataset_license).strip() or "NOASSERTION"
        if f"- Dataset license: `{exp}`" not in text:
            _fail("Markdown dataset license does not match JSON source.dataset_metadata")
    if expected_dataset_source_url is not None:
        exp = str(expected_dataset_source_url).strip() or "UNKNOWN"
        if f"- Dataset source URL: `{exp}`" not in text:
            _fail("Markdown dataset source URL does not match JSON source.dataset_metadata")

    def _section_has_bullet(section: str, *, lookahead: int = 40) -> bool:
        """Return True if section contains at least one list item.

        We keep this deliberately simple: the CLI report uses bullet lists for
        most sections. This check prevents empty sections that visually hide
        important fields (e.g. mode options, fallbacks, provenance warnings).
        """

        try:
            idx = lines.index(section)
        except ValueError:
            return False

        for line in lines[idx + 1 : idx + 1 + lookahead]:
            if line.startswith("## "):
                break
            if line.strip().startswith("- "):
                return True
        return False

    # Section presence: these headings are part of the CLI contract.
    required_sections = [
        "## Recommendation",
        "## Confidence",
        "## Uncertainty",
        "## Recommended Mode",
        "## Mode Options",
        "## Issues",
        "## Fallback Actions",
        "## Model Validation (Model vs Reality)",
        "## Decision Impact Simulation",
        "## Additional Decision Impacts",
        "## Explanation",
        "## Provenance Warnings",
    ]
    for section in required_sections:
        if section not in text:
            _fail(f"Markdown missing section: {section}")

    # Minimal content checks (avoid brittle full-text assertions).
    if "- Recommended mode:" not in text:
        _fail("Markdown missing recommended mode line")
    if expected_recommended_mode is not None:
        if f"- Recommended mode: `{expected_recommended_mode}`" not in text:
            _fail(
                "Markdown recommended mode does not match JSON: "
                f"expected={expected_recommended_mode!r}"
            )
    if "- Primary:" not in text or "- Score:" not in text:
        _fail("Markdown missing uncertainty primary/score lines")
    if expected_uncertainty_primary is not None:
        uncertainty_text = _section_text("## Uncertainty")
        if f"- Primary: `{expected_uncertainty_primary}`" not in uncertainty_text:
            _fail(
                "Markdown uncertainty primary does not match JSON: "
                f"expected={expected_uncertainty_primary!r}"
            )
    if expected_uncertainty_score is not None:
        uncertainty_text = _section_text("## Uncertainty")
        exp = str(expected_uncertainty_score)
        if f"- Score: `{exp}`" not in uncertainty_text:
            _fail("Markdown uncertainty score does not match JSON")
    impact_text = _section_text("## Decision Impact Simulation")
    if "## Decision Impact Simulation" in text and "- If wrong (expected loss):" not in impact_text:
        _fail("Markdown missing decision impact if-wrong loss line")

    # List sections should never be empty; they must at least render "- None".
    for section in (
        "## Mode Options",
        "## Issues",
        "## Fallback Actions",
        "## Additional Decision Impacts",
        "## Provenance Warnings",
    ):
        if not _section_has_bullet(section):
            _fail(f"Markdown section has no bullet list items: {section}")

    # Mode options should match JSON exactly line-by-line (but not necessarily ordering).
    if expected_mode_option_lines is not None:
        mode_text = _section_text("## Mode Options")
        if expected_mode_option_lines:
            if "- None" in mode_text:
                _fail("Markdown mode options rendered '- None' but JSON has options")
            for exp in expected_mode_option_lines:
                if exp not in mode_text:
                    _fail("Markdown mode option does not match JSON")
        else:
            if "- None" not in mode_text:
                _fail("Markdown mode options missing '- None' for empty JSON")

    # Additional decision impacts should match JSON exactly line-by-line.
    if expected_additional_impact_lines is not None:
        impacts_text = _section_text("## Additional Decision Impacts")
        if expected_additional_impact_lines:
            if "- None" in impacts_text:
                _fail("Markdown additional impacts rendered '- None' but JSON has impacts")
            for exp in expected_additional_impact_lines:
                if exp not in impacts_text:
                    _fail("Markdown additional decision impact does not match JSON")
        else:
            if "- None" not in impacts_text:
                _fail("Markdown additional impacts missing '- None' for empty JSON")

    if expected_explanation is not None:
        expl = str(expected_explanation).strip()
        if expl:
            # Be tolerant to markdown line wrapping/spacing differences.
            norm_text = re.sub(r"\s+", " ", text).strip()
            norm_expl = re.sub(r"\s+", " ", expl).strip()
            if norm_expl not in norm_text:
                _fail("Markdown explanation does not match JSON decision.explanation")


def validate_registry_v2(v2_path: Path) -> None:
    if not v2_path.exists():
        _fail(f"Missing v2 registry: {v2_path}")
    lines = v2_path.read_text(encoding="utf-8").splitlines()
    header = lines[0] if lines else ""
    required_cols = {
        "run_id",
        "timestamp_utc",
        "source_name",
        "dataset_license",
        "dataset_source_url",
        "preset",
        "confidence_scoring_version",
        "uncertainty_primary",
        "recommended_mode",
        "safe_mode_if_wrong_loss_s",
        "aggressive_mode_if_wrong_loss_s",
        "push_vs_conserve_if_wrong_loss_s",
    }

    header_cols = header.split(",") if header else []
    cols = set(header_cols)
    missing = sorted(required_cols - cols)
    if missing:
        _fail(f"v2 registry missing columns: {missing}")

    expected_order = [
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
    if header_cols[: len(expected_order)] != expected_order:
        _fail("v2 registry header order does not match canonical schema")


def _parse_float_maybe(s: str) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception as exc:
        raise ValueError(f"not a float: {s!r}") from exc


def _float_close(a: float | None, b: float | None, *, tol: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def validate_registry_v2_row(v2_path: Path, *, payload: dict) -> None:
    """Ensure the registry has a row for the JSON run_id.

    Also sanity-check that numeric columns remain parseable when present.
    """

    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        _fail("payload.run_id must be a non-empty string")

    with v2_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (r.get("run_id") or "").strip() == run_id]
    if not rows:
        _fail(f"v2 registry missing run_id row: {run_id}")

    r = rows[-1]

    # Cross-check key registry fields against JSON payload.
    if (r.get("timestamp_utc") or "").strip() != (payload.get("timestamp_utc") or "").strip():
        _fail(
            f"v2 registry timestamp_utc mismatch for run_id={run_id}: "
            f"json={payload.get('timestamp_utc')!r}, csv={r.get('timestamp_utc')!r}"
        )
    if (r.get("source_name") or "").strip() != ((payload.get("source") or {}).get("source_name") or ""):
        _fail(
            f"v2 registry source_name mismatch for run_id={run_id}: "
            f"json={(payload.get('source') or {}).get('source_name')!r}, csv={r.get('source_name')!r}"
        )

    # Dataset provenance snapshot (optional in JSON metadata, but required as CSV columns).
    meta = ((payload.get("source") or {}).get("dataset_metadata") or {})
    expected_license = (meta.get("license_spdx") or meta.get("license_note") or "NOASSERTION").strip()
    if (r.get("dataset_license") or "").strip() != expected_license:
        _fail(
            f"v2 registry dataset_license mismatch for run_id={run_id}: "
            f"json={expected_license!r}, csv={r.get('dataset_license')!r}"
        )
    expected_url = (meta.get("source_url") or "").strip()
    if (r.get("dataset_source_url") or "").strip() != expected_url:
        _fail(
            f"v2 registry dataset_source_url mismatch for run_id={run_id}: "
            f"json={expected_url!r}, csv={r.get('dataset_source_url')!r}"
        )
    if (r.get("preset") or "").strip() != (((payload.get("scenario") or {}).get("flags") or {}).get("preset") or ""):
        _fail(
            f"v2 registry preset mismatch for run_id={run_id}: "
            f"json={(((payload.get('scenario') or {}).get('flags') or {}).get('preset'))!r}, csv={r.get('preset')!r}"
        )
    if (r.get("confidence_scoring_version") or "").strip() != (payload.get("confidence_scoring_version") or ""):
        _fail(
            f"v2 registry confidence_scoring_version mismatch for run_id={run_id}: "
            f"json={payload.get('confidence_scoring_version')!r}, csv={r.get('confidence_scoring_version')!r}"
        )

    # Recommendation mapping.
    rec = ((payload.get("decision") or {}).get("recommendation") or {})
    if (r.get("recommendation_type") or "").strip() != (rec.get("type") or ""):
        _fail(
            f"v2 registry recommendation_type mismatch for run_id={run_id}: "
            f"json={rec.get('type')!r}, csv={r.get('recommendation_type')!r}"
        )
    if str(r.get("recommended_lap") or "").strip() != str(rec.get("recommended_lap") or "").strip():
        _fail(
            f"v2 registry recommended_lap mismatch for run_id={run_id}: "
            f"json={rec.get('recommended_lap')!r}, csv={r.get('recommended_lap')!r}"
        )

    # Confidence + uncertainty mapping.
    conf = ((payload.get("decision") or {}).get("confidence") or {})
    conf_row = _parse_float_maybe(r.get("confidence", ""))
    if not _float_close(conf_row, float(conf.get("confidence", 0.0))):
        _fail(
            f"v2 registry confidence mismatch for run_id={run_id}: "
            f"json={conf.get('confidence')!r}, csv={r.get('confidence')!r}"
        )

    if (r.get("risk_level") or "").strip() != str(conf.get("risk_level") or "").strip():
        _fail(
            f"v2 registry risk_level mismatch for run_id={run_id}: "
            f"json={conf.get('risk_level')!r}, csv={r.get('risk_level')!r}"
        )

    # Per-decision confidence values that the registry snapshots.
    decision_conf = conf.get("decision_confidence") or {}
    pit_row = _parse_float_maybe(r.get("pit_timing_confidence", ""))
    pit_json = _parse_float_maybe(decision_conf.get("pit_timing", ""))
    if pit_json is not None and pit_row is None:
        _fail(f"v2 registry missing pit_timing_confidence for run_id={run_id}")
    if pit_json is not None and pit_row is not None and not _float_close(pit_row, pit_json):
        _fail(
            f"v2 registry pit_timing_confidence mismatch for run_id={run_id}: "
            f"json={decision_conf.get('pit_timing')!r}, csv={r.get('pit_timing_confidence')!r}"
        )
    tyre_row = _parse_float_maybe(r.get("tyre_strategy_confidence", ""))
    tyre_json = _parse_float_maybe(decision_conf.get("tyre_strategy", ""))
    if tyre_json is not None and tyre_row is None:
        _fail(f"v2 registry missing tyre_strategy_confidence for run_id={run_id}")
    if tyre_json is not None and tyre_row is not None and not _float_close(tyre_row, tyre_json):
        _fail(
            f"v2 registry tyre_strategy_confidence mismatch for run_id={run_id}: "
            f"json={decision_conf.get('tyre_strategy')!r}, csv={r.get('tyre_strategy_confidence')!r}"
        )

    u = ((payload.get("decision") or {}).get("uncertainty") or {})
    if (r.get("uncertainty_primary") or "").strip() != str(u.get("primary_uncertainty") or "").strip():
        _fail(
            f"v2 registry uncertainty_primary mismatch for run_id={run_id}: "
            f"json={u.get('primary_uncertainty')!r}, csv={r.get('uncertainty_primary')!r}"
        )
    u_row = _parse_float_maybe(r.get("uncertainty_score", ""))
    if not _float_close(u_row, float(u.get("uncertainty_score", 0.0))):
        _fail(
            f"v2 registry uncertainty_score mismatch for run_id={run_id}: "
            f"json={u.get('uncertainty_score')!r}, csv={r.get('uncertainty_score')!r}"
        )

    # Mode selection.
    if (r.get("recommended_mode") or "").strip() != str((payload.get("decision") or {}).get("recommended_mode") or ""):
        _fail(
            f"v2 registry recommended_mode mismatch for run_id={run_id}: "
            f"json={((payload.get('decision') or {}).get('recommended_mode'))!r}, csv={r.get('recommended_mode')!r}"
        )

    # Conflict mapping.
    conflict = ((payload.get("decision") or {}).get("conflict") or {})
    conflict_score_row = _parse_float_maybe(r.get("conflict_score", ""))
    conflict_score_json = _parse_float_maybe(conflict.get("score", ""))
    if conflict_score_json is not None and conflict_score_row is None:
        _fail(f"v2 registry missing conflict_score for run_id={run_id}")
    if conflict_score_json is not None and conflict_score_row is not None and not _float_close(
        conflict_score_row, conflict_score_json
    ):
        _fail(
            f"v2 registry conflict_score mismatch for run_id={run_id}: "
            f"json={conflict.get('score')!r}, csv={r.get('conflict_score')!r}"
        )
    if (r.get("conflict_label") or "").strip() != str(conflict.get("label") or "").strip():
        _fail(
            f"v2 registry conflict_label mismatch for run_id={run_id}: "
            f"json={conflict.get('label')!r}, csv={r.get('conflict_label')!r}"
        )

    # Impacts mapping.
    di0 = payload.get("decision_impact") or {}
    impact_row = _parse_float_maybe(r.get("impact_if_wrong_loss_s", ""))
    if not _float_close(impact_row, _parse_float_maybe(di0.get("if_wrong_expected_loss_s", "")) or 0.0):
        _fail(
            f"v2 registry impact_if_wrong_loss_s mismatch for run_id={run_id}: "
            f"json={di0.get('if_wrong_expected_loss_s')!r}, csv={r.get('impact_if_wrong_loss_s')!r}"
        )

    pvc_json = None
    for di in payload.get("decision_impacts") or []:
        if di.get("decision") == "push_vs_conserve":
            pvc_json = di
            break
    if pvc_json is not None:
        pvc_expected = _parse_float_maybe(pvc_json.get("if_wrong_expected_loss_s", ""))
        pvc_row = _parse_float_maybe(r.get("push_vs_conserve_if_wrong_loss_s", ""))

        # Keep registry rows complete when the JSON includes this decision.
        if pvc_expected is not None and pvc_row is None:
            _fail(f"v2 registry missing push_vs_conserve_if_wrong_loss_s for run_id={run_id}")

        if pvc_expected is not None and pvc_row is not None and not _float_close(pvc_row, pvc_expected):
            _fail(
                f"v2 registry push_vs_conserve_if_wrong_loss_s mismatch for run_id={run_id}: "
                f"json={pvc_json.get('if_wrong_expected_loss_s')!r}, csv={r.get('push_vs_conserve_if_wrong_loss_s')!r}"
            )
        if (r.get("push_vs_conserve_risk_level") or "").strip() != str(pvc_json.get("risk_level") or "").strip():
            _fail(
                f"v2 registry push_vs_conserve_risk_level mismatch for run_id={run_id}: "
                f"json={pvc_json.get('risk_level')!r}, csv={r.get('push_vs_conserve_risk_level')!r}"
            )

    # Model validation mapping.
    mv = payload.get("model_validation") or {}
    if (r.get("model_validation_status") or "").strip() != str(mv.get("status") or "").strip():
        _fail(
            f"v2 registry model_validation_status mismatch for run_id={run_id}: "
            f"json={mv.get('status')!r}, csv={r.get('model_validation_status')!r}"
        )
    mv_mae_row = _parse_float_maybe(r.get("model_validation_mae_s", ""))
    mv_mae_json = _parse_float_maybe(mv.get("mean_abs_error_s", ""))
    if mv_mae_json is not None and mv_mae_row is None:
        _fail(f"v2 registry missing model_validation_mae_s for run_id={run_id}")
    if mv_mae_json is not None and mv_mae_row is not None and not _float_close(mv_mae_row, mv_mae_json):
        _fail(
            f"v2 registry model_validation_mae_s mismatch for run_id={run_id}: "
            f"json={mv.get('mean_abs_error_s')!r}, csv={r.get('model_validation_mae_s')!r}"
        )

    # Issues mapping: count should match.
    issues = ((payload.get("decision") or {}).get("issues") or [])
    try:
        issue_count_row = int(str(r.get("issue_count") or "0").strip() or "0")
    except Exception:
        issue_count_row = -1
    if issue_count_row != len(issues):
        _fail(
            f"v2 registry issue_count mismatch for run_id={run_id}: "
            f"json={len(issues)}, csv={r.get('issue_count')!r}"
        )

    expected_issues_str = ";".join(sorted({str(i.get("issue")) for i in issues if i.get("issue") is not None}))
    if (r.get("issues") or "").strip() != expected_issues_str:
        _fail(
            f"v2 registry issues mismatch for run_id={run_id}: "
            f"json={expected_issues_str!r}, csv={r.get('issues')!r}"
        )

    # Paths should match exact payload strings.
    if (r.get("json_path") or "").strip() != ((payload.get("artifacts") or {}).get("json") or ""):
        _fail(f"v2 registry json_path mismatch for run_id={run_id}")
    if (r.get("md_path") or "").strip() != ((payload.get("artifacts") or {}).get("markdown") or ""):
        _fail(f"v2 registry md_path mismatch for run_id={run_id}")

    json_path = Path(r.get("json_path") or "")
    md_path = Path(r.get("md_path") or "")
    if not json_path.exists():
        _fail(f"v2 registry json_path does not exist for run_id={run_id}: {json_path}")
    if not md_path.exists():
        _fail(f"v2 registry md_path does not exist for run_id={run_id}: {md_path}")

    # Enum-like columns: keep them bounded.
    if (r.get("recommendation_type") or "") not in {"PIT_NOW", "PIT_SOON", "EXTEND"}:
        _fail(f"v2 registry bad recommendation_type for run_id={run_id}: {r.get('recommendation_type')!r}")
    if (r.get("recommended_mode") or "") not in {"SAFE", "AGGRESSIVE", "UNKNOWN"}:
        _fail(f"v2 registry bad recommended_mode for run_id={run_id}: {r.get('recommended_mode')!r}")
    if (r.get("risk_level") or "") not in {"Low", "Medium", "Medium-High", "High"}:
        _fail(f"v2 registry bad risk_level for run_id={run_id}: {r.get('risk_level')!r}")
    if (r.get("conflict_label") or "") not in {"NONE", "LOW", "MEDIUM", "HIGH"}:
        _fail(f"v2 registry bad conflict_label for run_id={run_id}: {r.get('conflict_label')!r}")
    if (r.get("model_validation_status") or "") not in {"OK", "DEVIATION", "NO_DATA"}:
        _fail(
            f"v2 registry bad model_validation_status for run_id={run_id}: {r.get('model_validation_status')!r}"
        )
    pvcr = (r.get("push_vs_conserve_risk_level") or "").strip()
    if pvcr and pvcr not in {"LOW", "MEDIUM", "HIGH"}:
        _fail(f"v2 registry bad push_vs_conserve_risk_level for run_id={run_id}: {pvcr!r}")

    for col in (
        "confidence",
        "pit_timing_confidence",
        "tyre_strategy_confidence",
        "uncertainty_score",
        "model_validation_mae_s",
        "impact_if_wrong_loss_s",
        "push_vs_conserve_if_wrong_loss_s",
        "safe_mode_if_wrong_loss_s",
        "aggressive_mode_if_wrong_loss_s",
        "conflict_score",
    ):
        try:
            _parse_float_maybe(r.get(col, ""))
        except Exception as exc:
            _fail(f"v2 registry column {col!r} not parseable for run_id={run_id}: {exc}")

    # When mode options exist in JSON, we expect mode-loss columns to be present.
    mode_options = (payload.get("decision") or {}).get("mode_options") or []
    if mode_options:
        safe_expected = None
        aggressive_expected = None
        for opt in mode_options:
            mode = opt.get("mode")
            loss = _parse_float_maybe((opt.get("decision_impact") or {}).get("if_wrong_expected_loss_s", ""))
            if mode == "SAFE":
                safe_expected = loss
            elif mode == "AGGRESSIVE":
                aggressive_expected = loss

        safe_row = _parse_float_maybe(r.get("safe_mode_if_wrong_loss_s", ""))
        aggressive_row = _parse_float_maybe(r.get("aggressive_mode_if_wrong_loss_s", ""))

        if safe_expected is not None and safe_row is None:
            _fail(f"v2 registry missing safe_mode_if_wrong_loss_s for run_id={run_id}")
        if aggressive_expected is not None and aggressive_row is None:
            _fail(f"v2 registry missing aggressive_mode_if_wrong_loss_s for run_id={run_id}")

        # Keep it tolerant (string/float formatting differences), but ensure values match the JSON.
        if safe_expected is not None and safe_row is not None and not _float_close(safe_expected, safe_row):
            _fail(
                f"v2 registry safe_mode_if_wrong_loss_s mismatch for run_id={run_id}: "
                f"json={safe_expected}, csv={safe_row}"
            )
        if aggressive_expected is not None and aggressive_row is not None and not _float_close(
            aggressive_expected, aggressive_row
        ):
            _fail(
                f"v2 registry aggressive_mode_if_wrong_loss_s mismatch for run_id={run_id}: "
                f"json={aggressive_expected}, csv={aggressive_row}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate MDCE CLI artifacts (cheap sanity checks).")
    p.add_argument(
        "--outputs-dir",
        default="outputs/reports",
        help="Directory containing mdce_decision_run_*.json artifacts.",
    )
    p.add_argument(
        "--json",
        default=None,
        help="Optional explicit JSON artifact path. If omitted, uses latest in outputs-dir.",
    )
    p.add_argument(
        "--registry-v2",
        default=None,
        help="Expected v2 registry path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    json_path = Path(args.json) if args.json else _latest_run_json(outputs_dir)
    payload = _load_json(json_path)
    validate_payload(payload)

    # If the payload claims a JSON artifact path, it must match what we validated.
    if (payload.get("artifacts") or {}).get("json") and Path(payload["artifacts"]["json"]).resolve() != json_path.resolve():
        _fail(
            "Payload artifacts.json does not match the JSON file used for validation: "
            f"arg={str(json_path)!r}, payload={payload['artifacts']['json']!r}"
        )

    md_path = Path(payload["artifacts"]["markdown"])
    expected_mode = (payload.get("decision") or {}).get("recommended_mode")
    expected_primary = ((payload.get("decision") or {}).get("uncertainty") or {}).get("primary_uncertainty")
    expected_expl = (payload.get("decision") or {}).get("explanation")
    meta = ((payload.get("source") or {}).get("dataset_metadata") or {})
    expected_license = (meta.get("license_spdx") or meta.get("license_note") or "NOASSERTION")
    expected_url = (meta.get("source_url") or "UNKNOWN")

    rec = ((payload.get("decision") or {}).get("recommendation") or {})
    conf = ((payload.get("decision") or {}).get("confidence") or {})
    conflict = ((payload.get("decision") or {}).get("conflict") or {})
    impact = (payload.get("decision_impact") or {})
    unc = ((payload.get("decision") or {}).get("uncertainty") or {})
    mode_option_lines: list[str] = []
    for opt in (payload.get("decision") or {}).get("mode_options") or []:
        try:
            mode = opt.get("mode")
            orec = opt.get("recommendation") or {}
            oimp = opt.get("decision_impact") or {}
            mode_option_lines.append(
                "- "
                f"{mode}: {orec.get('type')} (lap {orec.get('recommended_lap')}), "
                f"if-wrong loss {oimp.get('if_wrong_expected_loss_s')}s (risk={oimp.get('risk_level')})"
            )
        except Exception:
            # Let the JSON schema validator fail on malformed shapes; keep this robust.
            continue

    additional_impact_lines: list[str] = []
    for di in payload.get("decision_impacts") or []:
        try:
            additional_impact_lines.append(
                f"- {di.get('decision')}: if-wrong loss `{di.get('if_wrong_expected_loss_s')}` s "
                f"(risk={di.get('risk_level')}, horizon={di.get('horizon_laps')} laps)"
            )
        except Exception:
            continue
    validate_markdown(
        md_path,
        run_id=payload["run_id"],
        expected_timestamp_utc=str(payload.get("timestamp_utc") or ""),
        expected_source_name=str(((payload.get("source") or {}).get("source_name") or "")),
        expected_confidence_scoring_version=str(payload.get("confidence_scoring_version") or ""),
        expected_dataset_license=str(expected_license),
        expected_dataset_source_url=str(expected_url),
        expected_recommendation_type=str(rec.get("type") or ""),
        expected_recommended_lap=int(rec.get("recommended_lap") or 0) if rec.get("recommended_lap") is not None else None,
        expected_expected_gain_loss_s=float(rec.get("expected_gain_loss_s") or 0.0)
        if rec.get("expected_gain_loss_s") is not None
        else None,
        expected_confidence=float(conf.get("confidence") or 0.0) if conf.get("confidence") is not None else None,
        expected_risk_level=str(conf.get("risk_level") or ""),
        expected_conflict_score=float(conflict.get("score") or 0.0) if conflict.get("score") is not None else None,
        expected_conflict_label=str(conflict.get("label") or ""),
        expected_impact_decision=str(impact.get("decision") or ""),
        expected_impact_horizon_laps=int(impact.get("horizon_laps") or 0) if impact.get("horizon_laps") is not None else None,
        expected_impact_if_right_gain_s=float(impact.get("if_right_expected_gain_s") or 0.0)
        if impact.get("if_right_expected_gain_s") is not None
        else None,
        expected_impact_if_wrong_loss_s=float(impact.get("if_wrong_expected_loss_s") or 0.0)
        if impact.get("if_wrong_expected_loss_s") is not None
        else None,
        expected_impact_risk_level=str(impact.get("risk_level") or ""),
        expected_recommended_mode=str(expected_mode) if expected_mode is not None else None,
        expected_uncertainty_primary=str(expected_primary) if expected_primary is not None else None,
        expected_uncertainty_score=float(unc.get("uncertainty_score") or 0.0)
        if unc.get("uncertainty_score") is not None
        else None,
        expected_mode_option_lines=mode_option_lines,
        expected_additional_impact_lines=additional_impact_lines,
        expected_explanation=str(expected_expl) if expected_expl is not None else None,
    )

    # Prefer the persisted v2 registry location when present.
    payload_v2 = (payload.get("artifacts", {}) or {}).get("registry_v2_csv")
    if args.registry_v2 is not None and payload_v2 is not None:
        if Path(args.registry_v2).resolve() != Path(payload_v2).resolve():
            _fail(
                "Registry v2 arg does not match payload artifacts.registry_v2_csv: "
                f"arg={args.registry_v2!r}, payload={payload_v2!r}"
            )

    v2_path = Path(payload_v2 or args.registry_v2 or "outputs/reports/MDCE_DECISION_RUN_REGISTRY_v2.csv")
    validate_registry_v2(v2_path)
    validate_registry_v2_row(v2_path, payload=payload)

    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
        raise
