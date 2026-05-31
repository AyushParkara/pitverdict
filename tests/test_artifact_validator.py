from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _write_min_valid_v2_payload(tmp: Path) -> tuple[dict, Path]:
    """Create a minimally valid v2 payload + on-disk artifact files.

    Returns: (payload, json_path)
    """

    root = tmp
    out_dir = root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = "20260101_000000"
    json_path = out_dir / f"mdce_decision_run_{run_id}.json"
    md_path = out_dir / f"mdce_decision_run_{run_id}.md"
    registry_path = out_dir / "MDCE_DECISION_RUN_REGISTRY.csv"
    registry_v2_path = out_dir / "MDCE_DECISION_RUN_REGISTRY_v2.csv"

    # Create placeholder artifacts to satisfy existence checks.
    md_path.write_text(
        "\n".join(
            [
                "# MDCE Decision Run",
                "",
                f"- Run ID: `{run_id}`",
                "- Dataset license: `NOASSERTION`",
                "- Dataset source URL: `UNKNOWN`",
                "",
                "## Recommendation",
                "",
                "## Confidence",
                "",
                "## Uncertainty",
                "",
                "- Primary: `none`",
                "- Score: `0.0`",
                "- Downstream decisions at risk: `[]`",
                "",
                "## Recommended Mode",
                "",
                "- Recommended mode: `UNKNOWN`",
                "",
                "## Mode Options",
                "",
                "- None",
                "",
                "## Issues",
                "",
                "- No major trust issues detected.",
                "",
                "## Fallback Actions",
                "",
                "- None",
                "",
                "## Model Validation (Model vs Reality)",
                "",
                "- Status: `NO_DATA`",
                "",
                "## Decision Impact Simulation",
                "",
                "- If wrong (expected loss): `0.0` s",
                "",
                "## Additional Decision Impacts",
                "",
                "- None",
                "",
                "## Explanation",
                "",
                "offline explanation",
                "",
                "## Provenance Warnings",
                "",
                "- None",
                "",
            ]
        ),
        encoding="utf-8",
    )
    json_path.write_text("{}", encoding="utf-8")
    registry_path.write_text("", encoding="utf-8")

    payload: dict = {
        "schema_version": "mdce_decision_run_v2",
        "run_id": run_id,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "project_root": str(root),
        "confidence_scoring_version": "confidence_v2_breakdown_and_per_decision",
        "source": {
            "source_name": "test",
            "real_columns": ["lap", "lap_time_s"],
            "derived_columns": [],
            "proxy_columns": [],
            "warnings": [],
            "dataset_metadata": {},
        },
        "scenario": {
            "flags": {
                "preset": "custom",
                "missing_telemetry": False,
                "tyre_signal_drift": False,
                "model_mismatch": False,
                "safety_car_phase": False,
                "weather_uncertainty": False,
            },
            "notes": [],
        },
        "decision": {
            "recommendation": {
                "type": "EXTEND",
                "recommended_lap": 5,
                "expected_gain_loss_s": 0.0,
                "base_reason": "x",
            },
            "confidence": {
                "confidence": 0.7,
                "risk_level": "Medium",
                "breakdown": {
                    "data_completeness": 1.0,
                    "signal_agreement": 1.0,
                    "model_alignment": 1.0,
                    "context_stability": 1.0,
                    "penalty_score": 1.0,
                },
                "decision_confidence": {"pit_timing": 0.7, "tyre_strategy": 0.7},
                "decision_risk_levels": {"pit_timing": "Medium", "tyre_strategy": "Medium"},
            },
            "uncertainty": {
                "primary_uncertainty": "none",
                "uncertainty_score": 0.0,
                "downstream_decisions_at_risk": [],
                "drivers": [],
            },
            "recommended_mode": "UNKNOWN",
            "mode_options": [],
            "conflict": {"score": 0.0, "label": "NONE"},
            "issues": [],
            "fallback_actions": [],
            "explanation": "offline explanation",
        },
        "model_validation": {
            "status": "NO_DATA",
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
            "notes": [],
        },
        "decision_impacts": [],
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
            "registry_csv": str(registry_path),
            "registry_v2_csv": str(registry_v2_path),
        },
    }

    # Create a canonical v2 registry with one matching row.
    import csv
    from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

    with registry_v2_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "timestamp_utc": payload["timestamp_utc"],
                "source_name": payload["source"]["source_name"],
                "dataset_license": "NOASSERTION",
                "dataset_source_url": "",
                "preset": payload["scenario"]["flags"]["preset"],
                "confidence_scoring_version": payload["confidence_scoring_version"],
                "recommendation_type": payload["decision"]["recommendation"]["type"],
                "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
                "confidence": payload["decision"]["confidence"]["confidence"],
                "risk_level": payload["decision"]["confidence"]["risk_level"],
                "pit_timing_confidence": payload["decision"]["confidence"]["decision_confidence"]["pit_timing"],
                "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"]["tyre_strategy"],
                "uncertainty_primary": payload["decision"]["uncertainty"]["primary_uncertainty"],
                "uncertainty_score": payload["decision"]["uncertainty"]["uncertainty_score"],
                "recommended_mode": payload["decision"]["recommended_mode"],
                "model_validation_status": payload["model_validation"]["status"],
                "model_validation_mae_s": payload["model_validation"]["mean_abs_error_s"],
                "impact_if_wrong_loss_s": payload["decision_impact"]["if_wrong_expected_loss_s"],
                "conflict_score": payload["decision"]["conflict"]["score"],
                "conflict_label": payload["decision"]["conflict"]["label"],
                "issue_count": 0,
                "issues": "",
                "json_path": str(json_path),
                "md_path": str(md_path),
            }
        )

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, json_path


class ArtifactValidatorTests(unittest.TestCase):
    def test_validate_payload_accepts_minimal_valid_payload(self) -> None:
        from tools.validate_mdce_outputs import validate_payload

        with tempfile.TemporaryDirectory() as tmp:
            payload, _json_path = _write_min_valid_v2_payload(Path(tmp))
            validate_payload(payload)

    def test_latest_run_json_prefers_v2(self) -> None:
        # If the outputs dir contains older v2 artifacts and newer legacy artifacts,
        # the validator should prefer the newest v2 artifact.
        from tools.validate_mdce_outputs import _latest_run_json

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            # Newest by filename (lexicographic timestamp) is v1.
            v2_path = out_dir / "mdce_decision_run_20260101_000000.json"
            v1_path = out_dir / "mdce_decision_run_20260101_000001.json"

            v2_path.write_text(json.dumps({"schema_version": "mdce_decision_run_v2"}), encoding="utf-8")
            v1_path.write_text(json.dumps({"schema_version": "mdce_decision_run_v1"}), encoding="utf-8")

            picked = _latest_run_json(out_dir)
            self.assertEqual(picked.resolve(), v2_path.resolve())

    def test_markdown_explanation_match_is_whitespace_tolerant(self) -> None:
        from tools.validate_mdce_outputs import validate_markdown

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "mdce_decision_run_20260101_000000.md"
            run_id = "20260101_000000"

            expected_mode = "SAFE"
            expected_primary = "pace_model"
            expected_expl = "Because signals disagree, we recommend SAFE mode and use lap-time trend as fallback."

            # Include the expected explanation with aggressive line breaks/spaces.
            md_path.write_text(
                "\n".join(
                    [
                        "# MDCE Decision Run",
                        "",
                        f"- Run ID: `{run_id}`",
                        "- Dataset license: `NOASSERTION`",
                        "- Dataset source URL: `UNKNOWN`",
                        "",
                        "## Recommendation",
                        "",
                        "- Type: `PIT_NOW`",
                        "",
                        "## Confidence",
                        "",
                        "- Confidence: `0.5`",
                        "",
                        "## Uncertainty",
                        "",
                        f"- Primary: `{expected_primary}`",
                        "- Score: `0.7`",
                        "- Downstream decisions at risk: `[]`",
                        "",
                        "## Confidence Breakdown",
                        "",
                        "- Breakdown: `{}`",
                        "",
                        "## Recommended Mode",
                        "",
                        f"- Recommended mode: `{expected_mode}`",
                        "",
                        "## Mode Options",
                        "",
                        "- None",
                        "",
                        "## Issues",
                        "",
                        "- No major trust issues detected.",
                        "",
                        "## Fallback Actions",
                        "",
                        "- None",
                        "",
                        "## Model Validation (Model vs Reality)",
                        "",
                        "- Status: `NO_DATA`",
                        "",
                        "## Decision Impact Simulation",
                        "",
                        "- If wrong (expected loss): `1.0` s",
                        "",
                        "## Additional Decision Impacts",
                        "",
                        "- None",
                        "",
                        "## Explanation",
                        "",
                        "Because  signals\n",
                        "disagree, we recommend SAFE mode\n",
                        "and use lap-time trend as fallback.",
                        "",
                        "## Provenance Warnings",
                        "",
                        "- None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            validate_markdown(
                md_path,
                run_id=run_id,
                expected_recommended_mode=expected_mode,
                expected_uncertainty_primary=expected_primary,
                expected_explanation=expected_expl,
            )

    def test_markdown_requires_core_sections_and_lines(self) -> None:
        from tools.validate_mdce_outputs import validate_markdown

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "mdce_decision_run_20260101_000010.md"
            run_id = "20260101_000010"

            # Minimal markdown that still satisfies the contract checks.
            md_path.write_text(
                "\n".join(
                    [
                        "# MDCE Decision Run",
                        "",
                        f"- Run ID: `{run_id}`",
                        "- Dataset license: `NOASSERTION`",
                        "- Dataset source URL: `UNKNOWN`",
                        "",
                        "## Recommendation",
                        "",
                        "## Confidence",
                        "",
                        "## Uncertainty",
                        "",
                        "- Primary: `none`",
                        "- Score: `0.0`",
                        "- Downstream decisions at risk: `[]`",
                        "",
                        "## Recommended Mode",
                        "",
                        "- Recommended mode: `UNKNOWN`",
                        "",
                        "## Mode Options",
                        "",
                        "- None",
                        "",
                        "## Issues",
                        "",
                        "- No major trust issues detected.",
                        "",
                        "## Fallback Actions",
                        "",
                        "- None",
                        "",
                        "## Model Validation (Model vs Reality)",
                        "",
                        "## Decision Impact Simulation",
                        "",
                        "- If wrong (expected loss): `0.0` s",
                        "",
                        "## Additional Decision Impacts",
                        "",
                        "- None",
                        "",
                        "## Explanation",
                        "",
                        "offline explanation",
                        "",
                        "## Provenance Warnings",
                        "",
                        "- None",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            validate_markdown(md_path, run_id=run_id, expected_explanation="offline explanation")

    def test_validator_fails_on_wrong_schema_version(self) -> None:
        from tools.validate_mdce_outputs import validate_payload

        with self.assertRaises(ValueError):
            validate_payload({"schema_version": "mdce_decision_run_v1"})

    def test_validator_rejects_relative_project_root(self) -> None:
        from tools.validate_mdce_outputs import validate_payload

        payload = _write_min_valid_v2_payload(Path(tempfile.mkdtemp()))[0]
        payload["project_root"] = "relative/path"
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_registry_v2_validator_rejects_missing_columns(self) -> None:
        from tools.validate_mdce_outputs import validate_registry_v2

        with tempfile.TemporaryDirectory() as tmp:
            v2 = Path(tmp) / "MDCE_DECISION_RUN_REGISTRY_v2.csv"
            # Intentionally incomplete header.
            v2.write_text("run_id,timestamp_utc\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2(v2)
            self.assertIn("missing columns", str(ctx.exception))

    def test_registry_v2_validator_rejects_wrong_header_order(self) -> None:
        from tools.validate_mdce_outputs import validate_registry_v2

        with tempfile.TemporaryDirectory() as tmp:
            v2 = Path(tmp) / "MDCE_DECISION_RUN_REGISTRY_v2.csv"
            # Provide required columns but in wrong order.
            v2.write_text(
                "timestamp_utc,run_id,source_name,dataset_license,dataset_source_url,preset,confidence_scoring_version,recommendation_type,recommended_lap,confidence,risk_level,pit_timing_confidence,tyre_strategy_confidence,uncertainty_primary,uncertainty_score,recommended_mode,safe_mode_if_wrong_loss_s,aggressive_mode_if_wrong_loss_s,model_validation_status,model_validation_mae_s,impact_if_wrong_loss_s,push_vs_conserve_if_wrong_loss_s,push_vs_conserve_risk_level,conflict_score,conflict_label,issue_count,issues,json_path,md_path\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2(v2)
            self.assertIn("header order", str(ctx.exception))

    def test_registry_v2_row_validator_fails_on_mismatch(self) -> None:
        from tools.validate_mdce_outputs import validate_registry_v2_row

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2 = root / "MDCE_DECISION_RUN_REGISTRY_v2.csv"
            json_path = root / "run.json"
            md_path = root / "run.md"
            json_path.write_text("{}", encoding="utf-8")
            md_path.write_text("x", encoding="utf-8")

            # Canonical v2 header + one row.
            import csv
            from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

            with v2.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                writer.writeheader()
                # Timestamp intentionally doesn't match payload below.
                writer.writerow(
                    {
                        "run_id": "20260101_000000",
                        "timestamp_utc": "2026-01-01T00:00:01Z",
                        "source_name": "src",
                        "dataset_license": "NOASSERTION",
                        "dataset_source_url": "",
                        "preset": "custom",
                        "confidence_scoring_version": "confidence_v2_breakdown_and_per_decision",
                        "recommendation_type": "EXTEND",
                        "recommended_lap": "5",
                        "confidence": "0.70",
                        "risk_level": "Medium",
                        "pit_timing_confidence": "0.70",
                        "tyre_strategy_confidence": "0.70",
                        "uncertainty_primary": "none",
                        "uncertainty_score": "0.00",
                        "recommended_mode": "UNKNOWN",
                        "model_validation_status": "NO_DATA",
                        "model_validation_mae_s": "0.0",
                        "impact_if_wrong_loss_s": "0.0",
                        "conflict_score": "0.0",
                        "conflict_label": "NONE",
                        "issue_count": "0",
                        "issues": "",
                        "json_path": str(json_path),
                        "md_path": str(md_path),
                    }
                )

            payload = {
                "run_id": "20260101_000000",
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "confidence_scoring_version": "confidence_v2_breakdown_and_per_decision",
                "source": {"source_name": "src"},
                "scenario": {"flags": {"preset": "custom"}},
                "decision": {
                    "recommendation": {"type": "EXTEND", "recommended_lap": 5},
                    "confidence": {"confidence": 0.70, "risk_level": "Medium", "decision_confidence": {"pit_timing": 0.70, "tyre_strategy": 0.70}},
                    "uncertainty": {"primary_uncertainty": "none", "uncertainty_score": 0.0},
                    "recommended_mode": "UNKNOWN",
                    "conflict": {"score": 0.0, "label": "NONE"},
                    "issues": [],
                    "mode_options": [],
                },
                "decision_impact": {"if_wrong_expected_loss_s": 0.0},
                "decision_impacts": [],
                "model_validation": {"status": "NO_DATA", "mean_abs_error_s": 0.0},
                "artifacts": {"json": str(json_path), "markdown": str(md_path)},
            }

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("timestamp_utc mismatch", str(ctx.exception))

    def test_registry_v2_row_validator_fails_on_non_parseable_numeric(self) -> None:
        from tools.validate_mdce_outputs import validate_registry_v2_row

        import csv
        from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2 = root / "MDCE_DECISION_RUN_REGISTRY_v2.csv"
            json_path = root / "run.json"
            md_path = root / "run.md"
            json_path.write_text("{}", encoding="utf-8")
            md_path.write_text("x", encoding="utf-8")

            with v2.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                writer.writeheader()
                writer.writerow(
                    {
                        "run_id": "20260101_000000",
                        "timestamp_utc": "2026-01-01T00:00:00Z",
                        "source_name": "src",
                        "dataset_license": "NOASSERTION",
                        "dataset_source_url": "",
                        "preset": "custom",
                        "confidence_scoring_version": "confidence_v2_breakdown_and_per_decision",
                        "recommendation_type": "EXTEND",
                        "recommended_lap": "5",
                        # Bad numeric value.
                        "confidence": "n/a",
                        "risk_level": "Medium",
                        "pit_timing_confidence": "0.70",
                        "tyre_strategy_confidence": "0.70",
                        "uncertainty_primary": "none",
                        "uncertainty_score": "0.00",
                        "recommended_mode": "UNKNOWN",
                        "model_validation_status": "NO_DATA",
                        "model_validation_mae_s": "0.0",
                        "impact_if_wrong_loss_s": "0.0",
                        "conflict_score": "0.0",
                        "conflict_label": "NONE",
                        "issue_count": "0",
                        "issues": "",
                        "json_path": str(json_path),
                        "md_path": str(md_path),
                    }
                )

            payload = {
                "run_id": "20260101_000000",
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "confidence_scoring_version": "confidence_v2_breakdown_and_per_decision",
                "source": {"source_name": "src"},
                "scenario": {"flags": {"preset": "custom"}},
                "decision": {
                    "recommendation": {"type": "EXTEND", "recommended_lap": 5},
                    "confidence": {"confidence": 0.70, "risk_level": "Medium", "decision_confidence": {"pit_timing": 0.70, "tyre_strategy": 0.70}},
                    "uncertainty": {"primary_uncertainty": "none", "uncertainty_score": 0.0},
                    "recommended_mode": "UNKNOWN",
                    "conflict": {"score": 0.0, "label": "NONE"},
                    "issues": [],
                    "mode_options": [],
                },
                "decision_impact": {"if_wrong_expected_loss_s": 0.0},
                "decision_impacts": [],
                "model_validation": {"status": "NO_DATA", "mean_abs_error_s": 0.0},
                "artifacts": {"json": str(json_path), "markdown": str(md_path)},
            }

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            # Error text comes from the float parser.
            self.assertIn("not a float", str(ctx.exception))

    def test_registry_v2_row_validator_fails_when_run_id_row_missing(self) -> None:
        from tools.validate_mdce_outputs import validate_registry_v2_row

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])
            # Replace registry with only header (no rows).
            header = v2.read_text(encoding="utf-8").splitlines()[0] + "\n"
            v2.write_text(header, encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("missing run_id row", str(ctx.exception))

    def test_registry_v2_row_validator_fails_on_mode_loss_mismatch(self) -> None:
        """If JSON has mode options, v2 CSV mode-loss columns must match."""

        import csv

        from tools.validate_mdce_outputs import validate_registry_v2_row
        from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])

            # Inject mode_options into JSON.
            payload["decision"]["mode_options"] = [
                {
                    "mode": "SAFE",
                    "recommendation": payload["decision"]["recommendation"],
                    "decision_impact": {
                        "decision": "pit_timing",
                        "horizon_laps": 3,
                        "if_right_expected_gain_s": 0.0,
                        "if_wrong_expected_loss_s": 1.2,
                        "risk_level": "LOW",
                        "assumptions": {"horizon_laps": 3.0},
                        "notes": [],
                    },
                },
                {
                    "mode": "AGGRESSIVE",
                    "recommendation": payload["decision"]["recommendation"],
                    "decision_impact": {
                        "decision": "pit_timing",
                        "horizon_laps": 3,
                        "if_right_expected_gain_s": 0.0,
                        "if_wrong_expected_loss_s": 2.4,
                        "risk_level": "LOW",
                        "assumptions": {"horizon_laps": 3.0},
                        "notes": [],
                    },
                },
            ]

            # Rewrite registry row, intentionally mismatching SAFE loss.
            run_id = payload["run_id"]
            with v2.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                w.writeheader()
                w.writerow(
                    {
                        "run_id": run_id,
                        "timestamp_utc": payload["timestamp_utc"],
                        "source_name": payload["source"]["source_name"],
                        "dataset_license": "NOASSERTION",
                        "dataset_source_url": "",
                        "preset": payload["scenario"]["flags"]["preset"],
                        "confidence_scoring_version": payload["confidence_scoring_version"],
                        "recommendation_type": payload["decision"]["recommendation"]["type"],
                        "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
                        "confidence": payload["decision"]["confidence"]["confidence"],
                        "risk_level": payload["decision"]["confidence"]["risk_level"],
                        "pit_timing_confidence": payload["decision"]["confidence"]["decision_confidence"]["pit_timing"],
                        "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"]["tyre_strategy"],
                        "uncertainty_primary": payload["decision"]["uncertainty"]["primary_uncertainty"],
                        "uncertainty_score": payload["decision"]["uncertainty"]["uncertainty_score"],
                        "recommended_mode": payload["decision"]["recommended_mode"],
                        # Wrong on purpose:
                        "safe_mode_if_wrong_loss_s": 9.9,
                        # Correct aggressive:
                        "aggressive_mode_if_wrong_loss_s": 2.4,
                        "model_validation_status": payload["model_validation"]["status"],
                        "model_validation_mae_s": payload["model_validation"]["mean_abs_error_s"],
                        "impact_if_wrong_loss_s": payload["decision_impact"]["if_wrong_expected_loss_s"],
                        "conflict_score": payload["decision"]["conflict"]["score"],
                        "conflict_label": payload["decision"]["conflict"]["label"],
                        "issue_count": 0,
                        "issues": "",
                        "json_path": payload["artifacts"]["json"],
                        "md_path": payload["artifacts"]["markdown"],
                    }
                )

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("safe_mode_if_wrong_loss_s mismatch", str(ctx.exception))

    def test_registry_v2_row_validator_fails_when_md_path_missing(self) -> None:
        from tools.validate_mdce_outputs import validate_registry_v2_row

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])
            md_path = Path(payload["artifacts"]["markdown"])
            md_path.unlink()

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("md_path does not exist", str(ctx.exception))

    def test_registry_v2_row_validator_rejects_bad_recommended_mode_enum(self) -> None:
        """Even if CSV matches JSON, enums must be bounded."""

        import csv

        from tools.validate_mdce_outputs import validate_registry_v2_row
        from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])

            payload["decision"]["recommended_mode"] = "FAST"

            with v2.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                w.writeheader()
                w.writerow(
                    {
                        "run_id": payload["run_id"],
                        "timestamp_utc": payload["timestamp_utc"],
                        "source_name": payload["source"]["source_name"],
                        "dataset_license": "NOASSERTION",
                        "dataset_source_url": "",
                        "preset": payload["scenario"]["flags"]["preset"],
                        "confidence_scoring_version": payload["confidence_scoring_version"],
                        "recommendation_type": payload["decision"]["recommendation"]["type"],
                        "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
                        "confidence": payload["decision"]["confidence"]["confidence"],
                        "risk_level": payload["decision"]["confidence"]["risk_level"],
                        "pit_timing_confidence": payload["decision"]["confidence"]["decision_confidence"]["pit_timing"],
                        "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"]["tyre_strategy"],
                        "uncertainty_primary": payload["decision"]["uncertainty"]["primary_uncertainty"],
                        "uncertainty_score": payload["decision"]["uncertainty"]["uncertainty_score"],
                        "recommended_mode": "FAST",
                        "model_validation_status": payload["model_validation"]["status"],
                        "model_validation_mae_s": payload["model_validation"]["mean_abs_error_s"],
                        "impact_if_wrong_loss_s": payload["decision_impact"]["if_wrong_expected_loss_s"],
                        "conflict_score": payload["decision"]["conflict"]["score"],
                        "conflict_label": payload["decision"]["conflict"]["label"],
                        "issue_count": 0,
                        "issues": "",
                        "json_path": payload["artifacts"]["json"],
                        "md_path": payload["artifacts"]["markdown"],
                    }
                )

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("bad recommended_mode", str(ctx.exception))

    def test_registry_v2_row_validator_fails_when_mode_losses_missing(self) -> None:
        import csv

        from tools.validate_mdce_outputs import validate_registry_v2_row
        from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])

            payload["decision"]["mode_options"] = [
                {
                    "mode": "SAFE",
                    "recommendation": payload["decision"]["recommendation"],
                    "decision_impact": {
                        "decision": "pit_timing",
                        "horizon_laps": 3,
                        "if_right_expected_gain_s": 0.0,
                        "if_wrong_expected_loss_s": 1.2,
                        "risk_level": "LOW",
                        "assumptions": {"horizon_laps": 3.0},
                        "notes": [],
                    },
                },
                {
                    "mode": "AGGRESSIVE",
                    "recommendation": payload["decision"]["recommendation"],
                    "decision_impact": {
                        "decision": "pit_timing",
                        "horizon_laps": 3,
                        "if_right_expected_gain_s": 0.0,
                        "if_wrong_expected_loss_s": 2.4,
                        "risk_level": "LOW",
                        "assumptions": {"horizon_laps": 3.0},
                        "notes": [],
                    },
                },
            ]

            with v2.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                w.writeheader()
                w.writerow(
                    {
                        "run_id": payload["run_id"],
                        "timestamp_utc": payload["timestamp_utc"],
                        "source_name": payload["source"]["source_name"],
                        "dataset_license": "NOASSERTION",
                        "dataset_source_url": "",
                        "preset": payload["scenario"]["flags"]["preset"],
                        "confidence_scoring_version": payload["confidence_scoring_version"],
                        "recommendation_type": payload["decision"]["recommendation"]["type"],
                        "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
                        "confidence": payload["decision"]["confidence"]["confidence"],
                        "risk_level": payload["decision"]["confidence"]["risk_level"],
                        "pit_timing_confidence": payload["decision"]["confidence"]["decision_confidence"]["pit_timing"],
                        "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"]["tyre_strategy"],
                        "uncertainty_primary": payload["decision"]["uncertainty"]["primary_uncertainty"],
                        "uncertainty_score": payload["decision"]["uncertainty"]["uncertainty_score"],
                        "recommended_mode": payload["decision"]["recommended_mode"],
                        # SAFE loss intentionally blank.
                        "safe_mode_if_wrong_loss_s": "",
                        "aggressive_mode_if_wrong_loss_s": "2.4",
                        "model_validation_status": payload["model_validation"]["status"],
                        "model_validation_mae_s": payload["model_validation"]["mean_abs_error_s"],
                        "impact_if_wrong_loss_s": payload["decision_impact"]["if_wrong_expected_loss_s"],
                        "conflict_score": payload["decision"]["conflict"]["score"],
                        "conflict_label": payload["decision"]["conflict"]["label"],
                        "issue_count": 0,
                        "issues": "",
                        "json_path": payload["artifacts"]["json"],
                        "md_path": payload["artifacts"]["markdown"],
                    }
                )

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("missing safe_mode_if_wrong_loss_s", str(ctx.exception))

    def test_registry_v2_row_validator_requires_push_vs_conserve_loss_when_present(self) -> None:
        """If JSON contains push_vs_conserve decision impact, registry must include its loss."""

        import csv

        from tools.validate_mdce_outputs import validate_registry_v2_row
        from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])

            payload["decision_impacts"] = [
                {
                    "decision": "push_vs_conserve",
                    "horizon_laps": 3,
                    "if_right_expected_gain_s": 0.0,
                    "if_wrong_expected_loss_s": 3.3,
                    "risk_level": "HIGH",
                    "assumptions": {"horizon_laps": 3.0},
                    "notes": [],
                }
            ]

            with v2.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                w.writeheader()
                w.writerow(
                    {
                        "run_id": payload["run_id"],
                        "timestamp_utc": payload["timestamp_utc"],
                        "source_name": payload["source"]["source_name"],
                        "dataset_license": "NOASSERTION",
                        "dataset_source_url": "",
                        "preset": payload["scenario"]["flags"]["preset"],
                        "confidence_scoring_version": payload["confidence_scoring_version"],
                        "recommendation_type": payload["decision"]["recommendation"]["type"],
                        "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
                        "confidence": payload["decision"]["confidence"]["confidence"],
                        "risk_level": payload["decision"]["confidence"]["risk_level"],
                        "pit_timing_confidence": payload["decision"]["confidence"]["decision_confidence"]["pit_timing"],
                        "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"]["tyre_strategy"],
                        "uncertainty_primary": payload["decision"]["uncertainty"]["primary_uncertainty"],
                        "uncertainty_score": payload["decision"]["uncertainty"]["uncertainty_score"],
                        "recommended_mode": payload["decision"]["recommended_mode"],
                        "model_validation_status": payload["model_validation"]["status"],
                        "model_validation_mae_s": payload["model_validation"]["mean_abs_error_s"],
                        "impact_if_wrong_loss_s": payload["decision_impact"]["if_wrong_expected_loss_s"],
                        # Leave push_vs_conserve loss blank on purpose.
                        "push_vs_conserve_if_wrong_loss_s": "",
                        "push_vs_conserve_risk_level": "HIGH",
                        "conflict_score": payload["decision"]["conflict"]["score"],
                        "conflict_label": payload["decision"]["conflict"]["label"],
                        "issue_count": 0,
                        "issues": "",
                        "json_path": payload["artifacts"]["json"],
                        "md_path": payload["artifacts"]["markdown"],
                    }
                )

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("missing push_vs_conserve_if_wrong_loss_s", str(ctx.exception))

    def test_registry_v2_row_validator_requires_pit_timing_confidence_when_present(self) -> None:
        import csv

        from tools.validate_mdce_outputs import validate_registry_v2_row
        from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])

            with v2.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                w.writeheader()
                row = {
                    "run_id": payload["run_id"],
                    "timestamp_utc": payload["timestamp_utc"],
                    "source_name": payload["source"]["source_name"],
                    "dataset_license": "NOASSERTION",
                    "dataset_source_url": "",
                    "preset": payload["scenario"]["flags"]["preset"],
                    "confidence_scoring_version": payload["confidence_scoring_version"],
                    "recommendation_type": payload["decision"]["recommendation"]["type"],
                    "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
                    "confidence": payload["decision"]["confidence"]["confidence"],
                    "risk_level": payload["decision"]["confidence"]["risk_level"],
                    # Leave blank.
                    "pit_timing_confidence": "",
                    "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"]["tyre_strategy"],
                    "uncertainty_primary": payload["decision"]["uncertainty"]["primary_uncertainty"],
                    "uncertainty_score": payload["decision"]["uncertainty"]["uncertainty_score"],
                    "recommended_mode": payload["decision"]["recommended_mode"],
                    "model_validation_status": payload["model_validation"]["status"],
                    "model_validation_mae_s": payload["model_validation"]["mean_abs_error_s"],
                    "impact_if_wrong_loss_s": payload["decision_impact"]["if_wrong_expected_loss_s"],
                    "conflict_score": payload["decision"]["conflict"]["score"],
                    "conflict_label": payload["decision"]["conflict"]["label"],
                    "issue_count": 0,
                    "issues": "",
                    "json_path": payload["artifacts"]["json"],
                    "md_path": payload["artifacts"]["markdown"],
                }
                w.writerow(row)

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("missing pit_timing_confidence", str(ctx.exception))

    def test_registry_v2_row_validator_requires_conflict_score_when_present(self) -> None:
        import csv

        from tools.validate_mdce_outputs import validate_registry_v2_row
        from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])

            with v2.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                w.writeheader()
                w.writerow(
                    {
                        "run_id": payload["run_id"],
                        "timestamp_utc": payload["timestamp_utc"],
                        "source_name": payload["source"]["source_name"],
                        "dataset_license": "NOASSERTION",
                        "dataset_source_url": "",
                        "preset": payload["scenario"]["flags"]["preset"],
                        "confidence_scoring_version": payload["confidence_scoring_version"],
                        "recommendation_type": payload["decision"]["recommendation"]["type"],
                        "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
                        "confidence": payload["decision"]["confidence"]["confidence"],
                        "risk_level": payload["decision"]["confidence"]["risk_level"],
                        "pit_timing_confidence": payload["decision"]["confidence"]["decision_confidence"]["pit_timing"],
                        "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"]["tyre_strategy"],
                        "uncertainty_primary": payload["decision"]["uncertainty"]["primary_uncertainty"],
                        "uncertainty_score": payload["decision"]["uncertainty"]["uncertainty_score"],
                        "recommended_mode": payload["decision"]["recommended_mode"],
                        "model_validation_status": payload["model_validation"]["status"],
                        "model_validation_mae_s": payload["model_validation"]["mean_abs_error_s"],
                        "impact_if_wrong_loss_s": payload["decision_impact"]["if_wrong_expected_loss_s"],
                        # Blank on purpose.
                        "conflict_score": "",
                        "conflict_label": payload["decision"]["conflict"]["label"],
                        "issue_count": 0,
                        "issues": "",
                        "json_path": payload["artifacts"]["json"],
                        "md_path": payload["artifacts"]["markdown"],
                    }
                )

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("missing conflict_score", str(ctx.exception))

    def test_registry_v2_row_validator_requires_model_validation_mae_when_present(self) -> None:
        import csv

        from tools.validate_mdce_outputs import validate_registry_v2_row
        from tools.run_mdce_decision import REGISTRY_V2_FIELDNAMES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, _json_path = _write_min_valid_v2_payload(root)
            v2 = Path(payload["artifacts"]["registry_v2_csv"])

            with v2.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=REGISTRY_V2_FIELDNAMES)
                w.writeheader()
                w.writerow(
                    {
                        "run_id": payload["run_id"],
                        "timestamp_utc": payload["timestamp_utc"],
                        "source_name": payload["source"]["source_name"],
                        "dataset_license": "NOASSERTION",
                        "dataset_source_url": "",
                        "preset": payload["scenario"]["flags"]["preset"],
                        "confidence_scoring_version": payload["confidence_scoring_version"],
                        "recommendation_type": payload["decision"]["recommendation"]["type"],
                        "recommended_lap": payload["decision"]["recommendation"]["recommended_lap"],
                        "confidence": payload["decision"]["confidence"]["confidence"],
                        "risk_level": payload["decision"]["confidence"]["risk_level"],
                        "pit_timing_confidence": payload["decision"]["confidence"]["decision_confidence"]["pit_timing"],
                        "tyre_strategy_confidence": payload["decision"]["confidence"]["decision_confidence"]["tyre_strategy"],
                        "uncertainty_primary": payload["decision"]["uncertainty"]["primary_uncertainty"],
                        "uncertainty_score": payload["decision"]["uncertainty"]["uncertainty_score"],
                        "recommended_mode": payload["decision"]["recommended_mode"],
                        "model_validation_status": payload["model_validation"]["status"],
                        # Blank on purpose.
                        "model_validation_mae_s": "",
                        "impact_if_wrong_loss_s": payload["decision_impact"]["if_wrong_expected_loss_s"],
                        "conflict_score": payload["decision"]["conflict"]["score"],
                        "conflict_label": payload["decision"]["conflict"]["label"],
                        "issue_count": 0,
                        "issues": "",
                        "json_path": payload["artifacts"]["json"],
                        "md_path": payload["artifacts"]["markdown"],
                    }
                )

            with self.assertRaises(ValueError) as ctx:
                validate_registry_v2_row(v2, payload=payload)
            self.assertIn("missing model_validation_mae_s", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
