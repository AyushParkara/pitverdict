from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import requests

from src.explanation_engine import explain_decision, mock_explanation
from src.models import (
    ConfidenceResult,
    DecisionImpactResult,
    RecommendationType,
    Severity,
    StrategyRecommendation,
    TrustIssue,
    UncertaintyResult,
)


class ExplanationEngineTests(unittest.TestCase):
    def _minimal_inputs(self):
        rec = StrategyRecommendation(
            recommendation_type=RecommendationType.PIT_NOW,
            recommended_lap=12,
            expected_gain_loss_s=1.2,
            base_reason="Tyre age high.",
        )
        conf = ConfidenceResult(
            confidence=0.42,
            risk_level="Medium-High",
            penalties=[],
            breakdown={},
            decision_confidence={},
            decision_risk_levels={},
        )
        return rec, conf

    def test_mock_explanation_includes_core_blocks(self) -> None:
        rec = StrategyRecommendation(
            recommendation_type=RecommendationType.PIT_NOW,
            recommended_lap=12,
            expected_gain_loss_s=1.2,
            base_reason="Tyre age high.",
        )
        conf = ConfidenceResult(
            confidence=0.42,
            risk_level="Medium-High",
            penalties=[],
            breakdown={"data_completeness": 0.8, "signal_agreement": 0.7, "model_alignment": 0.6, "context_stability": 0.5, "penalty_score": 0.4},
            decision_confidence={"pit_timing": 0.4, "tyre_strategy": 0.5},
            decision_risk_levels={"pit_timing": "High", "tyre_strategy": "Medium-High"},
        )
        issues = [
            TrustIssue(
                issue="missing_telemetry",
                severity=Severity.HIGH,
                affected_decisions=["pit_timing"],
                reason="Recent telemetry missing.",
                penalty=0.2,
            )
        ]
        uncertainty = UncertaintyResult(
            primary_uncertainty="missing_telemetry",
            uncertainty_score=0.5,
            downstream_decisions_at_risk=["pit_timing"],
            drivers=["missing_telemetry (penalty=0.2)"],
        )
        pit_impact = DecisionImpactResult(
            decision="pit_timing",
            horizon_laps=3,
            if_right_expected_gain_s=0.5,
            if_wrong_expected_loss_s=2.0,
            risk_level="MEDIUM",
            assumptions={"horizon_laps": 3.0},
            notes=["x"],
        )
        other_impact = DecisionImpactResult(
            decision="push_vs_conserve",
            horizon_laps=3,
            if_right_expected_gain_s=1.0,
            if_wrong_expected_loss_s=4.0,
            risk_level="HIGH",
            assumptions={"horizon_laps": 3.0},
            notes=["y"],
        )

        text = mock_explanation(
            recommendation=rec,
            confidence=conf,
            issues=issues,
            fallbacks=["Switch to SAFE mode."],
            uncertainty=uncertainty,
            recommended_mode="SAFE",
            decision_impact=pit_impact,
            decision_impacts=[other_impact],
        )
        self.assertIn("Recommendation:", text)
        self.assertIn("Decision confidence", text)
        self.assertIn("Mode: SAFE", text)
        self.assertIn("Uncertainty: primary=missing_telemetry", text)
        self.assertIn("Decision impacts:", text)
        self.assertIn("- pit_timing:", text)
        self.assertIn("- push_vs_conserve:", text)
        self.assertIn("missing_telemetry: Recent telemetry missing.", text)
        self.assertIn("Fallback:", text)

    def test_mock_explanation_follows_fixed_narrative_order(self) -> None:
        """Precision guard: the deterministic narrative must keep a fixed section order.

        Order contract:
          Recommendation -> Decision confidence/Mode -> Why (top drivers)
          -> Uncertainty -> Decision impacts -> Fallback
        """

        rec = StrategyRecommendation(
            recommendation_type=RecommendationType.PIT_NOW,
            recommended_lap=12,
            expected_gain_loss_s=1.2,
            base_reason="Tyre age high.",
        )
        conf = ConfidenceResult(
            confidence=0.42,
            risk_level="Medium-High",
            penalties=[],
            breakdown={},
            decision_confidence={},
            decision_risk_levels={},
        )
        issues = [
            TrustIssue(
                issue="missing_telemetry",
                severity=Severity.HIGH,
                affected_decisions=["pit_timing"],
                reason="Recent telemetry missing.",
                penalty=0.2,
            )
        ]
        uncertainty = UncertaintyResult(
            primary_uncertainty="missing_telemetry",
            uncertainty_score=0.5,
            downstream_decisions_at_risk=["pit_timing"],
            drivers=["missing_telemetry (penalty=0.2)"],
        )
        pit_impact = DecisionImpactResult(
            decision="pit_timing",
            horizon_laps=3,
            if_right_expected_gain_s=0.5,
            if_wrong_expected_loss_s=2.0,
            risk_level="MEDIUM",
            assumptions={"horizon_laps": 3.0},
            notes=["x"],
        )

        text = mock_explanation(
            recommendation=rec,
            confidence=conf,
            issues=issues,
            fallbacks=["Switch to SAFE mode."],
            uncertainty=uncertainty,
            recommended_mode="SAFE",
            decision_impact=pit_impact,
            decision_impacts=[],
        )

        markers = [
            "Recommendation:",
            "Decision confidence is",
            "Why (top drivers):",
            "Uncertainty:",
            "Decision impacts:",
            "Fallback:",
        ]
        positions = [text.find(m) for m in markers]
        # All markers must be present.
        self.assertTrue(all(p >= 0 for p in positions), f"Missing marker(s): {list(zip(markers, positions))}")
        # And they must appear in the contracted order.
        self.assertEqual(positions, sorted(positions), f"Narrative order broken: {list(zip(markers, positions))}")

    def test_mock_explanation_limits_why_to_top_three_drivers(self) -> None:
        """Why-section must surface only the top 3 drivers (by penalty), not dump all."""

        rec = StrategyRecommendation(
            recommendation_type=RecommendationType.PIT_SOON,
            recommended_lap=15,
            expected_gain_loss_s=0.5,
            base_reason="Rising degradation.",
        )
        conf = ConfidenceResult(
            confidence=0.3,
            risk_level="High",
            penalties=[],
            breakdown={},
            decision_confidence={},
            decision_risk_levels={},
        )
        issues = [
            TrustIssue(issue="no_usable_data", severity=Severity.HIGH, affected_decisions=["x"], reason="r1", penalty=0.50),
            TrustIssue(issue="missing_telemetry", severity=Severity.HIGH, affected_decisions=["x"], reason="r2", penalty=0.20),
            TrustIssue(issue="model_mismatch", severity=Severity.HIGH, affected_decisions=["x"], reason="r3", penalty=0.15),
            TrustIssue(issue="weather_uncertainty", severity=Severity.MEDIUM, affected_decisions=["x"], reason="r4", penalty=0.10),
            TrustIssue(issue="low_speed_consistency", severity=Severity.MEDIUM, affected_decisions=["x"], reason="r5", penalty=0.10),
        ]

        text = mock_explanation(
            recommendation=rec,
            confidence=conf,
            issues=issues,
            fallbacks=["Switch to SAFE mode."],
            uncertainty=None,
            recommended_mode="SAFE",
            decision_impact=None,
            decision_impacts=[],
        )

        why_block = text.split("Why (top drivers):", 1)[1].split("Uncertainty:", 1)[0]
        # Top 3 by penalty should be present.
        self.assertIn("no_usable_data:", why_block)
        self.assertIn("missing_telemetry:", why_block)
        self.assertIn("model_mismatch:", why_block)
        # The lower-penalty drivers should NOT appear in the Why block.
        self.assertNotIn("weather_uncertainty:", why_block)
        self.assertNotIn("low_speed_consistency:", why_block)

    def test_mock_explanation_handles_empty_issues_and_fallbacks(self) -> None:
        rec = StrategyRecommendation(
            recommendation_type=RecommendationType.EXTEND,
            recommended_lap=10,
            expected_gain_loss_s=0.0,
            base_reason="Stable pace.",
        )
        conf = ConfidenceResult(
            confidence=0.9,
            risk_level="Low",
            penalties=[],
            breakdown={},
            decision_confidence={},
            decision_risk_levels={},
        )
        text = mock_explanation(
            recommendation=rec,
            confidence=conf,
            issues=[],
            fallbacks=[],
            uncertainty=None,
            recommended_mode=None,
            decision_impact=None,
            decision_impacts=[],
        )
        self.assertIn("No major trust issues detected", text)
        self.assertIn("Uncertainty: none.", text)

    def test_explain_decision_prefers_mock_when_granite_disabled(self) -> None:
        rec = StrategyRecommendation(
            recommendation_type=RecommendationType.PIT_SOON,
            recommended_lap=15,
            expected_gain_loss_s=0.5,
            base_reason="Rising degradation.",
        )
        conf = ConfidenceResult(
            confidence=0.6,
            risk_level="Medium",
            penalties=[],
            breakdown={},
            decision_confidence={},
            decision_risk_levels={},
        )
        text = explain_decision(
            recommendation=rec,
            confidence=conf,
            issues=[],
            fallbacks=[],
            prefer_granite=False,
        )
        self.assertTrue(text.strip())
        self.assertIn("Recommendation:", text)

    def test_explain_decision_falls_back_when_granite_times_out(self) -> None:
        rec, conf = self._minimal_inputs()
        old_url = os.environ.get("WATSONX_GRANITE_URL")
        old_key = os.environ.get("WATSONX_API_KEY")
        try:
            os.environ["WATSONX_GRANITE_URL"] = "https://example.invalid/granite"
            os.environ["WATSONX_API_KEY"] = "test"

            with patch(
                "src.explanation_engine.requests.post",
                side_effect=requests.Timeout("timeout"),
            ):
                text = explain_decision(
                    recommendation=rec,
                    confidence=conf,
                    issues=[],
                    fallbacks=["Fallback A"],
                    prefer_granite=True,
                )
            self.assertIn("Recommendation:", text)
            self.assertIn("Fallback:", text)
        finally:
            if old_url is None:
                os.environ.pop("WATSONX_GRANITE_URL", None)
            else:
                os.environ["WATSONX_GRANITE_URL"] = old_url
            if old_key is None:
                os.environ.pop("WATSONX_API_KEY", None)
            else:
                os.environ["WATSONX_API_KEY"] = old_key

    def test_explain_decision_falls_back_when_granite_non_200(self) -> None:
        rec, conf = self._minimal_inputs()
        old_url = os.environ.get("WATSONX_GRANITE_URL")
        old_key = os.environ.get("WATSONX_API_KEY")
        try:
            os.environ["WATSONX_GRANITE_URL"] = "https://example.invalid/granite"
            os.environ["WATSONX_API_KEY"] = "test"

            class _Resp:
                def raise_for_status(self):
                    raise requests.HTTPError("500")

                def json(self):
                    return {"generated_text": "should not be used"}

            with patch("src.explanation_engine.requests.post", return_value=_Resp()):
                text = explain_decision(
                    recommendation=rec,
                    confidence=conf,
                    issues=[],
                    fallbacks=[],
                    prefer_granite=True,
                )
            self.assertIn("Recommendation:", text)
        finally:
            if old_url is None:
                os.environ.pop("WATSONX_GRANITE_URL", None)
            else:
                os.environ["WATSONX_GRANITE_URL"] = old_url
            if old_key is None:
                os.environ.pop("WATSONX_API_KEY", None)
            else:
                os.environ["WATSONX_API_KEY"] = old_key

    def test_explain_decision_falls_back_when_granite_invalid_json(self) -> None:
        rec, conf = self._minimal_inputs()
        old_url = os.environ.get("WATSONX_GRANITE_URL")
        old_key = os.environ.get("WATSONX_API_KEY")
        try:
            os.environ["WATSONX_GRANITE_URL"] = "https://example.invalid/granite"
            os.environ["WATSONX_API_KEY"] = "test"

            class _Resp:
                def raise_for_status(self):
                    return None

                def json(self):
                    raise ValueError("invalid json")

            with patch("src.explanation_engine.requests.post", return_value=_Resp()):
                text = explain_decision(
                    recommendation=rec,
                    confidence=conf,
                    issues=[],
                    fallbacks=[],
                    prefer_granite=True,
                )
            self.assertIn("Recommendation:", text)
        finally:
            if old_url is None:
                os.environ.pop("WATSONX_GRANITE_URL", None)
            else:
                os.environ["WATSONX_GRANITE_URL"] = old_url
            if old_key is None:
                os.environ.pop("WATSONX_API_KEY", None)
            else:
                os.environ["WATSONX_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
