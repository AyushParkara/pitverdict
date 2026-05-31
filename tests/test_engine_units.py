from __future__ import annotations

import unittest

from src.confidence_engine import risk_level, score_confidence, score_confidence_components, score_decision_confidence
from src.disagreement_engine import detect_multi_signal_disagreement
from src.model_validation_engine import validate_model_vs_reality
from src.models import LapRecord, Severity, TrustIssue
from src.uncertainty_engine import propagate_uncertainty


def _lr(
    lap: int,
    *,
    lap_time_s: float = 95.0,
    predicted_lap_time_s: float | None = None,
    tyre_temp_proxy_c: float = 95.0,
    speed_consistency: float = 0.98,
    missing: bool = False,
) -> LapRecord:
    return LapRecord(
        lap=lap,
        lap_time_s=float(lap_time_s),
        sector1_s=30.0,
        sector2_s=35.0,
        sector3_s=max(1.0, float(lap_time_s) - 65.0),
        tyre_compound="MEDIUM",
        tyre_age=lap,
        track_status="NORMAL",
        weather="DRY",
        gap_to_car_ahead_s=1.0,
        predicted_lap_time_s=float(predicted_lap_time_s if predicted_lap_time_s is not None else lap_time_s),
        tyre_temp_proxy_c=float(tyre_temp_proxy_c),
        speed_consistency=float(speed_consistency),
        missing=bool(missing),
    )


class ConfidenceEngineUnitTests(unittest.TestCase):
    def test_risk_level_boundaries(self) -> None:
        self.assertEqual(risk_level(0.75), "Low")
        self.assertEqual(risk_level(0.55), "Medium")
        self.assertEqual(risk_level(0.35), "Medium-High")
        self.assertEqual(risk_level(0.10), "High")

    def test_breakdown_and_overall_confidence_match_penalties(self) -> None:
        issues = [
            TrustIssue(
                issue="missing_telemetry",
                severity=Severity.MEDIUM,
                affected_decisions=["pit_timing"],
                reason="",
                penalty=0.2,
            )
        ]

        breakdown = score_confidence_components(issues)
        self.assertEqual(breakdown["data_completeness"], 0.8)
        self.assertEqual(breakdown["penalty_score"], 0.8)

        scored = score_confidence(issues, base_confidence=0.85)
        # min(component_factor ~= 0.95, penalty_factor=0.8) => 0.85*0.8 = 0.68
        self.assertAlmostEqual(scored.confidence, 0.68, places=2)
        self.assertEqual(scored.risk_level, "Medium")

    def test_decision_confidence_scopes_penalties(self) -> None:
        issues = [
            TrustIssue(
                issue="coverage_gap_track_gaps",
                severity=Severity.MEDIUM,
                affected_decisions=["traffic_rejoin_risk"],
                reason="",
                penalty=0.2,
            )
        ]
        by_domain = score_decision_confidence(issues, base_confidence=0.85)
        self.assertAlmostEqual(by_domain["pit_timing"], 0.85, places=2)
        self.assertLess(by_domain["traffic_rejoin_risk"], 0.85)


class DisagreementEngineUnitTests(unittest.TestCase):
    def test_detects_tyre_proxy_shift_with_stable_lap_times(self) -> None:
        # Second discrepancy rule: tyre proxy changes sharply while lap times stay stable.
        records = [_lr(i + 1, lap_time_s=95.0, tyre_temp_proxy_c=95.0 + (0.6 * i)) for i in range(6)]
        issues = detect_multi_signal_disagreement(records)
        self.assertTrue(any(i.issue == "multi_signal_disagreement" for i in issues))


class ModelValidationEngineUnitTests(unittest.TestCase):
    def test_no_data_when_all_records_missing(self) -> None:
        records = [_lr(1, missing=True), _lr(2, missing=True)]
        result = validate_model_vs_reality(records, window_laps=5)
        self.assertEqual(result.status, "NO_DATA")
        self.assertEqual(result.mean_abs_error_s, 0.0)
        self.assertEqual(result.deviations, [])

    def test_deviation_detected_and_penalty_applied(self) -> None:
        # Errors of 1.0s should breach threshold 0.75s.
        records = [_lr(i + 1, lap_time_s=96.0, predicted_lap_time_s=95.0) for i in range(5)]
        result = validate_model_vs_reality(records, window_laps=5, deviation_threshold_s=0.75)
        self.assertEqual(result.status, "DEVIATION")
        self.assertAlmostEqual(result.mean_abs_error_s, 1.0, places=2)
        self.assertGreater(result.recommended_confidence_penalty, 0.0)
        self.assertEqual(len(result.deviations), 5)


class UncertaintyEngineUnitTests(unittest.TestCase):
    def test_uncertainty_primary_and_score_and_downstream(self) -> None:
        issues = [
            TrustIssue(
                issue="a",
                severity=Severity.LOW,
                affected_decisions=["pit_timing", "all_strategy_decisions"],
                reason="",
                penalty=0.2,
            ),
            TrustIssue(
                issue="b",
                severity=Severity.MEDIUM,
                affected_decisions=["traffic_rejoin_risk"],
                reason="",
                penalty=0.15,
            ),
        ]
        u = propagate_uncertainty(issues)
        self.assertEqual(u.primary_uncertainty, "a")
        self.assertAlmostEqual(u.uncertainty_score, 0.58, places=2)
        self.assertEqual(u.downstream_decisions_at_risk, ["pit_timing", "traffic_rejoin_risk"])
        self.assertTrue(u.drivers)


if __name__ == "__main__":
    unittest.main()
