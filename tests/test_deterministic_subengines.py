from __future__ import annotations

import unittest

from src.coverage_gap_engine import detect_coverage_gaps
from src.decision_impact_engine import estimate_pit_timing_impact, estimate_push_vs_conserve_impact
from src.mode_engine import apply_mode_to_recommendation, build_mode_options
from src.models import (
    DecisionMode,
    DecisionImpactResult,
    RecommendationType,
    Severity,
    StrategyRecommendation,
    TrustIssue,
    UncertaintyResult,
)
from src.recommended_mode_engine import recommend_mode
from src.strategy_engine import make_strategy_recommendation


def _rec(
    lap: int,
    lap_time_s: float,
    *,
    speed_consistency: float = 0.95,
    gap_to_car_ahead_s: float = 1.0,
    tyre_compound: str = "MEDIUM",
    tyre_age: int | None = None,
    track_status: str = "NORMAL",
    sector_noise: float = 0.03,
):
    """Build a LapRecord with plausible defaults.

    We keep sector values *not* perfectly proportional to lap time so gap detection
    tests can isolate other signals independently.
    """

    from src.models import LapRecord

    # Arbitrary sector splits with a tiny lap-dependent wobble.
    s1 = 30.0 + (lap * sector_noise)
    s2 = 35.0 + (lap * sector_noise)
    s3 = max(1.0, float(lap_time_s) - s1 - s2)
    return LapRecord(
        lap=lap,
        lap_time_s=float(lap_time_s),
        sector1_s=float(s1),
        sector2_s=float(s2),
        sector3_s=float(s3),
        tyre_compound=str(tyre_compound),
        tyre_age=int(tyre_age if tyre_age is not None else lap),
        track_status=str(track_status),
        weather="DRY",
        gap_to_car_ahead_s=float(gap_to_car_ahead_s),
        predicted_lap_time_s=float(lap_time_s),
        tyre_temp_proxy_c=95.0,
        speed_consistency=float(speed_consistency),
    )


class ModeEngineTests(unittest.TestCase):
    def test_apply_mode_safe_promotes_pit_soon(self) -> None:
        self.assertEqual(
            apply_mode_to_recommendation(RecommendationType.PIT_SOON, DecisionMode.SAFE),
            RecommendationType.PIT_NOW,
        )
        self.assertEqual(
            apply_mode_to_recommendation(RecommendationType.EXTEND, DecisionMode.SAFE),
            RecommendationType.EXTEND,
        )

    def test_apply_mode_aggressive_softens_pit_now(self) -> None:
        self.assertEqual(
            apply_mode_to_recommendation(RecommendationType.PIT_NOW, DecisionMode.AGGRESSIVE),
            RecommendationType.PIT_SOON,
        )
        self.assertEqual(
            apply_mode_to_recommendation(RecommendationType.EXTEND, DecisionMode.AGGRESSIVE),
            RecommendationType.EXTEND,
        )

    def test_build_mode_options_adjusts_recommendations(self) -> None:
        base = StrategyRecommendation(
            recommendation_type=RecommendationType.PIT_SOON,
            recommended_lap=12,
            expected_gain_loss_s=0.0,
            base_reason="Baseline reason.",
        )

        def impact_fn(rec_type: RecommendationType) -> DecisionImpactResult:
            # Keep deterministic, and encode the recommendation type for easy assertions.
            return DecisionImpactResult(
                decision="pit_timing",
                horizon_laps=3,
                if_right_expected_gain_s=1.0,
                if_wrong_expected_loss_s=2.0 if rec_type == RecommendationType.EXTEND else 1.0,
                risk_level="LOW",
                assumptions={},
                notes=[rec_type.value],
            )

        options = build_mode_options(base, impact_fn)
        self.assertEqual({o.mode for o in options}, {DecisionMode.SAFE, DecisionMode.AGGRESSIVE})
        by_mode = {o.mode: o for o in options}
        self.assertEqual(by_mode[DecisionMode.SAFE].recommendation.recommendation_type, RecommendationType.PIT_NOW)
        self.assertIn("Mode-adjusted (SAFE):", by_mode[DecisionMode.SAFE].recommendation.base_reason)
        self.assertEqual(by_mode[DecisionMode.AGGRESSIVE].recommendation.recommendation_type, RecommendationType.PIT_SOON)


class StrategyEngineTests(unittest.TestCase):
    def test_strategy_extend_when_stable(self) -> None:
        records = [_rec(i + 1, 95.0 + (0.02 if i % 2 else 0.0)) for i in range(10)]
        result = make_strategy_recommendation(records)
        self.assertEqual(result.recommendation_type, RecommendationType.EXTEND)
        self.assertEqual(result.recommended_lap, records[-1].lap + 5)
        self.assertGreaterEqual(result.expected_gain_loss_s, 0.0)

    def test_strategy_pit_soon_when_degradation_rising(self) -> None:
        # Trend: +0.3s/lap over last 5 -> PIT_SOON.
        records = [_rec(i + 1, 95.0 + (0.3 * i)) for i in range(10)]
        result = make_strategy_recommendation(records)
        self.assertEqual(result.recommendation_type, RecommendationType.PIT_SOON)
        self.assertEqual(result.recommended_lap, records[-1].lap + 3)
        # expected_gain_loss_s is based on degradation * 5.
        self.assertGreaterEqual(result.expected_gain_loss_s, 0.0)

    def test_strategy_pit_now_when_high_tyre_age_or_sc(self) -> None:
        records = [_rec(i + 1, 95.0, tyre_age=19) for i in range(6)]
        result = make_strategy_recommendation(records)
        self.assertEqual(result.recommendation_type, RecommendationType.PIT_NOW)
        self.assertEqual(result.recommended_lap, records[-1].lap)

        sc_records = [_rec(i + 1, 95.0, tyre_age=12, track_status="SC") for i in range(6)]
        sc_result = make_strategy_recommendation(sc_records)
        self.assertEqual(sc_result.recommendation_type, RecommendationType.PIT_NOW)


class CoverageGapEngineTests(unittest.TestCase):
    def test_detects_gap_placeholder_stream(self) -> None:
        # Mostly 0.0 gaps in last 10 should trigger track gap coverage issue.
        records = [_rec(i + 1, 95.0 + (0.05 * i), gap_to_car_ahead_s=0.0) for i in range(10)]
        issues = detect_coverage_gaps(records)
        self.assertTrue(any(i.issue == "coverage_gap_track_gaps" for i in issues))

    def test_detects_unknown_tyre_compound_recently(self) -> None:
        records = [_rec(i + 1, 95.0 + (0.05 * i)) for i in range(7)]
        records += [_rec(8, 95.4, tyre_compound="UNKNOWN"), _rec(9, 95.45), _rec(10, 95.5)]
        issues = detect_coverage_gaps(records)
        self.assertTrue(any(i.issue == "coverage_gap_tyre_compound" for i in issues))


class DecisionImpactEngineTests(unittest.TestCase):
    def test_pit_timing_insufficient_data_is_zeroed(self) -> None:
        impact = estimate_pit_timing_impact([_rec(1, 95.0)], RecommendationType.PIT_NOW)
        self.assertEqual(impact.if_wrong_expected_loss_s, 0.0)
        self.assertEqual(impact.if_right_expected_gain_s, 0.0)

    def test_pit_timing_impact_regret_direction(self) -> None:
        # Degradation ~0.2s/lap -> extend regret 0.6 over horizon 3.
        records = [_rec(i + 1, 95.0 + (0.2 * i)) for i in range(6)]
        pit_now = estimate_pit_timing_impact(records, RecommendationType.PIT_NOW, horizon_laps=3)
        extend = estimate_pit_timing_impact(records, RecommendationType.EXTEND, horizon_laps=3)

        self.assertEqual(pit_now.if_wrong_expected_loss_s, 1.2)
        self.assertEqual(pit_now.risk_level, "LOW")
        self.assertAlmostEqual(pit_now.if_right_expected_gain_s, 0.6, places=2)

        self.assertAlmostEqual(extend.if_wrong_expected_loss_s, 0.6, places=2)
        self.assertAlmostEqual(extend.if_right_expected_gain_s, 1.2, places=2)

    def test_pit_timing_risk_level_medium_when_regret_large(self) -> None:
        records = [_rec(i + 1, 95.0 + (0.7 * i)) for i in range(6)]
        extend = estimate_pit_timing_impact(records, RecommendationType.EXTEND, horizon_laps=3)
        self.assertEqual(extend.risk_level, "MEDIUM")

    def test_push_vs_conserve_risk_level_high_on_degradation_and_inconsistency(self) -> None:
        records = [_rec(i + 1, 95.0 + (0.5 * i), speed_consistency=0.55) for i in range(6)]
        impact = estimate_push_vs_conserve_impact(records, horizon_laps=3)
        self.assertEqual(impact.risk_level, "HIGH")
        self.assertGreater(impact.if_wrong_expected_loss_s, 3.0)
        self.assertAlmostEqual(impact.if_right_expected_gain_s, 1.05, places=2)


class RecommendedModeEngineTests(unittest.TestCase):
    def test_recommend_mode_safe_on_high_severity_issue(self) -> None:
        issues = [
            TrustIssue(
                issue="x",
                severity=Severity.HIGH,
                affected_decisions=["pit_timing"],
                reason="",
                penalty=0.01,
            )
        ]
        self.assertEqual(recommend_mode(issues, None), DecisionMode.SAFE)

    def test_recommend_mode_safe_on_penalty_threshold(self) -> None:
        issues = [
            TrustIssue(issue="a", severity=Severity.LOW, affected_decisions=["pit_timing"], reason="", penalty=0.2),
            TrustIssue(issue="b", severity=Severity.LOW, affected_decisions=["pit_timing"], reason="", penalty=0.15),
        ]
        self.assertEqual(recommend_mode(issues, None), DecisionMode.SAFE)

    def test_recommend_mode_safe_on_uncertainty_threshold(self) -> None:
        uncertainty = UncertaintyResult(
            primary_uncertainty="weather",
            uncertainty_score=0.55,
            downstream_decisions_at_risk=["pit_timing"],
        )
        self.assertEqual(recommend_mode([], uncertainty), DecisionMode.SAFE)

    def test_recommend_mode_aggressive_when_clean(self) -> None:
        uncertainty = UncertaintyResult(
            primary_uncertainty="none",
            uncertainty_score=0.1,
            downstream_decisions_at_risk=[],
        )
        self.assertEqual(recommend_mode([], uncertainty), DecisionMode.AGGRESSIVE)


if __name__ == "__main__":
    unittest.main()
