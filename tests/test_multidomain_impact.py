from __future__ import annotations

import unittest
from io import StringIO

from src.data_loader import load_race_csv, load_sample_race
from src.decision_impact_engine import (
    estimate_aggressive_vs_safe_impact,
    estimate_safety_car_response_impact,
    estimate_stint_length_impact,
    estimate_traffic_rejoin_impact,
    estimate_tyre_compound_impact,
)


HEADER = (
    "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,"
    "track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency"
)


def _records(rows: list[str]) -> list:
    return load_race_csv(StringIO(HEADER + "\n" + "\n".join(rows)), source_name="impact csv").records


class NewImpactEstimatorTests(unittest.TestCase):
    def test_all_return_valid_shape_and_finite(self) -> None:
        import math

        records = load_sample_race()
        for fn, decision in [
            (estimate_stint_length_impact, "stint_length"),
            (estimate_tyre_compound_impact, "tyre_strategy"),
            (estimate_safety_car_response_impact, "safety_car_response"),
            (estimate_traffic_rejoin_impact, "traffic_rejoin_risk"),
            (estimate_aggressive_vs_safe_impact, "aggressive_vs_safe_strategy"),
        ]:
            out = fn(records, horizon_laps=3)
            self.assertEqual(out.decision, decision)
            self.assertIn(out.risk_level, {"LOW", "MEDIUM", "HIGH"})
            self.assertTrue(math.isfinite(out.if_wrong_expected_loss_s))
            self.assertGreaterEqual(out.if_wrong_expected_loss_s, 0.0)
            self.assertTrue(out.notes, "every estimate must carry honest notes")

    def test_safety_car_response_high_when_sc_present(self) -> None:
        rows = [f"{i},95.0,30.4,35.15,29.45,MEDIUM,{i},SC,DRY,1.0,95.0,100.0,0.98" for i in range(1, 7)]
        out = estimate_safety_car_response_impact(_records(rows))
        self.assertEqual(out.risk_level, "HIGH")
        self.assertGreaterEqual(out.if_wrong_expected_loss_s, 3.0)

    def test_safety_car_response_low_when_no_sc(self) -> None:
        rows = [f"{i},95.0,30.4,35.15,29.45,MEDIUM,{i},NORMAL,DRY,1.0,95.0,100.0,0.98" for i in range(1, 7)]
        out = estimate_safety_car_response_impact(_records(rows))
        self.assertEqual(out.risk_level, "LOW")

    def test_traffic_rejoin_flags_placeholder_gaps(self) -> None:
        # All gaps 0.0 -> placeholder path -> note must mention placeholder.
        rows = [f"{i},95.0,30.4,35.15,29.45,MEDIUM,{i},NORMAL,DRY,0.0,95.0,100.0,0.98" for i in range(1, 7)]
        out = estimate_traffic_rejoin_impact(_records(rows))
        self.assertTrue(any("placeholder" in n.lower() for n in out.notes))

    def test_tyre_compound_is_flagged_low_confidence(self) -> None:
        out = estimate_tyre_compound_impact(load_sample_race())
        self.assertTrue(any("low-confidence" in n.lower() for n in out.notes))

    def test_stint_length_scales_with_degradation(self) -> None:
        flat = [f"{i},95.0,30.4,35.15,29.45,MEDIUM,{i},NORMAL,DRY,1.0,95.0,100.0,0.98" for i in range(1, 7)]
        rising = [f"{i},{95.0 + 0.6*(i-1):.3f},30.4,35.15,29.45,MEDIUM,{i},NORMAL,DRY,1.0,95.0,100.0,0.98" for i in range(1, 7)]
        loss_flat = estimate_stint_length_impact(_records(flat)).if_wrong_expected_loss_s
        loss_rising = estimate_stint_length_impact(_records(rising)).if_wrong_expected_loss_s
        self.assertGreater(loss_rising, loss_flat)


class PipelineSevenDomainImpactTests(unittest.TestCase):
    def test_pipeline_emits_seven_domain_impacts(self) -> None:
        from src.models import ScenarioFlags
        from src.pipeline import analyze_decision

        result, _, _, _ = analyze_decision(load_sample_race(), ScenarioFlags(), prefer_granite=False)
        domains = {result.decision_impact.decision} | {di.decision for di in result.decision_impacts}
        expected = {
            "pit_timing",
            "push_vs_conserve",
            "stint_length",
            "tyre_strategy",
            "safety_car_response",
            "traffic_rejoin_risk",
            "aggressive_vs_safe_strategy",
        }
        self.assertEqual(domains, expected)


if __name__ == "__main__":
    unittest.main()
