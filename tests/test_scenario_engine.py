from __future__ import annotations

import unittest

from src.data_loader import load_sample_race
from src.models import ScenarioFlags
from src.scenario_engine import apply_scenarios


class ScenarioEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_sample_race()
        # The sample dataset is small; these tests assume it has enough laps.
        self.assertGreaterEqual(len(self.records), 8)

    def test_missing_telemetry_marks_last_two_laps(self) -> None:
        res = apply_scenarios(self.records, ScenarioFlags(missing_telemetry=True))
        self.assertEqual(len(res.records), len(self.records))
        self.assertTrue(res.records[-1].missing)
        self.assertTrue(res.records[-2].missing)
        self.assertFalse(any(r.missing for r in res.records[:-2]))
        self.assertTrue(any("missing telemetry" in n.lower() for n in res.notes))

    def test_safety_car_marks_last_three_laps(self) -> None:
        res = apply_scenarios(self.records, ScenarioFlags(safety_car_phase=True))
        last = res.records[-3:]
        self.assertTrue(all(r.track_status == "SC" for r in last))
        # Earlier laps shouldn't be forced to SC.
        self.assertTrue(any(r.track_status != "SC" for r in res.records[:-3]))
        self.assertTrue(any("safety car" in n.lower() for n in res.notes))

    def test_weather_uncertainty_sets_damp_and_reduces_speed_consistency(self) -> None:
        base_last5 = self.records[-5:]
        res = apply_scenarios(self.records, ScenarioFlags(weather_uncertainty=True))
        last5 = res.records[-5:]
        self.assertTrue(all(r.weather == "DAMP" for r in last5))
        for before, after in zip(base_last5, last5, strict=True):
            self.assertLessEqual(after.speed_consistency, before.speed_consistency)
        self.assertTrue(any("damp" in n.lower() for n in res.notes))

    def test_model_mismatch_makes_predictions_optimistic(self) -> None:
        res = apply_scenarios(self.records, ScenarioFlags(model_mismatch=True))
        # Only last 6 are affected per scenario.
        for before, after in zip(self.records[-6:], res.records[-6:], strict=True):
            self.assertLessEqual(after.predicted_lap_time_s, before.predicted_lap_time_s)
            # Scenario attempts to enforce a clear optimistic gap when possible.
            self.assertLessEqual(after.predicted_lap_time_s, after.lap_time_s - 1.0)
        self.assertTrue(any("optimistic" in n.lower() for n in res.notes))

    def test_tyre_signal_drift_holds_proxy_flat_over_last_six(self) -> None:
        res = apply_scenarios(self.records, ScenarioFlags(tyre_signal_drift=True))
        last6 = res.records[-6:]
        anchor = last6[0].tyre_temp_proxy_c
        self.assertTrue(all(abs(r.tyre_temp_proxy_c - anchor) < 1e-9 for r in last6))
        self.assertTrue(any("tyre proxy" in n.lower() for n in res.notes))


if __name__ == "__main__":
    unittest.main()
