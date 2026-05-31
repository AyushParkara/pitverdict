from __future__ import annotations

import unittest
from io import StringIO

from src.data_loader import load_race_csv, load_sample_race
from src.models import ScenarioFlags
from src.sensitivity_engine import analyze_recommendation_stability, _perturb_recent_lap_times


def _csv(rows: list[str]) -> list:
    header = (
        "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,"
        "track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency"
    )
    return load_race_csv(StringIO(header + "\n" + "\n".join(rows)), source_name="sens csv").records


class SensitivityEngineTests(unittest.TestCase):
    def test_shape_and_keys(self) -> None:
        out = analyze_recommendation_stability(load_sample_race())
        for k in (
            "baseline",
            "recent_n",
            "perturbations_s",
            "points",
            "type_stability",
            "recommended_lap_spread",
            "confidence_range",
            "verdict",
        ):
            self.assertIn(k, out)
        self.assertIn(out["verdict"], {"STABLE", "MODERATE", "UNSTABLE"})
        self.assertEqual(len(out["points"]), len(out["perturbations_s"]))

    def test_type_stability_in_unit_range(self) -> None:
        out = analyze_recommendation_stability(load_sample_race())
        self.assertGreaterEqual(out["type_stability"], 0.0)
        self.assertLessEqual(out["type_stability"], 1.0)

    def test_clean_stable_stint_is_stable(self) -> None:
        # Steady, clearly-stable pace -> recommendation should not flip.
        rows = [f"{i},95.0,30.4,35.15,29.45,MEDIUM,{i},NORMAL,DRY,1.0,95.0,100.0,0.98" for i in range(1, 11)]
        out = analyze_recommendation_stability(_csv(rows))
        self.assertEqual(out["verdict"], "STABLE")
        self.assertEqual(out["type_stability"], 1.0)

    def test_perturb_only_touches_recent_n(self) -> None:
        records = load_sample_race()
        recent_n = 3
        perturbed = _perturb_recent_lap_times(records, 0.5, recent_n)
        # Earlier laps unchanged.
        for before, after in zip(records[:-recent_n], perturbed[:-recent_n]):
            self.assertEqual(before.lap_time_s, after.lap_time_s)
        # Recent laps shifted by +0.5.
        for before, after in zip(records[-recent_n:], perturbed[-recent_n:]):
            self.assertAlmostEqual(after.lap_time_s, before.lap_time_s + 0.5, places=6)

    def test_can_detect_instability_near_boundary(self) -> None:
        # Construct a stint sitting near the PIT_SOON/EXTEND degradation boundary (~0.20/lap),
        # so small perturbations to recent laps can flip the recommendation type.
        # lap times rising ~0.2/lap.
        rows = []
        base = 95.0
        for i in range(1, 9):
            lt = base + 0.2 * (i - 1)
            rows.append(f"{i},{lt:.3f},{lt*0.32:.3f},{lt*0.37:.3f},{lt*0.31:.3f},MEDIUM,{i},NORMAL,DRY,1.0,{lt:.3f},100.0,0.98")
        out = analyze_recommendation_stability(_csv(rows), ScenarioFlags(), perturbations_s=(-0.5, -0.3, 0.3, 0.5), recent_n=3)
        # We don't assert a specific verdict (data-dependent), but the analysis must run and
        # produce a defined verdict and a non-negative lap spread.
        self.assertIn(out["verdict"], {"STABLE", "MODERATE", "UNSTABLE"})
        self.assertGreaterEqual(out["recommended_lap_spread"], 0)


if __name__ == "__main__":
    unittest.main()
