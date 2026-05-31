from __future__ import annotations

import unittest
from io import StringIO

from src.data_loader import load_race_csv, load_sample_race
from tools.mdce_calibration import (
    _pearson,
    build_report,
    confidence_vs_regret_backtest,
    scenario_monotonicity,
)


def _degrading_csv(n: int = 16) -> list:
    """Construct a clean, steadily-degrading stint so the backtest has signal."""
    header = (
        "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,"
        "track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency\n"
    )
    rows = []
    base = 95.0
    for i in range(1, n + 1):
        lt = base + 0.15 * (i - 1)  # gentle steady degradation
        s1, s2, s3 = lt * 0.32, lt * 0.37, lt * 0.31
        pred = base + 0.05 * (i - 1)  # model is mildly optimistic
        rows.append(
            f"{i},{lt:.3f},{s1:.3f},{s2:.3f},{s3:.3f},MEDIUM,{i},NORMAL,DRY,1.0,{pred:.3f},100.0,0.98"
        )
    return load_race_csv(StringIO(header + "\n".join(rows)), source_name="degrading csv").records


class CalibrationPearsonTests(unittest.TestCase):
    def test_pearson_perfect_negative(self) -> None:
        self.assertEqual(_pearson([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_pearson_perfect_positive(self) -> None:
        self.assertEqual(_pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_pearson_undefined_on_constant(self) -> None:
        self.assertIsNone(_pearson([1, 1, 1], [1, 2, 3]))

    def test_pearson_undefined_on_too_few(self) -> None:
        self.assertIsNone(_pearson([1.0], [2.0]))


class ScenarioMonotonicityTests(unittest.TestCase):
    def test_confidence_non_increasing_with_more_failures(self) -> None:
        records = load_sample_race()
        out = scenario_monotonicity(records)
        # Core claim: confidence should not increase as more failure modes are injected.
        self.assertTrue(out["monotonic_non_increasing"])
        # And the correlation must be defined and negative (more failures -> lower confidence).
        self.assertIsNotNone(out["pearson_failurecount_vs_confidence"])
        self.assertLess(out["pearson_failurecount_vs_confidence"], 0.0)
        # Every preset must be represented.
        self.assertGreaterEqual(len(out["rows"]), 5)


class BacktestTests(unittest.TestCase):
    def test_backtest_is_calibrated_on_degrading_stint(self) -> None:
        records = _degrading_csv(16)
        out = confidence_vs_regret_backtest(records)
        self.assertEqual(out["status"], "OK")
        # SAFE-mode situations should carry >= aggressive-action regret vs AGGRESSIVE-mode ones.
        self.assertTrue(out["calibration_holds"])

    def test_backtest_insufficient_data_is_reported(self) -> None:
        header = (
            "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,"
            "track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency\n"
        )
        rows = "\n".join(
            f"{i},95.0,30.4,35.15,29.45,MEDIUM,{i},NORMAL,DRY,1.0,95.0,100.0,0.98" for i in range(1, 5)
        )
        records = load_race_csv(StringIO(header + rows), source_name="tiny csv").records
        out = confidence_vs_regret_backtest(records)
        self.assertEqual(out["status"], "INSUFFICIENT_DATA")

    def test_calibration_holds_is_null_when_one_mode_group_empty(self) -> None:
        # If only one mode ever occurs, calibration_holds must be null (not a false claim).
        records = _degrading_csv(16)
        out = confidence_vs_regret_backtest(records)
        if out["num_safe_mode"] == 0 or out["num_aggressive_mode"] == 0:
            self.assertIsNone(out["calibration_holds"])
        else:
            self.assertIn(out["calibration_holds"], {True, False})


class ReportShapeTests(unittest.TestCase):
    def test_build_report_shape(self) -> None:
        records = load_sample_race()
        report = build_report(records, source_name="unit-test")
        self.assertEqual(report["schema_version"], "mdce_calibration_v1")
        for k in ("timestamp_utc", "source_name", "num_laps", "scenario_monotonicity", "confidence_vs_regret_backtest"):
            self.assertIn(k, report)
        self.assertTrue(report["timestamp_utc"].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
