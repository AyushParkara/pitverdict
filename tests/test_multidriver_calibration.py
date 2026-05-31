from __future__ import annotations

import unittest
from pathlib import Path

from tools.mdce_calibration_multi import DEFAULT_PARQUET, run_multi


class MultiDriverCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parquet = Path(__file__).resolve().parents[1] / DEFAULT_PARQUET
        if not self.parquet.exists():
            self.skipTest("Zenodo stg_laps parquet not available locally.")

    def test_runs_and_reports_summary_shape(self) -> None:
        report = run_multi(str(self.parquet), ["VER", "HAM"], decision_lap=None)
        self.assertEqual(report["schema_version"], "mdce_multidriver_calibration_v1")
        self.assertIn("summary", report)
        s = report["summary"]
        for k in (
            "drivers_evaluated",
            "monotonic_non_increasing_count",
            "calibration_evaluable",
            "calibration_holds_count",
        ):
            self.assertIn(k, s)
        # Each evaluated driver row carries the honest per-driver fields.
        for r in report["results"]:
            if "error" in r:
                continue
            for k in ("driver", "monotonic_non_increasing", "calibration_holds", "num_points"):
                self.assertIn(k, r)

    def test_monotonicity_is_reported_per_driver(self) -> None:
        report = run_multi(str(self.parquet), ["VER"], decision_lap=None)
        row = [r for r in report["results"] if r.get("driver") == "VER"][0]
        self.assertIn(row["monotonic_non_increasing"], {True, False})


if __name__ == "__main__":
    unittest.main()
