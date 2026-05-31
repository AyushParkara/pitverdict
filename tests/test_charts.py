from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_loader import load_sample_race
from tools.mdce_charts import generate_all


class ChartsTests(unittest.TestCase):
    def test_generates_png_files(self) -> None:
        records = load_sample_race()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "charts"
            produced = generate_all(records, out_dir)
            # At least the two always-on charts must exist.
            self.assertGreaterEqual(len(produced), 2)
            for p in produced:
                self.assertTrue(p.exists(), msg=f"missing {p}")
                self.assertGreater(p.stat().st_size, 1000, msg=f"suspiciously small PNG: {p}")
                # PNG magic header.
                with p.open("rb") as f:
                    self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n", msg=f"not a PNG: {p}")

    def test_expected_filenames_present(self) -> None:
        records = load_sample_race()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "charts"
            generate_all(records, out_dir)
            self.assertTrue((out_dir / "confidence_by_scenario.png").exists())
            self.assertTrue((out_dir / "lap_time_vs_model.png").exists())

    def test_confidence_over_laps_chart(self) -> None:
        from tools.mdce_charts import chart_confidence_over_laps

        records = load_sample_race()
        with tempfile.TemporaryDirectory() as tmp:
            out = chart_confidence_over_laps(records, Path(tmp) / "timeline.png")
            # Sample race has enough laps to produce a timeline.
            self.assertIsNotNone(out)
            self.assertTrue(out.exists())
            with out.open("rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
