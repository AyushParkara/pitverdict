from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path


class DataLoaderUnitTests(unittest.TestCase):
    def test_env_truthy_variants(self) -> None:
        from src.data_loader import _env_truthy

        name = "MDCE_TEST_TRUTHY"
        try:
            for v in ["1", "true", "TRUE", " yes ", "On", "y"]:
                os.environ[name] = v
                self.assertTrue(_env_truthy(name), v)
            for v in ["0", "false", "", "no", "off", "random"]:
                os.environ[name] = v
                self.assertFalse(_env_truthy(name), v)
        finally:
            os.environ.pop(name, None)

    def test_looks_noncommercial_license(self) -> None:
        from src.data_loader import _looks_noncommercial_license

        self.assertTrue(_looks_noncommercial_license("CC-BY-NC-4.0"))
        self.assertTrue(_looks_noncommercial_license("by-nc"))
        self.assertTrue(_looks_noncommercial_license("NonCommercial"))
        self.assertFalse(_looks_noncommercial_license("CC-BY-4.0"))
        self.assertFalse(_looks_noncommercial_license("MIT"))
        self.assertFalse(_looks_noncommercial_license(None))

    def test_default_real_data_requires_opt_in_metadata_missing_is_conservative(self) -> None:
        from src.data_loader import _default_real_data_requires_opt_in

        with tempfile.TemporaryDirectory() as tmp:
            real_path = Path(tmp) / "mdce_kaggle_weather_tyre_latest.csv"
            real_path.write_text("lap,lap_time_s\n1,95.0\n", encoding="utf-8")
            requires, hint = _default_real_data_requires_opt_in(real_path)
            self.assertTrue(requires)
            self.assertIsInstance(hint, str)
            self.assertIn("metadata missing", hint.lower())

    def test_default_real_data_requires_opt_in_metadata_unreadable_is_conservative(self) -> None:
        from src.data_loader import _default_real_data_requires_opt_in

        with tempfile.TemporaryDirectory() as tmp:
            real_path = Path(tmp) / "mdce_kaggle_weather_tyre_latest.csv"
            real_path.write_text("lap,lap_time_s\n1,95.0\n", encoding="utf-8")
            # Sidecar exists but is not valid JSON.
            real_path.with_suffix(".metadata.json").write_text("{ this is not json ", encoding="utf-8")
            requires, hint = _default_real_data_requires_opt_in(real_path)
            self.assertTrue(requires)
            self.assertIsInstance(hint, str)
            self.assertIn("unreadable", hint.lower())

    def test_default_real_data_requires_opt_in_detects_from_license_note(self) -> None:
        from src.data_loader import _default_real_data_requires_opt_in

        with tempfile.TemporaryDirectory() as tmp:
            real_path = Path(tmp) / "mdce_kaggle_weather_tyre_latest.csv"
            real_path.write_text("lap,lap_time_s\n1,95.0\n", encoding="utf-8")
            real_path.with_suffix(".metadata.json").write_text(
                json.dumps({"license_note": "CC BY-NC 4.0"}),
                encoding="utf-8",
            )
            requires, hint = _default_real_data_requires_opt_in(real_path)
            self.assertTrue(requires)
            self.assertIn("cc", (hint or "").lower())

    def test_read_sidecar_metadata_ignores_invalid_json(self) -> None:
        from src.data_loader import _read_sidecar_metadata

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_text("lap,lap_time_s\n1,95\n", encoding="utf-8")
            p.with_suffix(".metadata.json").write_text("not json", encoding="utf-8")
            self.assertEqual(_read_sidecar_metadata(p), {})

    def test_load_race_csv_sorts_by_lap_and_drops_invalid_rows(self) -> None:
        from src.data_loader import load_race_csv

        # Includes out-of-order laps and one invalid lap_time.
        csv_data = io.StringIO(
            "lap,lap_time_s,tyre_compound,tyre_age\n"
            "2,95.1,MEDIUM,2\n"
            "1,95.0,MEDIUM,1\n"
            "3,not_a_number,MEDIUM,3\n"
            "4,95.3,MEDIUM,4\n"
        )
        res = load_race_csv(csv_data, source_name="unit")
        laps = [r.lap for r in res.records]
        self.assertEqual(laps, [1, 2, 4])

    def test_load_race_csv_derives_sector_times_when_missing(self) -> None:
        from src.data_loader import load_race_csv

        res = load_race_csv(
            io.StringIO(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,100.0,MEDIUM,1\n"
                "2,101.0,MEDIUM,2\n"
            ),
            source_name="unit",
        )
        self.assertTrue(res.records)
        self.assertAlmostEqual(res.records[0].sector1_s, 32.0, places=3)
        self.assertAlmostEqual(res.records[0].sector2_s, 37.0, places=3)
        self.assertAlmostEqual(res.records[0].sector3_s, 31.0, places=3)
        self.assertIn("sector1_s", res.derived_columns)

    def test_load_race_csv_derives_weather_from_rainfall_if_present(self) -> None:
        from src.data_loader import load_race_csv

        res = load_race_csv(
            io.StringIO(
                "lap,lap_time_s,tyre_compound,tyre_age,rainfall\n"
                "1,95.0,MEDIUM,1,0.0\n"
                "2,95.4,MEDIUM,2,1.2\n"
            ),
            source_name="unit",
        )
        self.assertEqual([r.weather for r in res.records], ["DRY", "WET"])
        self.assertIn("weather", res.derived_columns)

    def test_load_race_csv_attaches_sidecar_metadata_for_paths(self) -> None:
        from src.data_loader import load_race_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.csv"
            p.write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.0,MEDIUM,1\n",
                encoding="utf-8",
            )
            p.with_suffix(".metadata.json").write_text(
                json.dumps({"license_spdx": "CC-BY-4.0", "source_url": "https://example.invalid"}),
                encoding="utf-8",
            )
            res = load_race_csv(p, source_name="path")
            self.assertEqual(res.dataset_metadata.get("license_spdx"), "CC-BY-4.0")


if __name__ == "__main__":
    unittest.main()
