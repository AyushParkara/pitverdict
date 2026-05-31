from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class DatasetValidatorTests(unittest.TestCase):
    def test_validator_accepts_minimal_csv(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.csv"
            p.write_text(
                "lap,lap_time_s\n"
                "1,95.1\n"
                "2,95.4\n",
                encoding="utf-8",
            )
            res = validate_mdce_csv(p)
            self.assertTrue(res.ok)
            self.assertFalse(res.errors)

    def test_validator_flags_missing_required_cols(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            p.write_text("lap\n1\n", encoding="utf-8")
            res = validate_mdce_csv(p)
            self.assertFalse(res.ok)
            self.assertTrue(any("Missing required columns" in e for e in res.errors))

    def test_validator_warns_on_placeholder_patterns(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w.csv"
            p.write_text(
                "lap,lap_time_s,sector1_s,sector2_s,sector3_s,gap_to_car_ahead_s,tyre_compound\n"
                "1,100,32,37,31,0.0,UNKNOWN\n"
                "2,102,32.64,37.74,31.62,0.0,UNKNOWN\n"
                "3,98,31.36,36.26,30.38,0.0,UNKNOWN\n"
                "4,101,32.32,37.37,31.31,0.0,UNKNOWN\n"
                "5,99,31.68,36.63,30.69,0.0,UNKNOWN\n"
                "6,103,32.96,38.11,31.93,0.0,UNKNOWN\n",
                encoding="utf-8",
            )
            res = validate_mdce_csv(p)
            self.assertTrue(res.ok)
            # Expect at least one placeholder-related warning.
            self.assertTrue(
                any(
                    "placeholder" in w.lower() or "proportional" in w.lower() or "unknown" in w.lower()
                    for w in res.warnings
                )
            )

    def test_validator_warns_on_invalid_gap_and_sector_cells(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.csv"
            p.write_text(
                "lap,lap_time_s,gap_to_car_ahead_s,sector1_s,sector2_s,sector3_s\n"
                "1,95.0,not_a_number,30,30,35\n"
                "2,95.0,-1,30,xx,35\n",
                encoding="utf-8",
            )
            res = validate_mdce_csv(p)
            self.assertTrue(res.ok)
            self.assertTrue(any("invalid gap_to_car_ahead_s" in w.lower() for w in res.warnings))
            self.assertTrue(any("negative" in w.lower() and "gap_to_car_ahead_s" in w for w in res.warnings))
            self.assertTrue(any("invalid sector2_s" in w.lower() for w in res.warnings))

    def test_validator_warns_when_sector_sum_differs_from_lap_time(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.csv"
            p.write_text(
                "lap,lap_time_s,sector1_s,sector2_s,sector3_s\n"
                # Sum=110 vs lap_time=95 should warn.
                "1,95.0,30,40,40\n",
                encoding="utf-8",
            )
            res = validate_mdce_csv(p)
            self.assertTrue(res.ok)
            self.assertTrue(any("sector sum" in w.lower() for w in res.warnings))

    def test_validator_warns_when_kaggle_csv_missing_sidecar_metadata(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mdce_kaggle_weather_tyre_latest.csv"
            p.write_text(
                "lap,lap_time_s\n"
                "1,95.1\n"
                "2,95.4\n",
                encoding="utf-8",
            )
            res = validate_mdce_csv(p)
            self.assertTrue(res.ok)
            self.assertTrue(any("Sidecar metadata missing" in w for w in res.warnings))

    def test_validator_warns_on_unsorted_and_duplicate_laps(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.csv"
            p.write_text(
                "lap,lap_time_s\n"
                "2,95.1\n"
                "1,95.0\n"
                "1,95.2\n",
                encoding="utf-8",
            )
            res = validate_mdce_csv(p)
            self.assertTrue(res.ok)
            self.assertTrue(any("not monotonically" in w.lower() for w in res.warnings))
            self.assertTrue(any("duplicate" in w.lower() for w in res.warnings))

    def test_validator_fails_on_unparseable_lap_time(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            p.write_text(
                "lap,lap_time_s\n"
                "1,not_a_number\n",
                encoding="utf-8",
            )
            res = validate_mdce_csv(p)
            self.assertFalse(res.ok)
            self.assertTrue(any("invalid lap/lap_time_s" in e.lower() for e in res.errors))

    def test_validator_fails_on_empty_csv(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.csv"
            p.write_text("lap,lap_time_s\n", encoding="utf-8")
            res = validate_mdce_csv(p)
            self.assertFalse(res.ok)
            self.assertTrue(any("no data rows" in e.lower() for e in res.errors))

    def test_validator_warns_when_sidecar_invalid_json(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mdce_kaggle_weather_tyre_latest.csv"
            p.write_text(
                "lap,lap_time_s\n"
                "1,95.1\n",
                encoding="utf-8",
            )
            # Sidecar exists but is not JSON.
            p.with_suffix(".metadata.json").write_text("{not json", encoding="utf-8")
            res = validate_mdce_csv(p)
            self.assertTrue(res.ok)
            self.assertTrue(any("could not be parsed" in w.lower() for w in res.warnings))

    def test_validator_warns_when_sidecar_missing_license_fields(self) -> None:
        from tools.validate_mdce_dataset import validate_mdce_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mdce_kaggle_weather_tyre_latest.csv"
            p.write_text(
                "lap,lap_time_s\n"
                "1,95.1\n",
                encoding="utf-8",
            )
            p.with_suffix(".metadata.json").write_text("{}", encoding="utf-8")
            res = validate_mdce_csv(p)
            self.assertTrue(res.ok)
            self.assertTrue(any("missing license" in w.lower() for w in res.warnings))

    def test_validator_exit_code_on_errors(self) -> None:
        # Integration-level check: CLI exits with code 2 when invalid.
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            p.write_text("lap\n1\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "tools/validate_mdce_dataset.py", "--input", str(p)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("ERROR:", proc.stderr)

    def test_validator_fail_on_warning_flag(self) -> None:
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "w.csv"
            # Force a warning via out-of-range lap_time_s.
            p.write_text(
                "lap,lap_time_s\n"
                "1,10\n"
                "2,10\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, "tools/validate_mdce_dataset.py", "--input", str(p), "--fail-on-warning"],
                text=True,
                capture_output=True,
            )
            self.assertEqual(proc.returncode, 3)
            self.assertIn("WARN:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
