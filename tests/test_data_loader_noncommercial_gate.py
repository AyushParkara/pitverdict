from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


class DataLoaderNonCommercialGateTests(unittest.TestCase):
    def test_default_loader_falls_back_without_opt_in(self) -> None:
        from src.data_loader import NONCOMMERCIAL_GUARD_WARNING_PREFIX, load_default_data_result

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Provide a NonCommercial prepared dataset under the temp root.
            processed_dir = root / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            real_csv = processed_dir / "mdce_kaggle_weather_tyre_latest.csv"
            real_csv.write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.1,MEDIUM,1\n"
                "2,95.4,MEDIUM,2\n",
                encoding="utf-8",
            )
            real_csv.with_suffix(".metadata.json").write_text(
                json.dumps({"license_spdx": "CC-BY-NC-4.0", "source_url": "https://example.invalid"}),
                encoding="utf-8",
            )

            # Provide a sample fallback dataset under the same temp root.
            sample_dir = root / "data"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_csv = sample_dir / "sample_race.csv"
            sample_csv.write_text(
                "\n".join(
                    [
                        "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency",
                        "1,96.8,30.9,35.2,30.7,MEDIUM,1,NORMAL,DRY,2.8,96.75,93.0,0.96",
                        "2,94.9,30.1,34.6,30.2,MEDIUM,2,NORMAL,DRY,2.6,94.85,94.0,0.97",
                        "3,94.4,29.9,34.4,30.1,MEDIUM,3,NORMAL,DRY,2.4,94.35,95.0,0.98",
                    ]
                ),
                encoding="utf-8",
            )

            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)
            res = load_default_data_result(root=root)
            self.assertEqual(res.source_name, "offline demo fallback")
            self.assertTrue(res.warnings)
            self.assertTrue(str(res.warnings[0]).startswith(NONCOMMERCIAL_GUARD_WARNING_PREFIX))

    def test_default_loader_falls_back_when_metadata_unreadable(self) -> None:
        from src.data_loader import NONCOMMERCIAL_GUARD_WARNING_PREFIX, load_default_data_result

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            processed_dir = root / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            real_csv = processed_dir / "mdce_kaggle_weather_tyre_latest.csv"
            real_csv.write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.1,MEDIUM,1\n",
                encoding="utf-8",
            )
            # Sidecar exists but cannot be parsed.
            real_csv.with_suffix(".metadata.json").write_text("{ not json", encoding="utf-8")

            sample_dir = root / "data"
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "sample_race.csv").write_text(
                "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency\n"
                "1,96.8,30.9,35.2,30.7,MEDIUM,1,NORMAL,DRY,2.8,96.75,93.0,0.96\n",
                encoding="utf-8",
            )

            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)
            res = load_default_data_result(root=root)
            self.assertEqual(res.source_name, "offline demo fallback")
            self.assertTrue(res.warnings)
            self.assertTrue(str(res.warnings[0]).startswith(NONCOMMERCIAL_GUARD_WARNING_PREFIX))

    def test_default_loader_uses_prepared_dataset_with_opt_in(self) -> None:
        from src.data_loader import load_default_data_result

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed_dir = root / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            real_csv = processed_dir / "mdce_kaggle_weather_tyre_latest.csv"
            real_csv.write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.1,MEDIUM,1\n"
                "2,95.4,MEDIUM,2\n",
                encoding="utf-8",
            )
            meta = {"license_spdx": "CC-BY-NC-4.0", "source_url": "https://example.invalid", "note": "x"}
            real_csv.with_suffix(".metadata.json").write_text(json.dumps(meta), encoding="utf-8")

            os.environ["MDCE_ALLOW_NONCOMMERCIAL_DATA"] = "1"
            try:
                res = load_default_data_result(root=root)
            finally:
                os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

            self.assertEqual(res.source_name, str(real_csv))
            self.assertIsInstance(res.dataset_metadata, dict)
            self.assertEqual(res.dataset_metadata.get("license_spdx"), "CC-BY-NC-4.0")
            # Should not be using the guarded fallback warnings.
            self.assertFalse(any(str(w).startswith("NonCommercial data guard:") for w in res.warnings or []))


if __name__ == "__main__":
    unittest.main()
