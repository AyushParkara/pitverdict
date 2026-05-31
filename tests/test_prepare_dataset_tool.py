from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


class PrepareDatasetToolTests(unittest.TestCase):
    def test_prepare_tool_writes_csv_and_metadata(self) -> None:
        """Hermetic test by patching the adapter call."""

        import sys

        import pandas as pd

        import tools.prepare_mdce_dataset as tool

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_csv = root / "out.csv"

            # Patch adapter used by the tool module.
            class _Prepared:
                def __init__(self) -> None:
                    self.frame = pd.DataFrame(
                        {
                            "lap": [1, 2],
                            "lap_time_s": [95.1, 95.4],
                            "tyre_compound": ["MEDIUM", "MEDIUM"],
                            "tyre_age": [1, 2],
                        }
                    )
                    self.metadata = {"license_spdx": "CC-BY-NC-4.0", "source_url": "https://example.invalid"}

            old_kaggle = tool.prepare_kaggle_weather_tyre
            old_fastf1 = tool.prepare_fastf1_race
            old_zenodo = tool.prepare_zenodo_stg_laps
            old_zenodo_gap = tool.prepare_zenodo_stg_laps_with_gap_proxy
            tool.prepare_kaggle_weather_tyre = lambda *args, **kwargs: _Prepared()  # type: ignore[assignment]
            tool.prepare_fastf1_race = lambda *args, **kwargs: _Prepared()  # type: ignore[assignment]
            tool.prepare_zenodo_stg_laps = lambda *args, **kwargs: _Prepared()  # type: ignore[assignment]
            tool.prepare_zenodo_stg_laps_with_gap_proxy = lambda *args, **kwargs: _Prepared()  # type: ignore[assignment]
            try:
                old_argv = sys.argv
                try:
                    sys.argv = [
                        "prepare_mdce_dataset.py",
                        "--source",
                        "kaggle-weather-tyre",
                        "--input",
                        str(root / "in.parquet"),
                        "--output",
                        str(out_csv),
                    ]
                    with redirect_stdout(StringIO()):
                        tool.main()
                finally:
                    sys.argv = old_argv
            finally:
                tool.prepare_kaggle_weather_tyre = old_kaggle  # type: ignore[assignment]
                tool.prepare_fastf1_race = old_fastf1  # type: ignore[assignment]
                tool.prepare_zenodo_stg_laps = old_zenodo  # type: ignore[assignment]
                tool.prepare_zenodo_stg_laps_with_gap_proxy = old_zenodo_gap  # type: ignore[assignment]

            self.assertTrue(out_csv.exists())
            meta_path = out_csv.with_suffix(".metadata.json")
            self.assertTrue(meta_path.exists())
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("license_spdx"), "CC-BY-NC-4.0")


if __name__ == "__main__":
    unittest.main()
