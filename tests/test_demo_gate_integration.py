from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class DemoGateIntegrationTests(unittest.TestCase):
    def test_demo_gate_runs_in_temp_root(self) -> None:
        """Integration smoke test: demo_gate can run end-to-end.

        Uses a temporary root with an explicit input CSV to avoid depending on
        whatever data might exist in the developer's working folder.
        """

        import json
        import os
        import sys

        import tools.demo_gate as gate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Minimal processed CSV + sidecar metadata.
            inp = root / "input.csv"
            inp.write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.1,MEDIUM,1\n"
                "2,95.4,MEDIUM,2\n"
                "3,95.9,MEDIUM,3\n",
                encoding="utf-8",
            )
            (root / "input.metadata.json").write_text(
                json.dumps({"license_spdx": "CC-BY-4.0", "source_url": "https://example.invalid"}),
                encoding="utf-8",
            )

            # Ensure NC opt-in isn't required for this test.
            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "demo_gate.py",
                    "--root",
                    str(root),
                    "--output-dir",
                    "outputs/reports",
                    "--input",
                    "input.csv",
                ]
                gate.main()
            finally:
                sys.argv = old_argv

            self.assertTrue(list(out_dir.glob("mdce_decision_run_*.json")))
            self.assertTrue(list(out_dir.glob("mdce_decision_run_*.md")))


if __name__ == "__main__":
    unittest.main()
