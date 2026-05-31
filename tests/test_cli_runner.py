from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


class CliRunnerTests(unittest.TestCase):
    def test_cli_writes_json_md_and_registry(self) -> None:
        # Import inside the test so the module path injection can happen as authored.
        import tools.run_mdce_decision as runner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs" / "reports"
            registry = root / "outputs" / "reports" / "MDCE_DECISION_RUN_REGISTRY.csv"
            v2_registry = root / "outputs" / "reports" / "MDCE_DECISION_RUN_REGISTRY_v2.csv"

            # Pre-create a legacy registry header so the runner exercises schema evolution logic.
            out_dir.mkdir(parents=True, exist_ok=True)
            registry.write_text(
                "run_id,timestamp_utc,source_name,recommendation_type,recommended_lap,confidence,risk_level,conflict_score,conflict_label,issue_count,issues,json_path,md_path\n",
                encoding="utf-8",
            )

            # Provide a tiny processed MDCE CSV so the test is hermetic.
            input_csv = root / "input.csv"
            input_csv.write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.1,MEDIUM,1\n"
                "2,95.4,MEDIUM,2\n"
                "3,95.9,MEDIUM,3\n",
                encoding="utf-8",
            )

            # Provide a sidecar metadata file so the runner can attach it.
            (root / "input.metadata.json").write_text(
                json.dumps({"license_spdx": "CC-BY-4.0", "source_url": "https://example.invalid/dataset"}),
                encoding="utf-8",
            )

            # Use sample fallback by not providing an input CSV.
            argv = [
                "run_mdce_decision.py",
                "--root",
                str(root),
                "--input",
                "input.csv",
                "--output-dir",
                "outputs/reports",
                "--registry-csv",
                "outputs/reports/MDCE_DECISION_RUN_REGISTRY.csv",
                "--no-granite",
            ]

            # Force-disable NonCommercial default data for hermetic tests.
            import os
            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

            # Patch argv for this run.
            import sys

            old_argv = sys.argv
            try:
                sys.argv = argv
                # Keep test output clean.
                with redirect_stdout(StringIO()):
                    runner.main()
            finally:
                sys.argv = old_argv

            self.assertTrue(out_dir.exists())
            self.assertTrue(registry.exists())

            json_files = sorted(out_dir.glob("mdce_decision_run_*.json"))
            md_files = sorted(out_dir.glob("mdce_decision_run_*.md"))
            self.assertGreaterEqual(len(json_files), 1)
            self.assertGreaterEqual(len(md_files), 1)

            payload = json.loads(json_files[-1].read_text(encoding="utf-8"))
            self.assertEqual(payload.get("schema_version"), "mdce_decision_run_v2")
            self.assertEqual(payload.get("project_root"), str(root))
            self.assertIn("source", payload)
            self.assertIn("dataset_metadata", payload["source"])
            self.assertIsInstance(payload["source"].get("dataset_metadata"), dict)
            self.assertEqual(payload["source"]["dataset_metadata"].get("license_spdx"), "CC-BY-4.0")

            # Markdown should surface the provenance fields for the demo.
            md_text = md_files[-1].read_text(encoding="utf-8")
            self.assertIn("- Dataset license: `CC-BY-4.0`", md_text)
            self.assertIn("- Dataset source URL: `https://example.invalid/dataset`", md_text)

        # Also verify fallback runs attach sample dataset metadata.
        import os
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs" / "reports"
            argv = [
                "run_mdce_decision.py",
                "--root",
                str(Path(__file__).resolve().parents[1]),
                "--output-dir",
                str(out_dir),
                "--registry-csv",
                str(root / "outputs" / "reports" / "MDCE_DECISION_RUN_REGISTRY.csv"),
                "--no-granite",
            ]
            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

            old_argv = sys.argv
            try:
                sys.argv = argv
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    runner.main()
            finally:
                sys.argv = old_argv

            json_files = sorted(out_dir.glob("mdce_decision_run_*.json"))
            payload = json.loads(json_files[-1].read_text(encoding="utf-8"))
            self.assertIn("dataset_metadata", payload["source"])
            self.assertEqual(payload["source"]["dataset_metadata"].get("license_spdx"), "NOASSERTION")

            md_files = sorted(out_dir.glob("mdce_decision_run_*.md"))
            md_text = md_files[-1].read_text(encoding="utf-8")
            self.assertIn("- Dataset license: `NOASSERTION`", md_text)

    def test_cli_prints_noncommercial_guard_warning_on_default_fallback(self) -> None:
        import tools.run_mdce_decision as runner

        import os
        import sys
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Ensure runner uses default loader (no --input), but sees a NonCommercial default dataset.
            processed_dir = root / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            (processed_dir / "mdce_kaggle_weather_tyre_latest.csv").write_text("lap,lap_time_s\n1,95.1\n", encoding="utf-8")
            (processed_dir / "mdce_kaggle_weather_tyre_latest.metadata.json").write_text(
                "{\"license_spdx\": \"CC-BY-NC-4.0\"}",
                encoding="utf-8",
            )

            # Provide the fallback sample dataset under the temp root so the test is hermetic.
            sample_dir = root / "data"
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "sample_race.csv").write_text(
                "\n".join(
                    [
                        "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency",
                        "1,96.8,30.9,35.2,30.7,MEDIUM,1,NORMAL,DRY,2.8,96.75,93.0,0.96",
                        "2,94.9,30.1,34.6,30.2,MEDIUM,2,NORMAL,DRY,2.6,94.85,94.0,0.97",
                        "3,94.4,29.9,34.4,30.1,MEDIUM,3,NORMAL,DRY,2.4,94.35,95.0,0.98",
                        "4,94.18,29.82,34.31,30.05,MEDIUM,4,NORMAL,DRY,2.3,94.13,96.0,0.98",
                        "5,94.3,29.9,34.4,30.0,MEDIUM,5,NORMAL,DRY,2.2,94.25,97.0,0.98",
                        "6,94.55,30.0,34.6,29.95,MEDIUM,6,NORMAL,DRY,2.1,94.5,98.0,0.98",
                    ]
                ),
                encoding="utf-8",
            )

            # Guarantee opt-in is disabled for the test.
            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

            argv = [
                "run_mdce_decision.py",
                "--root",
                str(root),
                "--output-dir",
                "outputs/reports",
                "--registry-csv",
                "outputs/reports/MDCE_DECISION_RUN_REGISTRY.csv",
                "--no-granite",
            ]

            old_argv = sys.argv
            try:
                sys.argv = argv
                out_buf = StringIO()
                err_buf = StringIO()
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    runner.main()
            finally:
                sys.argv = old_argv

            self.assertIn("NonCommercial data guard:", err_buf.getvalue())

    def test_cli_run_id_is_unique_within_same_second(self) -> None:
        """Regression: avoid JSON/MD overwrite when two runs happen in 1 second."""

        import tools.run_mdce_decision as runner

        import os
        import sys
        from contextlib import redirect_stderr, redirect_stdout
        from datetime import datetime, timezone
        from io import StringIO
        from unittest.mock import patch

        class _FakeDatetime:
            def __init__(self, times):
                self._it = iter(times)

            def now(self, tz=None):
                return next(self._it)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Minimal processed MDCE CSV so the run is hermetic.
            (root / "input.csv").write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.1,MEDIUM,1\n"
                "2,95.4,MEDIUM,2\n"
                "3,95.9,MEDIUM,3\n",
                encoding="utf-8",
            )

            argv = [
                "run_mdce_decision.py",
                "--root",
                str(root),
                "--input",
                "input.csv",
                "--output-dir",
                "outputs/reports",
                "--registry-csv",
                "outputs/reports/MDCE_DECISION_RUN_REGISTRY.csv",
                "--no-granite",
            ]

            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

            t1 = datetime(2026, 1, 1, 0, 0, 0, 111111, tzinfo=timezone.utc)
            t2 = datetime(2026, 1, 1, 0, 0, 0, 222222, tzinfo=timezone.utc)
            fake_dt = _FakeDatetime([t1, t2])

            old_argv = sys.argv
            try:
                sys.argv = argv
                with patch("tools.run_mdce_decision.datetime", fake_dt):
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        runner.main()
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        runner.main()
            finally:
                sys.argv = old_argv

            json_files = sorted(out_dir.glob("mdce_decision_run_*.json"))
            md_files = sorted(out_dir.glob("mdce_decision_run_*.md"))
            self.assertEqual(len(json_files), 2)
            self.assertEqual(len(md_files), 2)
            self.assertTrue(any("_111111" in p.name for p in json_files))
            self.assertTrue(any("_222222" in p.name for p in json_files))

    def test_cli_does_not_double_append_when_registry_is_v2(self) -> None:
        """Regression test: if caller points --registry-csv to v2, we only append once."""

        import tools.run_mdce_decision as runner

        import os
        import sys
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)

            v2_registry = out_dir / "MDCE_DECISION_RUN_REGISTRY_v2.csv"

            # Provide a tiny processed MDCE CSV so the test is hermetic.
            input_csv = root / "input.csv"
            input_csv.write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.1,MEDIUM,1\n"
                "2,95.4,MEDIUM,2\n"
                "3,95.9,MEDIUM,3\n",
                encoding="utf-8",
            )

            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

            argv = [
                "run_mdce_decision.py",
                "--root",
                str(root),
                "--input",
                "input.csv",
                "--output-dir",
                "outputs/reports",
                # Point legacy registry path to v2 file.
                "--registry-csv",
                "outputs/reports/MDCE_DECISION_RUN_REGISTRY_v2.csv",
                "--no-granite",
            ]

            old_argv = sys.argv
            try:
                sys.argv = argv
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    runner.main()
            finally:
                sys.argv = old_argv

            self.assertTrue(v2_registry.exists())
            # Header + exactly one row.
            lines = v2_registry.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)

    def test_v2_registry_header_migration_preserves_rows(self) -> None:
        """If the existing v2 registry has an old/shorter header, we rewrite it."""

        import tools.run_mdce_decision as runner

        import os
        import sys
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Create a v2 registry with a truncated header and one existing row.
            v2_registry = out_dir / "MDCE_DECISION_RUN_REGISTRY_v2.csv"
            v2_registry.write_text(
                "run_id,timestamp_utc\n"
                "old_run,2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )

            # Minimal processed CSV.
            (root / "input.csv").write_text(
                "lap,lap_time_s,tyre_compound,tyre_age\n"
                "1,95.1,MEDIUM,1\n"
                "2,95.4,MEDIUM,2\n"
                "3,95.9,MEDIUM,3\n",
                encoding="utf-8",
            )

            os.environ.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

            argv = [
                "run_mdce_decision.py",
                "--root",
                str(root),
                "--input",
                "input.csv",
                "--output-dir",
                "outputs/reports",
                # Point legacy registry somewhere else so we don't overwrite our v2 file.
                "--registry-csv",
                "outputs/reports/MDCE_DECISION_RUN_REGISTRY.csv",
                "--no-granite",
            ]

            old_argv = sys.argv
            try:
                sys.argv = argv
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    runner.main()
            finally:
                sys.argv = old_argv

            # After migration + append: header + old row + new run row.
            lines = v2_registry.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 3)
            self.assertTrue(lines[0].startswith("run_id,timestamp_utc,source_name,dataset_license,dataset_source_url,preset"))
            # Old row should still exist.
            self.assertIn("old_run", "\n".join(lines[1:]))


if __name__ == "__main__":
    unittest.main()
