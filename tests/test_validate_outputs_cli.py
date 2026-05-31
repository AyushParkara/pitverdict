from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


class ValidateOutputsCliTests(unittest.TestCase):
    def _run_mdce(self, root: Path) -> Path:
        """Run the headless CLI once; return latest JSON path."""

        import os
        import sys

        import tools.run_mdce_decision as runner

        out_dir = root / "outputs" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

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

        json_files = sorted(out_dir.glob("mdce_decision_run_*.json"))
        self.assertTrue(json_files)
        return json_files[-1]

    def test_validate_outputs_cli_ok(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                ]
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    validator.main()
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_payload_artifacts_json_mismatch(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)
            out_dir = root / "outputs" / "reports"

            payload = json.loads(json_path.read_text(encoding="utf-8"))

            # Create a decoy JSON file that exists, then point payload.artifacts.json at it.
            decoy = out_dir / "decoy.json"
            decoy.write_text("{}", encoding="utf-8")
            payload.setdefault("artifacts", {})["json"] = str(decoy)
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str(out_dir),
                    "--json",
                    str(json_path),
                ]
                with self.assertRaises(ValueError) as ctx:
                    # Validator prints to stderr then re-raises.
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("artifacts.json does not match", str(ctx.exception))
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_registry_v2_arg_mismatch_payload(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload_v2 = Path((payload.get("artifacts") or {}).get("registry_v2_csv") or "")
            self.assertTrue(payload_v2.exists())

            # Provide a different existing path to trigger the mismatch guard.
            alt_v2 = payload_v2.with_name("ALT_REGISTRY.csv")
            alt_v2.write_text(payload_v2.read_text(encoding="utf-8"), encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                    "--registry-v2",
                    str(alt_v2),
                ]
                with self.assertRaises(ValueError) as ctx:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("Registry v2 arg does not match", str(ctx.exception))
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_markdown_missing(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            md_path = Path((payload.get("artifacts") or {}).get("markdown") or "")
            self.assertTrue(md_path.exists())
            md_path.unlink()

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                ]
                with self.assertRaises(ValueError) as ctx:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("Markdown artifact not found", str(ctx.exception))
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_markdown_recommended_mode_mismatch(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            md_path = Path((payload.get("artifacts") or {}).get("markdown") or "")
            text = md_path.read_text(encoding="utf-8")
            # Flip the recommended mode line.
            if "- Recommended mode: `SAFE`" in text:
                text = text.replace("- Recommended mode: `SAFE`", "- Recommended mode: `AGGRESSIVE`")
            else:
                text = text.replace("- Recommended mode: `AGGRESSIVE`", "- Recommended mode: `SAFE`")
            md_path.write_text(text, encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                ]
                with self.assertRaises(ValueError) as ctx:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("Markdown recommended mode does not match", str(ctx.exception))
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_markdown_uncertainty_primary_mismatch(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            md_path = Path((payload.get("artifacts") or {}).get("markdown") or "")
            text = md_path.read_text(encoding="utf-8")
            # Force a mismatch: set primary to a value we don't use in JSON.
            text = text.replace("- Primary: `", "- Primary: `__MISMATCH__", 1)
            md_path.write_text(text, encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                ]
                with self.assertRaises(ValueError) as ctx:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("Markdown uncertainty primary does not match", str(ctx.exception))
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_markdown_uncertainty_score_mismatch(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            expected_score = str(((payload.get("decision") or {}).get("uncertainty") or {}).get("uncertainty_score"))
            self.assertNotEqual(expected_score, "None")

            md_path = Path((payload.get("artifacts") or {}).get("markdown") or "")
            text = md_path.read_text(encoding="utf-8")
            bad = "0.0" if expected_score != "0.0" else "1.0"
            text = text.replace(f"- Score: `{expected_score}`", f"- Score: `{bad}`", 1)
            md_path.write_text(text, encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                ]
                with self.assertRaises(ValueError) as ctx:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("Markdown uncertainty score does not match JSON", str(ctx.exception))
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_markdown_decision_impact_if_wrong_loss_mismatch(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            expected_loss = str((payload.get("decision_impact") or {}).get("if_wrong_expected_loss_s"))
            self.assertNotEqual(expected_loss, "None")

            md_path = Path((payload.get("artifacts") or {}).get("markdown") or "")
            text = md_path.read_text(encoding="utf-8")
            # Replace the numeric loss in the exact Markdown line.
            bad = "9999.0" if expected_loss != "9999.0" else "8888.0"
            text = text.replace(
                f"- If wrong (expected loss): `{expected_loss}` s",
                f"- If wrong (expected loss): `{bad}` s",
                1,
            )
            md_path.write_text(text, encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                ]
                with self.assertRaises(ValueError) as ctx:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("Markdown decision impact if-wrong loss does not match", str(ctx.exception))
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_markdown_confidence_risk_mismatch(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            expected_risk = str(((payload.get("decision") or {}).get("confidence") or {}).get("risk_level"))
            self.assertNotEqual(expected_risk, "None")

            md_path = Path((payload.get("artifacts") or {}).get("markdown") or "")
            text = md_path.read_text(encoding="utf-8")
            bad = "Low" if expected_risk != "Low" else "High"
            text = text.replace(f"- Risk: `{expected_risk}`", f"- Risk: `{bad}`", 1)
            md_path.write_text(text, encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                ]
                with self.assertRaises(ValueError) as ctx:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("Markdown risk level does not match JSON", str(ctx.exception))
            finally:
                sys.argv = old_argv

    def test_validate_outputs_fails_when_markdown_expected_gain_loss_mismatch(self) -> None:
        import sys

        import tools.validate_mdce_outputs as validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = self._run_mdce(root)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            expected_gl = str(((payload.get("decision") or {}).get("recommendation") or {}).get("expected_gain_loss_s"))
            self.assertNotEqual(expected_gl, "None")

            md_path = Path((payload.get("artifacts") or {}).get("markdown") or "")
            text = md_path.read_text(encoding="utf-8")
            bad = "1234.5" if expected_gl != "1234.5" else "4321.0"
            text = text.replace(
                f"- Expected gain/loss (heuristic): `{expected_gl}` seconds",
                f"- Expected gain/loss (heuristic): `{bad}` seconds",
                1,
            )
            md_path.write_text(text, encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_mdce_outputs.py",
                    "--outputs-dir",
                    str((root / "outputs" / "reports")),
                    "--json",
                    str(json_path),
                ]
                with self.assertRaises(ValueError) as ctx:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        validator.main()
                self.assertIn("Markdown expected gain/loss does not match JSON", str(ctx.exception))
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
