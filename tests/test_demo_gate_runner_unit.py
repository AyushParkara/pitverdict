from __future__ import annotations

import unittest
from unittest.mock import patch


class DemoGateRunnerUnitTests(unittest.TestCase):
    def test_demo_gate_passes_explicit_json_to_validator(self) -> None:
        """Avoid "latest json" races by passing --json to validate_mdce_outputs."""

        import sys

        import tools.demo_gate as gate

        calls: list[list[str]] = []

        def _fake_run(cmd, text=True, capture_output=False, env=None):
            calls.append(list(cmd))

            class R:
                def __init__(self, stdout: str, stderr: str, returncode: int):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode

            # First call is the analysis run; provide a JSON line.
            if "run_mdce_decision.py" in " ".join(cmd):
                return R(
                    "MDCE decision run complete\nJSON: out.json\nMD: out.md\n",
                    "",
                    0,
                )

            # Subsequent calls are validators.
            return R("OK\n", "", 0)

        old_argv = sys.argv
        try:
            sys.argv = ["demo_gate.py", "--root", "/tmp", "--output-dir", "outputs/reports"]
            # demo_gate imports subprocess inside helper functions, so patch the
            # stdlib entrypoint directly.
            with patch("subprocess.run", side_effect=_fake_run):
                gate.main()
        finally:
            sys.argv = old_argv

        validate_calls = [c for c in calls if "validate_mdce_outputs.py" in " ".join(c)]
        self.assertTrue(validate_calls)
        flat = " ".join(validate_calls[-1])
        self.assertIn("--json", flat)
        self.assertIn("out.json", flat)


if __name__ == "__main__":
    unittest.main()
