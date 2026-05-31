from __future__ import annotations

import unittest

from tools.mdce_fuzz import edge_case_csvs, run_all, run_one


class FuzzHarnessTests(unittest.TestCase):
    def test_no_hard_failures_across_all_edge_cases(self) -> None:
        """Robustness contract: no edge case may crash or emit invalid output.

        Acceptable outcomes per case: OK (valid result) or REJECTED_CLEANLY
        (controlled ValueError). UNEXPECTED_CRASH / INVALID_OUTPUT are failures.
        """

        report = run_all()
        self.assertTrue(
            report["passed"],
            msg=f"Fuzz hard failures: {report['hard_failures']}",
        )
        # Sanity: we actually exercised a meaningful number of cases.
        self.assertGreaterEqual(report["total_cases"], 10)

    def test_every_outcome_is_an_allowed_value(self) -> None:
        report = run_all()
        allowed = {"OK", "REJECTED_CLEANLY", "UNEXPECTED_CRASH", "INVALID_OUTPUT"}
        for r in report["results"]:
            self.assertIn(r["outcome"], allowed)

    def test_known_bad_inputs_are_rejected_cleanly(self) -> None:
        # Header-only and missing-required-column must be rejected, not crash.
        cases = edge_case_csvs()
        for name in ("header_only", "missing_required_col"):
            out = run_one(name, cases[name])
            self.assertEqual(out["outcome"], "REJECTED_CLEANLY", msg=f"{name} -> {out}")

    def test_minimal_required_cols_runs_ok(self) -> None:
        cases = edge_case_csvs()
        out = run_one("minimal_required_cols", cases["minimal_required_cols"])
        self.assertEqual(out["outcome"], "OK", msg=str(out))


if __name__ == "__main__":
    unittest.main()
