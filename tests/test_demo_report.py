from __future__ import annotations

import unittest

from src.data_loader import load_sample_race
from tools.mdce_demo_report import build_markdown


class DemoReportTests(unittest.TestCase):
    def test_report_contains_all_core_sections(self) -> None:
        records = load_sample_race()
        md = build_markdown(
            records,
            source_name="unit-test",
            dataset_metadata={"license_spdx": "CC-BY-4.0", "source_url": "https://example.invalid"},
            warnings=["Real column used: lap", "tyre_age derived from row order because it was not supplied."],
        )
        for section in (
            "# MDCE — Decision Confidence Report",
            "## 1) What MDCE Is",
            "## 2) Decision Run",
            "## 3) Evidence (does confidence behave sensibly?)",
            "## 4) Robustness (fuzz harness)",
            "## 5) Provenance & Safe Claims",
        ):
            self.assertIn(section, md, msg=f"Missing section: {section}")

    def test_report_filters_noise_warnings_but_keeps_real_notes(self) -> None:
        records = load_sample_race()
        md = build_markdown(
            records,
            source_name="unit-test",
            dataset_metadata={},
            warnings=[
                "Real column used: lap",
                "Derived column: weather",
                "Proxy column: tyre_temp_proxy_c",
                "tyre_age derived from row order because it was not supplied.",
            ],
        )
        # Provenance-noise prefixes should be filtered out of the report body notes.
        self.assertNotIn("Real column used: lap", md)
        # A genuine note should survive.
        self.assertIn("tyre_age derived from row order", md)

    def test_report_is_deterministic_modulo_timestamp(self) -> None:
        records = load_sample_race()
        md1 = build_markdown(records, source_name="x", dataset_metadata={}, warnings=[])
        md2 = build_markdown(records, source_name="x", dataset_metadata={}, warnings=[])

        def _strip_ts(text: str) -> str:
            return "\n".join(line for line in text.splitlines() if not line.startswith("_Generated "))

        self.assertEqual(_strip_ts(md1), _strip_ts(md2))


if __name__ == "__main__":
    unittest.main()
