from __future__ import annotations

import unittest

from src.data_loader import load_sample_race
from src.models import ScenarioFlags
from src.pipeline import analyze_decision
from src.scenario_presets import list_preset_names, resolve_preset


class PresetAndPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_sample_race()

    def test_all_presets_resolve(self) -> None:
        names = list_preset_names()
        self.assertTrue(names)
        for name in names:
            preset = resolve_preset(name)
            self.assertEqual(preset.name, name)

    def test_each_preset_runs_end_to_end(self) -> None:
        # Smoke test: every named preset should run the pipeline and return a stable shape.
        for name in list_preset_names():
            preset = resolve_preset(name)
            result, _, notes, conflict = analyze_decision(self.records, preset.flags, prefer_granite=False)
            self.assertTrue(result.explanation)

    def test_preset_expected_issue_keys_present(self) -> None:
        """Each preset should reliably induce its corresponding trust issue.

        This protects the demo: selecting a preset must visibly trigger the advertised guardrail.
        """

        from src.data_loader import load_sample_race
        from src.pipeline import analyze_decision
        from src.models import ScenarioFlags

        records = load_sample_race()
        cases = [
            (ScenarioFlags(missing_telemetry=True), "missing_telemetry"),
            (ScenarioFlags(model_mismatch=True), "optimistic_model_bias"),
            (ScenarioFlags(safety_car_phase=True), "safety_car_context"),
            (ScenarioFlags(weather_uncertainty=True), "weather_uncertainty"),
        ]
        for flags, expected in cases:
            with self.subTest(expected=expected):
                result, _, notes, conflict = analyze_decision(records, flags, prefer_granite=False)
                keys = {i.issue for i in result.issues}
                self.assertIn(expected, keys)
                self.assertIsInstance(notes, list)
                self.assertIsInstance(conflict, tuple)
                self.assertGreaterEqual(result.confidence.confidence, 0.0)
                self.assertLessEqual(result.confidence.confidence, 1.0)

    def test_scenario_flags_generally_reduce_confidence_vs_clean(self) -> None:
        # Invariant: turning on a scenario flag should not increase confidence.
        base, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)

        checks = {
            "missing_telemetry": ScenarioFlags(missing_telemetry=True),
            "tyre_signal_drift": ScenarioFlags(tyre_signal_drift=True),
            "model_mismatch": ScenarioFlags(model_mismatch=True),
            "safety_car_phase": ScenarioFlags(safety_car_phase=True),
            "weather_uncertainty": ScenarioFlags(weather_uncertainty=True),
        }
        for name, flags in checks.items():
            res, _, _, _ = analyze_decision(self.records, flags, prefer_granite=False)
            self.assertLessEqual(
                res.confidence.confidence,
                base.confidence.confidence,
                f"Expected {name} to not increase confidence",
            )

        # Strong case: the stack should decrease confidence substantially.
        stacked, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(missing_telemetry=True, model_mismatch=True, safety_car_phase=True, weather_uncertainty=True),
            prefer_granite=False,
        )
        self.assertLess(stacked.confidence.confidence, base.confidence.confidence)


if __name__ == "__main__":
    unittest.main()
