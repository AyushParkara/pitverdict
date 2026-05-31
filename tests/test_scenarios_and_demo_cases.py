from __future__ import annotations

import unittest

from src.data_loader import load_sample_race
from src.models import ScenarioFlags
from src.scenario_engine import apply_scenarios
from src.scenario_presets import list_preset_names, resolve_preset
from tools.demo_cases import run_cases, _assert_cases_differ


class NewPresetsTests(unittest.TestCase):
    def test_new_demo_presets_present_and_resolve(self) -> None:
        names = list_preset_names()
        for expected in ("sensor_failure", "conflicting_signals", "model_wrong", "extreme_conditions"):
            self.assertIn(expected, names)
            self.assertEqual(resolve_preset(expected).name, expected)

    def test_sensor_failure_combines_missing_and_tyre_drift(self) -> None:
        flags = resolve_preset("sensor_failure").flags
        self.assertTrue(flags.missing_telemetry)
        self.assertTrue(flags.tyre_signal_drift)

    def test_extreme_conditions_combines_weather_and_sc(self) -> None:
        flags = resolve_preset("extreme_conditions").flags
        self.assertTrue(flags.weather_uncertainty)
        self.assertTrue(flags.safety_car_phase)


class ScenarioLabelTests(unittest.TestCase):
    def test_applied_labels_are_structured(self) -> None:
        records = load_sample_race()
        res = apply_scenarios(records, ScenarioFlags(missing_telemetry=True, model_mismatch=True))
        self.assertTrue(res.applied)
        names = {a["scenario_name"] for a in res.applied}
        self.assertIn("missing_telemetry", names)
        self.assertIn("model_mismatch", names)
        for entry in res.applied:
            for key in ("scenario_name", "scenario_type", "changed"):
                self.assertIn(key, entry)
                self.assertIsInstance(entry[key], str)

    def test_no_flags_means_no_applied_labels(self) -> None:
        records = load_sample_race()
        res = apply_scenarios(records, ScenarioFlags())
        self.assertEqual(res.applied, [])
        # Backward-compat: notes still present as a list.
        self.assertIsInstance(res.notes, list)


class DemoCasesTests(unittest.TestCase):
    def test_demo_cases_produce_contrast(self) -> None:
        records = load_sample_race()
        rows = run_cases(records)
        # Four named cases.
        self.assertEqual([r["case"] for r in rows], [
            "case_1_normal",
            "case_2_bad_data",
            "case_3_model_mismatch",
            "case_4_extreme",
        ])
        # The curated cases must visibly differ and point the right direction.
        problems = _assert_cases_differ(rows)
        self.assertEqual(problems, [], msg=str(problems))

    def test_extreme_not_more_confident_than_normal(self) -> None:
        records = load_sample_race()
        rows = {r["case"]: r for r in run_cases(records)}
        self.assertLessEqual(
            rows["case_4_extreme"]["confidence"],
            rows["case_1_normal"]["confidence"],
        )


if __name__ == "__main__":
    unittest.main()
