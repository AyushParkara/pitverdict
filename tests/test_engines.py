from __future__ import annotations

import unittest
from io import StringIO

from src.data_loader import load_race_csv, load_sample_race
from src.dataset_adapters import prepare_kaggle_weather_tyre
from src.models import ScenarioFlags
from src.pipeline import analyze_decision


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_sample_race()

    def test_normal_data_has_usable_confidence(self) -> None:
        result, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        self.assertGreaterEqual(result.confidence.confidence, 0.55)
        self.assertIn("data_completeness", result.confidence.breakdown)
        self.assertIn("pit_timing", result.confidence.decision_confidence)
        self.assertTrue(result.explanation)

    def test_missing_telemetry_reduces_confidence(self) -> None:
        normal, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        missing, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(missing_telemetry=True),
            prefer_granite=False,
        )
        self.assertLess(missing.confidence.confidence, normal.confidence.confidence)
        self.assertTrue(any(issue.issue == "missing_telemetry" for issue in missing.issues))

    def test_high_uncertainty_preset_stack_drives_safe_mode(self) -> None:
        # Mirror the scenario preset behavior (stack multiple failure modes).
        result, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(missing_telemetry=True, model_mismatch=True, safety_car_phase=True, weather_uncertainty=True),
            prefer_granite=False,
        )
        self.assertIsNotNone(result.recommended_mode)
        self.assertEqual(result.recommended_mode.value, "SAFE")

    def test_confidence_monotonic_with_more_issues(self) -> None:
        # Turning on more scenario flags should not increase confidence.
        base, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        one, _, _, _ = analyze_decision(self.records, ScenarioFlags(model_mismatch=True), prefer_granite=False)
        many, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(model_mismatch=True, missing_telemetry=True, safety_car_phase=True, weather_uncertainty=True),
            prefer_granite=False,
        )
        self.assertLessEqual(one.confidence.confidence, base.confidence.confidence)
        self.assertLessEqual(many.confidence.confidence, one.confidence.confidence)

        # Decision-domain confidence should also remain monotonic.
        self.assertLessEqual(one.confidence.decision_confidence["pit_timing"], base.confidence.decision_confidence["pit_timing"])
        self.assertLessEqual(many.confidence.decision_confidence["pit_timing"], one.confidence.decision_confidence["pit_timing"])

    def test_model_mismatch_detected(self) -> None:
        result, _, _, conflict = analyze_decision(
            self.records,
            ScenarioFlags(model_mismatch=True),
            prefer_granite=False,
        )
        # The "model mismatch" scenario must be detected via model-vs-reality validation.
        # (We do not require the trend-based trust rule to trigger in every dataset.)
        self.assertIsNotNone(result.model_validation)
        self.assertEqual(result.model_validation.status, "DEVIATION")
        self.assertGreaterEqual(result.model_validation.mean_abs_error_s, 0.75)
        self.assertGreaterEqual(conflict[0], 0.0)

    def test_model_validation_attached(self) -> None:
        result, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        self.assertIsNotNone(result.model_validation)
        self.assertIn(result.model_validation.status, {"OK", "DEVIATION", "NO_DATA"})

    def test_decision_impact_attached(self) -> None:
        result, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        self.assertIsNotNone(result.decision_impact)
        self.assertEqual(result.decision_impact.decision, "pit_timing")

    def test_additional_decision_impacts_attached(self) -> None:
        result, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        self.assertTrue(result.decision_impacts)
        decisions = {d.decision for d in result.decision_impacts}
        self.assertIn("push_vs_conserve", decisions)

    def test_uncertainty_attached(self) -> None:
        result, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        self.assertIsNotNone(result.uncertainty)
        self.assertGreaterEqual(result.uncertainty.uncertainty_score, 0.0)

    def test_mode_options_attached(self) -> None:
        result, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        self.assertTrue(result.mode_options)
        modes = {o.mode.value for o in result.mode_options}
        self.assertEqual(modes, {"SAFE", "AGGRESSIVE"})

    def test_recommended_mode_safe_under_high_uncertainty(self) -> None:
        result, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(model_mismatch=True, missing_telemetry=True, safety_car_phase=True, weather_uncertainty=True),
            prefer_granite=False,
        )
        self.assertIsNotNone(result.recommended_mode)
        self.assertEqual(result.recommended_mode.value, "SAFE")

    def test_recommended_mode_aggressive_when_clean(self) -> None:
        # Use a tiny clean CSV so the test is not dependent on demo dataset quirks.
        from io import StringIO

        from src.data_loader import load_race_csv

        csv_data = StringIO(
            "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency\n"
            "1,95.0,30.2,35.0,29.8,MEDIUM,1,NORMAL,DRY,1.2,95.0,98.0,0.98\n"
            "2,95.1,30.3,35.0,29.8,MEDIUM,2,NORMAL,DRY,1.1,95.1,98.1,0.98\n"
            "3,95.0,30.2,34.9,29.9,MEDIUM,3,NORMAL,DRY,1.0,95.0,98.2,0.99\n"
            "4,95.1,30.3,35.0,29.8,MEDIUM,4,NORMAL,DRY,0.9,95.1,98.3,0.99\n"
            "5,95.0,30.2,34.9,29.9,MEDIUM,5,NORMAL,DRY,0.8,95.0,98.4,0.99\n"
        )
        loaded = load_race_csv(csv_data, source_name="clean csv")
        result, _, _, _ = analyze_decision(loaded.records, ScenarioFlags(), prefer_granite=False)
        self.assertIsNotNone(result.recommended_mode)
        self.assertEqual(result.recommended_mode.value, "AGGRESSIVE")

    def test_coverage_gap_detection_sector_placeholders(self) -> None:
        # sample_race.csv uses fixed proportional sector splits -> should be flagged.
        result, _, _, _ = analyze_decision(self.records, ScenarioFlags(), prefer_granite=False)
        self.assertTrue(any(i.issue == "coverage_gap_sector_times" for i in result.issues))

    def test_multi_signal_disagreement_detected_from_constructed_case(self) -> None:
        # Construct a case where actual pace degrades but model trend is flat and tyre proxy is flat.
        from io import StringIO

        from src.data_loader import load_race_csv

        csv_data = StringIO(
            "lap,lap_time_s,sector1_s,sector2_s,sector3_s,tyre_compound,tyre_age,track_status,weather,gap_to_car_ahead_s,predicted_lap_time_s,tyre_temp_proxy_c,speed_consistency\n"
            "1,95.00,30.0,35.0,30.0,MEDIUM,1,NORMAL,DRY,1.0,95.00,100.0,0.98\n"
            "2,95.20,30.1,35.1,30.0,MEDIUM,2,NORMAL,DRY,1.0,95.00,100.0,0.98\n"
            "3,95.45,30.2,35.2,30.1,MEDIUM,3,NORMAL,DRY,1.0,95.00,100.0,0.98\n"
            "4,95.75,30.3,35.3,30.2,MEDIUM,4,NORMAL,DRY,1.0,95.00,100.0,0.98\n"
            "5,96.10,30.4,35.4,30.3,MEDIUM,5,NORMAL,DRY,1.0,95.00,100.0,0.98\n"
            "6,96.50,30.5,35.5,30.5,MEDIUM,6,NORMAL,DRY,1.0,95.00,100.0,0.98\n"
        )
        loaded = load_race_csv(csv_data, source_name="disagreement csv")
        result, _, _, _ = analyze_decision(loaded.records, ScenarioFlags(), prefer_granite=False)
        self.assertTrue(any(i.issue == "multi_signal_disagreement" for i in result.issues))

    def test_high_uncertainty_adds_safe_mode_fallback(self) -> None:
        # Stack multiple issues via scenario flags; should add SAFE-mode nudge.
        result, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(model_mismatch=True, missing_telemetry=True, safety_car_phase=True, weather_uncertainty=True),
            prefer_granite=False,
        )
        self.assertTrue(any("SAFE mode" in a for a in result.fallback_actions))

    def test_tyre_signal_drift_detected(self) -> None:
        result, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(tyre_signal_drift=True),
            prefer_granite=False,
        )
        self.assertTrue(any(issue.issue == "tyre_lap_signal_conflict" for issue in result.issues))
        self.assertTrue(any("lap-time degradation" in action for action in result.fallback_actions))

    def test_safety_car_context_detected(self) -> None:
        result, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(safety_car_phase=True),
            prefer_granite=False,
        )
        self.assertTrue(any(issue.issue == "safety_car_context" for issue in result.issues))

    def test_real_csv_loader_derives_missing_optional_columns(self) -> None:
        csv_data = StringIO(
            "lap,lap_time_s,tyre_compound,tyre_age\n"
            "1,95.1,MEDIUM,1\n"
            "2,95.4,MEDIUM,2\n"
            "3,95.9,MEDIUM,3\n"
        )
        loaded = load_race_csv(csv_data, source_name="unit-test csv")
        self.assertEqual(len(loaded.records), 3)
        self.assertIsInstance(loaded.dataset_metadata, dict)
        self.assertIn("lap_time_s", loaded.real_columns)
        self.assertIn("predicted_lap_time_s", loaded.derived_columns)
        self.assertIn("tyre_temp_proxy_c", loaded.proxy_columns)
        self.assertTrue(any("derived from prior rolling" in warning for warning in loaded.warnings))

    def test_real_csv_loader_derives_weather_from_rainfall(self) -> None:
        csv_data = StringIO(
            "lap,lap_time_s,tyre_compound,tyre_age,rainfall,track_temp_c\n"
            "1,95.1,MEDIUM,1,0.0,31.0\n"
            "2,96.4,MEDIUM,2,1.2,30.5\n"
            "3,96.9,MEDIUM,3,0.0,30.0\n"
        )
        loaded = load_race_csv(csv_data, source_name="rain csv")
        self.assertEqual(loaded.records[0].weather, "DRY")
        self.assertEqual(loaded.records[1].weather, "WET")
        self.assertIn("rainfall", loaded.real_columns)
        self.assertIn("weather", loaded.derived_columns)

    def test_kaggle_weather_tyre_adapter_maps_real_columns(self) -> None:
        parquet_path = "data/raw/extracted/kaggle_naven_weather_tyre/f1_all.parquet"
        try:
            prepared = prepare_kaggle_weather_tyre(parquet_path, year=2023, round_number=22, driver_code="VER")
        except FileNotFoundError:
            self.skipTest("Downloaded Kaggle weather/tyre parquet is not available.")

        self.assertGreaterEqual(len(prepared.frame), 10)
        self.assertIn("lap_time_s", prepared.frame.columns)
        self.assertIn("track_temp_c", prepared.frame.columns)
        self.assertEqual(prepared.metadata["selected_driver_code"], "VER")
        self.assertEqual(prepared.metadata["selected_decision_lap"], 16)

    def test_real_csv_loader_rejects_missing_required_columns(self) -> None:
        csv_data = StringIO("lap,tyre_age\n1,1\n")
        with self.assertRaises(ValueError):
            load_race_csv(csv_data, source_name="bad csv")

    def test_every_trust_issue_has_a_fallback_action(self) -> None:
        """Guardrail: if we add a new TrustIssue key, we must define a fallback.

        The demo promise is: every warning/issue should have a concrete fallback.
        """

        from src.fallback_engine import FALLBACKS
        from src.pipeline import analyze_decision

        # Use a high-issue stack to maximize unique issue coverage.
        result, _, _, _ = analyze_decision(
            self.records,
            ScenarioFlags(
                missing_telemetry=True,
                tyre_signal_drift=True,
                model_mismatch=True,
                safety_car_phase=True,
                weather_uncertainty=True,
            ),
            prefer_granite=False,
        )
        issue_keys = {i.issue for i in result.issues}
        missing = sorted(k for k in issue_keys if k not in FALLBACKS)
        self.assertFalse(missing, f"Missing fallback mappings for issues: {missing}")


if __name__ == "__main__":
    unittest.main()
