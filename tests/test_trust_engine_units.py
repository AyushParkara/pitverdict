from __future__ import annotations

import unittest

from src.models import LapRecord
from src.trust_engine import conflict_score, detect_trust_issues


def _lr(
    lap: int,
    *,
    lap_time_s: float = 95.0,
    predicted_lap_time_s: float | None = None,
    tyre_temp_proxy_c: float = 95.0,
    speed_consistency: float = 0.98,
    track_status: str = "NORMAL",
    weather: str = "DRY",
    missing: bool = False,
) -> LapRecord:
    return LapRecord(
        lap=lap,
        lap_time_s=float(lap_time_s),
        sector1_s=30.0,
        sector2_s=35.0,
        sector3_s=max(1.0, float(lap_time_s) - 65.0),
        tyre_compound="MEDIUM",
        tyre_age=lap,
        track_status=str(track_status),
        weather=str(weather),
        gap_to_car_ahead_s=1.0,
        predicted_lap_time_s=float(predicted_lap_time_s if predicted_lap_time_s is not None else lap_time_s),
        tyre_temp_proxy_c=float(tyre_temp_proxy_c),
        speed_consistency=float(speed_consistency),
        missing=bool(missing),
    )


class TrustEngineUnitTests(unittest.TestCase):
    def test_no_usable_data(self) -> None:
        issues = detect_trust_issues([_lr(1, missing=True), _lr(2, missing=True)])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue, "no_usable_data")

    def test_missing_telemetry_detected(self) -> None:
        records = [_lr(i + 1) for i in range(6)]
        records[-1] = _lr(6, missing=True)
        issues = detect_trust_issues(records)
        self.assertTrue(any(i.issue == "missing_telemetry" for i in issues))

    def test_model_mismatch_detected_from_trend_gap(self) -> None:
        # Actual trend ~ +0.2s/lap over last 5, model trend ~0.0 => triggers.
        records = [_lr(i + 1, lap_time_s=95.0 + (0.2 * i), predicted_lap_time_s=95.0) for i in range(6)]
        issues = detect_trust_issues(records)
        self.assertTrue(any(i.issue == "model_mismatch" for i in issues))

    def test_optimistic_model_bias_detected_from_mean_gap(self) -> None:
        records = [_lr(i + 1, lap_time_s=95.0, predicted_lap_time_s=93.5) for i in range(6)]
        issues = detect_trust_issues(records)
        self.assertTrue(any(i.issue == "optimistic_model_bias" for i in issues))

    def test_tyre_lap_signal_conflict_detected(self) -> None:
        records = [_lr(i + 1, lap_time_s=95.0 + (0.25 * i), tyre_temp_proxy_c=100.0) for i in range(6)]
        issues = detect_trust_issues(records)
        self.assertTrue(any(i.issue == "tyre_lap_signal_conflict" for i in issues))

    def test_safety_car_context_detected(self) -> None:
        records = [_lr(i + 1) for i in range(6)]
        records[-1] = _lr(6, track_status="SC")
        issues = detect_trust_issues(records)
        self.assertTrue(any(i.issue == "safety_car_context" for i in issues))

    def test_weather_uncertainty_detected(self) -> None:
        records = [_lr(i + 1) for i in range(6)]
        records[-2] = _lr(5, weather="DAMP")
        issues = detect_trust_issues(records)
        self.assertTrue(any(i.issue == "weather_uncertainty" for i in issues))

    def test_low_speed_consistency_detected(self) -> None:
        records = [_lr(i + 1) for i in range(6)]
        records[-1] = _lr(6, speed_consistency=0.80)
        issues = detect_trust_issues(records)
        self.assertTrue(any(i.issue == "low_speed_consistency" for i in issues))


class ConflictScoreUnitTests(unittest.TestCase):
    def test_conflict_score_labels(self) -> None:
        # NONE
        records = [_lr(i + 1, lap_time_s=95.0 + (0.05 * i), predicted_lap_time_s=95.0 + (0.05 * i)) for i in range(5)]
        score, label = conflict_score(records)
        self.assertEqual(label, "NONE")
        self.assertAlmostEqual(score, 0.0, places=2)

        # LOW (>=0.20)
        records = [_lr(i + 1, lap_time_s=95.0 + (0.20 * i), predicted_lap_time_s=95.0 + (0.05 * i)) for i in range(5)]
        score, label = conflict_score(records)
        self.assertEqual(label, "LOW")
        self.assertGreaterEqual(score, 0.20)

        # MEDIUM (>=0.45)
        records = [_lr(i + 1, lap_time_s=95.0 + (0.40 * i), predicted_lap_time_s=95.0 + (0.05 * i)) for i in range(5)]
        score, label = conflict_score(records)
        self.assertEqual(label, "MEDIUM")
        self.assertGreaterEqual(score, 0.45)

        # HIGH (>=0.75)
        records = [_lr(i + 1, lap_time_s=95.0 + (0.55 * i), predicted_lap_time_s=95.0 + (0.05 * i)) for i in range(5)]
        score, label = conflict_score(records)
        self.assertEqual(label, "HIGH")
        self.assertGreaterEqual(score, 0.75)


if __name__ == "__main__":
    unittest.main()
