from __future__ import annotations

import unittest

from src.models import LapRecord, ScenarioFlags
from src.pipeline import analyze_decision


def _lr(
    lap: int,
    *,
    lap_time_s: float,
    predicted_lap_time_s: float,
    sector1_s: float,
    sector2_s: float,
    speed_consistency: float = 0.98,
) -> LapRecord:
    return LapRecord(
        lap=lap,
        lap_time_s=float(lap_time_s),
        sector1_s=float(sector1_s),
        sector2_s=float(sector2_s),
        sector3_s=max(1.0, float(lap_time_s) - float(sector1_s) - float(sector2_s)),
        tyre_compound="MEDIUM",
        tyre_age=lap,
        track_status="NORMAL",
        weather="DRY",
        gap_to_car_ahead_s=1.0,
        predicted_lap_time_s=float(predicted_lap_time_s),
        tyre_temp_proxy_c=95.0,
        speed_consistency=float(speed_consistency),
        missing=False,
    )


class PipelineModelDeviationGateTests(unittest.TestCase):
    def test_adds_model_deviation_when_validation_deviates_and_no_other_model_flags(self) -> None:
        # Construct a case where abs prediction error is high enough to trigger
        # model validation, but mean/trend checks in trust_engine do not.
        #
        # - actual lap_time_s is stable
        # - predicted alternates +/- 0.8s around actual => MAE=0.8 >= 0.75
        records: list[LapRecord] = []
        for i in range(1, 7):
            pred = 95.8 if i % 2 == 0 else 94.2
            # Sector values intentionally vary enough to not trigger "proportional placeholder" coverage warnings.
            s1 = 28.0 + (0.6 * i)
            s2 = 34.0 + (0.4 * i)
            records.append(_lr(i, lap_time_s=95.0, predicted_lap_time_s=pred, sector1_s=s1, sector2_s=s2))

        result, _, _, _ = analyze_decision(records, ScenarioFlags(), prefer_granite=False)
        keys = {i.issue for i in result.issues}
        self.assertIn("model_deviation", keys)

        md = next(i for i in result.issues if i.issue == "model_deviation")
        self.assertEqual(md.severity.value, "medium")
        # Penalty is derived from model_validation; assert the pipeline wires it through.
        self.assertIsNotNone(result.model_validation)
        self.assertAlmostEqual(md.penalty, result.model_validation.recommended_confidence_penalty, places=2)
        # With MAE=0.8 and threshold=0.75, penalty=round(0.05 + (0.05/2), 2)=0.07 (banker's rounding).
        self.assertAlmostEqual(md.penalty, 0.07, places=2)

    def test_does_not_add_model_deviation_when_trust_engine_already_flags_model_bias(self) -> None:
        # Here we make the model consistently optimistic, which should trigger
        # optimistic_model_bias in trust_engine. Even though model validation also
        # deviates, pipeline should avoid double-penalizing with model_deviation.
        records: list[LapRecord] = []
        for i in range(1, 7):
            # Mean(actual - predicted) = 2.0s => optimistic bias.
            s1 = 28.0 + (0.6 * i)
            s2 = 34.0 + (0.4 * i)
            records.append(_lr(i, lap_time_s=95.0, predicted_lap_time_s=93.0, sector1_s=s1, sector2_s=s2))

        result, _, _, _ = analyze_decision(records, ScenarioFlags(), prefer_granite=False)
        keys = {i.issue for i in result.issues}
        self.assertIn("optimistic_model_bias", keys)
        self.assertNotIn("model_deviation", keys)
        self.assertIsNotNone(result.model_validation)
        self.assertEqual(result.model_validation.status, "DEVIATION")


if __name__ == "__main__":
    unittest.main()
