from __future__ import annotations

from .data_loader import available_records, lap_time_trend
from .models import RecommendationType, StrategyRecommendation


def make_strategy_recommendation(records: list) -> StrategyRecommendation:
    """Create a deliberately simple baseline strategy recommendation."""
    usable = available_records(records)
    if not usable:
        return StrategyRecommendation(
            recommendation_type=RecommendationType.EXTEND,
            recommended_lap=0,
            expected_gain_loss_s=0.0,
            base_reason="No usable lap data is available.",
        )

    latest = usable[-1]
    degradation = lap_time_trend(usable, count=5)
    severe_deg = degradation >= 0.45
    rising_deg = degradation >= 0.20
    high_tyre_age = latest.tyre_age >= 18
    medium_tyre_age = latest.tyre_age >= 14
    safety_car_opportunity = latest.track_status in {"SC", "VSC"} and latest.tyre_age >= 12

    if severe_deg or high_tyre_age or safety_car_opportunity:
        recommendation_type = RecommendationType.PIT_NOW
        recommended_lap = latest.lap
        reason = "Tyre age and recent lap-time degradation make extending the stint risky."
    elif rising_deg or medium_tyre_age:
        recommendation_type = RecommendationType.PIT_SOON
        recommended_lap = latest.lap + 3
        reason = "Degradation is rising, but the decision can wait for a short confirmation window."
    else:
        recommendation_type = RecommendationType.EXTEND
        recommended_lap = latest.lap + 5
        reason = "Recent pace is stable enough to continue the stint."

    expected_gain_loss_s = round(max(0.0, degradation) * 5.0, 2)
    return StrategyRecommendation(
        recommendation_type=recommendation_type,
        recommended_lap=recommended_lap,
        expected_gain_loss_s=expected_gain_loss_s,
        base_reason=reason,
    )
