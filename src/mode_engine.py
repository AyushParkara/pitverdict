from __future__ import annotations

from dataclasses import replace

from .models import DecisionMode, ModeOption, RecommendationType


def apply_mode_to_recommendation(option_type: RecommendationType, mode: DecisionMode) -> RecommendationType:
    """Adjust a baseline recommendation based on decision mode.

    SAFE: prefer earlier, lower-regret actions.
    AGGRESSIVE: allow extending/pushing when reasonable.
    """

    if mode == DecisionMode.SAFE:
        if option_type == RecommendationType.PIT_SOON:
            return RecommendationType.PIT_NOW
        return option_type

    # AGGRESSIVE
    if option_type == RecommendationType.PIT_NOW:
        return RecommendationType.PIT_SOON
    return option_type


def build_mode_options(
    recommendation,
    impact_fn,
) -> list[ModeOption]:
    """Return SAFE and AGGRESSIVE options.

    impact_fn(recommendation_type)->DecisionImpactResult
    """

    options: list[ModeOption] = []
    for mode in (DecisionMode.SAFE, DecisionMode.AGGRESSIVE):
        rec_type = apply_mode_to_recommendation(recommendation.recommendation_type, mode)
        if rec_type == recommendation.recommendation_type:
            rec = recommendation
        else:
            rec = replace(
                recommendation,
                recommendation_type=rec_type,
                base_reason=f"Mode-adjusted ({mode.value}): {recommendation.base_reason}",
            )
        impact = impact_fn(rec.recommendation_type)
        options.append(ModeOption(mode=mode, recommendation=rec, decision_impact=impact))
    return options
