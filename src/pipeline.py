from __future__ import annotations

from .confidence_engine import CONFIDENCE_SCORING_VERSION, score_confidence
from .coverage_gap_engine import detect_coverage_gaps
from .decision_impact_engine import (
    estimate_aggressive_vs_safe_impact,
    estimate_pit_timing_impact,
    estimate_push_vs_conserve_impact,
    estimate_safety_car_response_impact,
    estimate_stint_length_impact,
    estimate_traffic_rejoin_impact,
    estimate_tyre_compound_impact,
)
from .disagreement_engine import detect_multi_signal_disagreement
from .explanation_engine import explain_decision
from .fallback_engine import fallback_actions
from .mode_engine import build_mode_options
from .model_validation_engine import validate_model_vs_reality
from .models import AnalysisResult, ScenarioFlags, Severity, TrustIssue
from .recommended_mode_engine import recommend_mode
from .scenario_engine import apply_scenarios
from .strategy_engine import make_strategy_recommendation
from .trust_engine import conflict_score, detect_trust_issues
from .uncertainty_engine import propagate_uncertainty


def analyze_decision(records: list, flags: ScenarioFlags, prefer_granite: bool = True) -> tuple[AnalysisResult, list, list[str], tuple[float, str]]:
    scenario = apply_scenarios(records, flags)
    recommendation = make_strategy_recommendation(scenario.records)
    issues = detect_trust_issues(scenario.records)

    # Data coverage gap detection (no loader/provenance dependency).
    # This flags placeholder-like patterns (e.g., derived sector splits, constant gaps).
    coverage_issues = detect_coverage_gaps(scenario.records)
    if coverage_issues:
        existing = {i.issue for i in issues}
        issues = [*issues, *[i for i in coverage_issues if i.issue not in existing]]

    disagreement_issues = detect_multi_signal_disagreement(scenario.records)
    if disagreement_issues:
        existing = {i.issue for i in issues}
        issues = [*issues, *[i for i in disagreement_issues if i.issue not in existing]]

    # Independent model-vs-reality validation (absolute error check).
    model_validation = validate_model_vs_reality(scenario.records)
    if model_validation.status.startswith("Model deviating"):
        already_model_flagged = any(i.issue in {"model_mismatch", "optimistic_model_bias"} for i in issues)
        # Avoid double-penalizing when an upstream trust rule already caught model issues.
        if not already_model_flagged and model_validation.recommended_confidence_penalty > 0.0:
            issues = [
                *issues,
                TrustIssue(
                    issue="model_deviation",
                    severity=Severity.HIGH if model_validation.max_abs_error_s >= 1.25 else Severity.MEDIUM,
                    affected_decisions=["pit_timing", "stint_length", "aggressive_vs_safe_strategy"],
                    reason=(
                        "Model deviation detected: recent predicted lap times diverge from actual pace. "
                        f"Avg error={model_validation.mean_abs_error_s}s, worst error={model_validation.max_abs_error_s}s."
                    ),
                    penalty=model_validation.recommended_confidence_penalty,
                ),
            ]

    confidence = score_confidence(issues)
    fallbacks = fallback_actions(issues)

    # Decision impact simulation (currently pit timing only).
    decision_impact = estimate_pit_timing_impact(
        scenario.records,
        recommendation.recommendation_type,
        horizon_laps=3,
    )

    # Additional impacts for other decision domains (bounded heuristics).
    # pit_timing is the primary `decision_impact`; these cover the remaining trust-map domains.
    decision_impacts = [
        estimate_push_vs_conserve_impact(scenario.records, horizon_laps=3),
        estimate_stint_length_impact(scenario.records, horizon_laps=3),
        estimate_tyre_compound_impact(scenario.records, horizon_laps=3),
        estimate_safety_car_response_impact(scenario.records, horizon_laps=3),
        estimate_traffic_rejoin_impact(scenario.records, horizon_laps=3),
        estimate_aggressive_vs_safe_impact(scenario.records, horizon_laps=3),
    ]

    uncertainty = propagate_uncertainty(issues)

    recommended_mode = recommend_mode(issues, uncertainty)

    mode_options = build_mode_options(
        recommendation,
        impact_fn=lambda rec_type: estimate_pit_timing_impact(scenario.records, rec_type, horizon_laps=3),
    )
    explanation = explain_decision(
        recommendation=recommendation,
        confidence=confidence,
        issues=issues,
        fallbacks=fallbacks,
        uncertainty=uncertainty,
        recommended_mode=recommended_mode.value if recommended_mode is not None else None,
        decision_impact=decision_impact,
        decision_impacts=decision_impacts,
        prefer_granite=prefer_granite,
    )
    result = AnalysisResult(
        recommendation=recommendation,
        confidence=confidence,
        issues=issues,
        fallback_actions=fallbacks,
        explanation=explanation,
        model_validation=model_validation,
        decision_impact=decision_impact,
        decision_impacts=decision_impacts,
        uncertainty=uncertainty,
        mode_options=mode_options,
        recommended_mode=recommended_mode,
    )
    # Expose scoring version via the extra return data so CLIs can persist it.
    # (We keep AnalysisResult unchanged to avoid breaking app/tests.)
    extra_notes = [
        f"confidence_scoring_version={CONFIDENCE_SCORING_VERSION}",
        f"model_validation_status={model_validation.status}",
        f"model_validation_mae_s={model_validation.mean_abs_error_s}",
        f"uncertainty_primary={uncertainty.primary_uncertainty}",
        f"uncertainty_score={uncertainty.uncertainty_score}",
        f"recommended_mode={recommended_mode.value}",
    ]
    return result, scenario.records, scenario.notes + extra_notes, conflict_score(scenario.records)
