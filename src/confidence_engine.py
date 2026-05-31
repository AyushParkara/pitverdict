from __future__ import annotations

from math import prod

from .models import ConfidenceResult, TrustIssue


CONFIDENCE_SCORING_VERSION = "confidence_v2_breakdown_and_per_decision"


def risk_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "Low"
    if confidence >= 0.55:
        return "Medium"
    if confidence >= 0.35:
        return "Medium-High"
    return "High"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _round2(x: float) -> float:
    return float(round(x, 2))


def _issue_present(issues: list[TrustIssue], name: str) -> bool:
    return any(i.issue == name for i in issues)


def _sum_penalties_for_decision(issues: list[TrustIssue], decision: str) -> float:
    return sum(i.penalty for i in issues if decision in i.affected_decisions or "all_strategy_decisions" in i.affected_decisions)


def score_confidence_components(issues: list[TrustIssue]) -> dict[str, float]:
    """Return deterministic component scores in [0,1].

    These are not ML-estimated probabilities. They are transparent engineering signals
    that explain how the overall confidence was formed.
    """

    # Components are derived from the existing TrustIssue penalties, grouped into
    # interpretable categories. This keeps the breakdown aligned with the actual
    # confidence computation (not a separate set of "made up" weights).

    total_penalty = sum(issue.penalty for issue in issues)

    completeness_issues = {"no_usable_data", "missing_telemetry"}
    alignment_issues = {"model_mismatch", "optimistic_model_bias", "model_deviation"}
    agreement_issues = {"tyre_lap_signal_conflict", "low_speed_consistency", "multi_signal_disagreement"}
    context_issues = {"safety_car_context", "weather_uncertainty"}

    # Coverage gaps reduce confidence because they represent missing/non-real inputs.
    # We treat them as part of completeness.
    coverage_issues = {"coverage_gap_sector_times", "coverage_gap_track_gaps", "coverage_gap_tyre_compound"}

    completeness_penalty = sum(i.penalty for i in issues if i.issue in completeness_issues | coverage_issues)
    alignment_penalty = sum(i.penalty for i in issues if i.issue in alignment_issues)
    agreement_penalty = sum(i.penalty for i in issues if i.issue in agreement_issues)
    context_penalty = sum(i.penalty for i in issues if i.issue in context_issues)

    data_completeness = _round2(_clamp01(1.0 - completeness_penalty))
    model_alignment = _round2(_clamp01(1.0 - alignment_penalty))
    signal_agreement = _round2(_clamp01(1.0 - agreement_penalty))
    context_stability = _round2(_clamp01(1.0 - context_penalty))

    # Penalized score as a sanity cross-check for the overall penalty mass.
    penalty_score = _round2(_clamp01(1.0 - min(1.0, total_penalty)))

    return {
        "data_completeness": data_completeness,
        "signal_agreement": signal_agreement,
        "model_alignment": model_alignment,
        "context_stability": context_stability,
        "penalty_score": penalty_score,
    }


def score_decision_confidence(issues: list[TrustIssue], base_confidence: float = 0.85) -> dict[str, float]:
    """Compute decision-domain confidence scores.

    Uses TrustIssue.affected_decisions to scope penalties. This makes the system
    behave like a real trust layer: some issues should only reduce certain decisions.
    """

    domains = [
        "pit_timing",
        "tyre_strategy",
        "stint_length",
        "push_vs_conserve",
        "safety_car_response",
        "traffic_rejoin_risk",
        "aggressive_vs_safe_strategy",
    ]
    out: dict[str, float] = {}
    for d in domains:
        penalty = _sum_penalties_for_decision(issues, d)
        # Use the same semantics as overall confidence: penalties reduce a bounded score.
        penalty_score = _clamp01(1.0 - min(1.0, penalty))
        out[d] = _round2(max(0.10, min(0.95, base_confidence * penalty_score)))
    return out


def score_confidence(issues: list[TrustIssue], base_confidence: float = 0.85) -> ConfidenceResult:
    breakdown = score_confidence_components(issues)

    # Confidence computation: deterministic and interpretable.
    # - Component scores explain *where* trust is lost.
    # - `penalty_score` ensures any unclassified issues still reduce confidence.
    component_factor = 1.0
    core_components = [
        breakdown.get("data_completeness", 1.0),
        breakdown.get("signal_agreement", 1.0),
        breakdown.get("model_alignment", 1.0),
        breakdown.get("context_stability", 1.0),
    ]
    if core_components:
        # Geometric mean keeps a single weak component from hiding behind strong ones.
        component_factor = float(prod(core_components) ** (1.0 / len(core_components)))

    penalty_factor = float(breakdown.get("penalty_score", 1.0))
    confidence = base_confidence * min(component_factor, penalty_factor)
    confidence = _round2(max(0.10, min(0.95, confidence)))
    decision_conf = score_decision_confidence(issues, base_confidence=base_confidence)
    decision_risk = {k: risk_level(v) for k, v in decision_conf.items()}
    return ConfidenceResult(
        confidence=confidence,
        risk_level=risk_level(confidence),
        penalties=issues,
        breakdown=breakdown,
        decision_confidence=decision_conf,
        decision_risk_levels=decision_risk,
    )
