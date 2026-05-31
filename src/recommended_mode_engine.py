from __future__ import annotations

from .models import DecisionMode, TrustIssue, UncertaintyResult


def recommend_mode(issues: list[TrustIssue], uncertainty: UncertaintyResult | None) -> DecisionMode:
    """Select SAFE vs AGGRESSIVE deterministically.

    Goal: reliability under uncertainty.

    - SAFE when uncertainty is high or there are high-severity trust issues.
    - AGGRESSIVE only when evidence is internally consistent.
    """

    if any(i.severity.value == "high" for i in issues):
        return DecisionMode.SAFE

    total_penalty = sum(i.penalty for i in issues)
    if total_penalty >= 0.35:
        return DecisionMode.SAFE

    if uncertainty is not None and uncertainty.uncertainty_score >= 0.55:
        return DecisionMode.SAFE

    return DecisionMode.AGGRESSIVE
