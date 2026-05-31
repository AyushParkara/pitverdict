from __future__ import annotations

from .models import TrustIssue, UncertaintyResult


def propagate_uncertainty(issues: list[TrustIssue]) -> UncertaintyResult:
    """Summarize primary uncertainty and downstream impact.

    Deterministic mapping from trust issues -> uncertainty drivers.
    This is not probabilistic forecasting; it is a structured explanation layer.
    """

    if not issues:
        return UncertaintyResult(
            primary_uncertainty="none",
            uncertainty_score=0.0,
            downstream_decisions_at_risk=[],
            drivers=[],
        )

    # Identify the highest-penalty driver as primary.
    primary = max(issues, key=lambda i: i.penalty)
    primary_uncertainty = primary.issue

    # Score: scaled fraction of total penalty mass, capped.
    total_penalty = sum(i.penalty for i in issues)
    uncertainty_score = round(min(1.0, total_penalty / 0.6), 2)

    downstream: set[str] = set()
    for i in issues:
        downstream.update(i.affected_decisions)
    downstream.discard("all_strategy_decisions")

    drivers = [
        f"{i.issue} (penalty={i.penalty})"
        for i in sorted(issues, key=lambda x: x.penalty, reverse=True)
    ]

    return UncertaintyResult(
        primary_uncertainty=primary_uncertainty,
        uncertainty_score=uncertainty_score,
        downstream_decisions_at_risk=sorted(downstream),
        drivers=drivers,
    )
