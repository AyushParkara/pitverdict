from __future__ import annotations

import os
from typing import Iterable

import requests

from .models import ConfidenceResult, DecisionImpactResult, StrategyRecommendation, TrustIssue, UncertaintyResult


def _format_issues(issues: Iterable[TrustIssue]) -> str:
    issue_lines = [f"- {issue.issue}: {issue.reason}" for issue in issues]
    return "\n".join(issue_lines) if issue_lines else "- No major trust issues detected."


def _format_top_drivers(issues: list[TrustIssue], *, top_n: int = 3) -> str:
    """Return the top trust-issue drivers, ranked by penalty then severity.

    The decision-intelligence narrative should lead with *why* confidence moved,
    not dump every issue. We surface the highest-impact drivers only.
    """

    if not issues:
        return "- No major trust issues detected."

    severity_weight = {"high": 3, "medium": 2, "low": 1}
    ordered = sorted(
        issues,
        key=lambda i: (-float(i.penalty), -severity_weight.get(i.severity.value, 0), i.issue),
    )
    top = ordered[:top_n]
    return "\n".join(f"- {i.issue}: {i.reason}" for i in top)


def mock_explanation(
    recommendation: StrategyRecommendation,
    confidence: ConfidenceResult,
    issues: list[TrustIssue],
    fallbacks: list[str],
    *,
    uncertainty: UncertaintyResult | None = None,
    recommended_mode: str | None = None,
    decision_impact: DecisionImpactResult | None = None,
    decision_impacts: list[DecisionImpactResult] | None = None,
) -> str:
    """Plain-English explanation. No machine codes, no internal labels."""
    rec_label = recommendation.recommendation_type.value.replace("_", " ").title()

    # --- Why (only reasons, not internal codes) ---
    if issues:
        severity_weight = {"high": 3, "medium": 2, "low": 1}
        top = sorted(issues, key=lambda i: (-float(i.penalty), -severity_weight.get(i.severity.value, 0)))
        top_reasons = [i.reason for i in top[:3]]
        why_block = "Why:\n" + "\n".join(f"- {r}" for r in top_reasons)
    else:
        why_block = "Why: No major concerns."

    # --- Uncertainty ---
    if uncertainty is None or uncertainty.uncertainty_score == 0:
        uncertainty_block = "Uncertainty is low."
    else:
        pct = int(uncertainty.uncertainty_score * 100)
        driver = uncertainty.primary_uncertainty.replace("_", " ")
        downstream = uncertainty.downstream_decisions_at_risk or []
        suffix = ""
        if downstream:
            clean = [d.replace("_", " ") for d in downstream]
            suffix = f" Decisions most affected: {', '.join(clean)}."
        uncertainty_block = f"Uncertainty is {pct}%. Main source: {driver}.{suffix}"

    # --- Impact table as readable lines ---
    all_impacts: list[tuple[str, float, str]] = []
    if decision_impact is not None:
        all_impacts.append(("Pit timing", decision_impact.if_wrong_expected_loss_s, decision_impact.risk_level))
    for di in decision_impacts or []:
        label = di.decision.replace("_", " ").title()
        all_impacts.append((label, di.if_wrong_expected_loss_s, di.risk_level))
    if all_impacts:
        impact_lines = [f"- {lbl}: {loss}s ({risk})" for lbl, loss, risk in all_impacts]
        impacts_block = "Estimated cost if the call is wrong:\n" + "\n".join(impact_lines)
    else:
        impacts_block = ""

    # --- Fallback ---
    fallback_text = " ".join(fallbacks) if fallbacks else "No fallback required — evidence is internally consistent."

    mode_text = recommended_mode or "balanced"

    return (
        f"The data recommends {rec_label} around lap {recommendation.recommended_lap}.\n"
        f"Confidence is {int(confidence.confidence * 100)}% with {confidence.risk_level} risk. "
        f"Recommended approach: {mode_text}.\n\n"
        f"{why_block}\n\n"
        f"{uncertainty_block}\n\n"
        f"{impacts_block}\n\n"
        f"Safest action: {fallback_text}"
    )


def _build_ollama_prompt(
    recommendation: StrategyRecommendation,
    confidence: ConfidenceResult,
    issues: list[TrustIssue],
    fallbacks: list[str],
    *,
    uncertainty: UncertaintyResult | None = None,
    recommended_mode: str | None = None,
    decision_impact: DecisionImpactResult | None = None,
    decision_impacts: list[DecisionImpactResult] | None = None,
) -> str:
    issue_text = _format_issues(issues)
    mode_text = recommended_mode or "balanced"
    unc_text = uncertainty.primary_uncertainty if uncertainty else "none"
    unc_score = uncertainty.uncertainty_score if uncertainty else 0.0
    impact_lines: list[str] = []
    if decision_impact is not None:
        impact_lines.append(f"pit_timing if-wrong loss {decision_impact.if_wrong_expected_loss_s}s (risk={decision_impact.risk_level})")
    for di in decision_impacts or []:
        impact_lines.append(f"{di.decision} if-wrong loss {di.if_wrong_expected_loss_s}s (risk={di.risk_level})")
    return (
        "You are a race engineer explaining a decision-confidence result to a strategist. "
        "Write 2-3 plain-English sentences. Do not use codes or labels like 'if-wrong loss' or 'mode'.\n"
        f"Recommendation: {recommendation.recommendation_type.value.replace('_', ' ')} around lap {recommendation.recommended_lap}.\n"
        f"Confidence: {int(confidence.confidence * 100)}%, risk: {confidence.risk_level}. Approach: {mode_text}.\n"
        f"Uncertainty source: {unc_text} (score: {unc_score}).\n"
        f"Issues:\n{issue_text}\n"
        f"Fallback: {'; '.join(fallbacks)}"
    )


def granite_explanation(
    recommendation: StrategyRecommendation,
    confidence: ConfidenceResult,
    issues: list[TrustIssue],
    fallbacks: list[str],
    *,
    uncertainty: UncertaintyResult | None = None,
    recommended_mode: str | None = None,
    decision_impact: DecisionImpactResult | None = None,
    decision_impacts: list[DecisionImpactResult] | None = None,
) -> str | None:
    """Call local Ollama or watsonx Granite to rewrite the explanation.

    Set `OLLAMA_MODEL=granite4.1:3b` (optionally `OLLAMA_HOST`) for local Ollama,
    OR set `WATSONX_GRANITE_URL` + `WATSONX_API_KEY` for IBM cloud Granite.
    Falls back to deterministic explanation if neither is configured.
    """

    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        prompt = _build_ollama_prompt(
            recommendation, confidence, issues, fallbacks,
            uncertainty=uncertainty, recommended_mode=recommended_mode,
            decision_impact=decision_impact, decision_impacts=decision_impacts,
        )
        payload = {
            "model": ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            resp = requests.post(f"{host}/api/chat", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        except requests.RequestException:
            return None
        except (ValueError, TypeError):
            return None

    api_url = os.getenv("WATSONX_GRANITE_URL")
    api_key = os.getenv("WATSONX_API_KEY")
    if not api_url or not api_key:
        return None

    impact_lines: list[str] = []
    if decision_impact is not None:
        impact_lines.append(
            f"pit_timing if-wrong loss {decision_impact.if_wrong_expected_loss_s}s (risk={decision_impact.risk_level})"
        )
    for di in decision_impacts or []:
        impact_lines.append(f"{di.decision} if-wrong loss {di.if_wrong_expected_loss_s}s (risk={di.risk_level})")

    payload = {
        "input": (
            "Explain this race-strategy confidence result in concise engineering language.\n"
            f"Recommendation: {recommendation.recommendation_type.value}, lap {recommendation.recommended_lap}\n"
            f"Confidence: {confidence.confidence}, risk: {confidence.risk_level}\n"
            f"Recommended mode: {recommended_mode or 'UNKNOWN'}\n"
            f"Uncertainty: {uncertainty.primary_uncertainty if uncertainty else 'none'} "
            f"(score={uncertainty.uncertainty_score if uncertainty else 0.0})\n"
            f"Decision impacts: {'; '.join(impact_lines) if impact_lines else 'none'}\n"
            f"Issues:\n{_format_issues(issues)}\n"
            f"Fallbacks: {'; '.join(fallbacks)}"
        )
    }
    try:
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return None
    except (ValueError, TypeError):
        return None

    if isinstance(data, dict):
        for key in ("generated_text", "output", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def explain_decision(
    recommendation: StrategyRecommendation,
    confidence: ConfidenceResult,
    issues: list[TrustIssue],
    fallbacks: list[str],
    *,
    uncertainty: UncertaintyResult | None = None,
    recommended_mode: str | None = None,
    decision_impact: DecisionImpactResult | None = None,
    decision_impacts: list[DecisionImpactResult] | None = None,
    prefer_granite: bool = True,
) -> str:
    if prefer_granite:
        explanation = granite_explanation(
            recommendation,
            confidence,
            issues,
            fallbacks,
            uncertainty=uncertainty,
            recommended_mode=recommended_mode,
            decision_impact=decision_impact,
            decision_impacts=decision_impacts,
        )
        if explanation:
            return explanation
    return mock_explanation(
        recommendation,
        confidence,
        issues,
        fallbacks,
        uncertainty=uncertainty,
        recommended_mode=recommended_mode,
        decision_impact=decision_impact,
        decision_impacts=decision_impacts,
    )
