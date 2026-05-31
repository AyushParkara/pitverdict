from __future__ import annotations

from .models import TrustIssue


FALLBACKS = {
    "missing_telemetry": "Avoid aggressive calls until fresh lap data arrives; use the last stable lap-time trend.",
    "model_mismatch": "Prefer a safer pit window and re-check the model against actual lap-time changes.",
    "model_deviation": "Treat model outputs as less reliable; re-check predicted lap times against recent actual pace before acting.",
    "optimistic_model_bias": "Do not extend only because the model is optimistic; compare against recent actual pace.",
    "tyre_lap_signal_conflict": "Use lap-time trend as fallback instead of the tyre-wear estimate.",
    "safety_car_context": "Reduce trust in pace-change estimates until normal racing pace resumes.",
    "weather_uncertainty": "Avoid long-stint confidence claims until conditions stabilize.",
    "low_speed_consistency": "Treat traffic/rejoin analysis as uncertain and verify with another timing signal.",
    "no_usable_data": "Play it safe: do not issue an aggressive strategy recommendation without usable data.",
    "coverage_gap_sector_times": "Avoid sector-based reasoning; use overall lap trend instead.",
    "coverage_gap_track_gaps": "Avoid traffic/rejoin claims; assume worst-case rejoin risk until gaps are available.",
    "coverage_gap_tyre_compound": "Avoid compound-specific strategy; treat tyre inputs as uncertain and be cautious with recommendations.",
    "multi_signal_disagreement": "Data sources disagree; pause aggressive actions and check again.",
    "lap_time_outlier": "Recent lap time looks unusual; avoid trend-based claims and verify track/traffic context before acting.",
}


def fallback_actions(issues: list[TrustIssue]) -> list[str]:
    actions: list[str] = []

    # Deterministic ordering: address the highest-impact uncertainty first.
    severity_weight = {"high": 3, "medium": 2, "low": 1}
    ordered = sorted(
        issues,
        key=lambda i: (-float(i.penalty), -severity_weight.get(i.severity.value, 0), i.issue),
    )

    # If uncertainty is materially high, nudge to SAFE mode as a global fallback.
    total_penalty = sum(i.penalty for i in issues)
    if total_penalty >= 0.45:
        actions.append("Switch to a safer approach until uncertainty reduces.")

    for issue in ordered:
        action = FALLBACKS.get(issue.issue)
        if action and action not in actions:
            actions.append(action)
    if not actions:
        actions.append("No fallback required; current decision evidence is internally consistent.")
    return actions
