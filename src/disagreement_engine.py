from __future__ import annotations

from statistics import mean

from .data_loader import available_records, recent_records
from .models import Severity, TrustIssue


def detect_multi_signal_disagreement(records: list) -> list[TrustIssue]:
    """Lightweight ensemble-style disagreement checks.

    This is *not* ML training. It's deterministic reasoning across multiple signals:

    - actual lap-time degradation (trend)
    - model predicted lap-time degradation (trend)
    - tyre proxy direction (trend)

    If these strongly disagree, confidence should drop.
    """

    usable = available_records(records)
    recent = recent_records(usable, count=6)
    if len(recent) < 4:
        return []

    # Trend proxies: positive means getting slower / hotter.
    actual_deltas = [recent[i + 1].lap_time_s - recent[i].lap_time_s for i in range(len(recent) - 1)]
    model_deltas = [recent[i + 1].predicted_lap_time_s - recent[i].predicted_lap_time_s for i in range(len(recent) - 1)]
    tyre_deltas = [recent[i + 1].tyre_temp_proxy_c - recent[i].tyre_temp_proxy_c for i in range(len(recent) - 1)]

    actual_trend = mean(actual_deltas)
    model_trend = mean(model_deltas)
    tyre_trend = mean(tyre_deltas)

    # Disagreement pattern: actual pace worsening but model is flat/optimistic
    # AND tyre proxy provides weak support.
    if actual_trend >= 0.25 and model_trend <= 0.05 and abs(tyre_trend) <= 0.08:
        return [
            TrustIssue(
                issue="multi_signal_disagreement",
                severity=Severity.MEDIUM,
                affected_decisions=["pit_timing", "stint_length", "push_vs_conserve"],
                reason=(
                    "Multiple signals disagree: actual pace degradation is rising, "
                    "but the model trend is flat/optimistic and the tyre proxy provides weak support."
                ),
                penalty=0.10,
            )
        ]

    # Disagreement pattern: tyre proxy suggests strong change but lap times stable.
    if abs(tyre_trend) >= 0.35 and abs(actual_trend) <= 0.05:
        return [
            TrustIssue(
                issue="multi_signal_disagreement",
                severity=Severity.MEDIUM,
                affected_decisions=["tyre_strategy", "pit_timing"],
                reason=(
                    "Multiple signals disagree: tyre proxy shifts sharply while lap-time trend remains stable."
                ),
                penalty=0.08,
            )
        ]

    return []
