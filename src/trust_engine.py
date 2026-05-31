from __future__ import annotations

from statistics import mean, median

from .data_loader import available_records, lap_time_trend, predicted_lap_trend, recent_records, tyre_temp_trend
from .models import Severity, TrustIssue


def detect_trust_issues(records: list) -> list[TrustIssue]:
    issues: list[TrustIssue] = []
    usable = available_records(records)
    recent = recent_records(records, count=5)

    if not usable:
        return [
            TrustIssue(
                issue="no_usable_data",
                severity=Severity.HIGH,
                affected_decisions=["all_strategy_decisions"],
                reason="No usable recent telemetry is available.",
                penalty=0.50,
            )
        ]

    if any(record.missing for record in records[-3:]):
        issues.append(
            TrustIssue(
                issue="missing_telemetry",
                severity=Severity.HIGH,
                affected_decisions=["pit_timing", "stint_length"],
                reason="Recent telemetry is missing, so the current lap trend is incomplete.",
                penalty=0.20,
            )
        )

    # Detect lap-time outliers in the recent window.
    # We intentionally do not treat these as "model mismatch" by default: a single
    # anomalous lap is often missing context (traffic/SC/yellow/mistake) rather than
    # systemic model drift.
    recent_for_outlier = recent_records(usable, count=8)
    if len(recent_for_outlier) >= 6:
        lap_times = [r.lap_time_s for r in recent_for_outlier]
        med = median(lap_times)
        abs_devs = [abs(x - med) for x in lap_times]
        mad = median(abs_devs) if abs_devs else 0.0
        # Robust scale estimate. If MAD is ~0 (too-flat window), fall back to a small
        # epsilon so we only trigger on truly large spikes.
        robust_sigma = 1.4826 * mad
        robust_sigma = max(robust_sigma, 0.15)
        last = recent_for_outlier[-1]
        z = (last.lap_time_s - med) / robust_sigma
        # Threshold tuned to catch obvious spikes (like +2s) without overfiring on noise.
        if z >= 6.0 and (last.lap_time_s - med) >= 1.2:
            issues.append(
                TrustIssue(
                    issue="lap_time_outlier",
                    severity=Severity.HIGH,
                    affected_decisions=["pit_timing", "stint_length", "push_vs_conserve", "aggressive_vs_safe_strategy"],
                    reason=(
                        "Recent lap time is a strong outlier vs the recent window. "
                        "This often indicates missing context (traffic/yellow/incident) rather than stable degradation."
                    ),
                    penalty=0.12,
                )
            )

    # Trend-based mismatch detection.
    # If we have a recent lap-time outlier, compute the trend on a "robust" window that
    # excludes the worst spike. This avoids mislabeling an acute anomaly as systemic drift.
    outlier_present = any(i.issue == "lap_time_outlier" for i in issues)
    trend_records = usable
    if outlier_present:
        window = recent_records(usable, count=6)
        if len(window) >= 5:
            worst = max(window, key=lambda r: r.lap_time_s)
            trend_records = [r for r in window if r is not worst]

    actual_trend = lap_time_trend(trend_records, count=5)
    model_trend = predicted_lap_trend(trend_records, count=5)
    mismatch_threshold = 0.18
    if outlier_present:
        # With outliers, require a stronger sustained divergence even after excluding the spike.
        mismatch_threshold = 0.24

    if actual_trend - model_trend >= mismatch_threshold:
        issues.append(
            TrustIssue(
                issue="model_mismatch",
                severity=Severity.HIGH,
                affected_decisions=["pit_timing", "stint_length", "aggressive_vs_safe_strategy"],
                reason="Actual lap times are worsening faster than the strategy model predicted.",
                penalty=0.15,
            )
        )

    if len(recent) >= 2:
        mean_actual = mean(record.lap_time_s for record in recent)
        mean_predicted = mean(record.predicted_lap_time_s for record in recent)
        if mean_actual - mean_predicted >= 1.0:
            issues.append(
                TrustIssue(
                    issue="optimistic_model_bias",
                    severity=Severity.MEDIUM,
                    affected_decisions=["pit_timing", "push_vs_conserve"],
                    reason="The model is consistently more optimistic than recent actual pace.",
                    penalty=0.10,
                )
            )

    tyre_trend = tyre_temp_trend(usable, count=5)
    if actual_trend >= 0.20 and abs(tyre_trend) <= 0.05:
        issues.append(
            TrustIssue(
                issue="tyre_lap_signal_conflict",
                severity=Severity.MEDIUM,
                affected_decisions=["tyre_strategy", "pit_timing"],
                reason="Lap times show degradation while the tyre proxy remains flat.",
                penalty=0.10,
            )
        )

    if any(record.track_status in {"SC", "VSC"} for record in records[-4:]):
        issues.append(
            TrustIssue(
                issue="safety_car_context",
                severity=Severity.MEDIUM,
                affected_decisions=["pit_timing", "tyre_strategy", "safety_car_response"],
                reason="Recent Safety Car context distorts lap-time and tyre-degradation assumptions.",
                penalty=0.10,
            )
        )

    if any(record.weather != "DRY" for record in records[-5:]):
        issues.append(
            TrustIssue(
                issue="weather_uncertainty",
                severity=Severity.MEDIUM,
                affected_decisions=["tyre_compound_choice", "push_vs_conserve", "stint_length"],
                reason="Recent weather state is uncertain, reducing trust in dry-running assumptions.",
                penalty=0.10,
            )
        )

    if recent and min(record.speed_consistency for record in recent) < 0.88:
        issues.append(
            TrustIssue(
                issue="low_speed_consistency",
                severity=Severity.MEDIUM,
                affected_decisions=["traffic_rejoin_risk", "push_vs_conserve"],
                reason="Speed consistency is below the expected range for a stable decision snapshot.",
                penalty=0.10,
            )
        )

    return issues


def conflict_score(records: list) -> tuple[float, str]:
    usable = available_records(records)
    actual_trend = lap_time_trend(usable, count=5)
    model_trend = predicted_lap_trend(usable, count=5)
    score = max(0.0, min(1.0, (actual_trend - model_trend) / 0.6))
    if score >= 0.75:
        label = "HIGH"
    elif score >= 0.45:
        label = "MEDIUM"
    elif score >= 0.20:
        label = "LOW"
    else:
        label = "NONE"
    return round(score, 2), label
