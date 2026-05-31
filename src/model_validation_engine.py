from __future__ import annotations

from statistics import mean

from .data_loader import available_records, recent_records
from .models import ModelDeviation, ModelValidationResult


def validate_model_vs_reality(
    records: list,
    *,
    window_laps: int = 5,
    deviation_threshold_s: float = 0.75,
) -> ModelValidationResult:
    """Validate model predictions against recent actual lap times.

    Deterministic and intentionally simple:

    - Computes MAE/max error over the last N usable laps.
    - Flags per-lap deviations above a threshold.
    - Suggests a confidence penalty based on error severity.
    """

    usable = available_records(records)
    recent = recent_records(usable, count=window_laps)
    if not recent:
        return ModelValidationResult(
            status="Not enough data to check",
            window_laps=window_laps,
            deviation_threshold_s=deviation_threshold_s,
            mean_abs_error_s=0.0,
            max_abs_error_s=0.0,
            deviations=[],
            recommended_confidence_penalty=0.0,
        )

    abs_errors = [abs(r.lap_time_s - r.predicted_lap_time_s) for r in recent]
    mae = mean(abs_errors) if abs_errors else 0.0
    max_err = max(abs_errors) if abs_errors else 0.0
    deviations: list[ModelDeviation] = []
    for r in recent:
        delta = r.lap_time_s - r.predicted_lap_time_s
        if abs(delta) >= deviation_threshold_s:
            deviations.append(
                ModelDeviation(
                    lap=r.lap,
                    expected_lap_time_s=float(r.predicted_lap_time_s),
                    actual_lap_time_s=float(r.lap_time_s),
                    delta_s=float(round(delta, 2)),
                )
            )

    status = "Model deviating from reality" if deviations else "Model matches reality"

    # Map error severity -> penalty. Keep bounded and deterministic.
    # - Below threshold: no penalty.
    # - Around ~1.0s MAE: noticeable penalty.
    # - Hard cap to avoid collapsing all confidence from this one check.
    penalty = 0.0
    if mae >= deviation_threshold_s:
        penalty = min(0.25, round(0.05 + ((mae - deviation_threshold_s) / 2.0), 2))

    return ModelValidationResult(
        status=status,
        window_laps=window_laps,
        deviation_threshold_s=deviation_threshold_s,
        mean_abs_error_s=round(float(mae), 2),
        max_abs_error_s=round(float(max_err), 2),
        deviations=deviations,
        recommended_confidence_penalty=penalty,
    )
