from __future__ import annotations

from dataclasses import replace

from .models import LapRecord, ScenarioFlags, ScenarioResult


def _replace_last(records: list[LapRecord], count: int, **changes: object) -> list[LapRecord]:
    if count <= 0:
        return records
    start = max(0, len(records) - count)
    updated = list(records)
    for index in range(start, len(records)):
        updated[index] = replace(updated[index], **changes)
    return updated


def apply_scenarios(records: list[LapRecord], flags: ScenarioFlags) -> ScenarioResult:
    """Apply transparent demo scenario mutations to the race data."""
    updated = list(records)
    notes: list[str] = []
    applied: list[dict] = []

    if flags.tyre_signal_drift:
        start = max(0, len(updated) - 6)
        anchor = updated[start].tyre_temp_proxy_c if updated else 100.0
        for index in range(start, len(updated)):
            updated[index] = replace(updated[index], tyre_temp_proxy_c=anchor)
        notes.append("Tyre proxy held flat while lap times continue to change.")
        applied.append(
            {
                "scenario_name": "tyre_signal_drift",
                "scenario_type": "conflicting_signal",
                "changed": "tyre_temp_proxy_c held flat over the last 6 laps",
            }
        )

    if flags.model_mismatch:
        start = max(0, len(updated) - 6)
        window = updated[start:]
        if window:
            # Force a strong mismatch on trend (not just a constant optimistic offset).
            # This avoids a pathological case where the preset can reduce penalties by
            # turning a HIGH-severity `model_mismatch` into a lower-penalty bias-only issue.
            #
            # Strategy: make model predictions unrealistically improve over the window while
            # staying optimistic vs actual. This reliably triggers `model_mismatch`.
            base = min(r.lap_time_s for r in window) - 1.50
            for index in range(start, len(updated)):
                record = updated[index]
                step = index - start
                target = base - (0.25 * step)
                optimistic_prediction = min(
                    record.predicted_lap_time_s,
                    record.lap_time_s - 1.50,
                    target,
                )
                updated[index] = replace(record, predicted_lap_time_s=optimistic_prediction)
        notes.append("Model predictions forced optimistic and improving against actual lap times.")
        applied.append(
            {
                "scenario_name": "model_mismatch",
                "scenario_type": "model_vs_reality",
                "changed": "predicted_lap_time_s forced optimistic/improving over the last 6 laps",
            }
        )

    if flags.safety_car_phase:
        updated = _replace_last(updated, 3, track_status="SC")
        notes.append("Recent laps marked as Safety Car, making tyre-wear estimates less reliable.")
        applied.append(
            {
                "scenario_name": "safety_car_phase",
                "scenario_type": "race_context",
                "changed": "track_status set to SC for the last 3 laps",
            }
        )

    if flags.weather_uncertainty:
        start = max(0, len(updated) - 5)
        for index in range(start, len(updated)):
            record = updated[index]
            updated[index] = replace(
                record,
                weather="DAMP",
                speed_consistency=max(0.0, record.speed_consistency - 0.08),
            )
        notes.append("Recent conditions marked as damp/uncertain.")
        applied.append(
            {
                "scenario_name": "weather_uncertainty",
                "scenario_type": "race_context",
                "changed": "weather set to DAMP and speed_consistency reduced over the last 5 laps",
            }
        )

    if flags.missing_telemetry:
        updated = _replace_last(updated, 2, missing=True)
        notes.append("Last two laps marked as missing telemetry.")
        applied.append(
            {
                "scenario_name": "missing_telemetry",
                "scenario_type": "sensor_failure",
                "changed": "last 2 laps marked missing=True",
            }
        )

    return ScenarioResult(records=updated, notes=notes, applied=applied)
