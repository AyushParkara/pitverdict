from __future__ import annotations

from dataclasses import dataclass

from .models import ScenarioFlags


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    description: str
    flags: ScenarioFlags


SCENARIO_PRESETS: dict[str, ScenarioPreset] = {
    "custom": ScenarioPreset(
        name="custom",
        description="Manually pick scenario flags.",
        flags=ScenarioFlags(),
    ),
    "missing_telemetry": ScenarioPreset(
        name="missing_telemetry",
        description="Last two laps marked missing telemetry.",
        flags=ScenarioFlags(missing_telemetry=True),
    ),
    "tyre_signal_drift": ScenarioPreset(
        name="tyre_signal_drift",
        description="Tyre proxy held flat while lap times change.",
        flags=ScenarioFlags(tyre_signal_drift=True),
    ),
    "model_mismatch": ScenarioPreset(
        name="model_mismatch",
        description="Model predictions made optimistic vs actual pace.",
        flags=ScenarioFlags(model_mismatch=True),
    ),
    "safety_car_phase": ScenarioPreset(
        name="safety_car_phase",
        description="Recent laps marked as Safety Car.",
        flags=ScenarioFlags(safety_car_phase=True),
    ),
    "weather_uncertainty": ScenarioPreset(
        name="weather_uncertainty",
        description="Recent conditions set to damp/uncertain.",
        flags=ScenarioFlags(weather_uncertainty=True),
    ),
    "high_uncertainty_stack": ScenarioPreset(
        name="high_uncertainty_stack",
        description="Stack multiple failure modes to force SAFE mode.",
        flags=ScenarioFlags(
            missing_telemetry=True,
            model_mismatch=True,
            safety_car_phase=True,
            weather_uncertainty=True,
        ),
    ),
    # Demo-oriented named scenarios (combinations of the base flags above).
    # These make the demo narrative clearer without adding new mutation logic.
    "sensor_failure": ScenarioPreset(
        name="sensor_failure",
        description="Sensor failure: recent telemetry missing and tyre proxy frozen.",
        flags=ScenarioFlags(missing_telemetry=True, tyre_signal_drift=True),
    ),
    "conflicting_signals": ScenarioPreset(
        name="conflicting_signals",
        description="Conflicting signals: tyre proxy held flat while lap times drift.",
        flags=ScenarioFlags(tyre_signal_drift=True),
    ),
    "model_wrong": ScenarioPreset(
        name="model_wrong",
        description="Model wrong: predictions optimistic vs actual degrading pace.",
        flags=ScenarioFlags(model_mismatch=True),
    ),
    "extreme_conditions": ScenarioPreset(
        name="extreme_conditions",
        description="Extreme conditions: sudden damp weather during a safety car phase.",
        flags=ScenarioFlags(weather_uncertainty=True, safety_car_phase=True),
    ),
}


_PRESET_ORDER = [
    "custom",
    "missing_telemetry",
    "tyre_signal_drift",
    "model_mismatch",
    "safety_car_phase",
    "weather_uncertainty",
    "high_uncertainty_stack",
    "sensor_failure",
    "conflicting_signals",
    "model_wrong",
    "extreme_conditions",
]


def _human_key(key: str) -> str:
    return key.replace("_", " ").title()


# Cache: mapping from human-readable name → raw key
_human_to_key: dict[str, str] = {
    _human_key(k): k for k in _PRESET_ORDER if k in SCENARIO_PRESETS
}


def list_preset_names() -> list[str]:
    """Return human-readable preset names for UI selectors."""
    return [_human_key(name) for name in _PRESET_ORDER if name in SCENARIO_PRESETS]


def resolve_preset(name: str) -> ScenarioPreset:
    raw = (name or "custom").strip()
    low = raw.lower()
    if low in SCENARIO_PRESETS:
        return SCENARIO_PRESETS[low]
    # Normalise to Title Case for human-readable lookup.
    key = _human_to_key.get(raw.replace("_", " ").title())
    if key is not None:
        return SCENARIO_PRESETS[key]
    return SCENARIO_PRESETS["custom"]
