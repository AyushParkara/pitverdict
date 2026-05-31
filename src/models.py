from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RecommendationType(str, Enum):
    PIT_NOW = "PIT_NOW"
    PIT_SOON = "PIT_SOON"
    EXTEND = "EXTEND"


class DecisionMode(str, Enum):
    SAFE = "SAFE"
    AGGRESSIVE = "AGGRESSIVE"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class LapRecord:
    lap: int
    lap_time_s: float
    sector1_s: float
    sector2_s: float
    sector3_s: float
    tyre_compound: str
    tyre_age: int
    track_status: str
    weather: str
    gap_to_car_ahead_s: float
    predicted_lap_time_s: float
    tyre_temp_proxy_c: float
    speed_consistency: float
    missing: bool = False


@dataclass(frozen=True)
class ScenarioFlags:
    missing_telemetry: bool = False
    tyre_signal_drift: bool = False
    model_mismatch: bool = False
    safety_car_phase: bool = False
    weather_uncertainty: bool = False


@dataclass(frozen=True)
class ScenarioResult:
    records: list[LapRecord]
    notes: list[str] = field(default_factory=list)
    # Structured, machine-readable labels for each applied scenario mutation.
    # Each entry: {"scenario_name": str, "scenario_type": str, "changed": str}.
    # Kept separate from human `notes` so existing consumers are unaffected.
    applied: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class DataLoadResult:
    records: list[LapRecord]
    source_name: str
    real_columns: list[str]
    derived_columns: list[str]
    proxy_columns: list[str]
    warnings: list[str] = field(default_factory=list)
    # Optional dataset-level provenance/licensing metadata (from sidecar `*.metadata.json` when present).
    dataset_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyRecommendation:
    recommendation_type: RecommendationType
    recommended_lap: int
    expected_gain_loss_s: float
    base_reason: str


@dataclass(frozen=True)
class TrustIssue:
    issue: str
    severity: Severity
    affected_decisions: list[str]
    reason: str
    penalty: float


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float
    risk_level: str
    penalties: list[TrustIssue]
    # Deterministic breakdown for explainability.
    # Scores are in [0.0, 1.0] where higher means more reliable evidence.
    breakdown: dict[str, float] = field(default_factory=dict)
    # Decision-specific confidence scores keyed by decision domain.
    decision_confidence: dict[str, float] = field(default_factory=dict)
    decision_risk_levels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelDeviation:
    lap: int
    expected_lap_time_s: float
    actual_lap_time_s: float
    delta_s: float


@dataclass(frozen=True)
class ModelValidationResult:
    status: str  # OK | DEVIATION | NO_DATA
    window_laps: int
    deviation_threshold_s: float
    mean_abs_error_s: float
    max_abs_error_s: float
    deviations: list[ModelDeviation] = field(default_factory=list)
    recommended_confidence_penalty: float = 0.0


@dataclass(frozen=True)
class DecisionImpactResult:
    decision: str
    horizon_laps: int
    if_right_expected_gain_s: float
    if_wrong_expected_loss_s: float
    risk_level: str
    assumptions: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModeOption:
    mode: DecisionMode
    recommendation: StrategyRecommendation
    decision_impact: DecisionImpactResult


@dataclass(frozen=True)
class UncertaintyResult:
    primary_uncertainty: str
    uncertainty_score: float  # 0..1 where higher means more uncertain
    downstream_decisions_at_risk: list[str]
    drivers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisResult:
    recommendation: StrategyRecommendation
    confidence: ConfidenceResult
    issues: list[TrustIssue]
    fallback_actions: list[str]
    explanation: str
    model_validation: ModelValidationResult | None = None
    decision_impact: DecisionImpactResult | None = None
    # Additional deterministic impact estimates for other decision domains.
    # Kept separate from `decision_impact` to avoid breaking existing UI/CLI fields.
    decision_impacts: list[DecisionImpactResult] = field(default_factory=list)
    uncertainty: UncertaintyResult | None = None
    mode_options: list[ModeOption] = field(default_factory=list)
    # System-selected recommended mode (deterministic) for decision-makers.
    recommended_mode: DecisionMode | None = None
