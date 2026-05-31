from pathlib import Path
import json
import os
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("MDCE_ROOT", "/content/drive/MyDrive/ibm_project_stuff/MDCE"))
if not ROOT.exists():
    raise FileNotFoundError(
        f"Expected project folder not found: {ROOT}. "
        "This add-on is folder-locked and will not scan MyDrive."
    )

DB_PATH = ROOT / "databases" / "mdce_f1.db"
REPORT_DIR = ROOT / "outputs" / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"
for folder in [REPORT_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print("wrote:", path)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def safe_mean(series, default=0.0):
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if len(series) else default


if not DB_PATH.exists():
    raise FileNotFoundError(f"Missing database: {DB_PATH}. Run main pipeline first.")

conn = sqlite3.connect(DB_PATH)
laps = pd.read_sql_query(
    """
    SELECT raceId, driverId, lap, milliseconds, year, round, name, code, Driver,
           Compound, TyreLife, Stint, AirTemp, TrackTemp, Humidity, Pressure,
           Rainfall, WindSpeed, position_x, grid
    FROM kaggle_weather_tyre_laps
    WHERE year = 2023 AND round = 22
    ORDER BY raceId, driverId, lap
    """,
    conn,
)
pit_stops = pd.read_sql_query(
    """
    SELECT p.raceId, p.driverId, p.stop, p.lap AS pit_lap, p.duration, d.code
    FROM jtrotman_pit_stops p
    LEFT JOIN jtrotman_drivers d ON p.driverId = d.driverId
    WHERE p.raceId = 1120
    ORDER BY p.raceId, p.driverId, p.stop
    """,
    conn,
)
conn.close()

for col in ["lap", "milliseconds", "TyreLife", "Stint", "AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed", "position_x", "grid"]:
    if col in laps.columns:
        laps[col] = pd.to_numeric(laps[col], errors="coerce")
laps["lap_time_s"] = laps["milliseconds"] / 1000.0
laps["tyre_age"] = laps["TyreLife"]
laps["compound"] = laps["Compound"].fillna("UNKNOWN").astype(str).str.upper()
laps["driver_code"] = laps["code"].fillna("UNK").astype(str).str.upper()
laps["circuit"] = laps["name"].fillna("UNKNOWN").astype(str)
laps["rainfall"] = pd.to_numeric(laps["Rainfall"], errors="coerce").fillna(0)
laps["track_temp"] = pd.to_numeric(laps["TrackTemp"], errors="coerce")
laps["air_temp"] = pd.to_numeric(laps["AirTemp"], errors="coerce")
laps["wind_speed"] = pd.to_numeric(laps["WindSpeed"], errors="coerce")


def build_context(df, decision_lap):
    history = df[pd.to_numeric(df["lap"], errors="coerce") <= decision_lap].sort_values("lap").copy()
    recent = history.tail(5)
    if history.empty:
        raise ValueError("No history rows for decision context.")
    lap_time_trend = 0.0
    if len(recent) >= 2:
        lap_time_trend = float((recent["lap_time_s"].iloc[-1] - recent["lap_time_s"].iloc[0]) / max(1, len(recent) - 1))
    return {
        "raceId": int(history["raceId"].iloc[-1]),
        "driverId": int(history["driverId"].iloc[-1]),
        "driver_code": str(history["driver_code"].iloc[-1]),
        "circuit": str(history["circuit"].iloc[-1]),
        "decision_lap": int(decision_lap),
        "compound": str(history["compound"].iloc[-1]),
        "tyre_age": float(history["tyre_age"].iloc[-1]),
        "lap_time_s": float(history["lap_time_s"].iloc[-1]),
        "lap_time_trend": lap_time_trend,
        "track_temp": safe_mean(recent["track_temp"], default=safe_mean(history["track_temp"])),
        "air_temp": safe_mean(recent["air_temp"], default=safe_mean(history["air_temp"])),
        "rainfall_recent": float(pd.to_numeric(recent["rainfall"], errors="coerce").fillna(0).max()) if len(recent) else 0.0,
        "wind_speed": safe_mean(recent["wind_speed"], default=safe_mean(history["wind_speed"])),
        "position": float(pd.to_numeric(history["position_x"], errors="coerce").iloc[-1]) if pd.notna(history["position_x"].iloc[-1]) else None,
        "grid": float(pd.to_numeric(history["grid"], errors="coerce").iloc[-1]) if pd.notna(history["grid"].iloc[-1]) else None,
        "history_rows": int(len(history)),
    }


def issue(name, severity, penalty, reason):
    return {"issue": name, "severity": severity, "penalty": float(penalty), "reason": reason}


def confidence_from_issues(base, issues):
    return clamp(base - sum(item["penalty"] for item in issues), 0.05, 0.95)


def risk_from_confidence(confidence):
    if confidence >= 0.75:
        return "Low"
    if confidence >= 0.60:
        return "Medium"
    if confidence >= 0.45:
        return "Medium-High"
    return "High"


def evaluate_pit_timing(ctx):
    issues = []
    if ctx["lap_time_trend"] >= 0.18:
        issues.append(issue("lap_time_degradation", "high", 0.12, "Recent lap times are worsening."))
    if ctx["tyre_age"] >= 15:
        issues.append(issue("older_tyre_stint", "medium", 0.08, "Tyre age is entering a public-data pit-risk window."))
    if ctx["rainfall_recent"] > 0:
        issues.append(issue("wet_weather_uncertainty", "medium", 0.10, "Rain signal can change pit timing assumptions."))
    confidence = confidence_from_issues(0.86, issues)
    if ctx["lap_time_trend"] >= 0.30 or ctx["tyre_age"] >= 16:
        recommendation = "PIT_NOW"
    elif ctx["lap_time_trend"] >= 0.15 or ctx["tyre_age"] >= 12:
        recommendation = "PIT_SOON"
    else:
        recommendation = "EXTEND_STINT"
    return recommendation, confidence, risk_from_confidence(confidence), issues


def evaluate_tyre_choice(ctx):
    issues = []
    if ctx["rainfall_recent"] > 0:
        recommendation = "CONSIDER_INTERMEDIATE_OR_WET"
        issues.append(issue("rain_detected", "high", 0.15, "Rainfall exists in recent public weather data."))
    elif ctx["track_temp"] >= 38:
        recommendation = "AVOID_OVERLY_SOFT_LONG_STINT"
        issues.append(issue("high_track_temperature", "medium", 0.08, "High track temperature increases thermal degradation risk."))
    elif ctx["compound"] == "SOFT" and ctx["tyre_age"] >= 10:
        recommendation = "MOVE_TO_MEDIUM_OR_HARD"
        issues.append(issue("soft_tyre_aging", "medium", 0.08, "Soft compound has meaningful age."))
    else:
        recommendation = "CURRENT_COMPOUND_ACCEPTABLE"
    confidence = confidence_from_issues(0.72, issues)
    return recommendation, confidence, risk_from_confidence(confidence), issues


def evaluate_stint_length(ctx):
    issues = []
    if ctx["history_rows"] < 8:
        issues.append(issue("short_history", "medium", 0.08, "Limited stint history reduces confidence."))
    if ctx["lap_time_trend"] >= 0.25:
        issues.append(issue("stint_degradation", "high", 0.15, "Lap-time trend suggests stint is degrading."))
    if ctx["tyre_age"] >= 18:
        issues.append(issue("long_tyre_age", "high", 0.15, "Tyre age is high for this public-data example."))
    if ctx["lap_time_trend"] >= 0.25 or ctx["tyre_age"] >= 18:
        recommendation = "SHORTEN_STINT"
    elif ctx["lap_time_trend"] <= 0.05 and ctx["tyre_age"] < 12:
        recommendation = "EXTEND_STINT"
    else:
        recommendation = "KEEP_TARGET_STINT"
    confidence = confidence_from_issues(0.76, issues)
    return recommendation, confidence, risk_from_confidence(confidence), issues


def evaluate_push_conserve(ctx):
    issues = []
    if ctx["lap_time_trend"] >= 0.20:
        recommendation = "CONSERVE"
        issues.append(issue("degradation_risk", "high", 0.12, "Pushing could worsen degradation."))
    elif ctx["tyre_age"] <= 8 and ctx["rainfall_recent"] == 0:
        recommendation = "PUSH"
    else:
        recommendation = "BALANCED_PACE"
    if ctx["wind_speed"] >= 8:
        issues.append(issue("wind_variability", "medium", 0.06, "Wind can affect pace consistency."))
    confidence = confidence_from_issues(0.70, issues)
    return recommendation, confidence, risk_from_confidence(confidence), issues


def evaluate_safety_car_response(ctx):
    issues = [
        issue("missing_live_track_status", "high", 0.20, "Live safety-car/VSC status is not available in this public-data decision context.")
    ]
    if ctx["tyre_age"] >= 12:
        recommendation = "PIT_IF_SAFETY_CAR_WINDOW_OPENS"
    else:
        recommendation = "HOLD_POSITION_UNLESS_CHEAP_STOP"
    confidence = confidence_from_issues(0.58, issues)
    return recommendation, confidence, risk_from_confidence(confidence), issues


def evaluate_aggressive_safe_strategy(ctx):
    issues = []
    if ctx["lap_time_trend"] >= 0.20 or ctx["rainfall_recent"] > 0:
        recommendation = "SAFE_STRATEGY"
        issues.append(issue("fragile_conditions", "high", 0.14, "Recent trend/weather makes aggressive strategy fragile."))
    elif ctx["position"] is not None and ctx["grid"] is not None and ctx["position"] > ctx["grid"]:
        recommendation = "AGGRESSIVE_RECOVERY_STRATEGY"
    else:
        recommendation = "BALANCED_STRATEGY"
    confidence = confidence_from_issues(0.68, issues)
    return recommendation, confidence, risk_from_confidence(confidence), issues


evaluators = {
    "pit_timing": evaluate_pit_timing,
    "tyre_choice": evaluate_tyre_choice,
    "stint_length": evaluate_stint_length,
    "push_conserve": evaluate_push_conserve,
    "safety_car_response": evaluate_safety_car_response,
    "aggressive_safe_strategy": evaluate_aggressive_safe_strategy,
}

validation_status = {
    "pit_timing": "validated_with_actual_pit_stop_labels_and_ml_backtest",
    "tyre_choice": "extension_ready_rule_layer_not_yet_label_validated",
    "stint_length": "extension_ready_rule_layer_not_yet_label_validated",
    "push_conserve": "extension_ready_rule_layer_not_yet_label_validated",
    "safety_car_response": "requires_live_or_lap_level_track_status_for_full_validation",
    "aggressive_safe_strategy": "extension_ready_rule_layer_not_yet_label_validated",
}

sample_cases = []
for driver_code in ["VER", "LEC", "HAM", "NOR"]:
    driver_rows = laps[laps["driver_code"].eq(driver_code)].copy()
    if driver_rows.empty:
        continue
    first_pit = pit_stops[pit_stops["code"].astype(str).str.upper().eq(driver_code)]["pit_lap"]
    decision_lap = int(pd.to_numeric(first_pit, errors="coerce").dropna().iloc[0]) if len(first_pit.dropna()) else int(driver_rows["lap"].max() // 3)
    ctx = build_context(driver_rows, decision_lap)
    for decision_type, fn in evaluators.items():
        recommendation, confidence, risk, issues = fn(ctx)
        sample_cases.append(
            {
                "driver_code": ctx["driver_code"],
                "circuit": ctx["circuit"],
                "decision_lap": ctx["decision_lap"],
                "decision_type": decision_type,
                "recommendation": recommendation,
                "confidence": round(confidence, 3),
                "risk": risk,
                "issue_count": len(issues),
                "issues": ", ".join(item["issue"] for item in issues) if issues else "none",
                "validation_status": validation_status[decision_type],
            }
        )

sample_df = pd.DataFrame(sample_cases)
sample_df.to_csv(REPORT_DIR / "mdce_multidecision_sample_outputs.csv", index=False)

registry = []
for decision_type in evaluators:
    registry.append(
        {
            "decision_type": decision_type,
            "current_status": validation_status[decision_type],
            "uses_real_public_data_now": True,
            "validated_with_outcome_labels_now": decision_type == "pit_timing",
            "extra_data_needed_for_strong_validation": {
                "pit_timing": "Already has pit-stop labels; can improve with live gaps and traffic.",
                "tyre_choice": "Need labelled optimal/actual tyre choice outcomes and compound performance by stint.",
                "stint_length": "Need stint outcome labels beyond first pit timing, plus traffic/rejoin loss.",
                "push_conserve": "Need command/pace intent labels or proxy objectives such as tyre survival vs lap-time gain.",
                "safety_car_response": "Need lap-level safety-car/VSC/track status and pit-window deltas.",
                "aggressive_safe_strategy": "Need objective outcome labels: position gained/lost, undercut/overcut success, risk outcome.",
            }[decision_type],
        }
    )
registry_df = pd.DataFrame(registry)
registry_df.to_csv(REPORT_DIR / "mdce_decision_capability_registry.csv", index=False)

write_json(
    REPORT_DIR / "mdce_multidecision_layer.json",
    {
        "safe_claim": "MDCE is a decision confidence layer. Pit timing is validated now; other decision types are extension-ready but require stronger labels before claiming predictive optimality.",
        "decision_capability_registry": registry,
        "sample_outputs": sample_cases,
    },
)

lines = [
    "# MDCE Multi-Decision Confidence Layer",
    "",
    "## Purpose",
    "",
    "This report separates the validated pit-window model from the broader MDCE decision-confidence framework.",
    "",
    "## Safe Claim",
    "",
    "MDCE is not only a pit-stop model. Pit timing is the currently validated public-data case. The same confidence/risk framework is prepared for tyre choice, stint length, push/conserve, safety-car response, and aggressive vs safe strategy, but those extensions need stronger labels before optimality claims.",
    "",
    "## Decision Capability Registry",
    "",
    "| Decision Type | Current Status | Outcome-Label Validated Now | Extra Data Needed |",
    "|---|---|---:|---|",
]
for row in registry:
    lines.append(
        f"| {row['decision_type']} | {row['current_status']} | {row['validated_with_outcome_labels_now']} | {row['extra_data_needed_for_strong_validation']} |"
    )
lines += [
    "",
    "## Sample Multi-Decision Outputs",
    "",
    "| Driver | Lap | Decision Type | Recommendation | Confidence | Risk | Issues |",
    "|---|---:|---|---|---:|---|---|",
]
for _, row in sample_df.iterrows():
    lines.append(
        f"| {row['driver_code']} | {row['decision_lap']} | {row['decision_type']} | {row['recommendation']} | {row['confidence']} | {row['risk']} | {row['issues']} |"
    )
lines += [
    "",
    "## Files Written",
    "",
    f"- `{REPORT_DIR / 'mdce_multidecision_sample_outputs.csv'}`",
    f"- `{REPORT_DIR / 'mdce_decision_capability_registry.csv'}`",
    f"- `{REPORT_DIR / 'mdce_multidecision_layer.json'}`",
]
(REPORT_DIR / "mdce_multidecision_layer.md").write_text("\n".join(lines), encoding="utf-8")

print("wrote:", REPORT_DIR / "mdce_multidecision_layer.md")
print(sample_df.to_string(index=False))
