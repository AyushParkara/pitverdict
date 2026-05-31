from pathlib import Path
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(os.environ.get("MDCE_ROOT", "/content/drive/MyDrive/ibm_project_stuff/MDCE"))
if not ROOT.exists():
    raise FileNotFoundError(
        f"Expected project folder not found: {ROOT}. "
        "This add-on is folder-locked and will not scan MyDrive."
    )

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"
CHART_DIR = ROOT / "outputs" / "charts"
MODEL_DIR = ROOT / "outputs" / "models"

for folder in [REPORT_DIR, CHART_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

training_path = PROCESSED_DIR / "pit_training_rows.parquet"
training = pd.read_parquet(training_path)
print("training shape:", training.shape)

target = "pit_within_3_laps"
numeric_features = [
    "lap",
    "lap_time_s",
    "position",
    "grid",
    "tyre_age",
    "stint",
    "air_temp",
    "track_temp",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "rolling_lap_avg_3",
    "rolling_lap_avg_5",
    "lap_time_delta",
    "degradation_3",
    "degradation_5",
    "lap_progress_ratio",
]
categorical_features = ["tyre_compound", "driver_code", "circuit"]
all_features = numeric_features + categorical_features

required_cols = all_features + [target, "year", "raceId", "driverId"]
missing = [col for col in required_cols if col not in training.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

train_2021 = training[training["year"].eq(2021)].dropna(subset=[target]).copy()
valid_2022 = training[training["year"].eq(2022)].dropna(subset=[target]).copy()
train_2021_2022 = training[training["year"].isin([2021, 2022])].dropna(subset=[target]).copy()
test_2023 = training[training["year"].eq(2023)].dropna(subset=[target]).copy()

print("train_2021:", train_2021.shape, "valid_2022:", valid_2022.shape, "test_2023:", test_2023.shape)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print("wrote:", path)


def build_model(feature_list):
    numeric = [col for col in feature_list if col in numeric_features]
    categorical = [col for col in feature_list if col in categorical_features]
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
                        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
                    ]
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate(y_true, scores, threshold):
    labels = (scores >= threshold).astype(int)
    row = {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(labels)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, scores)) if len(set(y_true)) == 2 else None,
    }
    return row


def choose_threshold_on_validation(scores, y_true):
    rows = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
        row = evaluate(y_true, scores, float(threshold))
        rows.append(row)
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["f1", "precision"], ascending=False).iloc[0].to_dict()
    return best, sweep


feature_sets = {
    "full_state_context": all_features,
    "no_schedule_progress": [col for col in all_features if col not in ["lap", "lap_progress_ratio"]],
    "no_identity_context": [col for col in all_features if col not in ["driver_code", "circuit"]],
    "state_only_no_schedule_no_identity": [
        col for col in all_features if col not in ["lap", "lap_progress_ratio", "driver_code", "circuit"]
    ],
    "no_tyre_signals": [col for col in all_features if col not in ["tyre_age", "tyre_compound", "stint"]],
    "degradation_weather_tyre_only": [
        "lap_time_s",
        "tyre_age",
        "stint",
        "air_temp",
        "track_temp",
        "humidity",
        "pressure",
        "rainfall",
        "wind_speed",
        "wind_direction",
        "rolling_lap_avg_3",
        "rolling_lap_avg_5",
        "lap_time_delta",
        "degradation_3",
        "degradation_5",
        "tyre_compound",
    ],
}

print("\nSTEP A: threshold-from-validation feature-set audit")
audit_rows = []
threshold_rows = []
trained_models = {}

for set_name, features in feature_sets.items():
    print("auditing feature set:", set_name, "features:", len(features))

    validation_model = build_model(features)
    validation_model.fit(train_2021[features], train_2021[target].astype(int))
    validation_scores = validation_model.predict_proba(valid_2022[features])[:, 1]
    best_threshold, sweep = choose_threshold_on_validation(validation_scores, valid_2022[target].astype(int).to_numpy())
    sweep["feature_set"] = set_name
    threshold_rows.append(sweep)

    final_model = build_model(features)
    final_model.fit(train_2021_2022[features], train_2021_2022[target].astype(int))
    test_scores = final_model.predict_proba(test_2023[features])[:, 1]

    test_at_validation_threshold = evaluate(test_2023[target].astype(int).to_numpy(), test_scores, best_threshold["threshold"])
    test_at_validation_threshold.update(
        {
            "feature_set": set_name,
            "feature_count": len(features),
            "threshold_source": "best_f1_on_2022_validation",
            "selected_threshold": best_threshold["threshold"],
            "validation_f1_at_selected_threshold": best_threshold["f1"],
            "validation_precision_at_selected_threshold": best_threshold["precision"],
            "validation_recall_at_selected_threshold": best_threshold["recall"],
        }
    )
    audit_rows.append(test_at_validation_threshold)

    test_at_050 = evaluate(test_2023[target].astype(int).to_numpy(), test_scores, 0.50)
    test_at_050.update(
        {
            "feature_set": set_name,
            "feature_count": len(features),
            "threshold_source": "fixed_0_50",
            "selected_threshold": 0.50,
            "validation_f1_at_selected_threshold": None,
            "validation_precision_at_selected_threshold": None,
            "validation_recall_at_selected_threshold": None,
        }
    )
    audit_rows.append(test_at_050)

    trained_models[set_name] = {"model": final_model, "features": features, "scores": test_scores, "threshold": best_threshold["threshold"]}

audit_df = pd.DataFrame(audit_rows).sort_values(["threshold_source", "average_precision", "f1"], ascending=[True, False, False])
threshold_df = pd.concat(threshold_rows, ignore_index=True)

audit_df.to_csv(REPORT_DIR / "bias_guard_feature_set_audit.csv", index=False)
threshold_df.to_csv(REPORT_DIR / "bias_guard_validation_threshold_sweeps.csv", index=False)

best_validation_threshold_row = audit_df[audit_df["threshold_source"].eq("best_f1_on_2022_validation")].sort_values(
    ["average_precision", "f1"], ascending=False
).iloc[0]
best_feature_set = str(best_validation_threshold_row["feature_set"])
joblib.dump(trained_models[best_feature_set]["model"], MODEL_DIR / "pit_window_bias_guard_best_model.joblib")
print("best bias-guard feature set:", best_feature_set)
print(audit_df.to_string(index=False))

plt.figure(figsize=(11, 6))
plot_df = audit_df[audit_df["threshold_source"].eq("best_f1_on_2022_validation")].copy()
plot_df = plot_df.sort_values("average_precision", ascending=True)
plt.barh(plot_df["feature_set"], plot_df["average_precision"])
plt.title("Bias Guard: 2023 Average Precision By Feature Set")
plt.xlabel("Average precision")
plt.tight_layout()
plt.savefig(CHART_DIR / "bias_guard_feature_set_ap.png", dpi=160)
plt.close()

plt.figure(figsize=(11, 6))
plot_f1 = audit_df[audit_df["threshold_source"].eq("best_f1_on_2022_validation")].sort_values("f1", ascending=True)
plt.barh(plot_f1["feature_set"], plot_f1["f1"])
plt.title("Bias Guard: 2023 F1 By Feature Set")
plt.xlabel("F1")
plt.tight_layout()
plt.savefig(CHART_DIR / "bias_guard_feature_set_f1.png", dpi=160)
plt.close()

print("\nSTEP B: group weak-spot audit for best feature set")
best_info = trained_models[best_feature_set]
test_scores = best_info["scores"]
threshold = best_info["threshold"]
test_labels = (test_scores >= threshold).astype(int)
group_eval = test_2023[["raceId", "driverId", "driver_code", "circuit", target]].copy()
group_eval["score"] = test_scores
group_eval["prediction"] = test_labels


def group_metrics(df, group_col, min_rows=100):
    rows = []
    for group_value, part in df.groupby(group_col):
        if len(part) < min_rows or len(set(part[target])) < 2:
            continue
        row = evaluate(part[target].astype(int).to_numpy(), part["score"].to_numpy(), threshold)
        row[group_col] = group_value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["f1", "average_precision"], ascending=True)


circuit_metrics = group_metrics(group_eval, "circuit", min_rows=150)
driver_metrics = group_metrics(group_eval, "driver_code", min_rows=150)
circuit_metrics.to_csv(REPORT_DIR / "bias_guard_circuit_classification_metrics.csv", index=False)
driver_metrics.to_csv(REPORT_DIR / "bias_guard_driver_classification_metrics.csv", index=False)

if not circuit_metrics.empty:
    plt.figure(figsize=(10, 6))
    plot_circuit = circuit_metrics.head(15).sort_values("f1", ascending=True)
    plt.barh(plot_circuit["circuit"], plot_circuit["f1"])
    plt.title("Bias Guard: Lowest Circuit F1 Scores")
    plt.xlabel("F1")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "bias_guard_lowest_circuit_f1.png", dpi=160)
    plt.close()

print("\nSTEP C: write bias guard report")

full_ap = float(
    audit_df[
        audit_df["feature_set"].eq("full_state_context")
        & audit_df["threshold_source"].eq("best_f1_on_2022_validation")
    ]["average_precision"].iloc[0]
)
no_schedule_ap = float(
    audit_df[
        audit_df["feature_set"].eq("no_schedule_progress")
        & audit_df["threshold_source"].eq("best_f1_on_2022_validation")
    ]["average_precision"].iloc[0]
)
schedule_dependency_gap = full_ap - no_schedule_ap

guard = {
    "best_feature_set": best_feature_set,
    "best_model_path": str(MODEL_DIR / "pit_window_bias_guard_best_model.joblib"),
    "threshold_selection_rule": "Threshold selected on 2022 validation only, then evaluated on untouched 2023 holdout.",
    "full_vs_no_schedule_average_precision_gap": schedule_dependency_gap,
    "feature_set_audit": audit_df.to_dict(orient="records"),
    "circuit_weak_spots": circuit_metrics.head(15).to_dict(orient="records") if not circuit_metrics.empty else [],
    "driver_weak_spots": driver_metrics.head(15).to_dict(orient="records") if not driver_metrics.empty else [],
    "interpretation": [
        "If no_schedule_progress drops sharply, the model relies heavily on race phase timing.",
        "This is not automatically leakage because lap and tyre age are known at decision time, but it means the model may learn typical strategy windows.",
        "A strong proof layer should report this dependency instead of hiding it.",
        "Thresholds are selected on 2022 validation, not on the 2023 test set, to avoid test-set threshold bias.",
    ],
}
write_json(REPORT_DIR / "bias_guard_audit.json", guard)

lines = [
    "# MDCE Bias and Leakage Guard Audit",
    "",
    "## Purpose",
    "",
    "This audit checks whether the pit-window model is relying too heavily on race-phase timing, driver/circuit identity, or other shortcut signals.",
    "",
    "## Best Bias-Guard Model",
    "",
    f"- Feature set: `{best_feature_set}`",
    f"- Model path: `{MODEL_DIR / 'pit_window_bias_guard_best_model.joblib'}`",
    f"- Threshold selection: `{guard['threshold_selection_rule']}`",
    "",
    "## Schedule Dependency Check",
    "",
    f"- Full feature AP: `{full_ap:.3f}`",
    f"- No schedule/progress AP: `{no_schedule_ap:.3f}`",
    f"- AP gap: `{schedule_dependency_gap:.3f}`",
    "",
    "Interpretation: a large positive gap means the model gets meaningful signal from lap/progress timing. That is allowed because lap is known at decision time, but it must be disclosed because it can reflect typical strategy-window learning.",
    "",
    "## Feature Set Audit",
    "",
    "| Feature Set | Threshold Source | Threshold | AP | ROC-AUC | F1 | Precision | Recall |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for _, row in audit_df.iterrows():
    lines.append(
        f"| {row['feature_set']} | {row['threshold_source']} | {row['selected_threshold']:.2f} | {row['average_precision']:.3f} | {row['roc_auc']:.3f} | {row['f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} |"
    )

lines += [
    "",
    "## Worst Circuit Classification Groups",
    "",
]
if circuit_metrics.empty:
    lines.append("- Not enough circuit rows to compute grouped circuit metrics.")
else:
    for _, row in circuit_metrics.head(10).iterrows():
        lines.append(f"- `{row['circuit']}`: F1 `{row['f1']:.3f}`, AP `{row['average_precision']:.3f}`, rows `{int(row['rows'])}`")

lines += [
    "",
    "## Files Written",
    "",
    f"- `{REPORT_DIR / 'bias_guard_feature_set_audit.csv'}`",
    f"- `{REPORT_DIR / 'bias_guard_validation_threshold_sweeps.csv'}`",
    f"- `{REPORT_DIR / 'bias_guard_circuit_classification_metrics.csv'}`",
    f"- `{REPORT_DIR / 'bias_guard_driver_classification_metrics.csv'}`",
    f"- `{REPORT_DIR / 'bias_guard_audit.json'}`",
    f"- `{CHART_DIR / 'bias_guard_feature_set_ap.png'}`",
    f"- `{CHART_DIR / 'bias_guard_feature_set_f1.png'}`",
    f"- `{CHART_DIR / 'bias_guard_lowest_circuit_f1.png'}`",
]
(REPORT_DIR / "bias_guard_audit.md").write_text("\n".join(lines), encoding="utf-8")

print("\nBIAS GUARD COMPLETE")
print("Best feature set:", best_feature_set)
print("Schedule dependency AP gap:", round(schedule_dependency_gap, 4))
print("Report:", REPORT_DIR / "bias_guard_audit.md")
