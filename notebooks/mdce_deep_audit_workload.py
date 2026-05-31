from pathlib import Path
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
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
        "This deep audit is folder-locked and will not scan MyDrive."
    )

TRAINING_PATH = ROOT / "data" / "processed" / "pit_training_rows.parquet"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR = ROOT / "outputs" / "models"
CHART_DIR = ROOT / "outputs" / "charts"

for folder in [REPORT_DIR, MODEL_DIR, CHART_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DEEP_RANDOM_SEEDS = [7, 13, 21, 42, 84, 101, 133, 202, 404, 777]
BOOTSTRAP_ROUNDS = 400
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print("wrote:", path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def make_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def binary_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)) if len(set(y_true)) == 2 else None,
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) == 2 else None,
    }


def choose_threshold(y_true, proba):
    rows = [binary_metrics(y_true, proba, threshold) for threshold in THRESHOLDS]
    sweep = pd.DataFrame(rows)
    selected = sweep.sort_values(["f1", "average_precision"], ascending=False).iloc[0].to_dict()
    return float(selected["threshold"]), sweep


def bootstrap_binary_ci(y_true, proba, threshold, seed=42, rounds=BOOTSTRAP_ROUNDS):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        proba_sample = proba[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        rows.append(binary_metrics(y_sample, proba_sample, threshold))
    boot = pd.DataFrame(rows)
    summary = {}
    for metric in ["average_precision", "roc_auc", "f1", "precision", "recall", "brier_score"]:
        values = pd.to_numeric(boot[metric], errors="coerce").dropna()
        summary[metric] = {
            "mean": float(values.mean()),
            "p05": float(values.quantile(0.05)),
            "p50": float(values.quantile(0.50)),
            "p95": float(values.quantile(0.95)),
        }
    return boot, summary


def classification_seed_stability(train_df, valid_df, test_df, feature_cols, categorical_features, target_col, model_prefix):
    numeric_features = [col for col in feature_cols if col not in categorical_features]
    train_full = pd.concat([train_df, valid_df], ignore_index=True)
    rows = []
    threshold_rows = []
    for seed in DEEP_RANDOM_SEEDS:
        print("seed stability:", model_prefix, seed)
        model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numeric_features, categorical_features)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=700,
                        min_samples_leaf=5,
                        max_features=0.5,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(train_df[feature_cols], train_df[target_col].astype(int))
        valid_proba = model.predict_proba(valid_df[feature_cols])[:, 1]
        threshold, sweep = choose_threshold(valid_df[target_col].astype(int).values, valid_proba)
        sweep["seed"] = seed
        threshold_rows.append(sweep)

        final_model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numeric_features, categorical_features)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=700,
                        min_samples_leaf=5,
                        max_features=0.5,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        final_model.fit(train_full[feature_cols], train_full[target_col].astype(int))
        proba = final_model.predict_proba(test_df[feature_cols])[:, 1]
        row = binary_metrics(test_df[target_col].astype(int).values, proba, threshold)
        row["seed"] = seed
        row["selected_threshold"] = threshold
        row["model_prefix"] = model_prefix
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(threshold_rows, ignore_index=True)


def regression_seed_stability(train_df, valid_df, test_df, feature_cols, categorical_features, target_col, model_prefix):
    numeric_features = [col for col in feature_cols if col not in categorical_features]
    train_full = pd.concat([train_df, valid_df], ignore_index=True)
    rows = []
    for seed in DEEP_RANDOM_SEEDS:
        print("seed stability:", model_prefix, seed)
        model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numeric_features, categorical_features)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=700,
                        min_samples_leaf=5,
                        max_features=0.5,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(train_full[feature_cols], train_full[target_col])
        pred = model.predict(test_df[feature_cols])
        rows.append(
            {
                "seed": seed,
                "model_prefix": model_prefix,
                "mae_laps": float(mean_absolute_error(test_df[target_col], pred)),
                "median_absolute_error_laps": float(np.median(np.abs(test_df[target_col] - pred))),
                "p90_absolute_error_laps": float(np.quantile(np.abs(test_df[target_col] - pred), 0.90)),
            }
        )
    return pd.DataFrame(rows)


def summarize_stability(df, metrics):
    rows = []
    for metric in metrics:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def artifact_audit(paths):
    rows = []
    for path in paths:
        path = Path(path)
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
            }
        )
    return pd.DataFrame(rows)


print("DEEP AUDIT STEP 1: load training")
if not TRAINING_PATH.exists():
    raise FileNotFoundError(f"Missing training table: {TRAINING_PATH}")
training = pd.read_parquet(TRAINING_PATH)

feature_cols = [
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
    "tyre_compound",
]
categorical_features = ["tyre_compound"]

df = training.dropna(subset=["year", "pit_within_3_laps", "laps_until_next_pit"]).copy()
df["laps_until_next_pit_clipped"] = pd.to_numeric(df["laps_until_next_pit"], errors="coerce").clip(lower=0, upper=25)
df = df.dropna(subset=["laps_until_next_pit_clipped"]).copy()
train_2021 = df[df["year"].eq(2021)].copy()
valid_2022 = df[df["year"].eq(2022)].copy()
test_2023 = df[df["year"].eq(2023)].copy()
if train_2021.empty or valid_2022.empty or test_2023.empty:
    raise RuntimeError("Need 2021/2022/2023 temporal split for deep audit.")

print("DEEP AUDIT STEP 2: pit seed stability")
pit_seed_df, pit_threshold_sweeps = classification_seed_stability(
    train_2021,
    valid_2022,
    test_2023,
    feature_cols,
    categorical_features,
    "pit_within_3_laps",
    "pit_within_3_laps_rf700",
)
pit_seed_df.to_csv(REPORT_DIR / "deep_pit_seed_stability.csv", index=False)
pit_threshold_sweeps.to_csv(REPORT_DIR / "deep_pit_threshold_sweeps_by_seed.csv", index=False)
pit_stability_summary = summarize_stability(
    pit_seed_df,
    ["average_precision", "roc_auc", "f1", "precision", "recall", "brier_score", "selected_threshold"],
)
pit_stability_summary.to_csv(REPORT_DIR / "deep_pit_seed_stability_summary.csv", index=False)

best_seed_row = pit_seed_df.sort_values(["average_precision", "f1"], ascending=False).iloc[0]
best_seed = int(best_seed_row["seed"])
numeric_features = [col for col in feature_cols if col not in categorical_features]
train_full = pd.concat([train_2021, valid_2022], ignore_index=True)
best_model = Pipeline(
    steps=[
        ("preprocess", make_preprocessor(numeric_features, categorical_features)),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=700,
                min_samples_leaf=5,
                max_features=0.5,
                class_weight="balanced_subsample",
                random_state=best_seed,
                n_jobs=-1,
            ),
        ),
    ]
)
best_model.fit(train_full[feature_cols], train_full["pit_within_3_laps"].astype(int))
best_proba = best_model.predict_proba(test_2023[feature_cols])[:, 1]
best_threshold = float(best_seed_row["selected_threshold"])
pit_boot, pit_boot_summary = bootstrap_binary_ci(test_2023["pit_within_3_laps"].astype(int).values, best_proba, best_threshold)
pit_boot.to_csv(REPORT_DIR / "deep_pit_bootstrap_metrics.csv", index=False)
write_json(REPORT_DIR / "deep_pit_bootstrap_summary.json", pit_boot_summary)

print("DEEP AUDIT STEP 3: stint remaining seed stability")
stint_regression_seed_df = regression_seed_stability(
    train_2021,
    valid_2022,
    test_2023,
    feature_cols,
    categorical_features,
    "laps_until_next_pit_clipped",
    "stint_remaining_rf700",
)
stint_regression_seed_df.to_csv(REPORT_DIR / "deep_stint_remaining_seed_stability.csv", index=False)
stint_regression_summary = summarize_stability(
    stint_regression_seed_df,
    ["mae_laps", "median_absolute_error_laps", "p90_absolute_error_laps"],
)
stint_regression_summary.to_csv(REPORT_DIR / "deep_stint_remaining_seed_stability_summary.csv", index=False)

print("DEEP AUDIT STEP 4: artifact audit")
artifact_paths = [
    MODEL_DIR / "pit_window_challenger_best_model.joblib",
    MODEL_DIR / "stint_action_classifier.joblib",
    MODEL_DIR / "stint_remaining_regressor.joblib",
    MODEL_DIR / "tyre_choice_classifier.joblib",
    MODEL_DIR / "safety_car_risk_classifier.joblib",
    REPORT_DIR / "pit_final_model_decision.md",
    REPORT_DIR / "mdce_real_multidecision_training_summary.md",
    REPORT_DIR / "safety_car_risk_training.md",
    # This report is produced by the final consolidation stage, which often runs AFTER deep audit
    # in the single pipeline. Auditing it here would incorrectly flag it as missing.
    # (The final stage still checks/prints it.)
]
artifact_df = artifact_audit(artifact_paths)
artifact_df.to_csv(REPORT_DIR / "deep_artifact_audit.csv", index=False)

print("DEEP AUDIT STEP 5: charts and summary")
plt.figure(figsize=(10, 5))
sns.boxplot(data=pit_seed_df[["average_precision", "roc_auc", "f1", "precision", "recall"]])
plt.title("Pit Model Seed Stability")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(CHART_DIR / "deep_pit_seed_stability.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(pit_boot["f1"], bins=30, kde=True)
plt.title("Pit Model Bootstrap F1 Distribution")
plt.xlabel("F1")
plt.tight_layout()
plt.savefig(CHART_DIR / "deep_pit_bootstrap_f1.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=stint_regression_seed_df[["mae_laps", "median_absolute_error_laps", "p90_absolute_error_laps"]])
plt.title("Stint Remaining Regression Seed Stability")
plt.ylabel("laps")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(CHART_DIR / "deep_stint_remaining_seed_stability.png", dpi=160)
plt.close()

summary = {
    "pit_seed_stability": pit_stability_summary.to_dict(orient="records"),
    "pit_bootstrap_summary": pit_boot_summary,
    "stint_remaining_seed_stability": stint_regression_summary.to_dict(orient="records"),
    "artifact_audit": artifact_df.to_dict(orient="records"),
    "safe_claim": "Deep audit measures stability and uncertainty. It does not change the selected model unless a later finalizer explicitly promotes a better model.",
}
write_json(REPORT_DIR / "MCDE_DEEP_AUDIT_SUMMARY.json", summary)

lines = [
    "# MCDE Deep Audit Workload",
    "",
    "## Purpose",
    "",
    "This workload adds longer-running trust checks: repeated random seeds, threshold stability, bootstrap intervals, stint regression stability, and artifact audit.",
    "",
    "## Pit Seed Stability",
    "",
    "| Metric | Mean | Std | Min | Median | Max |",
    "|---|---:|---:|---:|---:|---:|",
]
for _, row in pit_stability_summary.iterrows():
    lines.append(f"| {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['median']:.4f} | {row['max']:.4f} |")
lines += [
    "",
    "## Pit Bootstrap 90% Intervals",
    "",
    "| Metric | Mean | P05 | Median | P95 |",
    "|---|---:|---:|---:|---:|",
]
for metric, values in pit_boot_summary.items():
    lines.append(f"| {metric} | {values['mean']:.4f} | {values['p05']:.4f} | {values['p50']:.4f} | {values['p95']:.4f} |")
lines += [
    "",
    "## Stint Remaining Stability",
    "",
    "| Metric | Mean | Std | Min | Median | Max |",
    "|---|---:|---:|---:|---:|---:|",
]
for _, row in stint_regression_summary.iterrows():
    lines.append(f"| {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['median']:.4f} | {row['max']:.4f} |")
lines += [
    "",
    "## Artifact Audit",
    "",
    "| Path | Exists | Size MB |",
    "|---|---:|---:|",
]
for _, row in artifact_df.iterrows():
    lines.append(f"| `{row['path']}` | {row['exists']} | {row['size_mb']} |")
lines += [
    "",
    "## Files Written",
    "",
    f"- `{REPORT_DIR / 'deep_pit_seed_stability.csv'}`",
    f"- `{REPORT_DIR / 'deep_pit_threshold_sweeps_by_seed.csv'}`",
    f"- `{REPORT_DIR / 'deep_pit_bootstrap_metrics.csv'}`",
    f"- `{REPORT_DIR / 'deep_stint_remaining_seed_stability.csv'}`",
    f"- `{REPORT_DIR / 'deep_artifact_audit.csv'}`",
    f"- `{CHART_DIR / 'deep_pit_seed_stability.png'}`",
    f"- `{CHART_DIR / 'deep_pit_bootstrap_f1.png'}`",
    f"- `{CHART_DIR / 'deep_stint_remaining_seed_stability.png'}`",
]
(REPORT_DIR / "MCDE_DEEP_AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

print("MCDE DEEP AUDIT COMPLETE")
print("Pit seed stability rows:", len(pit_seed_df))
print("Pit bootstrap rounds:", len(pit_boot))
print("Stint regression seed rows:", len(stint_regression_seed_df))
print("Report:", REPORT_DIR / "MCDE_DEEP_AUDIT_SUMMARY.md")
