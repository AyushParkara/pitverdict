from pathlib import Path
import json
import os
import sqlite3

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(os.environ.get("MDCE_ROOT", "/content/drive/MyDrive/ibm_project_stuff/MDCE"))
if not ROOT.exists():
    raise FileNotFoundError(
        f"Expected project folder not found: {ROOT}. "
        "This notebook is folder-locked and will not scan MyDrive."
    )

DB_PATH = ROOT / "databases" / "mdce_f1.db"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR = ROOT / "outputs" / "models"
CHART_DIR = ROOT / "outputs" / "charts"

for folder in [REPORT_DIR, MODEL_DIR, CHART_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LABEL_NAMES = {"TrackStatus", "IsSafetyCar", "IsVSC", "Risk_SafetyCar"}


def clean_name(value):
    return str(value).strip()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print("wrote:", path)


def find_column(columns, candidates):
    lower_map = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for col in columns:
        lower = str(col).lower()
        if any(candidate.lower() in lower for candidate in candidates):
            return col
    return None


def find_safety_table(conn):
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
    rows = []
    for table in tables["name"].astype(str):
        try:
            cols = pd.read_sql_query(f'PRAGMA table_info("{table}")', conn)["name"].astype(str).tolist()
            count = int(pd.read_sql_query(f'SELECT COUNT(*) AS rows FROM "{table}"', conn).iloc[0]["rows"])
        except Exception:
            continue
        col_set = {col.lower() for col in cols}
        score = 0
        if "risk_safetycar".lower() in col_set:
            score += 10
        if "issafetycar".lower() in col_set:
            score += 5
        if "isvsc".lower() in col_set:
            score += 5
        if "trackstatus".lower() in col_set:
            score += 2
        if score:
            rows.append({"table": table, "rows": count, "score": score, "columns": cols})
    if not rows:
        return None, pd.DataFrame()
    audit = pd.DataFrame([{k: v for k, v in row.items() if k != "columns"} for row in rows]).sort_values(
        ["score", "rows"], ascending=False
    )
    best_table = audit.iloc[0]["table"]
    return best_table, audit


def derive_year(df):
    year_col = find_column(df.columns, ["year", "season"])
    if year_col is not None:
        year_values = pd.to_numeric(df[year_col], errors="coerce")
        if year_values.notna().sum() > 0:
            return year_values.astype("Int64")
    date_col = find_column(df.columns, ["date", "sessiondate", "eventdate"])
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().sum() > 0:
            return dates.dt.year.astype("Int64")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")


def build_target(df):
    cols = df.columns
    risk_col = find_column(cols, ["Risk_SafetyCar"])
    is_sc_col = find_column(cols, ["IsSafetyCar"])
    is_vsc_col = find_column(cols, ["IsVSC"])

    target_parts = []
    target_sources = []
    if risk_col is not None:
        risk = pd.to_numeric(df[risk_col], errors="coerce")
        if risk.notna().sum() > 0:
            if risk.max(skipna=True) <= 1:
                risk_label = (risk >= 0.5).astype("Int64")
            else:
                positive = risk[risk > 0]
                threshold = float(positive.median()) if len(positive) else 1.0
                risk_label = (risk >= threshold).astype("Int64")
            target_parts.append(risk_label.fillna(0).astype(int))
            target_sources.append(str(risk_col))
    if is_sc_col is not None:
        target_parts.append(pd.to_numeric(df[is_sc_col], errors="coerce").fillna(0).astype(int).clip(0, 1))
        target_sources.append(str(is_sc_col))
    if is_vsc_col is not None:
        target_parts.append(pd.to_numeric(df[is_vsc_col], errors="coerce").fillna(0).astype(int).clip(0, 1))
        target_sources.append(str(is_vsc_col))

    if not target_parts:
        return None, []
    target = target_parts[0].copy()
    for part in target_parts[1:]:
        target = ((target.astype(int) + part.astype(int)) > 0).astype(int)
    return target, target_sources


def select_features(df, target_sources, no_identity=False):
    blocked_tokens = [
        "risk_safetycar",
        "issafetycar",
        "isvsc",
        "trackstatus",
        "safety",
        "vsc",
    ]
    identity_tokens = ["driver", "team", "constructor", "event", "grand_prix", "session", "raceid", "driverid"]
    feature_cols = []
    for col in df.columns:
        lower = str(col).lower()
        if col in target_sources or any(token in lower for token in blocked_tokens):
            continue
        if no_identity and any(token in lower for token in identity_tokens):
            continue
        if lower in ["target", "target_year"]:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        feature_cols.append(col)

    numeric_features = []
    categorical_features = []
    for col in feature_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric_ratio = numeric.notna().mean()
        unique_count = df[col].nunique(dropna=True)
        if numeric_ratio >= 0.80:
            numeric_features.append(col)
        elif unique_count <= 50:
            categorical_features.append(col)

    return numeric_features, categorical_features


def make_model(estimator, numeric_features, categorical_features):
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
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
                ),
            ),
            ("classifier", estimator),
        ]
    )


def score_at_threshold(y_true, proba, threshold, model_name):
    pred = (proba >= threshold).astype(int)
    result = {
        "model": model_name,
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, proba)),
    }
    result["average_precision"] = float(average_precision_score(y_true, proba)) if len(set(y_true)) == 2 else None
    result["roc_auc"] = float(roc_auc_score(y_true, proba)) if len(set(y_true)) == 2 else None
    return result


def choose_threshold(y_valid, proba_valid, model_name):
    rows = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        rows.append(score_at_threshold(y_valid, proba_valid, threshold, model_name))
    sweep = pd.DataFrame(rows)
    valid = sweep[sweep["predicted_positive_rate"] > 0].copy()
    if valid.empty:
        valid = sweep
    selected = valid.sort_values(["f1", "average_precision"], ascending=False).iloc[0].to_dict()
    return float(selected["threshold"]), sweep


print("STEP SC-1: locate safety-car/risk table")
if not DB_PATH.exists():
    raise FileNotFoundError(f"Missing database: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
table_name, table_audit = find_safety_table(conn)
table_audit.to_csv(REPORT_DIR / "safety_car_training_table_audit.csv", index=False)
if table_name is None:
    conn.close()
    summary = {
        "status": "blocked_no_safety_car_label_table_found",
        "safe_claim": "Safety-car/risk training was not run because no table with IsSafetyCar/IsVSC/Risk_SafetyCar labels was found.",
    }
    write_json(REPORT_DIR / "safety_car_risk_model_metrics.json", summary)
    raise RuntimeError(summary["status"])
print("selected table:", table_name)
df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
conn.close()
print("safety table shape:", df.shape)

target, target_sources = build_target(df)
if target is None:
    summary = {
        "status": "blocked_no_safety_target_columns",
        "selected_table": table_name,
        "safe_claim": "Safety-car/risk training was not run because target columns could not be derived.",
    }
    write_json(REPORT_DIR / "safety_car_risk_model_metrics.json", summary)
    raise RuntimeError(summary["status"])

df = df.copy()
df["target_safety_risk"] = target.astype(int)
df["target_year"] = derive_year(df)
df = df.dropna(subset=["target_year"]).copy()
df["target_year"] = df["target_year"].astype(int)

year_counts = df.groupby("target_year")["target_safety_risk"].agg(["count", "sum", "mean"]).reset_index()
year_counts.to_csv(REPORT_DIR / "safety_car_year_target_distribution.csv", index=False)
print(year_counts.to_string(index=False))

years = sorted(df["target_year"].dropna().unique())
if len(years) < 3:
    summary = {
        "status": "blocked_need_at_least_three_years_for_temporal_split",
        "selected_table": table_name,
        "years": [int(x) for x in years],
        "target_sources": target_sources,
        "safe_claim": "Safety-car/risk model was blocked because strict temporal validation needs at least three seasons.",
    }
    write_json(REPORT_DIR / "safety_car_risk_model_metrics.json", summary)
    raise RuntimeError(summary["status"])

test_year = int(years[-1])
valid_year = int(years[-2])
train_years = [int(y) for y in years[:-2]]
train_df = df[df["target_year"].isin(train_years)].copy()
valid_df = df[df["target_year"].eq(valid_year)].copy()
test_df = df[df["target_year"].eq(test_year)].copy()

if train_df["target_safety_risk"].sum() < 10 or valid_df["target_safety_risk"].sum() < 3 or test_df["target_safety_risk"].sum() < 3:
    summary = {
        "status": "blocked_not_enough_positive_safety_events_for_strict_training",
        "selected_table": table_name,
        "train_positive": int(train_df["target_safety_risk"].sum()),
        "valid_positive": int(valid_df["target_safety_risk"].sum()),
        "test_positive": int(test_df["target_safety_risk"].sum()),
        "target_sources": target_sources,
    }
    write_json(REPORT_DIR / "safety_car_risk_model_metrics.json", summary)
    raise RuntimeError(summary["status"])

feature_sets = {}
for feature_set_name, no_identity in [("no_leakage_context", False), ("no_identity_no_leakage_context", True)]:
    numeric_features, categorical_features = select_features(df, target_sources, no_identity=no_identity)
    feature_sets[feature_set_name] = {
        "numeric": numeric_features[:80],
        "categorical": categorical_features[:20],
    }

candidates = {
    "rf_500_leaf5_balanced": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "extra_trees_700_leaf5_balanced": ExtraTreesClassifier(
        n_estimators=700,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "logistic_regression_balanced": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        # Use a solver that supports sparse matrices from OneHotEncoder.
        solver="saga",
        n_jobs=-1,
    ),
}

print("STEP SC-2: train safety-car/risk candidates")
validation_rows = []
threshold_sweeps = []
for feature_set_name, feature_config in feature_sets.items():
    features = feature_config["numeric"] + feature_config["categorical"]
    if not features:
        continue
    for candidate_name, estimator in candidates.items():
        model_name = f"{feature_set_name}_{candidate_name}"
        print("validating:", model_name, "features:", len(features))
        try:
            model = make_model(estimator, feature_config["numeric"], feature_config["categorical"])
            model.fit(train_df[features], train_df["target_safety_risk"])
            valid_proba = model.predict_proba(valid_df[features])[:, 1]
            threshold, sweep = choose_threshold(valid_df["target_safety_risk"].astype(int).values, valid_proba, model_name)
            sweep["candidate_model"] = model_name
            threshold_sweeps.append(sweep)
            row = score_at_threshold(valid_df["target_safety_risk"].astype(int).values, valid_proba, threshold, model_name)
            row["feature_set"] = feature_set_name
            row["feature_count"] = len(features)
            validation_rows.append(row)
        except Exception as exc:
            # Don't crash the whole stage because one candidate failed.
            print("candidate failed:", model_name, "error:", exc)
            continue

validation_df = pd.DataFrame(validation_rows).sort_values(["average_precision", "f1"], ascending=False)
if validation_df.empty:
    raise RuntimeError("No safety-car/risk candidate could be trained.")
validation_df.to_csv(REPORT_DIR / "safety_car_validation_leaderboard.csv", index=False)
pd.concat(threshold_sweeps, ignore_index=True).to_csv(REPORT_DIR / "safety_car_threshold_sweeps.csv", index=False)

best = validation_df.iloc[0].to_dict()
best_feature_config = feature_sets[best["feature_set"]]
best_features = best_feature_config["numeric"] + best_feature_config["categorical"]
best_candidate_key = best["model"].replace(f"{best['feature_set']}_", "")
final_estimator = candidates[best_candidate_key]
train_full = pd.concat([train_df, valid_df], ignore_index=True)
final_model = make_model(final_estimator, best_feature_config["numeric"], best_feature_config["categorical"])
final_model.fit(train_full[best_features], train_full["target_safety_risk"])
test_proba = final_model.predict_proba(test_df[best_features])[:, 1]
test_metrics = score_at_threshold(test_df["target_safety_risk"].astype(int).values, test_proba, best["threshold"], best["model"])

majority_pred_rate = float(train_full["target_safety_risk"].mean())
baseline_proba = np.full(len(test_df), majority_pred_rate)
baseline_metrics = score_at_threshold(test_df["target_safety_risk"].astype(int).values, baseline_proba, 0.5, "base_rate_baseline")

predictions = test_df[["target_year", "target_safety_risk"]].copy()
for col in ["EventName", "event", "Event", "Circuit", "TrackName", "Driver", "Team"]:
    actual = find_column(test_df.columns, [col])
    if actual is not None and actual not in predictions.columns:
        predictions[actual] = test_df[actual]
predictions["safety_risk_probability"] = test_proba
predictions["prediction"] = (test_proba >= float(best["threshold"])).astype(int)

# Use the actual holdout year in the filename. The previous hardcoded 2023 name was misleading
# because the temporal split is derived from whatever years exist in the dataset.
predictions_path = REPORT_DIR / f"safety_car_risk_predictions_{test_year}.csv"
predictions.to_csv(predictions_path, index=False)

model_path = MODEL_DIR / "safety_car_risk_classifier.joblib"
joblib.dump(final_model, model_path)

cm = confusion_matrix(test_df["target_safety_risk"].astype(int), predictions["prediction"], labels=[0, 1])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["no risk", "risk"], yticklabels=["no risk", "risk"])
plt.title("Safety-Car/Risk Classifier: Holdout Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(CHART_DIR / "safety_car_risk_confusion_matrix.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
validation_df.head(10).sort_values("average_precision").plot.barh(x="model", y="average_precision", legend=False)
plt.xlabel("Validation average precision")
plt.title("Safety-Car/Risk Candidate Leaderboard")
plt.tight_layout()
plt.savefig(CHART_DIR / "safety_car_risk_leaderboard.png", dpi=160)
plt.close()

promotion_gate = {
    "gate": "average_precision_gain_vs_base_rate >= 0.05 and holdout_f1 >= 0.20",
    "average_precision_gain_vs_base_rate": float(test_metrics["average_precision"] - test_metrics["positive_rate"]),
    "promoted": bool((test_metrics["average_precision"] - test_metrics["positive_rate"]) >= 0.05 and test_metrics["f1"] >= 0.20),
}

summary = {
    "status": "trained_and_promoted" if promotion_gate["promoted"] else "trained_not_promoted_gate_failed",
    "selected_table": table_name,
    "target_sources": target_sources,
    "target_definition": "target is positive when Risk_SafetyCar, IsSafetyCar, or IsVSC indicates safety-car/VSC/risk context.",
    "train_years": train_years,
    "validation_year": valid_year,
    "test_year": test_year,
    "best_model": best["model"],
    "feature_set": best["feature_set"],
    "threshold": float(best["threshold"]),
    "feature_count": int(len(best_features)),
    "model_path": str(model_path),
    "validation_leaderboard": validation_df.to_dict(orient="records"),
    "test_metrics": test_metrics,
    "baseline_metrics": baseline_metrics,
    "promotion_gate": promotion_gate,
    "predictions_path": str(predictions_path),
    "safe_claim": "This is a safety-car/risk context classifier from labelled public Mendeley data. It is not yet a complete pit-under-safety-car response optimizer.",
}
write_json(REPORT_DIR / "safety_car_risk_model_metrics.json", summary)

lines = [
    "# MDCE Safety-Car/Risk Training",
    "",
    "## Purpose",
    "",
    "This trains a real labelled safety-car/risk context model from tables containing `TrackStatus`, `IsSafetyCar`, `IsVSC`, and `Risk_SafetyCar`.",
    "",
    "## Selected Model",
    "",
    f"- Table: `{table_name}`",
    f"- Model: `{best['model']}`",
    f"- Feature set: `{best['feature_set']}`",
    f"- Threshold: `{float(best['threshold']):.2f}`",
    f"- Model path: `{model_path}`",
    f"- Train years: `{train_years}`",
    f"- Validation year: `{valid_year}`",
    f"- Test year: `{test_year}`",
    "",
    "## Holdout Metrics",
    "",
    f"- Positive rate: `{test_metrics['positive_rate']:.3f}`",
    f"- Average precision: `{test_metrics['average_precision']:.3f}`",
    f"- ROC-AUC: `{test_metrics['roc_auc']:.3f}`",
    f"- F1: `{test_metrics['f1']:.3f}`",
    f"- Precision: `{test_metrics['precision']:.3f}`",
    f"- Recall: `{test_metrics['recall']:.3f}`",
    f"- Brier score: `{test_metrics['brier_score']:.3f}`",
    "",
    "## Promotion Gate",
    "",
    f"- Gate: `{promotion_gate['gate']}`",
    f"- AP gain vs base rate: `{promotion_gate['average_precision_gain_vs_base_rate']:.3f}`",
    f"- Promoted: `{promotion_gate['promoted']}`",
    "",
    "## Safe Claim",
    "",
    summary["safe_claim"],
    "",
    "## Files Written",
    "",
    f"- `{REPORT_DIR / 'safety_car_risk_model_metrics.json'}`",
    f"- `{REPORT_DIR / 'safety_car_validation_leaderboard.csv'}`",
    f"- `{REPORT_DIR / 'safety_car_threshold_sweeps.csv'}`",
    f"- `{predictions_path}`",
    f"- `{CHART_DIR / 'safety_car_risk_confusion_matrix.png'}`",
    f"- `{CHART_DIR / 'safety_car_risk_leaderboard.png'}`",
]
(REPORT_DIR / "safety_car_risk_training.md").write_text("\n".join(lines), encoding="utf-8")

print("SAFETY CAR RISK TRAINING COMPLETE")
print("Status:", summary["status"])
print("Selected model:", best["model"])
print("Average precision:", round(test_metrics["average_precision"], 4))
print("F1:", round(test_metrics["f1"], 4))
print("Report:", REPORT_DIR / "safety_car_risk_training.md")
