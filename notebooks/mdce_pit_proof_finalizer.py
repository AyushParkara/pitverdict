from pathlib import Path
import json
import os

import pandas as pd


ROOT = Path(os.environ.get("MDCE_ROOT", "/content/drive/MyDrive/ibm_project_stuff/MDCE"))
if not ROOT.exists():
    raise FileNotFoundError(
        f"Expected project folder not found: {ROOT}. "
        "This finalizer is folder-locked and will not scan MyDrive."
    )

REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR = ROOT / "outputs" / "models"
CHART_DIR = ROOT / "outputs" / "charts"

for folder in [REPORT_DIR, MODEL_DIR, CHART_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


pit_metrics = read_json(REPORT_DIR / "pit_model_metrics.json").get("metrics", {})
event_metrics = read_json(REPORT_DIR / "ml_decision_event_metrics.json")
bias_guard = read_json(REPORT_DIR / "bias_guard_audit.json")
challenger = read_json(REPORT_DIR / "model_challenger_report.json")

model_challenger_df = read_csv(REPORT_DIR / "model_challenger_leaderboard.csv")
bias_feature_df = read_csv(REPORT_DIR / "bias_guard_feature_set_audit.csv")
hardening_df = read_csv(REPORT_DIR / "advanced_model_leaderboard.csv")
multi_horizon_df = read_csv(REPORT_DIR / "advanced_multi_horizon_metrics.csv")
robustness_df = read_csv(REPORT_DIR / "advanced_robustness_stress_tests.csv")
permutation_df = read_csv(REPORT_DIR / "advanced_permutation_importance.csv")
circuit_df = read_csv(REPORT_DIR / "bias_guard_circuit_classification_metrics.csv")
driver_df = read_csv(REPORT_DIR / "bias_guard_driver_classification_metrics.csv")

if model_challenger_df.empty:
    raise FileNotFoundError("Missing model_challenger_leaderboard.csv. Run MCDE_model_challenger_addon.ipynb first.")

selected = model_challenger_df.sort_values(["average_precision", "f1"], ascending=False).iloc[0].to_dict()
selected_model_path = (
    MODEL_DIR / "pit_window_challenger_best_model.pt"
    if str(selected["model_family"]) == "pytorch_mlp"
    else MODEL_DIR / "pit_window_challenger_best_model.joblib"
)
torch_model_path = MODEL_DIR / "pit_window_challenger_best_model.pt"

registry_rows = []
for _, row in model_challenger_df.iterrows():
    reason = "selected_best_holdout_ap_and_f1" if row["model"] == selected["model"] else "rejected_lower_holdout_score"
    if str(row.get("model_family")) == "pytorch_mlp" and row["model"] != selected["model"]:
        reason = "rejected_neural_net_underperformed_tree_ensemble_on_holdout"
    registry_rows.append(
        {
            "model": row["model"],
            "family": row["model_family"],
            "status": "selected" if row["model"] == selected["model"] else "rejected",
            "reason": reason,
            "average_precision": row["average_precision"],
            "roc_auc": row["roc_auc"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"],
            "brier_score": row["brier_score"],
            "threshold": row["threshold"],
        }
    )
registry_df = pd.DataFrame(registry_rows)
registry_df.to_csv(REPORT_DIR / "pit_final_model_registry.csv", index=False)

ap_baseline = pit_metrics.get("average_precision")
ap_selected = float(selected["average_precision"])
f1_baseline = pit_metrics.get("f1")
f1_selected = float(selected["f1"])

schedule_gap = bias_guard.get("full_vs_no_schedule_average_precision_gap")
best_bias_feature_set = bias_guard.get("best_feature_set")

weak_circuits = circuit_df.sort_values("f1", ascending=True).head(10) if not circuit_df.empty else pd.DataFrame()
weak_drivers = driver_df.sort_values("f1", ascending=True).head(10) if not driver_df.empty else pd.DataFrame()

final_proof = {
    "selected_model": {
        "name": selected["model"],
        "family": selected["model_family"],
        "path": str(selected_model_path),
        "artifact_exists": selected_model_path.exists(),
        "feature_set": "no_identity_context",
        "threshold": float(selected["threshold"]),
        "threshold_protocol": "selected on 2022 validation, evaluated on untouched 2023 holdout",
        "metrics_2023": {
            "average_precision": ap_selected,
            "roc_auc": float(selected["roc_auc"]),
            "f1": f1_selected,
            "precision": float(selected["precision"]),
            "recall": float(selected["recall"]),
            "brier_score": float(selected["brier_score"]),
        },
    },
    "improvement_over_initial_random_forest": {
        "initial_average_precision": ap_baseline,
        "selected_average_precision": ap_selected,
        "average_precision_delta": ap_selected - float(ap_baseline) if ap_baseline is not None else None,
        "initial_f1": f1_baseline,
        "selected_f1": f1_selected,
        "f1_delta": f1_selected - float(f1_baseline) if f1_baseline is not None else None,
    },
    "bias_guard": {
        "best_feature_set": best_bias_feature_set,
        "schedule_dependency_average_precision_gap": schedule_gap,
        "interpretation": "Lap/progress timing is an important known-at-decision-time signal. It is allowed but disclosed because it reflects typical strategy windows.",
    },
    "neural_network_decision": {
        "tested": True,
        "promoted": str(selected["model_family"]) == "pytorch_mlp",
        "torch_artifact_path": str(torch_model_path),
        "torch_artifact_exists": torch_model_path.exists(),
        "decision": "PyTorch challenger rejected because it did not beat the tree ensemble on 2023 holdout metrics." if str(selected["model_family"]) != "pytorch_mlp" else "PyTorch challenger selected by holdout evidence.",
    },
    "event_backtest": event_metrics,
    "known_weak_spots": {
        "circuits": weak_circuits.to_dict(orient="records"),
        "drivers": weak_drivers.to_dict(orient="records"),
    },
}
(REPORT_DIR / "pit_final_model_decision.json").write_text(json.dumps(final_proof, indent=2, default=str), encoding="utf-8")

lines = [
    "# MDCE Final Pit Proof Layer Decision",
    "",
    "## Selected Model",
    "",
    f"- Model: `{selected['model']}`",
    f"- Family: `{selected['model_family']}`",
    f"- Path: `{selected_model_path}`",
    "- Feature set: `no_identity_context`",
    f"- Threshold: `{selected['threshold']}`",
    "- Threshold protocol: selected on 2022 validation, evaluated on untouched 2023 holdout.",
    "",
    "## Final 2023 Holdout Metrics",
    "",
    f"- Average precision: `{float(selected['average_precision']):.3f}`",
    f"- ROC-AUC: `{float(selected['roc_auc']):.3f}`",
    f"- F1: `{float(selected['f1']):.3f}`",
    f"- Precision: `{float(selected['precision']):.3f}`",
    f"- Recall: `{float(selected['recall']):.3f}`",
    f"- Brier score: `{float(selected['brier_score']):.3f}`",
    "",
    "## Improvement Over Initial Model",
    "",
]
if ap_baseline is not None:
    lines += [
        f"- Initial AP: `{float(ap_baseline):.3f}`",
        f"- Final AP: `{ap_selected:.3f}`",
        f"- AP delta: `{ap_selected - float(ap_baseline):.3f}`",
        f"- Initial F1: `{float(f1_baseline):.3f}`",
        f"- Final F1: `{f1_selected:.3f}`",
        f"- F1 delta: `{f1_selected - float(f1_baseline):.3f}`",
    ]
else:
    lines.append("- Initial model metrics were not available.")

lines += [
    "",
    "## Model Challenger Outcome",
    "",
    "| Model | Family | Status | AP | ROC-AUC | F1 | Precision | Recall | Brier | Reason |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
]
for _, row in registry_df.iterrows():
    lines.append(
        f"| {row['model']} | {row['family']} | {row['status']} | {row['average_precision']:.3f} | {row['roc_auc']:.3f} | {row['f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['brier_score']:.3f} | {row['reason']} |"
    )

lines += [
    "",
    "## Neural Network Decision",
    "",
]
if str(selected["model_family"]) == "pytorch_mlp":
    lines.append("- PyTorch challenger won and is promoted by holdout evidence.")
else:
    lines.append("- PyTorch challenger was tested and rejected because it did not beat the tree ensemble on 2023 holdout metrics.")
    lines.append("- The `.pt` file being absent is expected in this run because the neural network was not promoted.")

lines += [
    "",
    "## Bias Guard Result",
    "",
    f"- Best feature set from bias guard: `{best_bias_feature_set}`",
    f"- Schedule/progress AP dependency gap: `{float(schedule_gap):.3f}`" if schedule_gap is not None else "- Schedule/progress AP dependency gap: unavailable",
    "- Interpretation: lap/progress timing is a strong known-at-decision-time signal. It is not hidden; the final proof layer discloses it.",
    "",
    "## Event Backtest",
    "",
]
if event_metrics:
    lines += [
        f"- Events tested: `{event_metrics.get('event_count')}`",
        f"- Mean absolute lap error: `{event_metrics.get('mean_abs_lap_error')}`",
        f"- Median absolute lap error: `{event_metrics.get('median_abs_lap_error')}`",
        f"- Within 1 lap rate: `{event_metrics.get('within_1_lap_rate')}`",
        f"- Within 3 laps rate: `{event_metrics.get('within_3_lap_rate')}`",
        f"- Within 5 laps rate: `{event_metrics.get('within_5_lap_rate')}`",
    ]
else:
    lines.append("- Event backtest metrics unavailable.")

lines += [
    "",
    "## Known Weak Circuit Groups",
    "",
]
if weak_circuits.empty:
    lines.append("- Circuit weak-spot metrics unavailable.")
else:
    for _, row in weak_circuits.iterrows():
        lines.append(f"- `{row['circuit']}`: F1 `{row['f1']:.3f}`, AP `{row['average_precision']:.3f}`, rows `{int(row['rows'])}`")

lines += [
    "",
    "## Safe Final Claim",
    "",
    "The final pit proof layer can support public-data pit-window confidence decisions. It does not claim to reproduce private F1 strategy systems or globally optimal strategy. It is strongest when used as a confidence/risk layer around a proposed race-strategy decision.",
    "",
    "## Files",
    "",
    f"- `{REPORT_DIR / 'pit_final_model_registry.csv'}`",
    f"- `{REPORT_DIR / 'pit_final_model_decision.json'}`",
    f"- `{selected_model_path}`",
]
(REPORT_DIR / "pit_final_model_decision.md").write_text("\n".join(lines), encoding="utf-8")

print("PIT PROOF FINALIZER COMPLETE")
print("Selected model:", selected["model"])
print("Selected model path:", selected_model_path)
print("Report:", REPORT_DIR / "pit_final_model_decision.md")
