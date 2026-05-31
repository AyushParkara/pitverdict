from pathlib import Path
import json
import os

import pandas as pd


ROOT = Path(os.environ.get("MDCE_ROOT", "/content/drive/MyDrive/ibm_project_stuff/MDCE"))
if not ROOT.exists():
    raise FileNotFoundError(
        f"Expected project folder not found: {ROOT}. "
        "This consolidator is folder-locked and will not scan MyDrive."
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


pit = read_json(REPORT_DIR / "pit_final_model_decision.json")
real_multi = read_json(REPORT_DIR / "mdce_real_multidecision_training_summary.json")
safety = read_json(REPORT_DIR / "safety_car_risk_model_metrics.json")
registry = read_csv(REPORT_DIR / "mdce_real_decision_training_registry.csv")

pit_selected = pit.get("selected_model", {})
pit_metrics = pit_selected.get("metrics_2023", {})
stint_action = real_multi.get("stint_action", {})
stint_remaining = real_multi.get("stint_remaining_regression", {})
tyre_choice = real_multi.get("tyre_choice", {})

tyre_metrics = tyre_choice.get("test_metrics") or {}
tyre_macro_f1 = tyre_metrics.get("macro_f1")
tyre_status = tyre_choice.get("status", "not_available")
if tyre_macro_f1 is not None and float(tyre_macro_f1) < 0.50:
    tyre_final_status = "trained_experimental_low_macro_f1"
else:
    tyre_final_status = tyre_status

safety_status = safety.get("status", "not_run_or_not_available")
safety_metrics = safety.get("test_metrics") or {}

# Optional extra artifacts from the safety-car stage.
safety_predictions_path = safety.get("predictions_path")

final_rows = [
    {
        "decision_layer": "pit_timing",
        "final_status": "strong_final_proof_layer",
        "model": pit_selected.get("name"),
        "model_path": pit_selected.get("path"),
        "primary_metric": "F1",
        "primary_metric_value": pit_metrics.get("f1"),
        "secondary_metric": "average_precision",
        "secondary_metric_value": pit_metrics.get("average_precision"),
        "claim_strength": "strong",
        "safe_claim": "Public-data pit-window confidence model; not private optimal strategy.",
    },
    {
        "decision_layer": "stint_length",
        "final_status": "trained_promoted_observed_behavior",
        "model": stint_action.get("best_model"),
        "model_path": str(MODEL_DIR / "stint_action_classifier.joblib"),
        "primary_metric": "macro_f1",
        "primary_metric_value": (stint_action.get("test_metrics") or {}).get("macro_f1"),
        "secondary_metric": "weighted_f1",
        "secondary_metric_value": (stint_action.get("test_metrics") or {}).get("weighted_f1"),
        "claim_strength": "medium_strong",
        "safe_claim": "Predicts observed next-pit horizon buckets, not guaranteed optimal stint length.",
    },
    {
        "decision_layer": "stint_remaining_regression",
        "final_status": "trained_promoted_observed_behavior",
        "model": stint_remaining.get("best_model"),
        "model_path": str(MODEL_DIR / "stint_remaining_regressor.joblib"),
        "primary_metric": "mae_laps",
        "primary_metric_value": (stint_remaining.get("test_metrics") or {}).get("mae_laps"),
        "secondary_metric": "mae_reduction_vs_baseline",
        "secondary_metric_value": (stint_remaining.get("promotion_gate") or {}).get("mae_reduction_vs_baseline"),
        "claim_strength": "medium",
        "safe_claim": "Estimates observed laps until next pit from public data.",
    },
    {
        "decision_layer": "tyre_choice",
        "final_status": tyre_final_status,
        "model": tyre_choice.get("best_model"),
        "model_path": str(MODEL_DIR / "tyre_choice_classifier.joblib") if tyre_choice.get("best_model") else "",
        "primary_metric": "macro_f1",
        "primary_metric_value": tyre_macro_f1,
        "secondary_metric": "weighted_f1",
        "secondary_metric_value": tyre_metrics.get("weighted_f1"),
        "claim_strength": "weak_experimental" if tyre_macro_f1 is not None and float(tyre_macro_f1) < 0.50 else "medium",
        "safe_claim": "Predicts observed next compound after pit stop; current macro F1 is weak, so do not present it as a mature proof layer.",
    },
    {
        "decision_layer": "safety_car_risk",
        "final_status": safety_status,
        "model": safety.get("best_model"),
        "model_path": safety.get("model_path"),
        "primary_metric": "average_precision",
        "primary_metric_value": safety_metrics.get("average_precision"),
        "secondary_metric": "f1",
        "secondary_metric_value": safety_metrics.get("f1"),
        "claim_strength": "conditional_on_promotion_gate",
        "safe_claim": safety.get("safe_claim", "Safety-car risk layer was not available."),
    },
    {
        "decision_layer": "push_conserve",
        "final_status": "blocked_missing_labels",
        "model": "",
        "model_path": "",
        "primary_metric": "",
        "primary_metric_value": None,
        "secondary_metric": "",
        "secondary_metric_value": None,
        "claim_strength": "blocked",
        "safe_claim": "Do not claim trained push/conserve decisions until command or intent labels exist.",
    },
    {
        "decision_layer": "aggressive_safe_strategy",
        "final_status": "blocked_missing_labels",
        "model": "",
        "model_path": "",
        "primary_metric": "",
        "primary_metric_value": None,
        "secondary_metric": "",
        "secondary_metric_value": None,
        "claim_strength": "blocked",
        "safe_claim": "Do not claim trained aggressive/safe decisions until objective outcome labels exist.",
    },
]

final_registry = pd.DataFrame(final_rows)
final_registry.to_csv(REPORT_DIR / "MCDE_FINAL_MODEL_REGISTRY.csv", index=False)

lines = [
    "# MCDE Final Display Report",
    "",
    "## Project",
    "",
    "Motorsport Decision Confidence Engine (MDCE) is a confidence and risk layer for motorsport strategy decisions under uncertain public data.",
    "",
    "The project is no longer only a pit-stop demo. Pit timing is the strongest proof layer, and additional real-labelled layers have been trained where labels are defensible.",
    "",
    "## Final Decision Layers",
    "",
    "| Layer | Status | Model | Primary Metric | Value | Claim Strength |",
    "|---|---|---|---|---:|---|",
]
for row in final_rows:
    value = "" if row["primary_metric_value"] is None else f"{float(row['primary_metric_value']):.3f}"
    lines.append(
        f"| {row['decision_layer']} | {row['final_status']} | {row['model'] or ''} | {row['primary_metric']} | {value} | {row['claim_strength']} |"
    )

lines += [
    "",
    "## Strongest Claims",
    "",
    f"- Pit timing: final model `{pit_selected.get('name')}` with F1 `{pit_metrics.get('f1')}` and AP `{pit_metrics.get('average_precision')}`.",
    f"- Stint length: model `{stint_action.get('best_model')}` with macro F1 `{(stint_action.get('test_metrics') or {}).get('macro_f1')}`.",
    f"- Stint remaining: model `{stint_remaining.get('best_model')}` with MAE `{(stint_remaining.get('test_metrics') or {}).get('mae_laps')}` laps.",
    "",
    "## Careful / Weak Claims",
    "",
    f"- Tyre choice: trained from actual post-pit compound labels, but macro F1 is `{tyre_macro_f1}`. Treat as experimental, not a top proof layer.",
    f"- Safety-car risk: status `{safety_status}`. Use only according to the promotion gate in `safety_car_risk_model_metrics.json`.",
    "- Push/conserve and aggressive/safe strategy remain blocked because the current dataset does not contain honest command/intent/outcome labels.",
    "",
    "## Do Not Claim",
    "",
    "- Do not claim MDCE beats Formula 1 team strategy systems.",
    "- Do not claim the tyre-choice model proves optimal compound choice.",
    "- Do not claim push/conserve or aggressive/safe strategy is trained.",
    "- Do not hide weak circuits, weak labels, or baseline comparisons.",
    "",
    "## Main Files For Display",
    "",
    f"- `{REPORT_DIR / 'MCDE_FINAL_DISPLAY_REPORT.md'}`",
    f"- `{REPORT_DIR / 'MCDE_FINAL_MODEL_REGISTRY.csv'}`",
    f"- `{REPORT_DIR / 'pit_final_model_decision.md'}`",
    f"- `{REPORT_DIR / 'mdce_real_multidecision_training_summary.md'}`",
    f"- `{REPORT_DIR / 'safety_car_risk_training.md'}`",
    f"- `{safety_predictions_path}`" if safety_predictions_path else "- Safety-car predictions CSV: not available",
    f"- `{CHART_DIR / 'real_multidecision_confusion_matrices.png'}`",
    f"- `{CHART_DIR / 'stint_remaining_regression_holdout.png'}`",
    f"- `{CHART_DIR / 'safety_car_risk_confusion_matrix.png'}`",
]

(REPORT_DIR / "MCDE_FINAL_DISPLAY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

checklist = [
    "# MCDE Final Presentation Checklist",
    "",
    "Open only these files during display:",
    "",
    "1. `MCDE_SINGLE_FINAL_PIPELINE.ipynb`",
    "2. `MCDE_FINAL_DISPLAY_REPORT.md`",
    "3. `MCDE_FINAL_MODEL_REGISTRY.csv`",
    "4. Final charts from `outputs/charts`",
    "",
    "Treat all other notebooks as development history, not presentation files.",
]
(REPORT_DIR / "MCDE_FINAL_PRESENTATION_CHECKLIST.md").write_text("\n".join(checklist), encoding="utf-8")

print("FINAL DISPLAY CONSOLIDATION COMPLETE")
print("Final report:", REPORT_DIR / "MCDE_FINAL_DISPLAY_REPORT.md")
print("Final registry:", REPORT_DIR / "MCDE_FINAL_MODEL_REGISTRY.csv")
print(final_registry.to_string(index=False))
