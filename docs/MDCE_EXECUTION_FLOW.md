# PitVerdict Execution Flow (Top-To-Bottom)

This is the operational, production-style flow for MDCE. It is ordered from raw input to outputs.

## A) Inputs

1. Processed MDCE CSV
   - Loaded via `src/data_loader.py` into `LapRecord[]`.
   - Provenance is tracked (real/derived/proxy) and surfaced as warnings.
   - NonCommercial default guard:
     - If the default prepared dataset (`data/processed/mdce_kaggle_weather_tyre_latest.csv`) is detected as NonCommercial (e.g. sidecar `*.metadata.json` has `license_spdx: CC-BY-NC-*`, or metadata is missing/unreadable for the Kaggle-derived default filename), MDCE will not auto-load it unless you opt in with `MDCE_ALLOW_NONCOMMERCIAL_DATA=1`.
   - Cheap input gate (recommended):

```bash
.venv/bin/python tools/validate_mdce_dataset.py --input <your_processed_mdce.csv>
```

2. Scenario flags (optional)
   - `ScenarioFlags` toggles demo failure modes.
   - Applied via `src/scenario_engine.py`.

## B) Core Intelligence Pipeline (Deterministic)

1. Scenario Engine
   - File: `src/scenario_engine.py`
   - Output: mutated `LapRecord[]` + scenario notes.

2. Baseline Strategy
   - File: `src/strategy_engine.py`
   - Output: `StrategyRecommendation` (simple baseline, explainable).

3. Coverage Gap Detection
   - File: `src/coverage_gap_engine.py`
   - Output: additional `TrustIssue[]` when inputs look like placeholders (e.g., derived sector splits, constant gaps).

4. Trust Issues
   - File: `src/trust_engine.py`
   - Output: `TrustIssue[]` + conflict score.

5. Multi-Signal Disagreement (Light Ensemble Reasoning)
   - File: `src/disagreement_engine.py`
   - Output: `TrustIssue[]` when independent signals disagree.

6. Model vs Reality Validation
   - File: `src/model_validation_engine.py`
   - Output: `ModelValidationResult` and (if needed) a scoped `model_deviation` issue.

7. Confidence (Overall + Per-Decision + Breakdown)
   - File: `src/confidence_engine.py`
   - Output: `ConfidenceResult` with:
     - `confidence` + `risk_level`
     - `breakdown` (component scores)
     - `decision_confidence` + `decision_risk_levels`

8. Uncertainty Propagation
   - File: `src/uncertainty_engine.py`
   - Output: `UncertaintyResult` (primary uncertainty + downstream decisions at risk).

9. Decision Impact Simulation
   - File: `src/decision_impact_engine.py`
   - Output: `DecisionImpactResult` (if-wrong loss estimate).
   - Current scope: pit timing.

10. SAFE/AGGRESSIVE Mode Options
   - File: `src/mode_engine.py`
   - Output: two `ModeOption`s (mode-adjusted recommendation + impact).

11. Recommended Mode (System Choice)
   - File: `src/recommended_mode_engine.py`
   - Output: `recommended_mode` (SAFE vs AGGRESSIVE) attached to `AnalysisResult`.

12. Fallback Actions
   - File: `src/fallback_engine.py`
   - Output: action list keyed off issues.

13. Explanation
   - File: `src/explanation_engine.py`
   - Output: deterministic explanation; Granite optional and explanation-only.

## C) Outputs

1. Streamlit app
   - File: `app.py`
   - Displays recommendation, confidence, issues, fallbacks, provenance.

2. CLI artifacts
   - File: `tools/run_mdce_decision.py`
   - Produces JSON + Markdown + append-only run registry CSV.
   - Canonical run registry: `outputs/reports/MDCE_DECISION_RUN_REGISTRY_v2.csv` (stable column order).
   - Contract gate (recommended): validate artifacts to ensure stable JSON/MD/registry schemas.

```bash
.venv/bin/python tools/validate_mdce_outputs.py --json outputs/reports/mdce_decision_run_<timestamp>.json
```
