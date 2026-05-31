# PitVerdict Master Document

This is the single canonical file for the entire PitVerdict project.

It contains:

- what MDCE is and is not
- how the codebase is structured
- how to run (app, CLI, Colab)
- data provenance and safe-claim boundaries
- what we have done so far (worklog summary)
- what remains (production TODO)

Operational reference:

- `MDCE_EXECUTION_FLOW.md` (ordered pipeline flow)

If anything conflicts across docs, treat **this file** as the source of truth.

## 1) Product Definition

Motorsport Decision Confidence Engine (MDCE) is a **decision-confidence / trust layer** for motorsport strategy decisions under uncertain and incomplete data.

MDCE focuses on controlled, defensible outputs:

- a baseline strategy recommendation (simple, transparent)
- confidence score + risk level
- explicit trust issues (what is wrong / missing / conflicting)
- fallback actions when signals disagree
- provenance: which inputs were real, derived, or proxy

Impact simulation semantics:

- MDCE impact simulation is **not** a full race simulator. It is a bounded heuristic layer for risk communication.
- `decision_impact` is the primary pit-timing impact estimate (kept for the original demo surface area).
- `decision_impacts` is a list of additional heuristic impact estimates for other decision domains (currently includes `push_vs_conserve`).

### What MDCE Is Not

MDCE is not:

- a full-race simulator or optimal strategy solver
- private F1 team telemetry validation
- a claim to beat real F1 team strategy systems
- a "magic AI" that generates unsupported decisions

## 2) Top-Level Architecture

Deterministic pipeline (used by both Streamlit and CLI):

```text
CSV (processed public or fallback demo)
  -> Data Loader (schema + provenance)
  -> Scenario Engine (demo failure modes)
  -> Baseline Strategy Engine (simple recommendation)
  -> Coverage Gap Engine (detect placeholder-like inputs)
  -> Model Validation (model vs reality deviation)
  -> Trust Engine (issues + penalties)
  -> Confidence Engine (confidence + risk)
  -> Uncertainty Engine (primary uncertainty + downstream impact)
  -> Decision Impact Engine (if-wrong loss estimates)
  -> Mode Engine (SAFE/AGGRESSIVE options)
  -> Fallback Engine (actions)
  -> Explanation Engine (Granite optional; otherwise deterministic)
```

Key rule:

> Python computes the evidence and the decision outputs. Granite, if enabled, only explains already-computed evidence.

## 3) Main Entry Points (What To Run)

If you only read one file in this repo, read this file (`MDCE_MASTER.md`).

### A) Streamlit UI

- File: `app.py`
- Run:

```bash
streamlit run app.py
```

If you want to use the prepared Kaggle-derived dataset as the default in the app, you must opt in (it is recorded as CC BY-NC 4.0):

```bash
export MDCE_ALLOW_NONCOMMERCIAL_DATA=1
```

Data behavior:

- If `data/processed/mdce_kaggle_weather_tyre_latest.csv` exists: it is used by default only when its license allows that (or you explicitly opt in for NonCommercial data via `MDCE_ALLOW_NONCOMMERCIAL_DATA=1`).
- Otherwise: falls back to `data/sample_race.csv` (offline demo fallback).
- You can always upload a processed MDCE CSV from the sidebar.

UI quality note:

- The UI shows a top-level "Data Quality" banner driven by detected trust issues (coverage gaps, missing telemetry, model deviation), conflict level, risk level, and recommended mode. This is the primary demo surface for the "trust layer" concept.

### B) Headless CLI (Production-Style)

- File: `tools/run_mdce_decision.py`
- Run (default outputs):

```bash
python3 tools/run_mdce_decision.py --output-dir outputs/reports
```

Note: use the repo virtualenv Python (`.venv/bin/python`) if you installed dependencies into the venv.

CLI writes:

- JSON: `outputs/reports/mdce_decision_run_<timestamp>.json`
- Markdown: `outputs/reports/mdce_decision_run_<timestamp>.md`
- Append-only registry: `outputs/reports/MDCE_DECISION_RUN_REGISTRY.csv`

Registry note:

- If the registry already exists with an older header, the runner will also write a full-schema registry alongside it:
  `outputs/reports/MDCE_DECISION_RUN_REGISTRY_v2.csv`

Artifact contract note (stability goal):

- The JSON/Markdown/registry outputs are treated as stable contracts.
- `decision.explanation` is required and is always present (Granite is explanation-only; otherwise deterministic).
- JSON includes persisted artifact paths under `payload.artifacts.*` (json/markdown/registry paths).
- The canonical append-only audit registry is `outputs/reports/MDCE_DECISION_RUN_REGISTRY_v2.csv` (stable column order).
- `run_id` is microsecond-precision to avoid artifact overwrite when multiple runs happen in the same second.
- To prevent silent regressions, validate artifacts after a CLI run:

```bash
.venv/bin/python tools/validate_mdce_outputs.py --json outputs/reports/mdce_decision_run_<timestamp>.json
```

Artifact contract basics (v2):

- JSON must have `schema_version=mdce_decision_run_v2` and include stable keys like `run_id`, `timestamp_utc`, `project_root`, `source`, `scenario`, `decision`, `model_validation`, `decision_impact`, `decision_impacts`, `artifacts`.
- Markdown must include the core sections validated by the artifact validator (recommendation, confidence, uncertainty, recommended mode, mode options, model validation, decision impacts, explanation).
- Registry v2 row must exist for the run_id and must be consistent with JSON for key fields (timestamp/source/preset/recommended_mode/etc).

Demo-friendly mirror (`plan_view`):

- The JSON also includes an **additive, non-breaking** `plan_view` object. This is an intentional flat mirror of the nested `decision.*` data, shaped for demo/explanation and for any consumer that prefers a single top-level view.
- `plan_view` does NOT change the stable v2 schema; it only re-exposes already-computed fields: `recommendation`, `confidence`, `risk_level`, `confidence_breakdown`, `conflict`, `decision_risk`, `impact`, `fallback`, `issues`, `recommended_mode`, `uncertainty`, `provenance`.
- Terminology note: `plan_view.conflict` is **multi-signal disagreement** (actual pace vs model prediction vs tyre proxy). It is deliberately NOT "multi-model" (we do not run an ensemble of competing ML models). The conflict note inside `plan_view` states this explicitly to keep claims defensible.

Root override (for Colab/Drive or alternate checkout location):

```bash
python3 tools/run_mdce_decision.py --root /content/drive/MyDrive/ibm_project_stuff/MDCE
```

### C) Public Dataset -> Processed MDCE CSV

- File: `tools/prepare_mdce_dataset.py`
- Current supported adapter: Kaggle weather/tyre parquet -> MDCE CSV

Validation (processed CSV hygiene):

```bash
python3 tools/validate_mdce_dataset.py --input data/processed/mdce_kaggle_weather_tyre_latest.csv
```

```bash
python3 tools/prepare_mdce_dataset.py \
  --source kaggle-weather-tyre \
  --input data/raw/extracted/kaggle_naven_weather_tyre/f1_all.parquet \
  --output data/processed/mdce_kaggle_weather_tyre_latest.csv
```

### D) Colab Single Final Pipeline (Model Proof Layers)

If you are running the model proof layers (pit proof finalizer, real multi-decision training, safety-car risk training, deep audit, final consolidation), the single notebook is:

- `MCDE_SINGLE_FINAL_PIPELINE.ipynb`

Important:

- The notebook now loads its stage sources from the `mdce_*.py` scripts in the project folder (not embedded snapshots), so script fixes apply automatically.

## 4) Codebase Map (Where Things Live)

Deterministic engine (used by app + CLI):

- `src/models.py`: core datatypes
- `src/data_loader.py`: CSV normalization, schema, provenance, derived/proxy fields
- `src/scenario_engine.py`: transparent demo failure-mode injections
- `src/strategy_engine.py`: simple baseline recommendation
- `src/trust_engine.py`: trust issues + penalties + conflict score
- `src/confidence_engine.py`: confidence score + risk level
- `src/fallback_engine.py`: fallback actions per issue
- `src/explanation_engine.py`: optional Granite, deterministic fallback explanation
- `src/pipeline.py`: ties everything together

App:

- `app.py`

Tools:

- `tools/prepare_mdce_dataset.py`: create processed MDCE CSV from a public dataset
- `tools/run_mdce_decision.py`: headless CLI run with artifacts + registry
- `tools/validate_mdce_outputs.py`: artifact contract validator (JSON/MD/registry)
- `tools/validate_mdce_dataset.py`: processed-CSV hygiene validator (schema + warnings)
- `tools/demo_gate.py`: one-command gate (CLI run + validators)
- `tools/mdce_calibration.py`: calibration evidence (scenario monotonicity + held-out mode-vs-regret backtest)
- `tools/mdce_fuzz.py`: robustness fuzz harness (deterministic hostile inputs; no crashes / no invalid output)
- `tools/mdce_demo_report.py`: one-page judge-facing Markdown report (what/run/calibration/robustness/provenance)

Tests:

- `tests/test_engines.py`: deterministic engine tests
- `tests/test_cli_runner.py`: hermetic CLI runner test
- `tests/test_calibration.py`: calibration tool (Pearson, monotonicity, held-out backtest)
- `tests/test_fuzz_harness.py`: robustness contract (no crashes / no invalid output)
- `tests/test_demo_report.py`: one-page report shape + determinism

### Evidence + Robustness (How To Run)

Evidence check (validated headline: scenario monotonicity):

- `tools/mdce_calibration.py` reports scenario monotonicity: confidence should fall as genuine failure modes are injected.
- `tools/mdce_calibration_multi.py` checks that monotonicity across multiple drivers from the bundled Zenodo race slice.
- Held-out regret is kept as an exploratory diagnostic only; it must not be presented as proof of calibration.

```bash
.venv/bin/python tools/mdce_calibration.py --input <processed_mdce.csv> --output-dir outputs/reports
```

Robustness fuzz harness (deterministic hostile inputs):

```bash
.venv/bin/python tools/mdce_fuzz.py --json-out outputs/reports/mdce_fuzz_report.json
```

One-page demo report (self-contained Markdown for judges):

```bash
.venv/bin/python tools/mdce_demo_report.py --input <processed_mdce.csv> --output outputs/reports/MDCE_DEMO_REPORT.md
```

## 5) Data Provenance + Claims

MDCE explicitly distinguishes:

- real/source columns (present in the provided dataset)
- derived columns (computed deterministically from real columns, with warnings)
- proxy columns (explicitly not private telemetry; used only as documented proxies)

Safe claim:

> MDCE can ingest public datasets mapped into its schema and produces confidence/risk plus provenance and fallbacks without claiming private telemetry.

Claims to avoid:

- "real tyre temperature telemetry" (we do not have it)
- "optimal strategy" (we do not solve that)
- "beats F1 teams" (not the goal, not evidenced)

## 6) What We Did So Far (Project Progress)

### Deterministic MDCE Engine (Repo)

- Built deterministic trust/confidence pipeline in `src/`.
- Built Streamlit UI for uploads + provenance display + scenario toggles.
- Built public dataset adapter for Kaggle weather/tyre parquet + a preparation CLI.
- Built headless production-style CLI runner that writes JSON/MD + an append-only registry CSV.
- Added unit tests for core behavior + hermetic CLI test.
- Added uncertainty-aware fallback ordering and SAFE-mode nudges when evidence is weak.

### Model Proof Layers (Colab/Drive Artifacts)

These run in Colab (and store artifacts under Drive paths). Summary of what has been produced on Drive previously:

- Pit proof layer finalized: selected RandomForest model with strong 2023 holdout metrics.
- Real-label multi-decision training:
  - stint action classifier promoted
  - stint remaining regressor promoted
  - tyre choice trained but weak (explicitly marked experimental)
- Safety-car risk layer:
  - trained but currently fails promotion gate due to rare-event imbalance
- Deep audit:
  - seed stability + bootstrap intervals + artifact audit
- Final display consolidation:
  - final markdown report + final model registry CSV

## 7) Current Production TODO

For the detailed operational checklist and ongoing worklog, see:

- `PRODUCTION_TODO_AND_WORKLOG.md`

Execution checklist (engineering TODO board):

- `MDCE_MASTER_TODO.md`

High-level next steps:

- Harden app UX (data quality banner, clearer warnings, stable layout)
- Standardize artifact schemas/registries across layers
- Improve safety-car/risk labeling + evaluation (without overclaiming)
- Re-check dataset licenses/terms and document them per source

## 8) Worklog (Key Recent Changes)

### 2026-05-27

- Safety-car training script hardened (sparse-safe logistic regression solver; candidate failures do not crash stage).
- Single final Colab notebook updated to load stage sources from scripts (no stale embedded snapshots).
- Requirements updated to include ML/reporting deps used by scripts.
- Streamlit UX sizing updated to use `width="stretch"` for charts/tables (Streamlit deprecated `use_container_width`).
- Added headless CLI run artifacts (JSON/MD) + append-only registry CSV.
- CLI: root-aware defaults; JSON includes `schema_version` (currently `mdce_decision_run_v2`) and `project_root`.
- Added confidence scoring version tag and a monotonicity regression test.

### 2026-05-27 (later)

- Added deterministic recommended mode selection (SAFE vs AGGRESSIVE) based on issues + uncertainty.
- Pipeline now artifacts `uncertainty_primary`, `uncertainty_score`, and `recommended_mode` in scenario notes.
- Fallbacks now have deterministic ordering and can nudge SAFE mode under high uncertainty.
