# PitVerdict Demo Runbook (3 Minutes)

This runbook is written to keep the demo defensible and repeatable.

## Demo Dataset (Standardized: Option B, disclosed gap-proxy)

The demo uses a real public dataset (Zenodo curated lap table) prepared into the
MDCE schema, with a **clearly disclosed** derived gap proxy. This is reproducible
offline (no network) from the in-repo parquet.

Generate it:

```bash
.venv/bin/python tools/prepare_mdce_dataset.py \
  --source zenodo-stg-laps-gap-proxy \
  --input data/raw/extracted/zenodo_2024_selected/anandm84-F1-telemetry-DE-592e162/docs/curation/sample_output/silver/stg_laps.parquet \
  --output /tmp/opencode/mdce_demo_optionB.csv
```

Disclosure (say this if asked): `gap_to_car_ahead_s` is a derived proxy computed
from cumulative lap times across drivers (`gap_to_car_ahead_method =
cumulative_lap_time_proxy` in the metadata). It is NOT an official timing-feed
gap. Sector times are real; track status defaults to NORMAL.

Honest behavior to expect: this real slice (HAM, round 1, decision lap 12)
contains a genuine lap-time outlier at lap 12. MDCE surfaces it via
`lap_time_outlier` and recommends SAFE even at baseline — this is correct, not a
bug.

Archived reference runs (for screenshots / fallback if live run fails):

```text
outputs/reports/demo_reference/   (clean baseline + high_uncertainty_stack)
```

Alternative (Option A, honest missing gaps, no proxy) if you prefer not to use a
derived gap at all: use `--source zenodo-stg-laps` (MDCE will flag the
`coverage_gap_track_gaps` issue instead).

## Pre-Flight (30 seconds)

1. Run the demo gate (headless). Against the standardized Option B dataset:

```bash
.venv/bin/python tools/demo_gate.py \
  --input /tmp/opencode/mdce_demo_optionB.csv \
  --preset high_uncertainty_stack
```

Or against the offline fallback dataset (no input needed):

```bash
.venv/bin/python tools/demo_gate.py --preset high_uncertainty_stack
```

Notes:

- `tools/demo_gate.py` validates the exact JSON artifact produced by the CLI (it parses the `JSON:` line) to avoid "latest file" races.
- The gate also enforces the stable artifact contract (JSON schema v2, required Markdown sections, and registry v2 row consistency).

2. If you plan to demo the prepared Kaggle-derived dataset as *default* data:

```bash
.venv/bin/python tools/demo_gate.py --allow-noncommercial --preset high_uncertainty_stack
```

Note: by default, MDCE will not auto-load NonCommercial datasets unless `MDCE_ALLOW_NONCOMMERCIAL_DATA=1` is set.

If you want this behavior in Streamlit as well:

```bash
export MDCE_ALLOW_NONCOMMERCIAL_DATA=1
streamlit run app.py
# If you didn't activate the venv:
# .venv/bin/python -m streamlit run app.py
```

## Demo Script (2:30)

### 1) What MDCE is (15 seconds)

Say:

"MDCE is a decision-confidence layer for race strategy. It does not claim to find the optimal strategy. It tells you whether the current recommendation is trustworthy when data, models, and race context disagree."

### 2) Baseline run (40 seconds)

In Streamlit (`streamlit run app.py`):

1. Keep preset = `custom` and toggles off.
2. Point out:
   - recommendation type + target lap
   - confidence + risk
   - confidence breakdown table
   - issues (should be minimal)
   - fallbacks (should be minimal)

Key line:

"This is deterministic: Python computes evidence, confidence, risk, and fallbacks. Granite is used only for explanation text."

### 3) Make it fail on purpose (70 seconds)

Pick preset `high_uncertainty_stack`.

Point out:

1. The Data Quality banner gets worse.
2. Confidence drops; risk increases.
3. Issues show why (missing telemetry, model mismatch/deviation, safety-car context, weather uncertainty).
4. Fallback actions appear and are ordered (SAFE mode nudge shows up when uncertainty is high).

Key line:

"Every detected issue comes with a concrete fallback action."

### 4) Provenance and safe claims (25 seconds)

Open the Data Provenance section and say:

"MDCE separates real/source fields from derived and proxy fields. If a signal is synthetic or a proxy, it is labelled, and decisions that depend on it get downscoped." 

If the NonCommercial warning appears:

"We explicitly guard NonCommercial datasets: the default prepared dataset is treated as CC BY-NC unless you opt in. The demo still runs offline with a fallback dataset." 

---

## Word-By-Word 3-Minute Script (read this verbatim)

Use this version for the recorded demo. Timestamps are guides, not hard cuts.
The honest framing of the evidence is intentional — do not overclaim.

**[0:00–0:20] Hook + what it is**

> "In Formula 1, teams already have great telemetry and great models. The real
> danger is trusting a strategy call when the data or the model is quietly wrong.
> This is MDCE — a decision-confidence layer. It doesn't try to find the optimal
> strategy. It answers one question: should we trust this recommendation right
> now, and if not, what's the safer action? All the reasoning is deterministic
> Python; IBM Granite is used only to phrase the explanation."

**[0:20–0:55] Baseline run** (Streamlit, preset = `custom`)

> "Here's a real public dataset mapped into our schema. MDCE gives a
> recommendation, a confidence score, and a risk level — with a green, yellow, or
> red trust indicator. It shows a confidence breakdown, and it separates real,
> derived, and proxy fields. Nothing here is presented as data we don't actually
> have."

(Point at: the metric row + the 🟢/🟡/🔴 indicator + confidence breakdown table.)

**[0:55–1:40] Break it on purpose** (switch preset to `high_uncertainty_stack`)

> "Now I inject realistic failures — missing telemetry, a model that's drifted
> from reality, a safety-car phase, and uncertain weather. Watch what happens:
> the data-quality banner turns red, confidence drops, risk rises, and the system
> switches to SAFE mode. Crucially, every issue it raises comes with a concrete
> fallback action — it doesn't just say 'something's wrong', it says what to do
> instead."

(Point at: banner, dropped confidence, issues table, fallback actions list.)

**[1:40–2:30] The differentiator — evidence it behaves sensibly**

> "Here's what makes this more than a dashboard. Scroll to Decision Evidence.
> First: as we add more genuine failures, confidence consistently drops — the
> correlation is about minus zero-nine, so the score tracks real degradation, not
> noise. This monotonicity holds across multiple drivers from the public race
> bundle, so it's the robust claim. Second, we show a held-out regret diagnostic,
> but we label it honestly as exploratory because it does not generalize across
> drivers. Third, a sensitivity check: if a
> tiny change in recent lap times would flip the recommendation, it flags the call
> as fragile. And it's robust — fourteen hostile inputs, zero crashes."

(Point at: monotonicity chart, then the exploratory regret diagnostic, then sensitivity verdict.)

> "To be clear and honest: this is bounded evidence on limited public data, not a
> statistical proof. The robust evidence is monotonicity; the regret backtest is
> diagnostic only."

**[2:30–3:00] Honest close**

> "So what is this? Other teams predict the best strategy. We built the layer that
> tells you when NOT to trust a strategy call — with the evidence to back it up.
> It assists strategists; it doesn't replace them. That's MDCE: a trust layer for
> decisions under uncertainty."

### Delivery tips

- Keep the failure injection live (toggle the preset on camera) — the drop is the moment.
- Say the monotonicity result first; do not lead with the regret diagnostic.
- Always include the one-line honesty caveat. It pre-empts the obvious judge question and signals maturity.



## Submission Checklist (fast)

- Run `tools/demo_gate.py` once and keep the latest `outputs/reports/mdce_decision_run_*.json`.
- Optional: keep the canonical run registry row in `outputs/reports/MDCE_DECISION_RUN_REGISTRY_v2.csv` for audit-friendly comparisons.
- Add screenshots following the shot-list above.
- Record the 3-minute demo following this runbook.
