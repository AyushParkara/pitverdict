# PitVerdict

**AI Builders Challenge — May Innovation Challenge: F1 Car Racing**

A decision-confidence / trust layer for race strategy decisions under uncertain and incomplete data. Built with IBM Granite, Python, and Streamlit.

---

## The Problem

Formula 1 teams already use advanced telemetry, dashboards, simulations, and strategy tools. The problem PitVerdict focuses on is narrower but critical:

> Even with strong models, live strategy can become fragile when telemetry is incomplete, model assumptions drift, or race context changes.

Existing tools tell you *what* to do. PitVerdict tells you *whether to trust it* — and if not, what safer option to take instead.

In the high-speed world of F1 racing, a bad strategy call can cost track position, pit window optimization, or even a race result. Teams need a layer that evaluates the reliability of their recommendations in real time, especially when:
- Telemetry data has gaps or placeholder values
- Model predictions drift from actual lap times
- Safety car periods or weather changes introduce uncertainty
- Multiple data signals (lap times, tyre degradation, track gaps) disagree

## AI / Technical Approach

PitVerdict uses a **deterministic decision intelligence pipeline** with IBM Granite for explanation:

### Core Pipeline (All Deterministic Python)

```text
CSV (processed public data)
  -> Data Loader (schema + provenance labelling)
  -> Scenario Engine (inject failure modes for testing)
  -> Baseline Strategy Engine (simple recommendation)
  -> Coverage Gap Engine (detect placeholder inputs)
  -> Model Validation (model vs reality deviation)
  -> Trust Engine (issues + penalties + multi-signal conflict)
  -> Confidence Engine (confidence score + risk level)
  -> Uncertainty Engine (primary uncertainty, downstream decisions at risk)
  -> Decision Impact Engine (if-wrong loss estimates for 7 decision domains)
  -> Mode Engine (SAFE / AGGRESSIVE mode options)
  -> Fallback Engine (concrete alternative actions)
  -> Explanation Engine (IBM Granite optional; deterministic fallback)
```

**Key design principle:** Python computes all evidence, confidence, risk, and fallbacks. IBM Granite is used **only** to phrase the explanation — the AI does not drive decisions, it explains them.

### IBM Technology Used

| Technology | Role |
|------------|------|
| **IBM Granite** (via watsonx) | Natural language explanation of computed evidence (optional; deterministic fallback included) |
| **Python/Streamlit** | Full pipeline engine + interactive dashboard |
| **Open source** | pandas, numpy, scikit-learn, plotly, matplotlib |

### Evidence the Confidence Is Meaningful

- **Scenario monotonicity (validated headline):** Confidence consistently drops as genuine failure modes are injected. Pearson correlation ~ -0.93 on real data across multiple drivers.
- **Held-out regret backtest (exploratory diagnostic):** Compares aggressive-action regret under SAFE vs AGGRESSIVE mode recommendations.
- **Robustness fuzz harness:** 14 deterministic hostile/edge-case inputs — every case produces a valid result or is cleanly rejected. No crashes, no NaN/Infinity, no out-of-range outputs.
- **Recommendation sensitivity:** Perturbs recent lap times by ±0.1–0.3s to test whether the recommendation flips easily.

### Decision Domains

PitVerdict scores confidence + risk across **7 strategy decision domains**:
1. Pit timing
2. Tyre compound choice
3. Stint length
4. Push vs conserve
5. Safety car response
6. Traffic / rejoin risk
7. Aggressive vs safe mode selection

## Why This Matters In The Context Of Racing

Formula 1 is a sport of milliseconds and marginal gains. Strategy calls — when to pit, which tyre to use, whether to push or conserve — directly impact race outcomes. But these decisions are made under extreme uncertainty:

- **Incomplete data:** Telemetry gaps from sensor issues or communication loss
- **Model drift:** Pre-race simulations diverge from actual track conditions
- **Context shifts:** Safety cars, weather changes, and yellow flags alter the risk landscape
- **Signal conflict:** Lap times, tyre degradation, and track position may tell different stories

PitVerdict addresses a gap no other F1 tool fills: instead of predicting the optimal strategy, it quantifies **how much to trust** a strategy recommendation at any given moment. This helps:
- **Strategists** identify when their models are becoming unreliable
- **Engineers** prioritize which data feeds need attention
- **Team leadership** make safer, more informed race decisions

The framework is demonstrated through pit-window confidence but generalizes to all race strategy decisions.

## Functioning Prototype

The Streamlit app provides:
- Interactive scenario controls (toggle failure modes)
- Real-time confidence, risk, and trust indicators
- Per-domain decision confidence (7 domains)
- Recommendation sensitivity analysis
- Evidence charts (monotonicity, calibration, confidence timeline)
- Dataset provenance labelling (real / derived / proxy columns)

### Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run tests (184 passing):
```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Project Structure

```
app.py                  # Streamlit dashboard (entry point)
README.md               # this file
LICENSE                 # MIT license
LICENSE-DATA.md         # Data attribution
requirements.txt        # runtime dependencies
.streamlit/config.toml  # UI theme

src/                    # Deterministic decision-intelligence engine (18 modules)
tools/                  # CLI, validators, evidence harnesses, charts
tests/                  # 184 deterministic tests (26 files)
notebooks/              # All notebooks + Colab stage-source scripts
data/                   # sample + prepared datasets (raw/processed gitignored)
outputs/                # generated reports/charts/registries (gitignored)
docs/                   # full documentation
```

## Safe Claims

- PitVerdict quantifies confidence in a race strategy recommendation under uncertainty.
- PitVerdict assists human decision-makers; it does not replace strategists.
- The prototype uses public real datasets with clearly labelled derived/proxy fields.
- The novelty is decision confidence, not raw telemetry validation.
- Granite explains already-computed evidence; it does not drive strategy decisions.

## License

Code: MIT (see [LICENSE](LICENSE)). Data: see [LICENSE-DATA.md](LICENSE-DATA.md) for per-source attribution.

## Demo Video
[Watch the 3-min demo](https://youtu.be/UaU5Kbepe60)

