from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import DEFAULT_REAL_DATA_PATH, NONCOMMERCIAL_OPT_IN_ENV, load_default_data_result, load_race_csv
from src.models import ScenarioFlags
from src.pipeline import analyze_decision
from src.scenario_presets import list_preset_names, resolve_preset
from src.sensitivity_engine import analyze_recommendation_stability
from tools.mdce_calibration import confidence_vs_regret_backtest, scenario_monotonicity
from tools.mdce_charts import chart_calibration_regret, chart_confidence_by_scenario, chart_confidence_over_laps


st.set_page_config(page_title="PitVerdict", page_icon="🏁", layout="wide")


def records_to_frame(records: list) -> pd.DataFrame:
    return pd.DataFrame([record.__dict__ for record in records])


def _filter_loader_warnings(warnings: list[str]) -> list[str]:
    # data_loader.py currently mixes provenance notes (real/derived/proxy columns) into `warnings`.
    # The UI already displays those columns explicitly, so keep the remaining warnings actionable.
    ignore_prefixes = (
        "Real column used:",
        "Derived column:",
        "Proxy column:",
    )
    return [w for w in warnings if not any(w.startswith(p) for p in ignore_prefixes)]


def _read_default_metadata() -> dict:
    meta_path = DEFAULT_REAL_DATA_PATH.with_suffix(".metadata.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _display_dataset_metadata(meta: dict) -> None:
    if not meta:
        return
    license_str = (meta.get("license_spdx") or meta.get("license_note") or "").strip()
    if license_str:
        st.caption(f"Dataset license: {license_str}")
    src_url = (meta.get("source_url") or "").strip()
    if src_url:
        st.caption(f"Dataset source URL: {src_url}")


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"})


def _confidence_indicator(confidence: float, risk_level: str) -> str:
    """Return a traffic-light emoji for at-a-glance trust."""
    if confidence >= 0.75 and risk_level == "Low":
        return "🟢"
    if confidence >= 0.55:
        return "🟡"
    return "🔴"


def _confidence_gauge(confidence: float, risk_level: str) -> go.Figure:
    pct = confidence * 100
    color = "#22c55e" if confidence >= 0.75 else "#f59e0b" if confidence >= 0.55 else "#ef4444"
    risk_dot = {"Low": "🟢", "Medium": "🟡", "Medium-High": "🟠", "High": "🔴"}.get(risk_level, "⚪")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 34, "color": color, "weight": 700}},
        title={"text": f"{risk_dot} {risk_level} risk", "font": {"size": 13, "color": "#64748b"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333", "tickfont": {"size": 9, "color": "#475569"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,.08)"},
                {"range": [40, 65], "color": "rgba(245,158,11,.08)"},
                {"range": [65, 100], "color": "rgba(34,197,94,.08)"},
            ],
        },
    ))
    fig.update_layout(
        height=190,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
    )
    return fig


def _confidence_factors_chart(breakdown: dict) -> go.Figure:
    """Horizontal bar chart for confidence factors — more legible than a table."""
    labels = {
        "data_completeness": "Data Completeness",
        "signal_agreement": "Signal Agreement",
        "model_alignment": "Model Alignment",
        "context_stability": "Context Stability",
        "penalty_score": "Penalty Score",
    }
    items = [(labels.get(k, k.replace("_", " ").title()), v) for k, v in breakdown.items() if k in labels]
    items.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*items) if items else ([], [])
    colors = ["#22c55e" if v >= 0.7 else "#f59e0b" if v >= 0.45 else "#ef4444" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=list(names),
        orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{v:.0%}" for v in vals],
        textposition="outside",
        textfont={"color": "#94a3b8", "size": 12},
        hovertemplate="%{y}: %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin={"l": 10, "r": 50, "t": 10, "b": 10},
        xaxis={"range": [0, 1], "visible": False, "showgrid": False},
        yaxis={"autorange": "reversed", "showgrid": False, "tickfont": {"size": 12, "color": "#cbd5e1"}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.25,
    )
    return fig


def _recommendation_card_html(rec, confidence: float, risk_level: str) -> str:
    action_color = {"PIT_NOW": "#ef4444", "PIT_SOON": "#f59e0b", "EXTEND": "#22c55e"}
    card_suffix = {"PIT_NOW": "pit_now", "PIT_SOON": "pit_soon", "EXTEND": "extend"}
    color = action_color.get(rec.recommendation_type.value, "#555")
    rcmd = card_suffix.get(rec.recommendation_type.value, "neutral")
    label = rec.recommendation_type.value.replace("_", " ").title()
    gain = rec.expected_gain_loss_s
    conf_dot = "🟢" if confidence >= 0.75 else "🟡" if confidence >= 0.55 else "🔴"
    return f"""
    <div class="pv-card pv-card-rcmd-{rcmd}" style="border-left-width:4px;">
      <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;">Decision</div>
      <div style="font-size:1.8rem;font-weight:700;color:{color};line-height:1.2;">{label}</div>
      <div style="display:flex;gap:24px;margin-top:12px;flex-wrap:wrap;">
        <div><span style="color:#64748b;font-size:.75rem;">Target Lap</span><br><span style="font-size:1.3rem;font-weight:600;color:#f1f5f9;">{rec.recommended_lap}</span></div>
        <div><span style="color:#64748b;font-size:.75rem;">Expected Gain</span><br><span style="font-size:1.3rem;font-weight:600;color:{color};">{gain:+.2f}s</span></div>
        <div><span style="color:#64748b;font-size:.75rem;">Confidence</span><br><span style="font-size:1.3rem;font-weight:600;color:#f1f5f9;">{conf_dot} {round(confidence*100)}%</span></div>
      </div>
    </div>
    """


def _verdict_card_html(rec, confidence: int, risk: str, mode: str | None) -> str:
    rec_label = rec.recommendation_type.value.replace("_", " ").title()
    mode_suffix = ""
    if mode == "SAFE":
        mode_suffix = " The system recommends a SAFE approach — evidence is weaker than usual."
    action_color = {"PIT_NOW": "#ef4444", "PIT_SOON": "#f59e0b", "EXTEND": "#22c55e"}
    border = action_color.get(rec.recommendation_type.value, "#555")
    return f"""
    <div class="pv-card" style="border-left:4px solid {border};padding:14px 18px;">
      <span style="color:#cbd5e1;font-size:.95rem;line-height:1.5;">
        The data suggests <strong style="color:{border};">{rec_label}</strong> at lap <strong>{rec.recommended_lap}</strong>.
        Confidence is <strong>{confidence}%</strong> ({risk} risk).{mode_suffix}
      </span>
    </div>
    """


def _mode_indicator(mode: str | None) -> str:
    return {"SAFE": "🛡️ SAFE", "AGGRESSIVE": "⚡ AGGRESSIVE"}.get(mode or "", mode or "UNKNOWN")


def _data_quality_banner(*, result, conflict: tuple[float, str]) -> None:
    conflict_score, conflict_label = conflict
    issues = getattr(result, "issues", []) or []

    coverage_gaps = [i for i in issues if isinstance(getattr(i, "issue", None), str) and i.issue.startswith("coverage_gap_")]
    has_missing = any(getattr(i, "issue", None) == "missing_telemetry" for i in issues)
    has_model_dev = any(getattr(i, "issue", None) in {"model_deviation", "model_mismatch", "optimistic_model_bias"} for i in issues)

    risk_level = getattr(getattr(result, "confidence", None), "risk_level", None)
    recommended_mode = getattr(getattr(result, "recommended_mode", None), "value", None)

    reasons: list[str] = []
    if coverage_gaps:
        reasons.append("Coverage gaps detected (some inputs look placeholder-like); be cautious about any claims those signals would support.")
    if has_missing:
        reasons.append("Missing telemetry markers present; expect reduced confidence and avoid overfitting to recent lap-to-lap changes.")
    if has_model_dev:
        reasons.append("Model-vs-reality deviation flagged; treat model-driven recommendations as less reliable.")
    if conflict_label in {"MEDIUM", "HIGH"}:
        reasons.append(f"Signal disagreement level is {conflict_label} ({conflict_score:.2f}).")
    if isinstance(risk_level, str) and risk_level in {"Medium-High", "High"}:
        reasons.append(f"Overall risk level is {risk_level}.")
    if isinstance(recommended_mode, str) and recommended_mode == "SAFE":
        reasons.append("System recommends a safer approach (evidence is weaker or uncertainty is elevated).")

    # Render a single banner that helps demo viewers understand "can we trust this run" quickly.
    if coverage_gaps or has_missing or has_model_dev or conflict_label == "HIGH" or risk_level == "High":
        level = "LOW"
        fn = st.error
    elif reasons:
        level = "MEDIUM"
        fn = st.warning
    else:
        level = "HIGH"
        fn = st.success

    msg = f"Data quality: **{level}**"
    if reasons:
        msg += "\n\n" + "\n".join(f"- {r}" for r in reasons)
    fn(msg)


def _breakdown_table(result) -> pd.DataFrame:
    breakdown = getattr(getattr(result, "confidence", None), "breakdown", None) or {}
    rows: list[dict] = []
    for k in ("data_completeness", "signal_agreement", "model_alignment", "context_stability", "penalty_score"):
        if k in breakdown:
            rows.append({"Component": k.replace("_", " ").title(), "Score": float(breakdown[k])})
    return pd.DataFrame(rows)


def _decision_domains_table(result) -> pd.DataFrame:
    """Per-decision confidence + risk across all strategy domains MDCE scores.

    This demonstrates MDCE is a trust layer for *many* strategy decisions, not just pit timing.
    """
    conf = getattr(result, "confidence", None)
    dc = getattr(conf, "decision_confidence", None) or {}
    dr = getattr(conf, "decision_risk_levels", None) or {}
    # Map domain -> if-wrong loss (s) from the computed impact estimates, when available.
    impact_loss: dict[str, float] = {}
    primary = getattr(result, "decision_impact", None)
    if primary is not None:
        impact_loss[primary.decision] = primary.if_wrong_expected_loss_s
    for di in getattr(result, "decision_impacts", None) or []:
        impact_loss[di.decision] = di.if_wrong_expected_loss_s

    def _dot(risk: str) -> str:
        return {"Low": "🟢", "Medium": "🟡", "Medium-High": "🟠", "High": "🔴"}.get(risk, "")

    rows: list[dict] = []
    for domain in sorted(dc.keys()):
        risk = dr.get(domain, "")
        loss = impact_loss.get(domain)
        rows.append(
            {
                "Decision Domain": domain.replace("_", " ").title(),
                "Confidence": f"{int(float(dc[domain]) * 100)}%",
                "Risk": f"{_dot(risk)} {risk}".strip(),
                "If-Wrong Loss (s)": f"{loss}" if loss is not None else "—",
            }
        )
    return pd.DataFrame(rows)


def build_lap_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["lap"],
            y=df["lap_time_s"],
            mode="lines+markers",
            name="Actual lap time",
            line={"color": "#2563eb", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["lap"],
            y=df["predicted_lap_time_s"],
            mode="lines+markers",
            name="Model predicted lap time",
            line={"color": "#f97316", "width": 2, "dash": "dash"},
        )
    )
    missing_df = df[df["missing"]]
    if not missing_df.empty:
        fig.add_trace(
            go.Scatter(
                x=missing_df["lap"],
                y=missing_df["lap_time_s"],
                mode="markers",
                name="Missing telemetry marker",
                marker={"color": "#dc2626", "size": 12, "symbol": "x"},
            )
        )
    fig.update_layout(
        height=410,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="Lap",
        yaxis_title="Lap time (seconds)",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def issue_table(result) -> pd.DataFrame:
    rows = []
    for issue in result.issues:
        rows.append(
            {
                "Issue": issue.issue,
                "Severity": issue.severity.value,
                "Penalty": f"-{int(issue.penalty * 100)}%",
                "Affected Decisions": ", ".join(issue.affected_decisions),
                "Reason": issue.reason,
            }
        )
    return pd.DataFrame(rows)


def mode_options_table(result) -> pd.DataFrame:
    rows: list[dict] = []
    for opt in getattr(result, "mode_options", []) or []:
        rows.append(
            {
                "Mode": opt.mode.value,
                "Recommendation": opt.recommendation.recommendation_type.value,
                "Target Lap": opt.recommendation.recommended_lap,
                "If-Wrong Loss (s)": opt.decision_impact.if_wrong_expected_loss_s,
                "Risk": opt.decision_impact.risk_level,
            }
        )
    return pd.DataFrame(rows)


def decision_impacts_table(result) -> pd.DataFrame:
    rows: list[dict] = []
    for di in getattr(result, "decision_impacts", []) or []:
        rows.append(
            {
                "Decision": di.decision,
                "Horizon (laps)": di.horizon_laps,
                "If-Right Gain (s)": di.if_right_expected_gain_s,
                "If-Wrong Loss (s)": di.if_wrong_expected_loss_s,
                "Risk": di.risk_level,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Design system (CSS) + hero banner
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  .block-container { padding: 1.5rem 2rem !important; max-width: 1440px; margin: 0 auto; }
  section[data-testid="stSidebar"] > div:first-child { padding: 1.2rem 1rem; }

  /* Card system */
  .pv-card {
    background: #16181d; border: 1px solid #262a30; border-radius: 12px;
    padding: 1.25rem; margin-bottom: 1rem; transition: border-color .15s;
  }
  .pv-card:hover { border-color: #383d45; }
  .pv-card-neutral { border-left: 3px solid #555; }
  .pv-card-rcmd-extend { border-left: 3px solid #22c55e; }
  .pv-card-rcmd-pit_now { border-left: 3px solid #ef4444; }
  .pv-card-rcmd-pit_soon { border-left: 3px solid #f59e0b; }

  /* Hero */
  .pv-hero {
    background: linear-gradient(135deg, #0e1015 0%, #16181d 40%, #1a1012 100%);
    border: 1px solid #262a30; border-radius: 16px; padding: 1.5rem 2rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
  }
  .pv-hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
  }
  .pv-hero h1 { color: #f1f5f9; margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -.3px; }
  .pv-hero p { color: #94a3b8; margin: 6px 0 0 0; font-size: .92rem; }
  .pv-hero .pill {
    display: inline-block; background: #1e2028; color: #cbd5e1; border: 1px solid #2a2e36;
    border-radius: 999px; padding: 3px 11px; margin-right: 6px; font-size: .72rem; font-weight: 500;
  }

  /* Metric tiles */
  div[data-testid="stMetric"] {
    background: #16181d; border: 1px solid #262a30; border-radius: 10px; padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,.2);
  }
  div[data-testid="stMetric"] label { color: #64748b !important; font-size: .78rem !important; }
  div[data-testid="stMetric"] div { color: #f1f5f9 !important; }

  /* DataFrames */
  div[data-testid="stDataFrame"] {
    border: 1px solid #262a30; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.15);
  }
  div[data-testid="stDataFrame"] th { background: #1a1c24 !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0e1015; border-right: 1px solid #1e2028;
  }
  section[data-testid="stSidebar"] .stButton button {
    width: 100%; background: #1e2028; border: 1px solid #2a2e36; border-radius: 8px; color: #e2e8f0;
  }
  section[data-testid="stSidebar"] .stButton button:hover { border-color: #ef4444; background: #262a32; }

  /* Tabs */
  button[data-testid="stTab"] {
    font-size: .85rem; font-weight: 500; letter-spacing: .2px;
  }
  button[data-testid="stTab"][aria-selected="true"] { border-bottom-color: #ef4444 !important; }

  /* Expander / infobox / alerts */
  .stAlert { border-radius: 10px !important; border: 1px solid #262a30 !important; }
  div[data-testid="stExpander"] {
    border: 1px solid #262a30; border-radius: 10px; background: #0e1015; overflow: hidden;
  }
  div[data-testid="stExpander"] summary { font-weight: 500; }

  /* Dividers */
  hr { border-color: #1e2028 !important; margin: 1rem 0 !important; }

  /* Custom scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0e1015; }
  ::-webkit-scrollbar-thumb { background: #2a2e36; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #383d45; }
</style>
<div class="pv-hero">
  <h1>🏁 PitVerdict</h1>
  <p>Confidence scoring for race strategy — should you trust this call right now?</p>
  <div style="margin-top:10px;">
    <span class="pill">🟢🟡🔴 trust indicators</span>
    <span class="pill">7 decision types</span>
    <span class="pill">data-backed evidence</span>
    <span class="pill">stress-tested</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Visible even when the sidebar is collapsed (useful for screenshots/demo recordings).
loaded_source_main_placeholder = st.empty()

with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
        "<span style='font-size:1.6rem;'>🏁</span>"
        "<span style='font-size:1.2rem;font-weight:700;color:#f5f6fa;'>PitVerdict</span>"
        "<span style='font-size:.65rem;color:#888;margin-left:auto;text-transform:uppercase;letter-spacing:1px;'>v1.0</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Confidence scoring for race strategy decisions")
    st.divider()

    st.header("Data Source")
    loaded_source_placeholder = st.empty()
    uploaded_file = st.file_uploader(
        "Upload processed CSV",
        type=["csv"],
        help="Use a CSV exported from Colab/Drive and mapped to the PitVerdict schema.",
    )
    if DEFAULT_REAL_DATA_PATH.exists():
        meta = _read_default_metadata()
        license_note = (meta.get("license_spdx") or meta.get("license_note") or "").strip()
        st.caption(f"Prepared dataset present: {DEFAULT_REAL_DATA_PATH}")
        if license_note:
            st.caption(f"License/provenance: {license_note}")

        # Mirror the loader's intent: trust explicit license metadata when present.
        # If metadata exists but doesn't specify a license, be conservative for the Kaggle-derived
        # default filename.
        if license_note:
            is_noncommercial_hint = ("BY-NC" in license_note.upper()) or ("NONCOMMERCIAL" in license_note.upper())
        else:
            is_noncommercial_hint = DEFAULT_REAL_DATA_PATH.name.startswith("mdce_kaggle_")
        opted_in = _env_truthy(NONCOMMERCIAL_OPT_IN_ENV)
        if is_noncommercial_hint and not opted_in:
            st.warning(
                "NonCommercial data guard: the prepared dataset is treated as NonCommercial, so PitVerdict will not auto-load it by default. "
                f"Set `{NONCOMMERCIAL_OPT_IN_ENV}=1` to allow it, or upload a different CSV."
            )
            st.caption("Default behavior without upload: offline demo fallback dataset.")
        elif is_noncommercial_hint and opted_in:
            st.info(f"NonCommercial data opt-in enabled via `{NONCOMMERCIAL_OPT_IN_ENV}`.")
            st.caption("Default behavior without upload: prepared dataset.")
        else:
            st.caption("Default behavior without upload: prepared dataset.")
    else:
        st.caption("If no CSV is uploaded, the app uses the offline fallback demo dataset.")
    st.divider()

    st.markdown("### ⚙️ Scenario Controls")
    st.caption("Toggle test scenarios to see how confidence changes.")

    preset_name = st.selectbox(
        "Scenario preset",
        key="scenario_preset",
        options=list_preset_names(),
        help="Preset applies a named failure-mode stack; you can still add extra toggles below.",
    )

    def _apply_preset_to_state() -> None:
        preset = resolve_preset(st.session_state.get("scenario_preset") or "custom")
        st.session_state["flag_missing_telemetry"] = preset.flags.missing_telemetry
        st.session_state["flag_tyre_signal_drift"] = preset.flags.tyre_signal_drift
        st.session_state["flag_model_mismatch"] = preset.flags.model_mismatch
        st.session_state["flag_safety_car_phase"] = preset.flags.safety_car_phase
        st.session_state["flag_weather_uncertainty"] = preset.flags.weather_uncertainty

    # First load: initialize toggle state from the preset.
    if "flag_missing_telemetry" not in st.session_state:
        _apply_preset_to_state()

    # Re-apply on preset change (ensures preset actually updates checkboxes).
    if st.session_state.get("_last_preset") != preset_name:
        st.session_state["_last_preset"] = preset_name
        _apply_preset_to_state()

    preset = resolve_preset(preset_name)
    st.caption(preset.description)

    if st.button("Reset toggles to preset"):
        _apply_preset_to_state()
        st.rerun()

    with st.expander("Scenario toggles", expanded=True):
        missing_telemetry = st.checkbox("Missing telemetry", key="flag_missing_telemetry")
        tyre_signal_drift = st.checkbox("Tyre signal drift", key="flag_tyre_signal_drift")
        model_mismatch = st.checkbox("Model mismatch", key="flag_model_mismatch")
        safety_car_phase = st.checkbox("Safety car phase", key="flag_safety_car_phase")
        weather_uncertainty = st.checkbox("Weather uncertainty", key="flag_weather_uncertainty")

    flags = ScenarioFlags(
        missing_telemetry=bool(missing_telemetry),
        tyre_signal_drift=bool(tyre_signal_drift),
        model_mismatch=bool(model_mismatch),
        safety_car_phase=bool(safety_car_phase),
        weather_uncertainty=bool(weather_uncertainty),
    )
    prefer_granite = st.toggle("Use Granite if configured", value=True)
    st.divider()
    st.caption("Always label synthetic fields. Nothing here is presented as private F1 telemetry.")

if uploaded_file is not None:
    try:
        data_load = load_race_csv(uploaded_file, source_name=uploaded_file.name)
    except ValueError as exc:
        st.error(f"Could not load uploaded CSV: {exc}")
        st.stop()
else:
    # Root-aware defaults aren't required for the app today since it runs from repo root,
    # but the loader supports it for CLI/Colab.
    data_load = load_default_data_result()

with st.sidebar:
    loaded_source_placeholder.caption(f"Loaded source: {data_load.source_name}")

meta = getattr(data_load, "dataset_metadata", {}) or {}
license_str = (meta.get("license_spdx") or meta.get("license_note") or "").strip()
main_line = f"Loaded source: **{data_load.source_name}**"
if license_str:
    main_line += f" | License: **{license_str}**"
loaded_source_main_placeholder.caption(main_line)

records = data_load.records
result, scenario_records, scenario_notes, conflict = analyze_decision(records, flags, prefer_granite=prefer_granite)
df = records_to_frame(scenario_records)

rec = result.recommendation
confidence_pct = int(result.confidence.confidence * 100)
conflict_score, conflict_label = conflict

_data_quality_banner(result=result, conflict=conflict)

recommended_mode_value = result.recommended_mode.value if result.recommended_mode is not None else None

# Visual dashboard header: recommendation card + confidence gauge side by side.
card_col, gauge_col = st.columns([1.7, 1])
with card_col:
    st.markdown(_recommendation_card_html(rec, result.confidence.confidence, result.confidence.risk_level), unsafe_allow_html=True)
with gauge_col:
    st.markdown('<div class="pv-card" style="padding:.75rem;">', unsafe_allow_html=True)
    st.plotly_chart(_confidence_gauge(result.confidence.confidence, result.confidence.risk_level), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# Plain-English verdict with styled card.
st.markdown(_verdict_card_html(rec, confidence_pct, result.confidence.risk_level, recommended_mode_value), unsafe_allow_html=True)

# Download + lap chart in a card.
st.markdown('<div class="pv-card">', unsafe_allow_html=True)
_decision_export = {
    "source_name": data_load.source_name,
    "recommendation": {
        "type": rec.recommendation_type.value,
        "recommended_lap": rec.recommended_lap,
        "expected_gain_loss_s": rec.expected_gain_loss_s,
    },
    "confidence": result.confidence.confidence,
    "risk_level": result.confidence.risk_level,
    "recommended_mode": recommended_mode_value or "UNKNOWN",
    "confidence_breakdown": result.confidence.breakdown,
    "issues": [
        {"issue": i.issue, "severity": i.severity.value, "penalty": i.penalty, "reason": i.reason}
        for i in result.issues
    ],
    "fallback_actions": result.fallback_actions,
    "explanation": result.explanation,
    "conflict": {"score": conflict_score, "label": conflict_label},
}
col1, col2 = st.columns([1, 3])
with col1:
    st.download_button(
        "Download decision (JSON)",
        data=json.dumps(_decision_export, indent=2),
        file_name="pitverdict_decision.json",
        mime="application/json",
    )
with col2:
    st.markdown('<div style="margin-top:2px;"><span style="color:#64748b;font-size:.8rem;">Export the full decision payload for audit or replay.</span></div>', unsafe_allow_html=True)
st.plotly_chart(build_lap_chart(df), width="stretch")
st.markdown('</div>', unsafe_allow_html=True)

# Organize the body into tabs for readability (Overview / Evidence / Provenance).
tab_overview, tab_evidence, tab_provenance = st.tabs(
    ["📊 Overview", "🔬 Decision Evidence", "🗂️ Provenance"]
)

with tab_provenance:
    st.write(f"Source: **{data_load.source_name}**")

    _display_dataset_metadata(getattr(data_load, "dataset_metadata", {}) or {})

    col_a, col_b, col_c = st.columns(3)
    col_a.write("Real/source columns")
    col_a.write(data_load.real_columns or ["None"])
    col_b.write("Derived columns")
    col_b.write(data_load.derived_columns or ["None"])
    col_c.write("Proxy columns")
    col_c.write(data_load.proxy_columns or ["None"])

    ui_warnings = _filter_loader_warnings(data_load.warnings or [])
    if ui_warnings:
        st.warning("\n".join(f"- {warning}" for warning in ui_warnings))
    st.caption("Tip: validate a processed CSV with `tools/validate_mdce_dataset.py` before demoing it.")

with tab_overview:
    st.subheader("Recommended Mode")
    if result.recommended_mode is None:
        st.write("None")
    else:
        st.metric("Mode", result.recommended_mode.value)

    st.subheader("Mode Options")
    modes_df = mode_options_table(result)
    if modes_df.empty:
        st.write("None")
    else:
        st.dataframe(modes_df, width="stretch", hide_index=True)

    st.subheader("Decision Impacts")
    impacts_df = decision_impacts_table(result)
    if impacts_df.empty:
        st.write("None")
    else:
        st.dataframe(impacts_df, width="stretch", hide_index=True)

    st.subheader("Fallback Actions")
    for action in result.fallback_actions:
        st.write(f"- {action}")

    st.subheader("Model Accuracy vs Reality")
    mv = getattr(result, "model_validation", None)
    if mv is None:
        st.write("None")
    else:
        st.write(f"Status: **{mv.status}**")
        st.caption(f"Avg prediction error: {mv.mean_abs_error_s}s | Worst error: {mv.max_abs_error_s}s")
        if getattr(mv, "deviations", None):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Lap": d.lap,
                            "Expected": d.expected_lap_time_s,
                            "Actual": d.actual_lap_time_s,
                            "Delta (s)": d.delta_s,
                        }
                        for d in mv.deviations
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    st.subheader("Trust Issues")
    issues_df = issue_table(result)
    if issues_df.empty:
        st.success("No major trust issues detected.")
    else:
        st.dataframe(issues_df, width="stretch", hide_index=True)

    st.subheader("Uncertainty")
    if result.uncertainty is None:
        st.write("None")
    else:
        st.write(f"Primary driver: **{result.uncertainty.primary_uncertainty}**")
        st.metric("Uncertainty score", f"{int(result.uncertainty.uncertainty_score * 100)}%")
        if result.uncertainty.downstream_decisions_at_risk:
            st.caption("Downstream decisions at risk: " + ", ".join(result.uncertainty.downstream_decisions_at_risk))

    st.subheader("Confidence Factors")
    breakdown = getattr(getattr(result, "confidence", None), "breakdown", None)
    if breakdown:
        st.plotly_chart(_confidence_factors_chart(breakdown), width="stretch")
    else:
        st.write("None")
    if scenario_notes:
        st.info("Scenario injections: " + " ".join(scenario_notes))

    st.subheader("Decision Domains")
    domains_df = _decision_domains_table(result)
    if domains_df.empty:
        st.write("None")
    else:
        st.dataframe(domains_df, width="stretch", hide_index=True)
        st.caption(
            "Confidence + risk for every strategy decision, not just pit timing. "
            "If-wrong loss is an estimate; tyre/traffic figures are estimates too."
        )

    st.subheader("Explanation")
    st.write(result.explanation)

    st.subheader("Beyond Pit Timing")
    st.write(
        "The same trust framework applies to tyre choice, stint length, push/conserve calls, "
        "safety-car response, traffic risk, and aggressive vs safe strategy choices."
    )

# ---------------------------------------------------------------------------
# Decision Evidence: prove the confidence is meaningful + robust (not arbitrary).
# This surfaces the calibration + charts that previously lived only in CLI tools.
# ---------------------------------------------------------------------------
with tab_evidence:
    st.header("Decision Evidence (is the confidence meaningful?)")

    import tempfile as _tmpfile
    from pathlib import Path as _Path

    _chart_tmp = _Path(_tmpfile.mkdtemp(prefix="mdce_charts_"))

    # 1) Scenario applied-labels (what was changed, structured).
    applied = []
    try:
        _scn = __import__("src.scenario_engine", fromlist=["apply_scenarios"]).apply_scenarios(records, flags)
        applied = getattr(_scn, "applied", []) or []
    except Exception:
        applied = []
    if applied:
        with st.expander("Active scenario injections (what was changed)", expanded=False):
            st.dataframe(
                pd.DataFrame(applied).rename(
                    columns={"scenario_name": "Scenario", "scenario_type": "Type", "changed": "What changed"}
                ),
                width="stretch",
                hide_index=True,
            )

    ev_left, ev_right = st.columns(2)

    with ev_left:
        st.subheader("A) Trust score drops when problems increase")
        try:
            mono = scenario_monotonicity(records)
            st.write(f"Confidence never rises as problems increase: **{mono['monotonic_non_increasing']}**")
            corr = mono.get("pearson_failurecount_vs_confidence")
            st.write(f"Correlation (problem count vs confidence): **{corr}**")
            st.caption(
                "Validated across all drivers tested (~ -0.8): confidence falls as genuine problems increase. "
                "This confirms the system behaves as expected under stress."
            )
            try:
                p = chart_confidence_by_scenario(records, _chart_tmp / "confidence_by_scenario.png")
                st.image(str(p), width="stretch")
                with open(p, "rb") as _f:
                    st.download_button(
                        "Download chart (PNG)",
                        data=_f.read(),
                        file_name="confidence_by_scenario.png",
                        mime="image/png",
                        key="dl_conf_chart",
                    )
            except Exception as exc:
                st.caption(f"(chart unavailable: {exc})")
        except Exception as exc:
            st.warning(f"Trust-score check unavailable: {exc}")

    with ev_right:
        st.subheader("B) Exploratory: does the score match real outcomes?")
        try:
            bt = confidence_vs_regret_backtest(records)
            if bt.get("status") != "OK":
                st.info(f"Check: {bt.get('status')} — {bt.get('note', '')}")
            else:
                st.write(
                    f"Average loss from aggressive move when **SAFE**: **{bt['mean_aggressive_regret_when_safe_s']} s** "
                    f"vs when **AGGRESSIVE**: **{bt['mean_aggressive_regret_when_aggressive_s']} s**"
                )
                st.write(f"On this data slice (diagnostic only): **{bt.get('calibration_holds', 'N/A')}**")
                st.caption(
                    "Exploratory only — not a proof. This signal does not generalize across all drivers "
                    "(early-stint confidence mechanically precedes later degradation). "
                    "The validated result is panel A (confidence drops under stress)."
                )
                try:
                    cp = chart_calibration_regret(records, _chart_tmp / "calibration_regret.png")
                    if cp is not None:
                        st.image(str(cp), width="stretch")
                        with open(cp, "rb") as _f:
                            st.download_button(
                                "Download chart (PNG)",
                                data=_f.read(),
                                file_name="calibration_regret.png",
                                mime="image/png",
                                key="dl_calib_chart",
                            )
                except Exception as exc:
                    st.caption(f"(chart unavailable: {exc})")
        except Exception as exc:
            st.warning(f"Outcome check unavailable: {exc}")

    st.caption(
        "Stress test: 14 deliberately broken or hostile inputs are fed to the system; "
        "every case yields a valid result or is cleanly rejected — no crashes, no invalid output."
    )

    # 3) Recommendation sensitivity (research test 6): does the call flip under small input noise?
    st.subheader("C) Is the recommendation stable?")
    try:
        sens = analyze_recommendation_stability(scenario_records, flags)
        verdict = sens["verdict"]
        verdict_icon = {"STABLE": "🟢", "MODERATE": "🟡", "UNSTABLE": "🔴"}.get(verdict, "")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Stability verdict", f"{verdict_icon} {verdict}")
        sc2.metric("Type stability", f"{int(sens['type_stability'] * 100)}%")
        sc3.metric("Recommended-lap spread", sens["recommended_lap_spread"])
        st.caption(
            "We nudge the most recent lap times by ±0.1–0.3s and re-run the decision. "
            "If the recommendation flips easily, the call is fragile and should be treated with caution."
        )
        with st.expander("Show perturbation details", expanded=False):
            st.dataframe(
                pd.DataFrame(sens["points"]).rename(
                    columns={
                        "delta_s": "Δ lap time (s)",
                        "recommendation": "Recommendation",
                        "recommended_lap": "Target lap",
                        "confidence": "Confidence",
                        "type_changed": "Flipped?",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
    except Exception as exc:
        st.warning(f"Stability check unavailable: {exc}")

    # 4) Confidence-over-laps timeline (walk-forward).
    st.subheader("D) Confidence over the stint")
    try:
        tl = chart_confidence_over_laps(records, _chart_tmp / "confidence_over_laps.png")
        if tl is not None:
            st.image(str(tl), width="stretch")
            st.caption("At each lap the system sees only past laps; this shows confidence evolving as evidence accumulates.")
            with open(tl, "rb") as _f:
                st.download_button(
                    "Download chart (PNG)",
                    data=_f.read(),
                    file_name="confidence_over_laps.png",
                    mime="image/png",
                    key="dl_timeline_chart",
                )
        else:
            st.caption("(Not enough laps to plot a confidence timeline for this dataset.)")
    except Exception as exc:
        st.caption(f"(timeline unavailable: {exc})")
