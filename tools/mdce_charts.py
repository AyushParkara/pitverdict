from __future__ import annotations

"""MDCE demo charts.

Generates deterministic PNG charts for the README / demo / submission evidence.
Uses a non-interactive matplotlib backend so it works headless (CI/Colab/server).

Charts produced:
  1) confidence_by_scenario.png
     - Confidence per scenario preset (ordered by injected failure count).
     - Visually shows confidence dropping as genuine failure modes increase.
  2) lap_time_vs_model.png
     - Actual lap time vs model-predicted lap time across the stint.
     - Missing-telemetry laps marked.
  3) calibration_regret.png
     - Mean realized aggressive-action regret when MDCE said SAFE vs AGGRESSIVE.
     - Visual proof the recommended mode is calibrated to real downstream risk.

All inputs come from the deterministic pipeline + calibration tool. No network.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_default_data_result, load_race_csv  # noqa: E402
from tools.mdce_calibration import confidence_vs_regret_backtest, scenario_monotonicity  # noqa: E402


def chart_confidence_by_scenario(records: list, out_path: Path) -> Path:
    mono = scenario_monotonicity(records)
    rows = sorted(mono["rows"], key=lambda r: (r["injected_failures"], -r["confidence"]))
    labels = [r["preset"] for r in rows]
    confidences = [r["confidence"] for r in rows]
    failures = [r["injected_failures"] for r in rows]

    # Color by failure count (more failures -> warmer/redder).
    max_f = max(failures) if failures else 1
    colors = [plt.cm.RdYlGn(1.0 - (f / max_f if max_f else 0.0)) for f in failures]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, confidences, color=colors, edgecolor="#333", linewidth=0.5)
    for bar, c in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{c:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Decision confidence")
    ax.set_title("MDCE confidence drops as failure modes increase")
    ax.axhline(0.55, color="#888", linestyle="--", linewidth=1, label="Medium-risk threshold")
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def chart_lap_time_vs_model(records: list, out_path: Path) -> Path:
    usable = records
    laps = [r.lap for r in usable]
    actual = [r.lap_time_s for r in usable]
    predicted = [r.predicted_lap_time_s for r in usable]
    missing_laps = [r.lap for r in usable if getattr(r, "missing", False)]
    missing_vals = [r.lap_time_s for r in usable if getattr(r, "missing", False)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(laps, actual, marker="o", color="#2563eb", linewidth=2, label="Actual lap time")
    ax.plot(laps, predicted, marker="o", color="#f97316", linewidth=1.5, linestyle="--", label="Model predicted")
    if missing_laps:
        ax.scatter(missing_laps, missing_vals, color="#dc2626", s=80, marker="x", zorder=5, label="Missing telemetry")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Actual vs model-predicted lap time")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def chart_calibration_regret(records: list, out_path: Path) -> Path | None:
    bt = confidence_vs_regret_backtest(records)
    if bt.get("status") != "OK":
        return None

    safe_v = bt["mean_aggressive_regret_when_safe_s"]
    aggr_v = bt["mean_aggressive_regret_when_aggressive_s"]
    cats: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    if safe_v is not None:
        cats.append(f"SAFE\n(n={bt['num_safe_mode']})")
        vals.append(safe_v)
        colors.append("#16a34a")
    if aggr_v is not None:
        cats.append(f"AGGRESSIVE\n(n={bt['num_aggressive_mode']})")
        vals.append(aggr_v)
        colors.append("#dc2626")

    if not vals:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(cats, vals, color=colors, edgecolor="#333", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean realized aggressive-action regret (s)")
    ax.set_title("Held-out: pushing was costlier exactly when MDCE said SAFE")
    holds = bt.get("calibration_holds")
    ax.text(
        0.5, -0.18, f"calibration_holds = {holds}", transform=ax.transAxes,
        ha="center", fontsize=9, color="#444",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def chart_confidence_over_laps(records: list, out_path: Path, *, min_history: int = 5) -> Path | None:
    """Walk-forward confidence: at each lap k, MDCE sees only laps[0..k].

    Shows how decision confidence evolves across the stint as evidence accumulates.
    Returns None if there is not enough history to plot.
    """
    from src.models import ScenarioFlags
    from src.pipeline import analyze_decision

    usable = [r for r in records if not getattr(r, "missing", False)]
    n = len(usable)
    if n < min_history + 1:
        return None

    laps: list[int] = []
    confs: list[float] = []
    for k in range(min_history - 1, n):
        history = usable[: k + 1]
        result, *_ = analyze_decision(history, ScenarioFlags(), prefer_granite=False)
        laps.append(int(history[-1].lap))
        confs.append(float(result.confidence.confidence))

    if len(laps) < 2:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(laps, confs, marker="o", color="#2563eb", linewidth=2, label="Decision confidence")
    ax.axhspan(0.75, 1.0, color="#16a34a", alpha=0.08)
    ax.axhspan(0.55, 0.75, color="#eab308", alpha=0.08)
    ax.axhspan(0.0, 0.55, color="#dc2626", alpha=0.08)
    ax.axhline(0.75, color="#16a34a", linestyle="--", linewidth=0.8)
    ax.axhline(0.55, color="#dc2626", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Lap (decision point)")
    ax.set_ylabel("Decision confidence")
    ax.set_title("Confidence over the stint (walk-forward; only past laps seen)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def generate_all(records: list, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    produced.append(chart_confidence_by_scenario(records, out_dir / "confidence_by_scenario.png"))
    produced.append(chart_lap_time_vs_model(records, out_dir / "lap_time_vs_model.png"))
    regret = chart_calibration_regret(records, out_dir / "calibration_regret.png")
    if regret is not None:
        produced.append(regret)
    timeline = chart_confidence_over_laps(records, out_dir / "confidence_over_laps.png")
    if timeline is not None:
        produced.append(timeline)
    return produced


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MDCE demo charts (PNG).")
    p.add_argument("--root", default=None, help="Optional project root for default data resolution.")
    p.add_argument("--input", default=None, help="Optional processed MDCE CSV. If omitted, uses default loader.")
    p.add_argument("--output-dir", default="outputs/charts", help="Directory to write PNG charts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path
        data_load = load_race_csv(input_path, source_name=str(input_path))
    else:
        data_load = load_default_data_result(root=root)

    produced = generate_all(data_load.records, out_dir)
    print(f"MDCE charts: wrote {len(produced)} file(s)")
    for p in produced:
        print(" -", p)


if __name__ == "__main__":
    main()
