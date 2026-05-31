from __future__ import annotations

"""Recommendation sensitivity / stability analysis.

Research test 6 ("recommendation sensitivity"): if small perturbations to recent
inputs flip the recommendation, the decision is fragile and confidence should be
treated cautiously.

This is deterministic and explainable:
  - Perturb the most recent N lap times by a fixed set of small deltas.
  - Re-run the deterministic pipeline for each perturbation.
  - Measure how often the recommendation TYPE changes, how far the recommended
    lap moves, and how much confidence varies.

It does NOT modify the pipeline; it only probes it. Kept out of `pipeline.py`
so the main decision path stays unchanged.
"""

from dataclasses import replace

from .models import ScenarioFlags
from .pipeline import analyze_decision


DEFAULT_PERTURBATIONS_S = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)


def _perturb_recent_lap_times(records: list, delta_s: float, recent_n: int) -> list:
    if not records:
        return records
    n = len(records)
    start = max(0, n - recent_n)
    updated = list(records)
    for i in range(start, n):
        r = updated[i]
        updated[i] = replace(r, lap_time_s=max(0.001, float(r.lap_time_s) + float(delta_s)))
    return updated


def analyze_recommendation_stability(
    records: list,
    flags: ScenarioFlags | None = None,
    *,
    perturbations_s: tuple[float, ...] = DEFAULT_PERTURBATIONS_S,
    recent_n: int = 3,
) -> dict:
    """Probe recommendation stability under small recent-input perturbations."""
    flags = flags or ScenarioFlags()

    base, *_ = analyze_decision(records, flags, prefer_granite=False)
    base_type = base.recommendation.recommendation_type.value
    base_lap = int(base.recommendation.recommended_lap)
    base_conf = float(base.confidence.confidence)

    points: list[dict] = []
    for delta in perturbations_s:
        perturbed = _perturb_recent_lap_times(records, delta, recent_n)
        res, *_ = analyze_decision(perturbed, flags, prefer_granite=False)
        rtype = res.recommendation.recommendation_type.value
        points.append(
            {
                "delta_s": float(delta),
                "recommendation": rtype,
                "recommended_lap": int(res.recommendation.recommended_lap),
                "confidence": float(res.confidence.confidence),
                "type_changed": rtype != base_type,
            }
        )

    n = len(points)
    same_type = sum(1 for p in points if not p["type_changed"])
    type_stability = round(same_type / n, 3) if n else 1.0

    laps = [p["recommended_lap"] for p in points] + [base_lap]
    lap_spread = max(laps) - min(laps)

    confs = [p["confidence"] for p in points] + [base_conf]
    conf_range = round(max(confs) - min(confs), 3)

    if type_stability >= 0.999 and lap_spread <= 2:
        verdict = "STABLE"
    elif type_stability >= 0.6:
        verdict = "MODERATE"
    else:
        verdict = "UNSTABLE"

    return {
        "baseline": {
            "recommendation": base_type,
            "recommended_lap": base_lap,
            "confidence": base_conf,
        },
        "recent_n": recent_n,
        "perturbations_s": list(perturbations_s),
        "points": points,
        "type_stability": type_stability,
        "recommended_lap_spread": lap_spread,
        "confidence_range": conf_range,
        "verdict": verdict,
        "interpretation": (
            "If small input perturbations flip the recommendation (low type_stability or large "
            "lap spread), the decision is fragile and should be treated with extra caution."
        ),
    }
