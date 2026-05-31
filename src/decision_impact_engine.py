from __future__ import annotations

from statistics import mean

from .data_loader import available_records, recent_records
from .models import DecisionImpactResult, RecommendationType


def _risk_level_from_loss(loss_s: float) -> str:
    if loss_s >= 3.0:
        return "HIGH"
    if loss_s >= 1.5:
        return "MEDIUM"
    return "LOW"


def estimate_pit_timing_impact(
    records: list,
    recommendation_type: RecommendationType,
    *,
    horizon_laps: int = 3,
) -> DecisionImpactResult:
    """Simulate the time impact if pit-timing decision is wrong.

    This is not a full race simulator. It produces a tight, deterministic estimate
    for demo-grade risk communication.

    Heuristic model:

    - Compute recent per-lap degradation (actual trend).
    - If we extend when we should have pitted: loss ~ degradation * horizon.
    - If we pit now but should have extended: loss ~ smaller constant + missed track position.
    """

    usable = available_records(records)
    recent = recent_records(usable, count=5)
    if len(recent) < 2:
        return DecisionImpactResult(
            decision="pit_timing",
            horizon_laps=horizon_laps,
            if_right_expected_gain_s=0.0,
            if_wrong_expected_loss_s=0.0,
            risk_level="LOW",
            assumptions={"horizon_laps": float(horizon_laps)},
            notes=["Insufficient recent data to simulate decision impact."],
        )

    # Simple degradation proxy: positive means getting slower.
    deltas = [recent[i + 1].lap_time_s - recent[i].lap_time_s for i in range(len(recent) - 1)]
    degradation_per_lap = max(0.0, mean(deltas))

    # Base pit cost is not known (no full model), so we only compare relative losses.
    # These constants are illustrative and bounded.
    pit_now_regret_s = 1.2
    extend_regret_s = degradation_per_lap * horizon_laps

    if recommendation_type in {RecommendationType.PIT_NOW, RecommendationType.PIT_SOON}:
        # If we're wrong and should extend: regret is smaller/constant.
        wrong_loss = pit_now_regret_s
        right_gain = extend_regret_s
        notes = [
            f"Assuming extending would cost ~{round(degradation_per_lap, 2)}s/lap over {horizon_laps} laps.",
            "If pitting is wrong, regret assumes small time loss from track-position disruption.",
        ]
    else:
        # If we're wrong and should pit: regret grows with degradation.
        wrong_loss = extend_regret_s
        right_gain = pit_now_regret_s
        notes = [
            f"Assuming degradation continues at ~{round(degradation_per_lap, 2)}s/lap for {horizon_laps} laps.",
            "If extending is wrong, regret approximates cumulative pace loss.",
        ]

    wrong_loss = round(float(wrong_loss), 2)
    right_gain = round(float(right_gain), 2)
    return DecisionImpactResult(
        decision="pit_timing",
        horizon_laps=horizon_laps,
        if_right_expected_gain_s=right_gain,
        if_wrong_expected_loss_s=wrong_loss,
        risk_level=_risk_level_from_loss(wrong_loss),
        assumptions={
            "degradation_per_lap_s": round(float(degradation_per_lap), 3),
            "pit_now_regret_s": pit_now_regret_s,
            "horizon_laps": float(horizon_laps),
        },
        notes=notes,
    )


def estimate_push_vs_conserve_impact(
    records: list,
    *,
    horizon_laps: int = 3,
) -> DecisionImpactResult:
    """Estimate expected regret for a push vs conserve call.

    Deterministic, local heuristic using only available MDCE schema fields.

    - If pushing is wrong: regret comes from amplified degradation (if tyre/pace already drifting)
      and amplified inconsistency (proxy for traffic/driver variability).
    - If conserving is wrong: regret largely comes from missed lap-time opportunity.

    We do not have fuel/ERS/engine/position deltas; this is bounded and illustrative.
    """

    usable = available_records(records)
    recent = recent_records(usable, count=6)
    if len(recent) < 3:
        return DecisionImpactResult(
            decision="push_vs_conserve",
            horizon_laps=horizon_laps,
            if_right_expected_gain_s=0.0,
            if_wrong_expected_loss_s=0.0,
            risk_level="LOW",
            assumptions={"horizon_laps": float(horizon_laps)},
            notes=["Insufficient recent data to simulate push vs conserve impact."],
        )

    # Pace drift proxy: positive means lap time is getting slower.
    deltas = [recent[i + 1].lap_time_s - recent[i].lap_time_s for i in range(len(recent) - 1)]
    degradation_per_lap = max(0.0, mean(deltas))

    # Consistency proxy is already in [0,1]; lower means more volatile.
    min_consistency = min(r.speed_consistency for r in recent)
    inconsistency = max(0.0, 1.0 - float(min_consistency))

    # Baseline "push" gain is small and bounded; we avoid claiming big pace deltas.
    # If pace is already degrading or inconsistent, pushing has higher downside.
    push_gain_s = 0.35 * horizon_laps
    push_wrong_loss_s = (0.7 + (2.2 * degradation_per_lap) + (1.6 * inconsistency)) * horizon_laps

    conserve_gain_s = 0.0
    conserve_wrong_loss_s = push_gain_s

    # Choose the worse-case loss as the risk communication number.
    wrong_loss = max(push_wrong_loss_s, conserve_wrong_loss_s)
    right_gain = max(push_gain_s, conserve_gain_s)

    wrong_loss = round(float(wrong_loss), 2)
    right_gain = round(float(right_gain), 2)
    return DecisionImpactResult(
        decision="push_vs_conserve",
        horizon_laps=horizon_laps,
        if_right_expected_gain_s=right_gain,
        if_wrong_expected_loss_s=wrong_loss,
        risk_level=_risk_level_from_loss(wrong_loss),
        assumptions={
            "degradation_per_lap_s": round(float(degradation_per_lap), 3),
            "min_speed_consistency": round(float(min_consistency), 3),
            "horizon_laps": float(horizon_laps),
            "push_gain_per_lap_s": 0.35,
        },
        notes=[
            f"Degradation proxy ~{round(degradation_per_lap, 2)}s/lap over recent window.",
            f"Speed consistency proxy min={round(min_consistency, 2)}.",
            "Push/conserve impact is heuristic due to lack of fuel/ERS/position signals.",
        ],
    )


def _recent_degradation(records: list, count: int = 5) -> float:
    usable = available_records(records)
    recent = recent_records(usable, count=count)
    if len(recent) < 2:
        return 0.0
    deltas = [recent[i + 1].lap_time_s - recent[i].lap_time_s for i in range(len(recent) - 1)]
    return max(0.0, mean(deltas))


def _zeroed(decision: str, horizon_laps: int, note: str) -> DecisionImpactResult:
    return DecisionImpactResult(
        decision=decision,
        horizon_laps=horizon_laps,
        if_right_expected_gain_s=0.0,
        if_wrong_expected_loss_s=0.0,
        risk_level="LOW",
        assumptions={"horizon_laps": float(horizon_laps)},
        notes=[note],
    )


def estimate_stint_length_impact(records: list, *, horizon_laps: int = 3) -> DecisionImpactResult:
    """Regret of mis-judging stint length (extending too long).

    Bounded heuristic: extending into rising degradation costs ~ degradation * horizon.
    Strong-ish: degradation is directly observable from lap times.
    """
    usable = available_records(records)
    if len(recent_records(usable, count=5)) < 2:
        return _zeroed("stint_length", horizon_laps, "Insufficient data to estimate stint-length impact.")
    deg = _recent_degradation(records)
    wrong_loss = round(deg * horizon_laps, 2)
    right_gain = round(deg * horizon_laps * 0.5, 2)
    return DecisionImpactResult(
        decision="stint_length",
        horizon_laps=horizon_laps,
        if_right_expected_gain_s=right_gain,
        if_wrong_expected_loss_s=wrong_loss,
        risk_level=_risk_level_from_loss(wrong_loss),
        assumptions={"degradation_per_lap_s": round(float(deg), 3), "horizon_laps": float(horizon_laps)},
        notes=[
            f"Extending the stint into ~{round(deg, 2)}s/lap degradation over {horizon_laps} laps.",
            "Heuristic: no tyre-life model; regret approximates cumulative pace loss.",
        ],
    )


def estimate_tyre_compound_impact(records: list, *, horizon_laps: int = 3) -> DecisionImpactResult:
    """Compound-choice risk. WEAK/heuristic: MDCE has no compound performance data.

    We can only proxy from degradation + whether the compound is known. Explicitly
    flagged low-confidence so it is never presented as a real compound model.
    """
    usable = available_records(records)
    recent = recent_records(usable, count=5)
    if len(recent) < 2:
        return _zeroed("tyre_strategy", horizon_laps, "Insufficient data to estimate tyre-compound impact.")
    deg = _recent_degradation(records)
    unknown = any(str(r.tyre_compound).upper() in {"UNKNOWN", ""} for r in recent[-3:])
    # Bounded, deliberately small; bump slightly if compound is unknown (more uncertainty).
    wrong_loss = round((deg * horizon_laps * 0.6) + (0.5 if unknown else 0.0), 2)
    return DecisionImpactResult(
        decision="tyre_strategy",
        horizon_laps=horizon_laps,
        if_right_expected_gain_s=0.0,
        if_wrong_expected_loss_s=wrong_loss,
        risk_level=_risk_level_from_loss(wrong_loss),
        assumptions={"degradation_per_lap_s": round(float(deg), 3), "compound_unknown": float(unknown), "horizon_laps": float(horizon_laps)},
        notes=[
            "LOW-CONFIDENCE heuristic: MDCE has no compound performance model.",
            "Proxy from degradation" + (" + unknown compound penalty." if unknown else "."),
        ],
    )


def estimate_safety_car_response_impact(records: list, *, horizon_laps: int = 3) -> DecisionImpactResult:
    """Regret of mishandling a Safety Car / VSC window.

    Uses real `track_status`. Pitting under SC typically saves a large bounded amount
    (reduced effective pit loss). Missing that window is the regret.
    """
    horizon = horizon_laps
    sc_recent = any(getattr(r, "track_status", "") in {"SC", "VSC"} for r in records[-4:])
    if sc_recent:
        wrong_loss = 8.0  # illustrative, bounded: forfeited SC pit-time saving
        notes = [
            "Safety Car / VSC detected in recent laps.",
            "Illustrative bounded estimate: not pitting under SC can forfeit a large time saving.",
        ]
    else:
        wrong_loss = 0.5
        notes = ["No recent SC/VSC; safety-car response impact is low."]
    wrong_loss = round(float(wrong_loss), 2)
    return DecisionImpactResult(
        decision="safety_car_response",
        horizon_laps=horizon,
        if_right_expected_gain_s=round(wrong_loss * 0.5, 2),
        if_wrong_expected_loss_s=wrong_loss,
        risk_level=_risk_level_from_loss(wrong_loss),
        assumptions={"sc_recent": float(sc_recent), "horizon_laps": float(horizon)},
        notes=notes,
    )


def estimate_traffic_rejoin_impact(records: list, *, horizon_laps: int = 3) -> DecisionImpactResult:
    """Traffic / rejoin risk. Proxy-dependent: relies on gap_to_car_ahead + consistency.

    If gap data looks like a placeholder (mostly ~0), we downscope and say so.
    """
    usable = available_records(records)
    recent = recent_records(usable, count=6)
    if len(recent) < 3:
        return _zeroed("traffic_rejoin_risk", horizon_laps, "Insufficient data to estimate traffic/rejoin impact.")
    gaps = [float(r.gap_to_car_ahead_s) for r in recent]
    placeholder = (sum(1 for g in gaps if abs(g) <= 0.01) / len(gaps)) >= 0.9
    min_consistency = min(float(r.speed_consistency) for r in recent)
    inconsistency = max(0.0, 1.0 - min_consistency)
    if placeholder:
        wrong_loss = round(0.5 + 1.5 * inconsistency, 2)
        notes = [
            "Gap-to-car-ahead looks like a placeholder (mostly 0.0); estimate is downscoped.",
            "Traffic/rejoin risk should be treated as uncertain until real gap data is available.",
        ]
    else:
        small_gap_factor = max(0.0, 1.0 - (min(gaps) / 2.0)) if gaps else 0.0
        wrong_loss = round(1.0 + (2.0 * inconsistency) + small_gap_factor, 2)
        notes = [
            f"Min gap ~{round(min(gaps), 2)}s; inconsistency proxy {round(inconsistency, 2)}.",
            "Heuristic: rejoining near traffic with volatile pace raises rejoin risk.",
        ]
    return DecisionImpactResult(
        decision="traffic_rejoin_risk",
        horizon_laps=horizon_laps,
        if_right_expected_gain_s=0.0,
        if_wrong_expected_loss_s=wrong_loss,
        risk_level=_risk_level_from_loss(wrong_loss),
        assumptions={"gap_placeholder": float(placeholder), "min_speed_consistency": round(min_consistency, 3), "horizon_laps": float(horizon_laps)},
        notes=notes,
    )


def estimate_aggressive_vs_safe_impact(records: list, *, horizon_laps: int = 3) -> DecisionImpactResult:
    """Regret of choosing AGGRESSIVE when conditions are degrading.

    Closely related to push_vs_conserve; surfaced as its own domain for the trust map.
    """
    usable = available_records(records)
    recent = recent_records(usable, count=6)
    if len(recent) < 3:
        return _zeroed("aggressive_vs_safe_strategy", horizon_laps, "Insufficient data to estimate aggressive/safe impact.")
    deg = _recent_degradation(records, count=6)
    min_consistency = min(float(r.speed_consistency) for r in recent)
    inconsistency = max(0.0, 1.0 - min_consistency)
    wrong_loss = round((0.6 + (2.0 * deg) + (1.4 * inconsistency)) * horizon_laps / 2.0, 2)
    return DecisionImpactResult(
        decision="aggressive_vs_safe_strategy",
        horizon_laps=horizon_laps,
        if_right_expected_gain_s=round(0.3 * horizon_laps, 2),
        if_wrong_expected_loss_s=wrong_loss,
        risk_level=_risk_level_from_loss(wrong_loss),
        assumptions={"degradation_per_lap_s": round(float(deg), 3), "min_speed_consistency": round(min_consistency, 3), "horizon_laps": float(horizon_laps)},
        notes=[
            "Regret of pushing when pace is degrading / inconsistent.",
            "Related to push_vs_conserve; bounded heuristic (no fuel/ERS/position model).",
        ],
    )
