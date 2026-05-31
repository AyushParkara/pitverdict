from __future__ import annotations

from statistics import pstdev, mean

from .data_loader import available_records, recent_records
from .models import Severity, TrustIssue


def _is_effectively_constant(values: list[float], *, tol: float) -> bool:
    if len(values) < 3:
        return False
    return pstdev(values) <= tol


def detect_coverage_gaps(records: list) -> list[TrustIssue]:
    """Detect likely data coverage gaps from the already-mapped LapRecord stream.

    Constraint: we do not rely on loader provenance. This keeps the intelligence
    layer self-contained and still able to flag "derived/placeholder" patterns.
    """

    issues: list[TrustIssue] = []
    usable = available_records(records)
    recent = recent_records(usable, count=10)
    if not recent:
        return issues

    # 1) Sector timing coverage.
    # If sector splits are perfectly proportional to lap time across the window,
    # it's a strong hint they are synthetic/derived placeholders.
    lap_times = [r.lap_time_s for r in recent]
    if all(t > 0 for t in lap_times):
        r1 = [r.sector1_s / r.lap_time_s for r in recent]
        r2 = [r.sector2_s / r.lap_time_s for r in recent]
        r3 = [r.sector3_s / r.lap_time_s for r in recent]
        # Placeholder sector splits in MDCE are produced by the loader using fixed ratios:
        # sector1=0.32*lap, sector2=0.37*lap, sector3=0.31*lap.
        #
        # Real sector data can be stable, but it should not consistently match those
        # synthetic ratios. We therefore use a *fingerprint* check:
        # the ratios are in a tight band AND they are close to the synthetic constants.
        def _range(vals: list[float]) -> float:
            return max(vals) - min(vals) if vals else 0.0

        mean_r1, mean_r2, mean_r3 = mean(r1), mean(r2), mean(r3)
        close_to_synth = (
            abs(mean_r1 - 0.32) <= 0.01
            and abs(mean_r2 - 0.37) <= 0.02
            and abs(mean_r3 - 0.31) <= 0.02
        )
        tight_band = _range(r1) <= 0.005 and _range(r2) <= 0.010 and _range(r3) <= 0.015
        if close_to_synth and tight_band:
            issues.append(
                TrustIssue(
                    issue="coverage_gap_sector_times",
                    severity=Severity.MEDIUM,
                    affected_decisions=["traffic_rejoin_risk", "push_vs_conserve"],
                    reason="Sector times look like estimated values rather than real data; be cautious with sector-level analysis.",
                    penalty=0.06,
                )
            )

    # 2) Gap-to-car-ahead coverage.
    # A flat 0.0 stream is usually a missing placeholder.
    gaps = [r.gap_to_car_ahead_s for r in recent]
    if gaps and sum(1 for g in gaps if abs(g) <= 0.01) / len(gaps) >= 0.9:
        issues.append(
            TrustIssue(
                issue="coverage_gap_track_gaps",
                severity=Severity.MEDIUM,
                affected_decisions=["traffic_rejoin_risk"],
                reason="Gap-to-car-ahead is mostly 0.0 (placeholder). Traffic/rejoin risk estimates are unreliable.",
                penalty=0.08,
            )
        )

    # 3) Tyre compound coverage.
    if any(str(r.tyre_compound).upper() in {"UNKNOWN", ""} for r in recent[-3:]):
        issues.append(
            TrustIssue(
                issue="coverage_gap_tyre_compound",
                severity=Severity.MEDIUM,
                affected_decisions=["tyre_strategy", "tyre_compound_choice"],
                reason="Tyre compound is unknown in the recent window; compound-dependent decisions are unreliable.",
                penalty=0.07,
            )
        )

    return issues
