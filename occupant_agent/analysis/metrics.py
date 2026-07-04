"""
Validation metrics for OccupantAgent behavioral and energy accuracy.

Two-tier validation design:
  Tier 1 (behavioral): KL-divergence, KS-test — compare simulated vs. ATUS
    time-at-activity distributions.
  Tier 2 (energy): CVRMSE, MBE — compare simulated vs. measured energy profiles.

All functions accept numpy arrays or lists and return plain Python floats.
No side effects; safe to call in parallel evaluation loops.

References
──────────
  ASHRAE Guideline 14-2014: CVRMSE < 30%, MBE < 10% for whole-building monthly.
  KL divergence: Kullback & Leibler (1951). MBE/CVRMSE: ASHRAE 14, Eq. 4 & 5.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# ── Tier 1: behavioral ────────────────────────────────────────────────────────

def compute_kl(
    p: Sequence[float],
    q: Sequence[float],
    epsilon: float = 1e-9,
) -> float:
    """
    KL divergence KL(P ‖ Q) — simulated (P) vs. reference (Q).

    Both sequences are treated as probability distributions (normalized if needed).
    Zero-valued bins in Q are regularized by epsilon.

    Args:
        p:       Simulated distribution (e.g., hourly fraction at each activity).
        q:       Reference distribution (ATUS empirical).
        epsilon: Floor for Q bins to avoid log(0).

    Returns:
        KL divergence in nats. Lower is better; 0 = identical distributions.
    """
    p_arr = list(p)
    q_arr = list(q)
    if len(p_arr) != len(q_arr):
        raise ValueError(f"Length mismatch: len(p)={len(p_arr)}, len(q)={len(q_arr)}")

    p_sum = sum(p_arr)
    q_sum = sum(q_arr)
    if p_sum == 0 or q_sum == 0:
        raise ValueError("Cannot compute KL divergence for all-zero distribution.")
    p_norm = [x / p_sum for x in p_arr]
    q_norm = [max(x / q_sum, epsilon) for x in q_arr]

    return sum(
        pi * math.log(pi / qi)
        for pi, qi in zip(p_norm, q_norm)
        if pi > 0
    )


def compute_ks(
    p: Sequence[float],
    q: Sequence[float],
) -> float:
    """
    Kolmogorov–Smirnov statistic D = max|CDF_P(x) - CDF_Q(x)|.

    Treats inputs as (already-ordered) probability mass functions and computes
    the maximum absolute difference between their empirical CDFs.

    **Ordering contract**: the KS statistic is order-dependent. Both p and q
    must use the same fixed category order on every call. In evaluate.py this
    is enforced by the `categories` list defined at the call site. Changing
    the order produces a different numeric result, so callers must not vary it
    across runs or strata.

    Args:
        p: Simulated PMF. Must be in the same fixed category order as q.
        q: Reference PMF. Must be in the same fixed category order as p.

    Returns:
        KS statistic in [0, 1]. Lower is better.
    """
    p_arr = list(p)
    q_arr = list(q)
    if len(p_arr) != len(q_arr):
        raise ValueError(f"Length mismatch: len(p)={len(p_arr)}, len(q)={len(q_arr)}")

    p_sum = sum(p_arr) or 1.0
    q_sum = sum(q_arr) or 1.0
    p_norm = [x / p_sum for x in p_arr]
    q_norm = [x / q_sum for x in q_arr]

    cum_p, cum_q, max_d = 0.0, 0.0, 0.0
    for pi, qi in zip(p_norm, q_norm):
        cum_p += pi
        cum_q += qi
        max_d = max(max_d, abs(cum_p - cum_q))
    return max_d


def compute_kl_by_hour(
    simulated_counts: dict[int, dict[str, float]],
    reference_pcts: dict[int, dict[str, float]],
    epsilon: float = 1e-9,
) -> dict[int, float]:
    """
    Compute per-hour KL divergence over activity categories.

    Args:
        simulated_counts: {hour: {category: count}} from simulation log.
        reference_pcts:   {hour: {category: weighted_pct}} from ATUS CSVs.

    Returns:
        {hour: kl_divergence} for each hour in simulated_counts.
    """
    results: dict[int, float] = {}
    all_categories = sorted(
        {cat for h in simulated_counts.values() for cat in h}
        | {cat for h in reference_pcts.values() for cat in h}
    )
    for h, sim_h in simulated_counts.items():
        ref_h = reference_pcts.get(h, {})
        p = [sim_h.get(c, 0.0) for c in all_categories]
        q = [ref_h.get(c, epsilon) for c in all_categories]
        try:
            results[h] = compute_kl(p, q, epsilon=epsilon)
        except ValueError:
            results[h] = float("nan")
    return results


# ── Tier 2: energy ────────────────────────────────────────────────────────────

def compute_cvrmse(
    measured: Sequence[float],
    simulated: Sequence[float],
) -> float:
    """
    Coefficient of Variation of Root Mean Square Error (CV-RMSE).

    ASHRAE Guideline 14 definition:
        CVRMSE = RMSE / mean(measured)

    Acceptance threshold: < 30% for whole-building monthly energy.

    Args:
        measured:   Measured energy values (kWh or W — must be consistent).
        simulated:  Simulated energy values in the same units and order.

    Returns:
        CVRMSE as a fraction (multiply by 100 for %). Lower is better.
    """
    measured_arr = list(measured)
    simulated_arr = list(simulated)
    if len(measured_arr) != len(simulated_arr):
        raise ValueError(
            f"Length mismatch: len(measured)={len(measured_arr)}, "
            f"len(simulated)={len(simulated_arr)}"
        )
    n = len(measured_arr)
    if n == 0:
        raise ValueError("Empty sequences.")

    mean_m = sum(measured_arr) / n
    if mean_m == 0:
        raise ValueError("Mean of measured values is zero; CVRMSE undefined.")

    mse = sum((m - s) ** 2 for m, s in zip(measured_arr, simulated_arr)) / n
    return math.sqrt(mse) / mean_m


def compute_mbe(
    measured: Sequence[float],
    simulated: Sequence[float],
) -> float:
    """
    Mean Bias Error (MBE) — signed systematic bias.

    ASHRAE Guideline 14 definition:
        MBE = mean(simulated - measured) / mean(measured)

    Acceptance threshold: |MBE| < 10% for whole-building monthly energy.
    Positive = simulation over-predicts; negative = under-predicts.

    Args:
        measured:   Measured energy values.
        simulated:  Simulated energy values.

    Returns:
        MBE as a signed fraction (multiply by 100 for %). Near 0 is better.
    """
    measured_arr = list(measured)
    simulated_arr = list(simulated)
    if len(measured_arr) != len(simulated_arr):
        raise ValueError(
            f"Length mismatch: len(measured)={len(measured_arr)}, "
            f"len(simulated)={len(simulated_arr)}"
        )
    n = len(measured_arr)
    if n == 0:
        raise ValueError("Empty sequences.")

    mean_m = sum(measured_arr) / n
    if mean_m == 0:
        raise ValueError("Mean of measured values is zero; MBE undefined.")

    return sum(s - m for m, s in zip(measured_arr, simulated_arr)) / n / mean_m
