"""
regime_smoother.py — Hysteresis Filter for Regime Transitions
==============================================================
Prevents regime whipsawing by requiring a new regime label to persist
for `confirm_days` consecutive trading days before adopting it.

Why this matters:
    Without smoothing, a noisy classifier might flip between BULLISH and
    BEARISH on consecutive days, generating excessive portfolio turnover
    and transaction costs. The hysteresis filter acts as a low-pass filter
    on the regime signal.

    The tradeoff: longer confirm_days reduces false positives (unnecessary
    regime switches) but increases response lag to genuine regime changes.
    ProjectR uses 3 days — reduced from ProjectV's 5 days to lower lag
    while still filtering one-day noise events.

Algorithm:
    Walk forward through the raw label series.
    Track the current confirmed regime and a candidate regime.
    When a new label appears, start counting consecutive days.
    If it holds for confirm_days days, adopt it as confirmed.
    If the candidate resets before confirm_days, discard it.
"""

from __future__ import annotations

import pandas as pd


def confirm_regime(
    labels: pd.Series,
    confirm_days: int,
) -> pd.Series:
    """
    Apply hysteresis filter to raw regime labels.

    Args:
        labels:       Raw daily regime label series (e.g. from classifier)
        confirm_days: Consecutive days required to adopt a new regime

    Returns:
        Smoothed regime label series (same index as input)

    Example:
        Raw:      [BULL, BULL, BEAR, BULL, BEAR, BEAR, BEAR]
        confirm=3: [BULL, BULL, BULL, BULL, BULL, BULL, BEAR]

        The single BEAR on day 3 and BEAR on days 5-6 don't trigger
        a transition because they don't persist for 3 days.
        Only the BEAR starting day 5 that holds through day 7 confirms.
    """
    if len(labels) == 0:
        return labels.copy()

    confirmed      = labels.iloc[0]
    candidate      = labels.iloc[0]
    candidate_count = 1

    result = [confirmed]

    for i in range(1, len(labels)):
        current = labels.iloc[i]

        if current == confirmed:
            # Still in confirmed regime — reset any pending candidate
            candidate       = confirmed
            candidate_count = 1
            result.append(confirmed)

        elif current == candidate:
            # Candidate persisting — increment counter
            candidate_count += 1
            if candidate_count >= confirm_days:
                confirmed = candidate
            result.append(confirmed)

        else:
            # New candidate — start fresh count
            candidate       = current
            candidate_count = 1
            # Note: if confirm_days == 1, adopt immediately
            if candidate_count >= confirm_days:
                confirmed = candidate
            result.append(confirmed)

    return pd.Series(result, index=labels.index, dtype=object)


def regime_transition_log(
    smoothed_labels: pd.Series,
) -> pd.DataFrame:
    """
    Extract all regime transitions from a smoothed label series.

    Useful for:
        - Analyzing how often the system switches regimes
        - Measuring response time to market events
        - Debugging the classifier + smoother pipeline

    Args:
        smoothed_labels: Output of confirm_regime()

    Returns:
        DataFrame with columns: date, from_regime, to_regime, days_in_prior_regime
    """
    transitions = []
    prev_label  = smoothed_labels.iloc[0]
    days_in_regime = 1

    for dt, label in smoothed_labels.iloc[1:].items():
        if label != prev_label:
            transitions.append({
                "date":                dt,
                "from_regime":         prev_label,
                "to_regime":           label,
                "days_in_prior_regime": days_in_regime,
            })
            prev_label     = label
            days_in_regime = 1
        else:
            days_in_regime += 1

    return pd.DataFrame(transitions)
