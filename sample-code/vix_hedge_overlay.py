import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HedgeSignal:
    vix_level: float
    vix_3m_level: float
    term_structure_inverted: bool
    vix_spike: bool
    drawdown_trigger: bool
    hedge_intensity: float
    hedge_active: bool
    trigger_reason: Optional[str]


SOFT_THRESHOLD = 0.20
HARD_THRESHOLD = 0.45
VIX_SPIKE_THRESHOLD = 0.30
DRAWDOWN_TRIGGER = 0.07
TERM_STRUCTURE_INVERSION_THRESHOLD = 1.0


def compute_vix_signal(
    vix_level: float,
    vix_3m_level: float,
    portfolio_drawdown: float,
    vix_1w_change: float,
) -> float:
    """
    Combines VIX level, term structure, spike velocity, and portfolio
    drawdown into a single hedge intensity score between 0 and 1.

    [Core signal construction logic is not public]
    """
    term_structure_ratio = vix_level / vix_3m_level if vix_3m_level > 0 else 1.0
    term_inverted = term_structure_ratio >= TERM_STRUCTURE_INVERSION_THRESHOLD

    spike = vix_1w_change >= VIX_SPIKE_THRESHOLD
    dd_trigger = abs(portfolio_drawdown) >= DRAWDOWN_TRIGGER

    signals_active = sum([term_inverted, spike, dd_trigger])
    base_intensity = signals_active / 3.0

    return float(np.clip(base_intensity, 0.0, 1.0))


def get_hedge_signal(
    vix_level: float,
    vix_3m_level: float,
    portfolio_drawdown: float,
    vix_1w_change: float,
    soft_threshold: float = SOFT_THRESHOLD,
    hard_threshold: float = HARD_THRESHOLD,
) -> HedgeSignal:
    term_inverted = (vix_level / vix_3m_level) >= TERM_STRUCTURE_INVERSION_THRESHOLD if vix_3m_level > 0 else False
    spike = vix_1w_change >= VIX_SPIKE_THRESHOLD
    dd_trigger = abs(portfolio_drawdown) >= DRAWDOWN_TRIGGER

    intensity = compute_vix_signal(vix_level, vix_3m_level, portfolio_drawdown, vix_1w_change)
    active = intensity >= soft_threshold

    reason = None
    if active:
        triggers = []
        if term_inverted:
            triggers.append("term_structure_inverted")
        if spike:
            triggers.append("vix_spike")
        if dd_trigger:
            triggers.append("drawdown_trigger")
        reason = " + ".join(triggers) if triggers else "composite_threshold"

    return HedgeSignal(
        vix_level=vix_level,
        vix_3m_level=vix_3m_level,
        term_structure_inverted=term_inverted,
        vix_spike=spike,
        drawdown_trigger=dd_trigger,
        hedge_intensity=intensity,
        hedge_active=active,
        trigger_reason=reason,
    )


def apply_hedge_to_weights(
    weights: dict[str, float],
    hedge_signal: HedgeSignal,
    soft_threshold: float = SOFT_THRESHOLD,
    hard_threshold: float = HARD_THRESHOLD,
) -> dict[str, float]:
    """
    Scales down position weights based on hedge intensity.
    Soft threshold: partial reduction.
    Hard threshold: significant gross scale reduction.

    [Position sizing under hedge conditions is not public]
    """
    if not hedge_signal.hedge_active:
        return weights

    intensity = hedge_signal.hedge_intensity

    if intensity >= hard_threshold:
        scale = 0.5
    elif intensity >= soft_threshold:
        scale = 1.0 - ((intensity - soft_threshold) / (hard_threshold - soft_threshold)) * 0.3
    else:
        scale = 1.0

    return {ticker: w * scale for ticker, w in weights.items()}
