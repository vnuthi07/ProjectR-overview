import numpy as np
import pandas as pd
from typing import Optional


def blend_weights(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    blend_ratio: float,
) -> dict[str, float]:
    all_assets = set(old_weights) | set(new_weights)
    blended = {}
    for asset in all_assets:
        old = old_weights.get(asset, 0.0)
        new = new_weights.get(asset, 0.0)
        blended[asset] = (1 - blend_ratio) * old + blend_ratio * new
    total = sum(abs(v) for v in blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


class RegimeTransitionSmoother:
    """
    Smooths portfolio weight transitions at regime boundaries to reduce
    whipsaw from abrupt regime changes. Blends old and new regime weights
    over a configurable number of rebalance periods.

    Crisis override: when VIX exceeds the crisis threshold, new BEARISH
    weights are applied immediately with no smoothing.
    """

    def __init__(
        self,
        n_blend_periods: int = 3,
        crisis_vix_threshold: float = 30.0,
    ):
        self.n_blend_periods = n_blend_periods
        self.crisis_vix_threshold = crisis_vix_threshold

        self._current_regime: Optional[str] = None
        self._transition_step: int = 0
        self._weights_at_transition: Optional[dict] = None

    def is_crisis(self, vix_level: float) -> bool:
        return vix_level >= self.crisis_vix_threshold

    def get_weights(
        self,
        target_weights: dict[str, float],
        new_regime: str,
        vix_level: float,
    ) -> dict[str, float]:
        regime_changed = new_regime != self._current_regime

        if self.is_crisis(vix_level):
            self._current_regime = new_regime
            self._transition_step = 0
            self._weights_at_transition = None
            return target_weights

        if regime_changed:
            self._weights_at_transition = self._last_weights or target_weights
            self._transition_step = 1
            self._current_regime = new_regime

        if self._transition_step > 0 and self._transition_step <= self.n_blend_periods:
            blend_ratio = self._transition_step / self.n_blend_periods
            smoothed = blend_weights(
                self._weights_at_transition, target_weights, blend_ratio
            )
            self._transition_step += 1
            self._last_weights = smoothed
            return smoothed

        self._last_weights = target_weights
        return target_weights

    def reset(self) -> None:
        self._current_regime = None
        self._transition_step = 0
        self._weights_at_transition = None
        self._last_weights = None


def compute_transition_statistics(regime_series: pd.Series) -> dict:
    transitions = (regime_series != regime_series.shift(1)).sum() - 1
    total_periods = len(regime_series)
    regime_counts = regime_series.value_counts()
    regime_fractions = regime_counts / total_periods

    return {
        "total_transitions": int(transitions),
        "transitions_per_year": float(transitions / (total_periods / 252)),
        "regime_distribution": regime_fractions.to_dict(),
        "most_common_regime": regime_counts.idxmax(),
    }
