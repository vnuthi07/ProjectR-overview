import pandas as pd
import numpy as np


REGIME_LOOKBACKS = {
    "BULLISH_LOWVOL":   {"primary": 252, "secondary": 126},
    "BULLISH_HIGHVOL":  {"primary": 252, "secondary": 126},
    "SIDEWAYS_LOWVOL":  {"primary": 63,  "secondary": 21},
    "SIDEWAYS_HIGHVOL": {"primary": 63,  "secondary": 21},
    "BEARISH_LOWVOL":   {"primary": 126, "secondary": 63},
    "BEARISH_HIGHVOL":  {"primary": 126, "secondary": 63},
}

PRIMARY_WEIGHT = 0.7
SECONDARY_WEIGHT = 0.3


def compute_momentum_score(
    prices: pd.Series,
    primary_lookback: int,
    secondary_lookback: int,
    skip_days: int = 21,
) -> float:
    """
    Computes a blended momentum score from two lookback windows.
    skip_days removes the most recent month to avoid short-term reversal.
    """
    if len(prices) < primary_lookback + skip_days:
        return np.nan

    def lookback_return(lb: int) -> float:
        start_idx = -(lb + skip_days)
        end_idx = -skip_days if skip_days > 0 else len(prices)
        p_start = prices.iloc[start_idx]
        p_end = prices.iloc[end_idx]
        if p_start <= 0:
            return np.nan
        return float(p_end / p_start - 1)

    primary = lookback_return(primary_lookback)
    secondary = lookback_return(secondary_lookback)

    if np.isnan(primary) or np.isnan(secondary):
        return np.nan

    return PRIMARY_WEIGHT * primary + SECONDARY_WEIGHT * secondary


def compute_regime_adaptive_scores(
    price_data: pd.DataFrame,
    regime: str,
    skip_days: int = 21,
) -> pd.Series:
    """
    Computes momentum scores for all assets using lookback periods
    appropriate for the current market regime.
    """
    if regime not in REGIME_LOOKBACKS:
        raise ValueError(f"Unknown regime: {regime}. Must be one of {list(REGIME_LOOKBACKS)}")

    lookbacks = REGIME_LOOKBACKS[regime]
    primary = lookbacks["primary"]
    secondary = lookbacks["secondary"]

    scores = {}
    for ticker in price_data.columns:
        scores[ticker] = compute_momentum_score(
            price_data[ticker], primary, secondary, skip_days
        )

    return pd.Series(scores).dropna()


def normalize_scores(scores: pd.Series) -> pd.Series:
    """Cross-sectional z-score normalization."""
    mean = scores.mean()
    std = scores.std()
    if std == 0:
        return pd.Series(0.0, index=scores.index)
    return (scores - mean) / std


def rank_scores(scores: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank normalization."""
    return scores.rank(pct=True)
