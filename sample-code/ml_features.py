import pandas as pd
import numpy as np
from typing import Optional


def regime_duration_feature(
    regime_series: pd.Series,
    current_date: pd.Timestamp,
    current_regime: str,
) -> float:
    """Days current regime has been active — longer regimes are more stable."""
    past = regime_series.loc[:current_date]
    if len(past) == 0:
        return 0.0
    reversed_regimes = past[::-1]
    duration = 0
    for r in reversed_regimes:
        if r == current_regime:
            duration += 1
        else:
            break
    return float(duration)


def regime_transition_probability(
    regime_series: pd.Series,
    current_date: pd.Timestamp,
    window: int = 60,
) -> float:
    """Rolling frequency of regime changes — high = unstable environment."""
    past = regime_series.loc[:current_date].iloc[-window:]
    if len(past) < 2:
        return 0.0
    transitions = (past != past.shift(1)).sum() - 1
    return float(transitions / len(past))


def cross_sectional_rank(
    scores: pd.Series,
) -> pd.Series:
    """Percentile rank within universe — removes level bias."""
    return scores.rank(pct=True)


def bollinger_zscore(
    prices: pd.Series,
    window: int = 20,
) -> float:
    """
    Distance from moving average in standard deviation units.
    Negative = oversold, Positive = overbought.
    """
    if len(prices) < window:
        return np.nan
    rolling_mean = prices.iloc[-window:].mean()
    rolling_std = prices.iloc[-window:].std()
    if rolling_std == 0:
        return 0.0
    return float((prices.iloc[-1] - rolling_mean) / rolling_std)


def volatility_acceleration(
    returns: pd.Series,
    short_window: int = 5,
    long_window: int = 21,
) -> float:
    """
    Ratio of short-term to long-term realized volatility.
    > 1 = vol expanding, < 1 = vol contracting.
    """
    if len(returns) < long_window:
        return np.nan
    short_vol = returns.iloc[-short_window:].std() * np.sqrt(252)
    long_vol = returns.iloc[-long_window:].std() * np.sqrt(252)
    if long_vol == 0:
        return 1.0
    return float(short_vol / long_vol)


def distance_from_52w_high(prices: pd.Series) -> float:
    """Fraction below 52-week high — captures mean reversion potential."""
    if len(prices) < 252:
        window = prices
    else:
        window = prices.iloc[-252:]
    high = window.max()
    current = prices.iloc[-1]
    if high == 0:
        return 0.0
    return float((current - high) / high)


def correlation_to_top_performer(
    asset_returns: pd.Series,
    top_performer_returns: pd.Series,
    window: int = 21,
) -> float:
    """
    Rolling correlation to current top momentum asset.
    High correlation = crowding risk.
    """
    if len(asset_returns) < window:
        return np.nan
    r1 = asset_returns.iloc[-window:]
    r2 = top_performer_returns.iloc[-window:]
    aligned = pd.concat([r1, r2], axis=1).dropna()
    if len(aligned) < 5:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def build_ml_feature_vector(
    ticker: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_series: pd.Series,
    current_date: pd.Timestamp,
    current_regime: str,
    momentum_scores: pd.Series,
) -> dict:
    """
    Assembles the full orthogonal feature vector for a single asset.
    Features are designed to be independent of raw momentum signals.

    [Feature weights, selection logic, and model architecture are not public]
    """
    p = prices[ticker].loc[:current_date]
    r = returns[ticker].loc[:current_date]

    top_ticker = momentum_scores.idxmax()
    top_returns = returns[top_ticker].loc[:current_date] if top_ticker != ticker else r

    features = {
        "regime_duration": regime_duration_feature(regime_series, current_date, current_regime),
        "regime_transition_prob": regime_transition_probability(regime_series, current_date),
        "bb_zscore": bollinger_zscore(p),
        "vol_acceleration": volatility_acceleration(r),
        "distance_from_52w_high": distance_from_52w_high(p),
        "cross_sectional_rank": float(momentum_scores.rank(pct=True).get(ticker, np.nan)),
        "correlation_to_leader": correlation_to_top_performer(r, top_returns),
    }

    return features


def build_feature_matrix(
    tickers: list[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_series: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    momentum_scores_history: dict,
) -> pd.DataFrame:
    """Builds the full feature matrix across all assets and rebalance dates."""
    rows = []
    for date in rebalance_dates:
        if date not in regime_series.index:
            continue
        regime = regime_series.loc[date]
        scores = momentum_scores_history.get(date, pd.Series(dtype=float))

        for ticker in tickers:
            try:
                features = build_ml_feature_vector(
                    ticker, prices, returns, regime_series,
                    date, regime, scores
                )
                features["ticker"] = ticker
                features["date"] = date
                rows.append(features)
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index(["date", "ticker"])
    return df
