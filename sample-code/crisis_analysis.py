"""
crisis_analysis.py — Crisis Period Performance Analysis
========================================================
Analyzes how ProjectR performed during historically significant
market stress events.

This is the analysis every quant interviewer asks about:
    "How did you do in 2008? In COVID? In the 2022 rate shock?"

A strategy that looks great in-sample but collapses in crises is
not a strategy — it's a bull market survivor. ProjectR's regime
classifier is specifically designed to go defensive before drawdowns
compound. This module measures whether it actually did.

Key questions answered per crisis period:
    1. Did the strategy protect capital vs SPY?
    2. How quickly did the regime classifier go defensive?
       (regime_response_lag_days — lower is better)
    3. What was average gross exposure during the stress period?
    4. What was the dominant regime detected?
    5. How does max drawdown compare to benchmark?

Crisis periods covered:
    GFC peak:         Sep 2008 – Mar 2009  (Lehman, peak drawdown)
    GFC full:         Oct 2007 – Jun 2009  (full bear market)
    Euro crisis:      Jul 2011 – Oct 2011  (sovereign debt contagion)
    China shock:      Aug 2015 – Sep 2015  (flash crash)
    COVID crash:      Feb 19 – Mar 23 2020 (fastest bear market ever)
    COVID recovery:   Mar 23 – Aug 2020    (V-shaped recovery)
    Rate shock 2022:  Jan – Dec 2022       (worst bonds/stocks year in decades)
    SVB crisis:       Mar – May 2023       (regional banking contagion)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical crisis periods — dates are inclusive market-session bounds
CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "GFC_peak":        ("2008-09-01", "2009-03-31"),
    "GFC_full":        ("2007-10-01", "2009-06-30"),
    "Euro_crisis":     ("2011-07-01", "2011-10-31"),
    "China_shock":     ("2015-08-01", "2015-09-30"),
    "COVID_crash":     ("2020-02-19", "2020-03-23"),
    "COVID_recovery":  ("2020-03-23", "2020-08-31"),
    "Rate_shock_2022": ("2022-01-01", "2022-12-31"),
    "SVB_crisis":      ("2023-03-01", "2023-05-31"),
}

# Regime labels that constitute a "defensive" state
DEFENSIVE_REGIMES: set[str] = {
    "BEARISH_HIGHVOL",
    "BEARISH_LOWVOL",
    "CHOPPY_HIGHVOL",
}


def crisis_performance_report(
    strategy_equity: pd.Series,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    regime_series: pd.Series | None = None,
    weights_df: pd.DataFrame | None = None,
    crisis_periods: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Generate per-crisis performance comparison vs benchmark.

    Args:
        strategy_equity:   Cumulative equity curve
        strategy_returns:  Daily strategy returns
        benchmark_returns: Daily benchmark (SPY) returns
        regime_series:     Daily regime labels (optional)
                           Used to compute response lag and dominant regime
        weights_df:        Daily portfolio weights DataFrame (optional)
                           Used to compute average gross exposure
        crisis_periods:    Custom crisis dict (default: CRISIS_PERIODS)

    Returns:
        pd.DataFrame with one row per crisis period, columns:
            period_name, start, end, n_days,
            strategy_return, benchmark_return, excess_return,
            strategy_maxdd, benchmark_maxdd, dd_reduction,
            dominant_regime, avg_gross_exposure,
            regime_response_lag_days

    Interpretation guide:
        excess_return > 0:      Strategy outperformed benchmark
        dd_reduction > 0:       Strategy had shallower drawdown
        regime_response_lag:    Days until defensive regime triggered
                                (<5 days is excellent, >20 days is slow)
        avg_gross_exposure:     How much the system de-risked
                                (<0.7 in bear market = meaningful defense)
    """
    if crisis_periods is None:
        crisis_periods = CRISIS_PERIODS

    # Build benchmark equity curve
    bench_eq = (1.0 + benchmark_returns).cumprod()
    bench_eq = bench_eq / bench_eq.iloc[0]

    rows = []

    for name, (start, end) in crisis_periods.items():
        # Check data coverage
        mask = (strategy_equity.index >= start) & (strategy_equity.index <= end)
        if mask.sum() < 5:
            continue

        # Strategy metrics
        strat_sub = strategy_equity.loc[start:end]
        strat_ret = float(strat_sub.iloc[-1] / strat_sub.iloc[0] - 1.0) \
            if len(strat_sub) >= 2 else np.nan

        rolling_max = strat_sub.cummax()
        strat_dd = float(((strat_sub / rolling_max) - 1.0).min()) \
            if len(strat_sub) >= 2 else np.nan

        # Benchmark metrics
        bench_sub = bench_eq.loc[start:end]
        bench_ret = float(bench_sub.iloc[-1] / bench_sub.iloc[0] - 1.0) \
            if len(bench_sub) >= 2 else np.nan

        bench_roll_max = bench_sub.cummax()
        bench_dd = float(((bench_sub / bench_roll_max) - 1.0).min()) \
            if len(bench_sub) >= 2 else np.nan

        # Derived metrics
        excess = strat_ret - bench_ret \
            if not (np.isnan(strat_ret) or np.isnan(bench_ret)) else np.nan
        dd_red = strat_dd - bench_dd \
            if not (np.isnan(strat_dd) or np.isnan(bench_dd)) else np.nan

        # Regime analysis (if available)
        dom_regime = "N/A"
        avg_exp    = np.nan
        lag        = np.nan

        if regime_series is not None:
            reg_sub = regime_series.loc[start:end]
            if not reg_sub.empty:
                dom_regime = str(reg_sub.mode().iloc[0])

            # Response lag: days from crisis start to first defensive regime
            reg_from_start = regime_series.loc[start:]
            lag = np.nan
            for i, label in enumerate(reg_from_start):
                if label in DEFENSIVE_REGIMES:
                    lag = float(i)
                    break

        if weights_df is not None:
            w_sub = weights_df.loc[start:end]
            if not w_sub.empty:
                avg_exp = float(w_sub.abs().sum(axis=1).mean())

        rows.append({
            "period_name":              name,
            "start":                    start,
            "end":                      end,
            "n_days":                   int(mask.sum()),
            "strategy_return":          strat_ret,
            "benchmark_return":         bench_ret,
            "excess_return":            excess,
            "strategy_maxdd":           strat_dd,
            "benchmark_maxdd":          bench_dd,
            "dd_reduction":             dd_red,
            "dominant_regime":          dom_regime,
            "avg_gross_exposure":       avg_exp,
            "regime_response_lag_days": lag,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["_sort"] = pd.to_datetime(df["start"])
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
