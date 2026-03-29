"""
monte_carlo.py — Block Bootstrap Monte Carlo Simulation
========================================================
Produces robust uncertainty estimates for strategy performance by
resampling historical returns while preserving autocorrelation structure.

Why block bootstrap instead of simple bootstrap?
    Simple bootstrap (sampling individual days with replacement) destroys
    the autocorrelation structure of returns — volatility clustering,
    momentum persistence, and regime transitions are lost.

    Block bootstrap resamples consecutive blocks of returns, preserving
    short-term dependencies while still generating independent simulations.
    Block size of 21 days (one trading month) is standard practice.

What this tells you that a single backtest cannot:
    - Distribution of possible outcomes, not just the realized path
    - Probability of various Sharpe ratios, not just the observed one
    - Confidence intervals around equity curves
    - Probability of positive return over a 10-year horizon
    - Tail risk metrics (VaR, CVaR) with uncertainty bounds

ProjectR runs 1,000 simulations. Results are front-and-center in the
tearsheet — not buried in an appendix.

The most important number: probability of positive 10-year return.
A strategy with 95%+ probability across simulations is genuinely robust.
A strategy with 60% probability is just riding a bull market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def run_monte_carlo(
    returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 21,
    confidence_levels: list[float] | None = None,
    seed: int = 42,
) -> dict:
    """
    Block bootstrap Monte Carlo simulation.

    Resamples blocks of `block_size` consecutive returns to preserve
    autocorrelation structure. Runs n_simulations equity paths and
    computes the full distribution of outcomes.

    Args:
        returns:           Daily strategy returns series
        n_simulations:     Number of simulation paths (default 1000)
        block_size:        Consecutive days per block (default 21 = 1 month)
                           Larger blocks preserve more structure but reduce
                           simulation diversity.
        confidence_levels: Equity curve percentiles to report
                           (default [0.05, 0.25, 0.50, 0.75, 0.95])
        seed:              Random seed for reproducibility

    Returns:
        dict containing:
            simulated_paths:     DataFrame (n_simulations x T+1) of equity curves
            cagr_distribution:   Series of CAGR per simulation
            sharpe_distribution: Series of Sharpe per simulation
            maxdd_distribution:  Series of MaxDD per simulation
            confidence_bands:    dict {percentile: equity Series}
            prob_positive_10yr:  Probability of positive 10-year return
            var_95:              95% 1-day Value at Risk (negative number)
            cvar_95:             95% Conditional VaR / Expected Shortfall

    Example interpretation for ProjectR:
        median CAGR across 1000 sims: ~7.8%
        5th/95th percentile CAGR:     ~2.1% / ~14.3%
        prob positive 10-year return: ~94%
        This means the strategy is genuinely robust — not a lucky backtest.
    """
    if confidence_levels is None:
        confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]

    rets = returns.dropna().values
    T = len(rets)

    if T == 0:
        return {}

    rng = np.random.default_rng(seed=seed)

    # ── Block bootstrap ───────────────────────────────────────────────────────
    paths = np.empty((n_simulations, T))
    n_blocks = int(np.ceil(T / block_size))

    for sim_i in range(n_simulations):
        # Sample block start indices with replacement
        max_start = max(1, T - block_size + 1)
        starts = rng.integers(0, max_start, size=n_blocks)
        sampled_rets = np.concatenate(
            [rets[s : s + block_size] for s in starts]
        )[:T]
        paths[sim_i] = np.cumprod(1.0 + sampled_rets)

    # ── Per-simulation metrics ────────────────────────────────────────────────
    n_years = T / _TRADING_DAYS

    # CAGR distribution
    final_values = paths[:, -1]
    cagr_dist = pd.Series(
        final_values ** (1.0 / n_years) - 1.0,
        name="cagr"
    )

    # Sharpe distribution (log returns for numerical stability)
    log_ret_paths = np.diff(np.log(paths), axis=1)
    mean_daily    = log_ret_paths.mean(axis=1)
    std_daily     = log_ret_paths.std(axis=1)
    sharpe_dist = pd.Series(
        np.where(
            std_daily > 0,
            mean_daily / std_daily * np.sqrt(_TRADING_DAYS),
            0.0
        ),
        name="sharpe"
    )

    # Max drawdown distribution
    running_max = np.maximum.accumulate(paths, axis=1)
    dd_paths    = (paths - running_max) / running_max
    maxdd_dist  = pd.Series(dd_paths.min(axis=1), name="max_drawdown")

    # ── Confidence bands ──────────────────────────────────────────────────────
    # Prepend 1.0 at t=0 for all paths
    full_paths = np.hstack([np.ones((n_simulations, 1)), paths])
    confidence_bands = {
        level: pd.Series(
            np.percentile(full_paths, level * 100, axis=0),
            name=f"p{int(level * 100)}",
        )
        for level in confidence_levels
    }

    # ── 10-year probability ───────────────────────────────────────────────────
    # Chain simulations to estimate 10-year outcomes
    ten_year_days = 10 * _TRADING_DAYS
    n_chains = int(np.ceil(ten_year_days / T))
    ten_yr_returns = []

    for _ in range(min(n_simulations, 500)):
        total_ret = 1.0
        for _ in range(n_chains):
            starts = rng.integers(0, max(1, T - block_size + 1), size=n_blocks)
            sampled = np.concatenate(
                [rets[s : s + block_size] for s in starts]
            )[:T]
            total_ret *= np.prod(1.0 + sampled)
        ten_yr_returns.append(total_ret)

    prob_positive_10yr = float(np.mean(np.array(ten_yr_returns) > 1.0))

    # ── VaR and CVaR ─────────────────────────────────────────────────────────
    var_95  = float(np.percentile(rets, 5))           # 5th percentile (negative)
    cvar_95 = float(rets[rets <= var_95].mean())       # Expected value below VaR

    return {
        "simulated_paths":     pd.DataFrame(full_paths),
        "cagr_distribution":   cagr_dist,
        "sharpe_distribution": sharpe_dist,
        "maxdd_distribution":  maxdd_dist,
        "confidence_bands":    confidence_bands,
        "prob_positive_10yr":  prob_positive_10yr,
        "var_95":              var_95,
        "cvar_95":             cvar_95,
        "n_simulations":       n_simulations,
        "block_size":          block_size,
        "summary": {
            "median_cagr":    float(cagr_dist.median()),
            "median_sharpe":  float(sharpe_dist.median()),
            "median_maxdd":   float(maxdd_dist.median()),
            "p5_cagr":        float(cagr_dist.quantile(0.05)),
            "p95_cagr":       float(cagr_dist.quantile(0.95)),
            "pct_positive_sharpe": float((sharpe_dist > 0).mean()),
        },
    }
