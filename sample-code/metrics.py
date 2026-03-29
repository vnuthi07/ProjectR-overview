"""
metrics.py — Comprehensive Performance Metrics Suite
=====================================================
Full performance reporting for ProjectR's systematic equity strategy.

Covers return metrics, risk-adjusted metrics, drawdown analysis,
benchmark-relative statistics, and trading efficiency metrics.

This is the metrics layer that powers ProjectR's Streamlit dashboard
and multi-page tearsheet (PNG + PDF output).

Key metrics reported:
    Return:     CAGR, total return, best/worst day/month/year
    Risk-adj:   Sharpe, Sortino, Calmar, Omega ratio
    Drawdown:   Max DD, avg DD, max DD duration, current DD
    Benchmark:  Alpha, beta, information ratio, tracking error,
                up/down capture ratios
    Trading:    Avg turnover, avg holding period, win rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover: pd.Series,
    rf_rate: float = 0.0,
) -> dict:
    """
    Compute full performance metric suite for a systematic strategy.

    Args:
        equity:            Cumulative equity curve (normalized to 1.0 at start)
        returns:           Daily return series
        benchmark_returns: Daily benchmark return series (e.g. SPY)
        turnover:          Daily portfolio turnover series
        rf_rate:           Annual risk-free rate (default 0.0)

    Returns:
        dict containing all performance metrics

    Example output (ProjectR 2003-2025):
        {
            'cagr': 0.084,
            'sharpe_ratio': 0.86,
            'sortino_ratio': 0.90,
            'calmar_ratio': 0.61,
            'max_drawdown': -0.136,
            'annualized_vol': 0.100,
            'alpha_vs_benchmark': 0.060,
            'win_rate': 0.44,
            ...
        }
    """
    metrics: dict = {}

    # ── Return metrics ────────────────────────────────────────────────────────
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_days = len(returns)
    n_years = n_days / _TRADING_DAYS
    cagr = float(
        (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0
    ) if n_years > 0 else 0.0
    ann_vol = float(returns.std() * np.sqrt(_TRADING_DAYS))

    metrics["total_return"]   = total_return
    metrics["cagr"]           = cagr
    metrics["annualized_vol"] = ann_vol
    metrics["best_day"]       = float(returns.max())
    metrics["worst_day"]      = float(returns.min())

    # Monthly and annual return extremes
    monthly_rets = (1 + returns).resample("ME").prod() - 1
    metrics["best_month"]  = float(monthly_rets.max())
    metrics["worst_month"] = float(monthly_rets.min())

    annual_rets = (1 + returns).resample("YE").prod() - 1
    metrics["best_year"]  = float(annual_rets.max())
    metrics["worst_year"] = float(annual_rets.min())

    # ── Risk-adjusted metrics ─────────────────────────────────────────────────
    excess_returns = returns - rf_rate / _TRADING_DAYS

    # Sharpe ratio
    sharpe = float(
        excess_returns.mean() / excess_returns.std() * np.sqrt(_TRADING_DAYS)
    ) if excess_returns.std() > 0 else 0.0
    metrics["sharpe_ratio"] = sharpe

    # Sortino ratio — penalizes only downside volatility
    downside = returns[returns < 0]
    downside_std = float(downside.std() * np.sqrt(_TRADING_DAYS)) if len(downside) > 0 else np.nan
    sortino = float(
        excess_returns.mean() * _TRADING_DAYS / downside_std
    ) if downside_std and downside_std > 0 else np.nan
    metrics["sortino_ratio"] = sortino

    # Omega ratio — probability-weighted ratio of gains to losses
    threshold = rf_rate / _TRADING_DAYS
    gains  = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns <= threshold]).sum()
    metrics["omega_ratio"] = float(gains / losses) if losses > 0 else np.nan

    # ── Drawdown metrics ──────────────────────────────────────────────────────
    rolling_max = equity.cummax()
    drawdown_series = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown_series.min())
    metrics["max_drawdown"]     = max_dd
    metrics["current_drawdown"] = float(drawdown_series.iloc[-1])
    metrics["avg_drawdown"]     = float(
        drawdown_series[drawdown_series < 0].mean()
    ) if (drawdown_series < 0).any() else 0.0

    # Calmar ratio — CAGR / abs(max drawdown)
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else np.nan
    metrics["calmar_ratio"] = calmar

    # Drawdown duration analysis
    in_dd = drawdown_series < 0
    dd_durations = []
    count = 0
    for flag in in_dd:
        if flag:
            count += 1
        else:
            if count > 0:
                dd_durations.append(count)
            count = 0
    if count > 0:
        dd_durations.append(count)

    metrics["max_drawdown_duration"] = int(max(dd_durations)) if dd_durations else 0
    metrics["avg_drawdown_duration"] = float(np.mean(dd_durations)) if dd_durations else 0.0

    # ── Benchmark-relative metrics ────────────────────────────────────────────
    common = returns.index.intersection(benchmark_returns.index)
    r = returns.loc[common]
    b = benchmark_returns.loc[common]

    if len(common) > 1:
        # OLS beta and alpha
        cov_matrix = np.cov(r.values, b.values)
        var_b = cov_matrix[1, 1]
        beta  = float(cov_matrix[0, 1] / var_b) if var_b > 0 else 0.0
        alpha = float((r.mean() - beta * b.mean()) * _TRADING_DAYS)  # annualized

        # Information ratio and tracking error
        te = float((r - b).std() * np.sqrt(_TRADING_DAYS))
        ir = float(((r - b).mean() * _TRADING_DAYS) / te) if te > 0 else np.nan

        # Up/down capture ratios
        up_mask = b > 0
        dn_mask = b < 0
        up_cap = float(r[up_mask].mean() / b[up_mask].mean()) \
            if up_mask.sum() > 0 and b[up_mask].mean() != 0 else np.nan
        dn_cap = float(r[dn_mask].mean() / b[dn_mask].mean()) \
            if dn_mask.sum() > 0 and b[dn_mask].mean() != 0 else np.nan
    else:
        beta = alpha = te = ir = up_cap = dn_cap = np.nan

    metrics["beta_vs_benchmark"]  = beta
    metrics["alpha_vs_benchmark"] = alpha
    metrics["information_ratio"]  = ir
    metrics["tracking_error"]     = te
    metrics["up_capture"]         = up_cap
    metrics["down_capture"]       = dn_cap

    # ── Trading efficiency metrics ────────────────────────────────────────────
    metrics["avg_turnover"]       = float(turnover.mean()) if len(turnover) > 0 else 0.0
    metrics["total_turnover"]     = float(turnover.sum()) if len(turnover) > 0 else 0.0
    avg_turn = metrics["avg_turnover"]
    metrics["avg_holding_period"] = float(1.0 / avg_turn) if avg_turn > 0 else np.nan
    metrics["win_rate"]           = float((returns > 0).mean())

    return metrics
