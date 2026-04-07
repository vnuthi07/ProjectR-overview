import numpy as np
import pandas as pd
from typing import Optional


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    total = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    return float(total ** (1 / n_years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float((excess.mean() / downside) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean())


def alpha_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    r = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    cov = np.cov(r, b)
    beta = cov[0, 1] / cov[1, 1]
    alpha = annualized_return(r, periods_per_year) - beta * annualized_return(
        b, periods_per_year
    )
    return float(alpha), float(beta)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    def _sharpe(x):
        if x.std() == 0:
            return 0.0
        return (x.mean() / x.std()) * np.sqrt(periods_per_year)

    return returns.rolling(window).apply(_sharpe, raw=False)


def rolling_drawdown(returns: pd.Series) -> pd.Series:
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max


def full_tearsheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    metrics = {
        "cagr": annualized_return(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar": calmar_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "annualized_vol": annualized_volatility(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurt()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "best_year": float(
            returns.resample("YE").apply(lambda x: (1 + x).prod() - 1).max()
        ),
        "worst_year": float(
            returns.resample("YE").apply(lambda x: (1 + x).prod() - 1).min()
        ),
    }

    if benchmark is not None:
        a, b = alpha_beta(returns, benchmark, periods_per_year)
        corr = returns.corr(benchmark)
        metrics["alpha"] = a
        metrics["beta"] = b
        metrics["spy_correlation"] = corr

    return metrics


def annual_returns(returns: pd.Series) -> pd.Series:
    return returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
