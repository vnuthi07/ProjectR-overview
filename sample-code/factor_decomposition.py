import numpy as np
import pandas as pd
from dataclasses import dataclass
import statsmodels.api as sm


FACTOR_TICKERS = {
    "SPY": "US Equity Market",
    "QQQ": "Growth/Tech",
    "IEF": "Intermediate Duration",
    "GLD": "Gold/Inflation Hedge",
    "UUP": "US Dollar",
}


@dataclass
class FactorDecompositionResult:
    alpha_annualized: float
    alpha_tstat: float
    alpha_pvalue: float
    r_squared: float
    betas: dict[str, float]
    beta_tstats: dict[str, float]
    beta_pvalues: dict[str, float]
    residual_sharpe: float
    information_ratio: float


def run_factor_decomposition(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> FactorDecompositionResult:
    aligned = pd.concat([strategy_returns, factor_returns], axis=1).dropna()
    y = aligned.iloc[:, 0]
    X = sm.add_constant(aligned.iloc[:, 1:])

    model = sm.OLS(y, X).fit()

    alpha_daily = model.params["const"]
    alpha_annualized = (1 + alpha_daily) ** periods_per_year - 1

    factor_names = [c for c in aligned.columns[1:]]
    betas = {name: float(model.params[name]) for name in factor_names}
    beta_tstats = {name: float(model.tvalues[name]) for name in factor_names}
    beta_pvalues = {name: float(model.pvalues[name]) for name in factor_names}

    residuals = model.resid
    residual_sharpe = float(
        residuals.mean() / residuals.std() * np.sqrt(periods_per_year)
    )

    factor_component = model.fittedvalues - alpha_daily
    ir = float(
        alpha_daily / residuals.std() * np.sqrt(periods_per_year)
    ) if residuals.std() > 0 else 0.0

    return FactorDecompositionResult(
        alpha_annualized=alpha_annualized,
        alpha_tstat=float(model.tvalues["const"]),
        alpha_pvalue=float(model.pvalues["const"]),
        r_squared=float(model.rsquared),
        betas=betas,
        beta_tstats=beta_tstats,
        beta_pvalues=beta_pvalues,
        residual_sharpe=residual_sharpe,
        information_ratio=ir,
    )


def rolling_factor_betas(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    aligned = pd.concat([strategy_returns, factor_returns], axis=1).dropna()
    factor_names = aligned.columns[1:].tolist()
    results = []

    for i in range(window, len(aligned)):
        window_data = aligned.iloc[i - window : i]
        y = window_data.iloc[:, 0]
        X = sm.add_constant(window_data.iloc[:, 1:])
        try:
            model = sm.OLS(y, X).fit()
            row = {"date": aligned.index[i]}
            row.update({name: model.params[name] for name in factor_names})
            row["alpha"] = model.params["const"] * 252
            row["r_squared"] = model.rsquared
            results.append(row)
        except Exception:
            continue

    return pd.DataFrame(results).set_index("date")


def print_factor_decomposition(result: FactorDecompositionResult) -> None:
    print("Factor Decomposition Results")
    print(f"{'─' * 50}")
    print(f"Annualized Alpha : {result.alpha_annualized:>8.2%}  (t={result.alpha_tstat:.2f}, p={result.alpha_pvalue:.3f})")
    print(f"R-Squared        : {result.r_squared:>8.3f}")
    print(f"Residual Sharpe  : {result.residual_sharpe:>8.3f}")
    print(f"Info Ratio       : {result.information_ratio:>8.3f}")
    print()
    print(f"{'Factor':<15} {'Beta':>8} {'t-stat':>8} {'p-value':>8}")
    print(f"{'─' * 50}")
    for name in result.betas:
        print(
            f"{name:<15} {result.betas[name]:>8.3f} "
            f"{result.beta_tstats[name]:>8.2f} "
            f"{result.beta_pvalues[name]:>8.3f}"
        )
