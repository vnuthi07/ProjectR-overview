import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass
class MonteCarloResult:
    n_simulations: int
    block_size: int
    sharpe_distribution: np.ndarray
    cagr_distribution: np.ndarray
    max_dd_distribution: np.ndarray
    sharpe_p5: float
    sharpe_p50: float
    sharpe_p95: float
    cagr_p5: float
    cagr_p50: float
    cagr_p95: float
    max_dd_p5: float
    max_dd_p50: float
    max_dd_p95: float
    prob_positive_sharpe: float
    prob_sharpe_above_1: float


def block_bootstrap(returns: pd.Series, block_size: int = 21) -> pd.Series:
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    start_indices = np.random.randint(0, n - block_size, size=n_blocks)
    blocks = [returns.iloc[i : i + block_size].values for i in start_indices]
    simulated = np.concatenate(blocks)[:n]
    return pd.Series(simulated, index=returns.index)


def run_monte_carlo(
    returns: pd.Series,
    metric_fn: Callable[[pd.Series], dict],
    n_simulations: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> MonteCarloResult:
    np.random.seed(seed)
    sharpes = np.zeros(n_simulations)
    cagrs = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)

    for i in range(n_simulations):
        simulated = block_bootstrap(returns, block_size)
        metrics = metric_fn(simulated)
        sharpes[i] = metrics["sharpe"]
        cagrs[i] = metrics["cagr"]
        max_dds[i] = metrics["max_drawdown"]

    return MonteCarloResult(
        n_simulations=n_simulations,
        block_size=block_size,
        sharpe_distribution=sharpes,
        cagr_distribution=cagrs,
        max_dd_distribution=max_dds,
        sharpe_p5=float(np.percentile(sharpes, 5)),
        sharpe_p50=float(np.percentile(sharpes, 50)),
        sharpe_p95=float(np.percentile(sharpes, 95)),
        cagr_p5=float(np.percentile(cagrs, 5)),
        cagr_p50=float(np.percentile(cagrs, 50)),
        cagr_p95=float(np.percentile(cagrs, 95)),
        max_dd_p5=float(np.percentile(max_dds, 5)),
        max_dd_p50=float(np.percentile(max_dds, 50)),
        max_dd_p95=float(np.percentile(max_dds, 95)),
        prob_positive_sharpe=float((sharpes > 0).mean()),
        prob_sharpe_above_1=float((sharpes > 1.0).mean()),
    )


def print_monte_carlo_summary(result: MonteCarloResult) -> None:
    print("Monte Carlo Simulation Results")
    print(f"{'─' * 45}")
    print(f"Simulations : {result.n_simulations:,}")
    print(f"Block size  : {result.block_size} days")
    print()
    print(f"{'Metric':<20} {'P5':>8} {'P50':>8} {'P95':>8}")
    print(f"{'─' * 45}")
    print(f"{'Sharpe':<20} {result.sharpe_p5:>8.3f} {result.sharpe_p50:>8.3f} {result.sharpe_p95:>8.3f}")
    print(f"{'CAGR':<20} {result.cagr_p5:>8.1%} {result.cagr_p50:>8.1%} {result.cagr_p95:>8.1%}")
    print(f"{'Max Drawdown':<20} {result.max_dd_p5:>8.1%} {result.max_dd_p50:>8.1%} {result.max_dd_p95:>8.1%}")
    print()
    print(f"P(Sharpe > 0) : {result.prob_positive_sharpe:.1%}")
    print(f"P(Sharpe > 1) : {result.prob_sharpe_above_1:.1%}")
