import pandas as pd
import numpy as np
from dataclasses import dataclass


CRISIS_PERIODS = {
    "GFC": ("2008-09-01", "2009-03-31"),
    "US Debt Downgrade": ("2011-07-01", "2011-10-31"),
    "China/Oil Selloff": ("2015-08-01", "2016-02-29"),
    "Q4 2018 Selloff": ("2018-10-01", "2018-12-31"),
    "COVID Crash": ("2020-02-01", "2020-03-31"),
    "2022 Bear Market": ("2022-01-01", "2022-10-31"),
}


@dataclass
class CrisisPeriodResult:
    name: str
    start: str
    end: str
    strategy_return: float
    benchmark_return: float
    outperformance: float
    strategy_max_dd: float
    benchmark_max_dd: float
    strategy_sharpe: float
    trading_days: int


def analyze_crisis_period(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    name: str,
    start: str,
    end: str,
) -> CrisisPeriodResult:
    s = strategy_returns.loc[start:end]
    b = benchmark_returns.loc[start:end]

    def total_return(r):
        return float((1 + r).prod() - 1)

    def period_max_dd(r):
        cum = (1 + r).cumprod()
        return float(((cum - cum.cummax()) / cum.cummax()).min())

    def period_sharpe(r):
        if r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))

    strat_ret = total_return(s)
    bench_ret = total_return(b)

    return CrisisPeriodResult(
        name=name,
        start=start,
        end=end,
        strategy_return=strat_ret,
        benchmark_return=bench_ret,
        outperformance=strat_ret - bench_ret,
        strategy_max_dd=period_max_dd(s),
        benchmark_max_dd=period_max_dd(b),
        strategy_sharpe=period_sharpe(s),
        trading_days=len(s),
    )


def run_crisis_analysis(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods: dict = None,
) -> list[CrisisPeriodResult]:
    if periods is None:
        periods = CRISIS_PERIODS

    results = []
    for name, (start, end) in periods.items():
        try:
            result = analyze_crisis_period(
                strategy_returns, benchmark_returns, name, start, end
            )
            results.append(result)
        except Exception:
            continue

    return results


def print_crisis_summary(results: list[CrisisPeriodResult]) -> None:
    print(f"{'Crisis Period Analysis'}")
    print(f"{'─' * 75}")
    print(
        f"{'Period':<25} {'Strategy':>10} {'SPY':>10} "
        f"{'Alpha':>10} {'Strat DD':>10} {'Days':>6}"
    )
    print(f"{'─' * 75}")

    for r in results:
        print(
            f"{r.name:<25} {r.strategy_return:>10.1%} {r.benchmark_return:>10.1%} "
            f"{r.outperformance:>+10.1%} {r.strategy_max_dd:>10.1%} {r.trading_days:>6}"
        )

    avg_alpha = np.mean([r.outperformance for r in results])
    print(f"{'─' * 75}")
    print(f"{'Average outperformance':<25} {avg_alpha:>+10.1%}")
