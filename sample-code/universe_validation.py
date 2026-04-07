import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class UniverseValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    alpha_tickers: list[str] = field(default_factory=list)
    hedge_tickers: list[str] = field(default_factory=list)
    overlap: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    low_liquidity: list[str] = field(default_factory=list)
    n_alpha: int = 0
    n_hedge: int = 0


def validate_universe(
    alpha_tickers: list[str],
    hedge_tickers: list[str],
    price_data: pd.DataFrame,
    min_history_days: int = 252,
    min_adv_usd: float = 50_000_000,
) -> UniverseValidationResult:
    result = UniverseValidationResult(
        is_valid=True,
        alpha_tickers=alpha_tickers,
        hedge_tickers=hedge_tickers,
        n_alpha=len(alpha_tickers),
        n_hedge=len(hedge_tickers),
    )

    overlap = set(alpha_tickers) & set(hedge_tickers)
    if overlap:
        result.errors.append(
            f"Universe overlap detected — tickers appear in both alpha and hedge: {sorted(overlap)}"
        )
        result.overlap = sorted(overlap)
        result.is_valid = False

    all_tickers = alpha_tickers + hedge_tickers
    missing = [t for t in all_tickers if t not in price_data.columns]
    if missing:
        result.errors.append(f"Missing price data for: {missing}")
        result.missing_data = missing
        result.is_valid = False

    present = [t for t in all_tickers if t in price_data.columns]
    for ticker in present:
        series = price_data[ticker].dropna()
        if len(series) < min_history_days:
            result.warnings.append(
                f"{ticker} has only {len(series)} days of history (min: {min_history_days})"
            )

    if not result.is_valid:
        return result

    if "volume" in price_data.columns.get_level_values(0) if isinstance(
        price_data.columns, pd.MultiIndex
    ) else False:
        for ticker in alpha_tickers:
            try:
                avg_volume = price_data["volume"][ticker].rolling(20).mean().iloc[-1]
                avg_price = price_data["close"][ticker].iloc[-1]
                adv = avg_volume * avg_price
                if adv < min_adv_usd:
                    result.warnings.append(
                        f"{ticker} ADV ${adv:,.0f} below minimum ${min_adv_usd:,.0f}"
                    )
                    result.low_liquidity.append(ticker)
            except Exception:
                continue

    return result


def assert_no_overlap(alpha_tickers: list[str], hedge_tickers: list[str]) -> None:
    overlap = set(alpha_tickers) & set(hedge_tickers)
    if overlap:
        raise ValueError(
            f"Universe overlap is not permitted. "
            f"Tickers in both alpha and hedge universes: {sorted(overlap)}. "
            f"This would create undefined behavior in portfolio construction."
        )


def check_data_freshness(
    price_data: pd.DataFrame,
    max_stale_days: int = 5,
) -> dict[str, int]:
    today = pd.Timestamp.today().normalize()
    staleness = {}
    for col in price_data.columns:
        last_valid = price_data[col].last_valid_index()
        if last_valid is not None:
            days_stale = (today - last_valid).days
            if days_stale > max_stale_days:
                staleness[col] = days_stale
    return staleness


def print_validation_summary(result: UniverseValidationResult) -> None:
    status = "PASS" if result.is_valid else "FAIL"
    print(f"Universe Validation: {status}")
    print(f"{'─' * 40}")
    print(f"Alpha tickers : {result.n_alpha}")
    print(f"Hedge tickers : {result.n_hedge}")

    if result.overlap:
        print(f"Overlap       : {result.overlap}")

    if result.errors:
        print()
        print("Errors:")
        for e in result.errors:
            print(f"  ✗ {e}")

    if result.warnings:
        print()
        print("Warnings:")
        for w in result.warnings:
            print(f"  ⚠ {w}")

    if result.is_valid and not result.warnings:
        print("All checks passed.")
