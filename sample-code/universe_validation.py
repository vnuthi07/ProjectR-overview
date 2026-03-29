"""
universe_validation.py — Universe Construction and Validation
=============================================================
Enforces strict constraints on the alpha and hedge universe composition.

One of the key architectural improvements in ProjectR over ProjectV:
    ProjectV had overlapping tickers in the alpha and hedge sleeves —
    TLT, IEF, SHY, GLD, SLV, UUP appeared in both simultaneously.
    This caused double-counting in portfolio construction (the system
    could be long TLT in the alpha sleeve for momentum AND long TLT
    in the hedge sleeve for defense, with weights adding up in an
    uncontrolled way).

    ProjectR enforces disjoint universes with a hard assertion at startup.
    If any ticker appears in both sleeves, the system raises ValueError
    before running a single backtest step.

Design principle:
    Fail loudly at startup rather than silently producing wrong results.
    A system that runs without error but produces subtly wrong portfolio
    weights is harder to debug than one that crashes immediately with
    a clear error message.

Additional validation:
    Tickers with inception dates after 2007 trigger warnings because
    they have limited backtest history — their early performance is
    missing, which can bias full-period metrics.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Approximate inception years for ETFs with limited history.
# Tickers listed after 2007 will have incomplete backtest coverage.
_KNOWN_LATE_STARTERS: dict[str, int] = {
    "MTUM": 2013,   # iShares Momentum Factor
    "USMV": 2011,   # iShares Min Vol
    "QUAL": 2013,   # iShares Quality Factor
    "VLUE": 2013,   # iShares Value Factor
    "SPHB": 2011,   # PowerShares S&P 500 High Beta
    "SGOL": 2009,   # Aberdeen Physical Gold
    "EMB":  2007,   # iShares EM Bond
    "BTC-USD": 2014, # Bitcoin
}

_WARN_AFTER_YEAR = 2007


def validate_universe(alpha: list[str], hedge: list[str]) -> None:
    """
    Validate universe composition at startup.

    Raises:
        ValueError: If alpha or hedge list is empty
        ValueError: If any ticker appears in both alpha and hedge sleeves

    Logs warnings for:
        Tickers with known inception dates after 2007

    This validation runs once before any backtest computation.
    Better to fail immediately with a clear error than to run a 20-year
    backtest with silently wrong portfolio weights.

    Args:
        alpha: List of alpha sleeve tickers
        hedge: List of hedge sleeve tickers
    """
    if not alpha:
        raise ValueError("Alpha sleeve is empty — at least one ticker required.")
    if not hedge:
        raise ValueError("Hedge sleeve is empty — at least one ticker required.")

    # Hard assertion: no overlap allowed
    overlap = set(alpha) & set(hedge)
    if overlap:
        raise ValueError(
            f"Alpha and hedge sleeves must be strictly disjoint.\n"
            f"Overlapping tickers found: {sorted(overlap)}\n"
            f"Fix: move each ticker to exactly one sleeve."
        )

    # Warn about tickers with limited history
    all_tickers = list(alpha) + list(hedge)
    for ticker in all_tickers:
        inception = _KNOWN_LATE_STARTERS.get(ticker)
        if inception is not None and inception > _WARN_AFTER_YEAR:
            log.warning(
                "Ticker '%s' listed ~%d — backtest data before this date "
                "is unavailable. Early-period metrics may be biased.",
                ticker,
                inception,
            )

    log.info(
        "Universe validated: %d alpha tickers, %d hedge tickers, 0 overlaps.",
        len(alpha),
        len(hedge),
    )


def get_all_tickers(
    alpha: list[str],
    hedge: list[str],
    benchmark: str,
) -> list[str]:
    """
    Return deduplicated union of all tickers needed for the backtest.

    Preserves order: alpha first, then hedge, then benchmark if not
    already present. Used to build a single price data download request.

    Args:
        alpha:     Alpha sleeve tickers
        hedge:     Hedge sleeve tickers
        benchmark: Benchmark ticker (e.g. 'SPY')

    Returns:
        Deduplicated list of all required tickers
    """
    seen:   set[str]  = set()
    result: list[str] = []

    for ticker in alpha + hedge + [benchmark]:
        if ticker not in seen:
            seen.add(ticker)
            result.append(ticker)

    return result
