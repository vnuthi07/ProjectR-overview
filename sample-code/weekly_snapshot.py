import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class WeeklySnapshot:
    week_ending: str
    week_return: float
    week_spy_return: float
    cumulative_return: float
    cumulative_spy_return: float
    sharpe_since_inception: float
    max_dd_since_inception: float
    current_regime: str
    vix_level: float
    hedge_active: bool
    current_positions: list[dict]
    positions_added: list[str]
    positions_removed: list[str]
    narrative: str
    portfolio_value: float
    inception_value: float


def compute_sharpe_since_inception(
    daily_returns: list[float],
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> float:
    if len(daily_returns) < min_periods:
        return 0.0
    r = np.array(daily_returns)
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def compute_max_dd_since_inception(cumulative_values: list[float]) -> float:
    if len(cumulative_values) < 2:
        return 0.0
    values = np.array(cumulative_values)
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return float(drawdown.min())


def compute_week_return(
    performance_log: list[dict],
    week_ending_date: str,
) -> tuple[float, float]:
    df = pd.DataFrame(performance_log)
    df["date"] = pd.to_datetime(df["date"])
    week_end = pd.Timestamp(week_ending_date)
    week_start = week_end - timedelta(days=7)

    week_data = df[(df["date"] > week_start) & (df["date"] <= week_end)]
    if len(week_data) == 0:
        return 0.0, 0.0

    strategy_week = float((1 + pd.Series(week_data["daily_return"].values)).prod() - 1)
    spy_week = float((1 + pd.Series(week_data["spy_daily_return"].values)).prod() - 1)
    return strategy_week, spy_week


def get_positions_delta(
    current_positions: list[dict],
    previous_positions: list[dict],
) -> tuple[list[str], list[str]]:
    current_tickers = {p["ticker"] for p in current_positions}
    previous_tickers = {p["ticker"] for p in previous_positions}
    added = sorted(current_tickers - previous_tickers)
    removed = sorted(previous_tickers - current_tickers)
    return added, removed


def generate_narrative(snapshot: dict) -> str:
    week_ret = snapshot["week_return"]
    spy_ret = snapshot["week_spy_return"]
    outperformance = week_ret - spy_ret
    direction = "outperforming" if outperformance > 0 else "underperforming"
    regime = snapshot["current_regime"]
    hedge = snapshot["hedge_active"]
    added = snapshot["positions_added"]
    removed = snapshot["positions_removed"]

    n_long = sum(1 for p in snapshot["current_positions"] if p.get("side") == "long")
    n_short = sum(1 for p in snapshot["current_positions"] if p.get("side") == "short")

    narrative = (
        f"ProjectR returned {week_ret:+.1%} this week vs SPY {spy_ret:+.1%}, "
        f"{direction} by {abs(outperformance):.1%}pp. "
        f"Current regime: {regime} with {n_long} long and {n_short} short positions. "
    )

    if hedge:
        narrative += "Hedge overlay was active this week. "
    else:
        narrative += "No hedge activation this week. "

    if added or removed:
        changes = []
        if added:
            changes.append(f"Added {', '.join(added)}")
        if removed:
            changes.append(f"Removed {', '.join(removed)}")
        narrative += ". ".join(changes) + "."

    return narrative.strip()


def build_weekly_snapshot(
    performance_log: list[dict],
    rebalance_log: list[dict],
    week_ending: str,
) -> WeeklySnapshot:
    if not rebalance_log:
        raise ValueError("No rebalance data available")

    latest_rebalance = rebalance_log[-1]
    previous_rebalance = rebalance_log[-2] if len(rebalance_log) > 1 else {}

    week_return, week_spy_return = compute_week_return(performance_log, week_ending)

    all_values = [entry["portfolio_value"] for entry in performance_log]
    all_returns = [entry["daily_return"] for entry in performance_log]
    inception_value = all_values[0] if all_values else 100_000

    cumulative_return = float(all_values[-1] / inception_value - 1) if all_values else 0.0
    spy_values = [entry.get("spy_value", inception_value) for entry in performance_log]
    cumulative_spy_return = float(spy_values[-1] / spy_values[0] - 1) if spy_values else 0.0

    current_positions = latest_rebalance.get("target_weights", {})
    previous_positions = previous_rebalance.get("target_weights", {}) if previous_rebalance else {}

    positions_list = [
        {
            "ticker": ticker,
            "weight": weight,
            "side": "long" if weight > 0 else "short",
        }
        for ticker, weight in current_positions.items()
    ]

    added, removed = get_positions_delta(
        [{"ticker": t} for t in current_positions],
        [{"ticker": t} for t in previous_positions],
    )

    snapshot_dict = {
        "week_ending": week_ending,
        "week_return": week_return,
        "week_spy_return": week_spy_return,
        "cumulative_return": cumulative_return,
        "cumulative_spy_return": cumulative_spy_return,
        "sharpe_since_inception": compute_sharpe_since_inception(all_returns),
        "max_dd_since_inception": compute_max_dd_since_inception(all_values),
        "current_regime": latest_rebalance.get("regime", "UNKNOWN"),
        "vix_level": latest_rebalance.get("vix", 0.0),
        "hedge_active": latest_rebalance.get("hedge_active", False),
        "current_positions": positions_list,
        "positions_added": added,
        "positions_removed": removed,
        "portfolio_value": all_values[-1] if all_values else inception_value,
        "inception_value": inception_value,
        "narrative": "",
    }

    snapshot_dict["narrative"] = generate_narrative(snapshot_dict)
    return WeeklySnapshot(**snapshot_dict)


def append_snapshot(snapshot: WeeklySnapshot, output_path: str) -> None:
    path = Path(output_path)
    existing = []
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing.append(asdict(snapshot))
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def print_snapshot_summary(snapshot: WeeklySnapshot) -> None:
    print(f"Weekly Snapshot — {snapshot.week_ending}")
    print(f"{'─' * 50}")
    print(f"Week Return       : {snapshot.week_return:>+8.2%}  (SPY {snapshot.week_spy_return:>+.2%})")
    print(f"Since Inception   : {snapshot.cumulative_return:>+8.2%}  (SPY {snapshot.cumulative_spy_return:>+.2%})")
    print(f"Sharpe (live)     : {snapshot.sharpe_since_inception:>8.3f}")
    print(f"Max DD (live)     : {snapshot.max_dd_since_inception:>8.2%}")
    print(f"Regime            : {snapshot.current_regime}")
    print(f"VIX               : {snapshot.vix_level:>8.1f}")
    print(f"Hedge Active      : {'Yes' if snapshot.hedge_active else 'No'}")
    print(f"Portfolio Value   : ${snapshot.portfolio_value:>12,.0f}")
    print()
    print(f"Positions ({len(snapshot.current_positions)}):")
    for p in sorted(snapshot.current_positions, key=lambda x: abs(x["weight"]), reverse=True):
        side_label = "LONG " if p["side"] == "long" else "SHORT"
        print(f"  {side_label}  {p['ticker']:<8} {p['weight']:>+.1%}")
    if snapshot.positions_added:
        print(f"\nAdded   : {', '.join(snapshot.positions_added)}")
    if snapshot.positions_removed:
        print(f"Removed : {', '.join(snapshot.positions_removed)}")
    print()
    print(f"Narrative: {snapshot.narrative}")
