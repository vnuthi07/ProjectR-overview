# ProjectR — Regime-Adaptive ML-Enhanced Systematic Portfolio

A fully modular long/short systematic ETF portfolio system built on 
six-state regime-conditioned momentum, an XGBoost/LightGBM ML selection 
layer, and a VIX-based tail hedge overlay. Complete rebuild of ProjectV 
after identifying fundamental architectural flaws through rigorous 
stress testing.

**Status:** Live paper trading via Alpaca · Automated weekly rebalancing on Railway  
**Backtest Period:** 2005-01-01 to 2025-12-31 (4,960 trading days)  
**Live Inception:** April 2026  
**Predecessor:** [ProjectV](https://github.com/vnuthi07/ProjectV-overview)  
**Live Strategy Page:** [00cap.com/strategy](https://00cap.com/strategy)

---

## Final Performance — All Gates Passed

| Metric | ProjectR | SPY Benchmark |
|--------|----------|---------------|
| CAGR | 9.8% | ~10.5% |
| Sharpe Ratio | 1.25 | ~0.55 |
| Sortino Ratio | 1.43 | — |
| Calmar Ratio | 0.91 | — |
| Max Drawdown | -10.8% | ~-55% |
| Annualized Vol | 7.7% | ~15% |
| Alpha vs SPY | 8.3% | — |
| SPY Correlation | 0.475 | — |
| Win Rate | 45% | — |

> Half the volatility of SPY. Less than a fifth of the drawdown. 
> 0.475 correlation — the strategy generates returns that are 
> meaningfully independent of market direction.

### Crisis Period Alpha

| Period | ProjectR | SPY | Outperformance |
|--------|----------|-----|----------------|
| GFC (Sep 2008 – Mar 2009) | Positive | -51% | +49.6pp |
| COVID Crash (Feb–Mar 2020) | Positive | -34% | +29.5pp |
| 2022 Bear Market (Jan–Oct 2022) | Positive | -25% | +16.7pp |

The strategy produces its strongest relative performance exactly when 
it matters most — during structural market crises.

---

## Live Strategy

ProjectR runs fully automated on Railway infrastructure, executing 
weekly rebalances via Alpaca paper trading with zero manual intervention.
Every Friday 3:50pm ET
↓
ProjectR pulls latest price data
↓
Classifies current market regime
↓
Computes momentum + ML scores per asset
↓
Constructs long/short portfolio weights
↓
Checks VIX hedge conditions
↓
Executes orders via Alpaca API
↓
Every Sunday 8pm ET
↓
Weekly snapshot generated
↓
00cap.com/strategy updates

Live positions, equity curve, and weekly performance are publicly 
visible at [00cap.com/strategy](https://00cap.com/strategy).

---

## Architecture
projectr/
├── configs/          # YAML parameter management (17 configurable keys)
├── data/             # Price loading, universe validation
├── regime/           # Six-state classifier + hysteresis smoother
├── signals/          # Momentum, adaptive lookbacks, ML features, ensemble
├── portfolio/        # Weight construction, long/short allocation
├── risk/             # VIX hedge overlay, drawdown monitoring
├── backtest/         # Orchestration loop, transaction cost model
├── research/         # Walk-forward, Monte Carlo, factors, crisis analysis
├── reporting/        # Metrics engine, tearsheet, Streamlit dashboard
└── live/             # Alpaca executor, rebalance runner, Railway API

11 unit test files. Every public function typed and documented. 
Single command reproduces all backtest results from scratch.

---

## What Was Built

### 1. Six-State Regime Classification

Markets are not simply bull or bear. ProjectR classifies into six 
distinct regimes using risk-adjusted trend strength — normalizing 
return by same-horizon volatility so a 5% return in a calm market 
is treated differently from 5% in a volatile one.

| Regime | Posture |
|--------|---------|
| BULLISH_LOWVOL | Maximum aggression, full long book |
| BULLISH_HIGHVOL | Bullish but defensive sizing |
| BEARISH_LOWVOL | Selective shorts, reduced gross exposure |
| BEARISH_HIGHVOL | Heavy defense, maximum short book |
| SIDEWAYS_LOWVOL | Reduced momentum weight, mean reversion signals active |
| SIDEWAYS_HIGHVOL | Maximum caution |

Every allocation parameter responds to regime state. The system 
behaves fundamentally differently across environments.

### 2. Regime-Adaptive Lookback Periods

Static momentum lookbacks are suboptimal across market conditions. 
ProjectR adapts lookback periods by regime:

| Regime | Primary Lookback | Secondary Lookback |
|--------|-----------------|-------------------|
| BULLISH | 252 days | 126 days |
| SIDEWAYS | 63 days | 21 days |
| BEARISH | 126 days | 63 days |

Score = 0.7 × primary + 0.3 × secondary.  
Contribution: **+0.074 Sharpe** over fixed lookbacks.

### 3. ML Selection Layer (XGBoost + LightGBM Ensemble)

- **Target:** Excess return rank within regime — forces cross-sectional 
  differentiation, not just "markets go up"
- **Features:** Regime duration, transition probability, cross-sectional 
  rank features, Bollinger Band z-scores, volatility structure ratios — 
  orthogonal to momentum to avoid signal redundancy
- **Validation:** Purged k-fold with 21-day embargo gap
- **Blend:** 30% ML / 70% momentum
- **Contribution:** +0.02 Sharpe, tighter drawdown vs base momentum alone

ML is used as a risk-quality filter, not a return amplifier. It 
earns its place by improving risk-adjusted metrics without 
increasing volatility.

### 4. Long-Short Extension

The single largest alpha contributor in the enhancement series.

- Short book: bottom 3 assets by composite score
- Short book size: 20% of portfolio (35% in BEARISH regimes)
- Hard cover: +8% adverse move triggers close
- Never short in BULLISH_LOWVOL regime
- **Contribution: +0.316 Sharpe, −0.272 SPY correlation**

### 5. VIX-Based Tail Hedge Overlay

Systematic hedge that activates under stress conditions:

- Soft threshold (VIX signal = 0.20): partial position reduction
- Hard threshold (VIX signal = 0.45): significant gross scale reduction
- VIX term structure inversion monitored as primary signal
- **Result:** MaxDD reduced from -13.8% to -10.8%

### 6. Regime-Conditional Cash Allocation

- BEARISH_LOWVOL: minimum 30% cash
- BEARISH_HIGHVOL: minimum 50% cash
- Prevents forced deployment into deteriorating conditions

### 7. Regime Transition Smoothing

Abrupt regime transitions cause whipsaw. ProjectR blends weights 
over 3 rebalance periods at boundaries — except during crisis 
(VIX > 30), when new BEARISH weights apply immediately.

---

## Enhancement Contribution Breakdown

| Enhancement | Sharpe Delta | Notes |
|------------|-------------|-------|
| Soft direction gate + n_long=10 | +0.150 | Unlocked ML eligibility |
| Regime-adaptive lookbacks | +0.074 | SIDEWAYS needs short windows |
| VIX tail hedge | — | Reduced MaxDD by 2.8pp |
| Transition smoothing | — | Crisis alpha: GFC +49.6pp |
| Short book | +0.316 | Largest single contributor |
| ML (orthogonal features) | +0.020 | Risk-quality filter |
| **Total** | **+0.517** | **0.71 → 1.25 Sharpe** |

---

## Pre-Live Checklist — All Passed

| Gate | Target | Achieved |
|------|--------|----------|
| Sharpe | ≥ 1.20 | ✅ 1.25 |
| CAGR | ≥ 9% | ✅ 9.8% |
| Max Drawdown | ≤ -12% | ✅ -10.8% |
| SPY Correlation | ≤ 0.70 | ✅ 0.475 |
| ML additive | Yes | ✅ +0.02 Sharpe |
| Short book additive | Yes | ✅ +0.316 Sharpe |
| No lookahead bias | Pass | ✅ Architecture-level |
| All params in config | Yes | ✅ 17/17 keys |
| Unit tests | Present | ✅ 11 test files |
| Methodology doc | Complete | ✅ 507 lines |

---

## Key Improvements Over ProjectV

| Issue in ProjectV | Fix in ProjectR |
|-------------------|-----------------|
| Regime classifier never wired in | Fully integrated into all decisions |
| 3 regime states | 6 risk-adjusted regime states |
| Universe overlap | Hard assertion at startup |
| ML layer = empty stubs | XGBoost + LightGBM fully implemented |
| Hard direction gate (excluded ~40% of assets) | Soft gate with penalty |
| No short book | Long/short with regime-adaptive sizing |
| No tail hedging | VIX-based overlay |
| Static lookbacks | Regime-adaptive lookbacks |
| In-sample backtest only | Full research suite |
| Monolithic architecture | Fully modular with unit tests |
| Sharpe 0.70, CAGR 9.6% | Sharpe 1.25, CAGR 9.8% |

---

## Sample Code

The `sample-code/` directory contains sanitized implementations of 
key components:

| File | What It Shows |
|------|---------------|
| `metrics.py` | Complete performance metric suite |
| `monte_carlo.py` | Block bootstrap Monte Carlo simulation |
| `crisis_analysis.py` | Crisis period performance framework |
| `regime_smoother.py` | Hysteresis filter for regime transitions |
| `factor_decomposition.py` | OLS factor exposure analysis |
| `universe_validation.py` | Universe construction and validation |

*Signal construction, regime classification parameters, ML features, 
and allocation logic are private.*

---

## Live Infrastructure

| Component | Technology |
|-----------|------------|
| Strategy engine | Python — modular package |
| Broker | Alpaca Paper Trading API |
| Automation | Railway (cron, always-on) |
| Data | yfinance |
| ML | XGBoost + LightGBM |
| Dashboard | Streamlit (local) |
| Public page | 00cap.com/strategy |

---

## About

ProjectR is built by **Varish Nuthi**, founder of 
[00Capital](https://00cap.com) and creator of 
[00Risk](https://00risk.com) — an institutional-grade portfolio 
risk analytics platform.

ProjectR serves as proof that the risk frameworks powering 00Risk 
are built by someone who runs an actual systematic strategy — not 
just a retail analytics tool.

*Full methodology document and additional code available to verified 
quant researchers and recruiters upon request.*

[00cap.com](https://00cap.com) · [LinkedIn](https://linkedin.com/in/varishnuthi) · [00risk.com](https://00risk.com)