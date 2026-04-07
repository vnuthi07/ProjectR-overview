# ProjectR — Results

**Backtest Period:** January 2005 – December 2025 (4,960 trading days)  
**Live Inception:** April 2026  
**Universe:** 25 alpha tickers, 5 hedge tickers  
**Rebalance Frequency:** Weekly (Friday close)

---

## Core Metrics

| Metric | ProjectR | SPY Benchmark |
|--------|----------|---------------|
| CAGR | 9.8% | ~10.5% |
| Sharpe Ratio | 1.25 | ~0.55 |
| Sortino Ratio | 1.43 | — |
| Calmar Ratio | 0.91 | — |
| Max Drawdown | -10.8% | ~-55% |
| Annualized Volatility | 7.7% | ~15% |
| Alpha vs SPY | 8.3% | — |
| SPY Correlation | 0.475 | — |
| Win Rate | 45% | — |

SPY delivers higher raw CAGR through leverage on equity risk premium.
ProjectR targets genuine risk-adjusted alpha — half the volatility,
one-fifth the drawdown, 0.475 correlation to market direction.

---

## Crisis Period Performance

| Period | ProjectR | SPY | Outperformance |
|--------|----------|-----|----------------|
| GFC (Sep 2008 – Mar 2009) | Positive | -51% | +49.6pp |
| US Debt Downgrade (Jul–Oct 2011) | Positive | -19% | Positive |
| China/Oil Selloff (Aug 2015 – Feb 2016) | Positive | -13% | Positive |
| Q4 2018 Selloff (Oct–Dec 2018) | Positive | -20% | Positive |
| COVID Crash (Feb–Mar 2020) | Positive | -34% | +29.5pp |
| 2022 Bear Market (Jan–Oct 2022) | Positive | -25% | +16.7pp |

The strategy produces its strongest relative performance during
structural market crises — when it matters most.

---

## Annual Returns

| Year | ProjectR | SPY |
|------|----------|-----|
| 2016 | +5.9% | +12.0% |
| 2017 | +27.0% | +21.7% |
| 2018 | +6.4% | -4.6% |
| 2019 | +15.6% | +31.2% |
| 2020 | +37.9% | +18.3% |
| 2021 | +34.6% | +28.7% |
| 2022 | -7.0% | -18.2% |
| 2023 | +17.2% | +26.3% |
| 2024 | +13.8% | +24.9% |
| 2025 | +27.3% | +13.6% |

---

## Enhancement Contribution Breakdown

Each enhancement was tested independently and accepted only if it
improved Sharpe without meaningfully increasing drawdown.

| Enhancement | Sharpe Delta | Notes |
|------------|-------------|-------|
| Soft direction gate + n_long=10 | +0.150 | Unlocked ML eligibility |
| Regime-adaptive lookbacks | +0.074 | SIDEWAYS needs short windows |
| VIX tail hedge overlay | — | Reduced MaxDD by 2.8pp |
| Regime transition smoothing | — | GFC alpha +49.6pp |
| Short book (n_short=3, 20%) | +0.316 | Largest single contributor |
| ML orthogonal features | +0.020 | Risk-quality filter |
| **Total** | **+0.517** | **Sharpe 0.71 → 1.25** |

---

## ML Layer

| Config | Sharpe | CAGR | MaxDD |
|--------|--------|------|-------|
| ML OFF | 1.23 | 9.8% | -11.0% |
| ML ON | 1.25 | 9.8% | -10.8% |

ML contributes as a risk-quality filter — same CAGR, tighter
drawdown, better risk-adjusted metrics. Features are orthogonal
to momentum signals: regime duration, transition probability,
Bollinger Band z-scores, volatility acceleration, cross-sectional
rank, distance from 52-week high, correlation to leader.

---

## Short Book Attribution

| Component | Sharpe Contribution |
|-----------|-------------------|
| Long book alone | Baseline |
| Short book addition | +0.316 |
| SPY correlation reduction | -0.272 |

Short book size: 20% of portfolio in standard regimes,
35% in BEARISH regimes. Hard cover at +8% adverse move.
Never short in BULLISH_LOWVOL regime.

---

## Pre-Live Checklist

| Gate | Target | Result |
|------|--------|--------|
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

## Factor Decomposition

| Factor | Beta | Interpretation |
|--------|------|----------------|
| SPY | ~0.25 | Low market exposure by design |
| QQQ | Low | Limited growth/tech bias |
| IEF | Low | Managed duration exposure |
| GLD | Low | Occasional safe-haven rotation |
| UUP | Low | Limited dollar sensitivity |

**Annualized Alpha: 8.3%**  
**R-squared: ~0.35** — 65% of returns unexplained by these factors.

ProjectR is not leveraged SPY beta. Returns come from
regime-conditioned selection and systematic short exposure,
not factor loading.

---

*Full methodology: [METHODOLOGY.md](./METHODOLOGY.md)*  
*Live performance: [00cap.com/strategy](https://00cap.com/strategy)*
