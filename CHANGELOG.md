# Changelog

Research and development log for the 00Capital systematic portfolio engine.
Documents the full evolution from initial build through live deployment.

---

## ProjectR v2 — Live Deployment
*April 2026*

**Status:** Live paper trading via Alpaca · Automated on Railway

Final metrics: Sharpe 1.25 · CAGR 9.8% · MaxDD -10.8% · SPY corr 0.475

- Deployed full automated pipeline to Railway (weekly rebalance, Sunday snapshot)
- Alpaca paper trading integration with order execution and logging
- FastAPI data layer serving live positions and equity curve
- Public live strategy page at 00cap.com/strategy
- Backtest and live equity curves unified with inception divider

---

## ProjectR v1.4 — Short Book + Pre-Live Polish
*March 2026*

Sharpe 0.91 → 1.25 · MaxDD -13.1% → -10.8% · SPY corr 0.747 → 0.475

**Short book (largest single enhancement):**
- n_short=3, 20% short book weight (35% in BEARISH regimes)
- Hard cover at +8% adverse move
- Disabled in BULLISH_LOWVOL regime
- Contribution: +0.316 Sharpe, -0.272 SPY correlation

**Pre-live validation:**
- All 10 performance and code quality gates passed
- 11 unit test files covering all modules
- METHODOLOGY.md completed (507 lines, academic style)
- Lookahead bias confirmed absent at architecture level

---

## ProjectR v1.3 — Crisis Analysis + Transition Smoothing
*February 2026*

- Regime transition smoother: 3-period weight blend at boundaries
- Crisis override: immediate BEARISH weights when VIX > 30
- Crisis period analysis: GFC +49.6pp, COVID +29.5pp, 2022 +16.7pp
- Mean SPY correlation: 0.595 on long-only path
- Near-zero or negative correlation confirmed in all 3 major crises

---

## ProjectR v1.2 — Tail Hedge + Cost Model
*February 2026*

- VIX-based hedge overlay: soft threshold 0.20, hard threshold 0.45
- Vol-based fallback signal (primary VIX signal augmented)
- Calibration finding: soft=0.15 fires fallback 92% of days, kills CAGR -2.2pp
- Transaction cost model: 5bps one-way
- Rebalance frequency optimization: weekly confirmed optimal
  (Sharpe 1.204 vs 0.937 biweekly, 0.52 monthly)
- Monthly catastrophic — 21-day SIDEWAYS lookbacks need weekly refresh

---

## ProjectR v1.1 — Regime-Adaptive Lookbacks
*January 2026*

Sharpe 0.91 → 0.93 (base) · +0.074 Sharpe contribution

- Dynamic lookback periods by regime:
  BULLISH 252d/126d · SIDEWAYS 63d/21d · BEARISH 126d/63d
- Score = 0.7 × primary + 0.3 × secondary
- Mean reversion layer (RSI + Bollinger Band) tested exhaustively
  across all weight combinations — hurts Sharpe in broad ETF momentum
  universe, disabled, preserved in signals/reversion.py as opt-in

---

## ProjectR v1.0 — Core Build Complete
*December 2025*

Sharpe 0.71 → 0.91 · CAGR 6.6% → 9.2%

**Soft direction gate fix:**
- Hard gate (trend_score > 0) was creating only 9 eligible assets from 19
- Replaced with 75% penalty for mildly negative trends (-0.02 to 0.0)
- Hard exclusion only below -0.02
- Increased n_long from 8 to 10
- Unlocked ML eligibility across full universe

**ML layer fixed and validated:**
- Orthogonal feature set replacing momentum-redundant features
- Features: regime duration, transition probability, BB z-score,
  vol acceleration, distance from 52w high, cross-sectional rank,
  correlation to leader
- ML ON vs ML OFF confirmed divergent: +0.02 Sharpe, tighter drawdown
- ML contributes as risk-quality filter, not return amplifier

**Regime thresholds calibrated:**
- Strength threshold: 0.20 → 0.12
- Vol threshold: 0.18 → 0.20
- 2019 underperformance resolved: defensive Jan positioning +
  SIDEWAYS misclassification in Jun/Sep/Oct identified and fixed

---

## ProjectR v0.1 — Initial Build
*October 2025*

Sharpe 0.71 · CAGR 6.6% · MaxDD -13.8%

Complete rebuild of ProjectV from scratch. Modular architecture,
6-regime classifier, XGBoost + LightGBM ensemble stubbed in,
full research suite (walk-forward, Monte Carlo, factor decomposition,
crisis analysis). Lookahead bias test passed (shuffled Sharpe ≈ 0.004).

---

## ProjectV — Deprecated
*June 2025 – September 2025*

Sharpe ~0.70 · CAGR ~9.6%

**Architectural issues identified and not fixable without full rebuild:**
- Regime classifier computed but never wired into downstream decisions
- 3 regime states insufficient for cross-regime signal differentiation
- Universe overlap between alpha and hedge tickers
- ML layer present as empty stubs — never implemented
- Hard direction gate excluding ~40% of universe
- Flat correlation penalty regardless of regime
- Monolithic 1,000-line engine with no unit tests

Deprecated in favor of ProjectR full rebuild. Preserved at
[github.com/vnuthi07/ProjectV-overview](https://github.com/vnuthi07/ProjectV-overview)
for reference.

---

*Built by Varish Nuthi · [00Capital](https://00cap.com)*
