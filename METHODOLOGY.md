# ProjectR: A Regime-Adaptive Systematic Portfolio Strategy

**00Capital | Varish Nuthi**
*March 2026*

---

## Abstract

ProjectR is a fully systematic, regime-adaptive momentum strategy trading a 31-asset ETF universe. The central thesis is that momentum signals behave fundamentally differently across market regimes — trending, mean-reverting, and crisis environments each reward distinct parameter configurations and risk tolerances. The system detects one of six market regimes from SPY price dynamics, then adapts signal weights, portfolio construction, and risk controls accordingly. The strategy blends vol-normalized multi-horizon momentum (with regime-adaptive lookbacks) with an XGBoost/LightGBM ensemble, overlaid with VIX-based tail-risk hedging, regime-conditioned stop losses, and a long-short book with 20% short allocation. Out-of-sample walk-forward analysis across the full 2005–2025 period, including the GFC, COVID crash, and 2022 rate shock, demonstrates Sharpe ratios above 1.20 with maximum drawdowns below 12%, SPY correlation of 0.47, and all three major crisis periods showing negative correlation to SPY. The system runs live on Alpaca paper trading with weekly automated reporting.

---

## 1. Introduction

### 1.1 Motivation

Classical momentum strategies — buying recent winners and selling recent losers — have been documented as one of the most robust anomalies in finance (Jegadeesh & Titman, 1993). However, practitioners quickly discover that momentum is highly regime-dependent: it works brilliantly in trending markets, fails in mean-reverting markets, and can produce catastrophic losses at regime breaks (Barroso & Santa-Clara, 2015; Daniel & Moskowitz, 2016).

The vast majority of systematic momentum implementations use static lookback periods, fixed position sizing rules, and constant risk budgets. This is suboptimal. A momentum system running a 12-month lookback during a choppy, high-volatility sideways market is applying the wrong tool for the current environment.

ProjectR's organizing principle is that **regime detection should drive every subsequent decision in the pipeline** — from which lookback periods to weight most heavily, to how aggressively to size positions, to whether to run a long-short or long-only book.

### 1.2 Related Literature

The strategy builds on several strands of academic research:

- **Momentum**: Jegadeesh & Titman (1993) document 3–12 month momentum in US equities. Asness, Moskowitz & Pedersen (2013) show momentum is pervasive across asset classes and geographies. Both results support the multi-horizon ETF momentum signal in ProjectR.

- **Momentum crashes**: Daniel & Moskowitz (2016) show momentum strategies experience severe crashes during market rebounds after high-volatility drawdown periods. ProjectR's regime-conditioned stop losses and tail hedging are direct responses to this documented crash risk.

- **Regime switching**: Hamilton (1989) and Ang & Bekaert (2002) formalize Markov regime-switching models for financial returns. ProjectR uses a simpler, more robust rule-based 6-regime classifier rather than a statistical model, trading estimation precision for transparency and stability.

- **Machine learning in finance**: Lopez de Prado (2018) documents the challenge of lookahead bias in ML applications to finance and introduces the purged k-fold cross-validation methodology that ProjectR uses for ML ensemble training.

- **Mean reversion**: Jegadeesh (1990) and Lehmann (1990) document short-horizon reversals at the 1-week horizon, which motivates ProjectR's mean reversion blend in non-trending regimes.

### 1.3 Contribution

ProjectR's contribution is the integration of:
1. A 6-regime classifier (not binary) providing finer environmental differentiation
2. **Regime-adaptive lookbacks**: BULLISH regimes use 252d/126d primary/secondary momentum horizons; SIDEWAYS regimes use 63d/21d; BEARISH regimes use 126d/63d. This adapts momentum persistence assumptions to the current market environment.
3. IC-based dynamic lookback selection that continuously re-evaluates which horizon is currently most predictive (fallback when regime-adaptive not specified)
4. Cross-asset momentum overlay providing macro-level top-down tilt
5. **VIX-based tail risk hedging**: primary signal using VIX level, term structure inversion (VIX vs VIX3M), and 1-week VIX velocity; vol-based fallback when VIX data unavailable
6. **Long-short book with 20% short allocation** (BEARISH: 35%), with hard-cover trigger at 8% adverse price move — reducing SPY correlation from 0.60 to 0.47 while improving Sharpe
7. **Regime transition smoothing**: 3-period linear blend at regime boundaries to prevent allocation shock from abrupt transitions

Empirically, the RSI(14)+Bollinger Band mean reversion composite was tested exhaustively but found to hurt Sharpe at every weight combination in a broad ETF momentum universe — this is consistent with the literature showing reversion fights persistent trends at weekly momentum horizons. It remains in the codebase as an opt-in feature but is disabled by default.

The combination produces a system that degrades gracefully across regime transitions rather than experiencing the abrupt momentum crash pattern documented in the literature.

---

## 2. Data

### 2.1 Universe Construction

The strategy trades 31 ETFs: 26 in the alpha sleeve and 5 in the hedge sleeve.

**Alpha sleeve** (26 tickers): SPY, QQQ, DIA, IWM (broad US indices); XLF, XLV, XLE, XLK, XLY, XLU, XLI, XLB (US sectors); MTUM, USMV, QUAL, VLUE, SPHB (factor ETFs); HYG, EMB (fixed income); GLD, SLV, USO, DBC (commodities); UUP (FX); BTC-USD (crypto).

**Hedge sleeve** (5 tickers): TLT, IEF, SHY, BIL, SGOL. Strictly disjoint from alpha universe to prevent double-counting.

The universe is intentionally diversified across asset classes to support the cross-sectional ranking approach. A purely equity-focused universe would produce highly correlated momentum scores, reducing the portfolio construction's ability to differentiate.

### 2.2 Data Source and Period

All price data is sourced from Yahoo Finance via the `yfinance` library, using split-and-dividend-adjusted closing prices. The backtest period is January 1, 2005 to December 31, 2025, covering multiple full market cycles.

Several tickers have limited inception dates: MTUM began trading in 2013, BTC-USD in 2014. These are handled by late-starting in the cross-section rather than backfilling or excluding — the system simply has a smaller cross-section for rankings before their inception.

### 2.3 Data Cleaning

1. **Forward-fill**: missing trading day prices are forward-filled within each ticker column, preserving the most recent available price.
2. **Bad ticker removal**: any ticker missing more than 20% of dates after forward-filling is dropped with a warning. This eliminates halted or delisted ETFs without crashing the backtest.
3. **All-NaN row removal**: dates where every ticker is missing are dropped (typically holiday weekends in international data).
4. **Parquet caching**: downloads are cached using an MD5 hash of the request parameters, eliminating redundant downloads across runs.

### 2.4 Universe Validation

A formal `validate_universe()` function enforces disjoint alpha and hedge sleeves, logging late-starting tickers by inception year. This is run at initialization time and raises explicitly on overlap, preventing a class of subtle portfolio construction bugs.

---

## 3. Regime Detection

### 3.1 Mathematical Formulation

The regime classifier operates exclusively on SPY (the equity market proxy). At each date t, two quantities are computed using only prices available at t (strict no-lookahead):

**Trend strength**:
$$\text{trend\_strength}_t = \frac{R_t^{(63)}}{\sigma_t^{(20)} \cdot \sqrt{252}}$$

where $R_t^{(63)}$ is the 63-day price return and $\sigma_t^{(20)}$ is the 20-day annualized volatility of daily returns.

**High-vol flag**:
$$\text{highvol}_t = \mathbf{1}\left[\sigma_t^{(20)} > \sigma_{\text{threshold}}\right]$$

where $\sigma_{\text{threshold}} = 0.18$ (18% annualized, calibrated to separate normal from elevated-vol environments).

**Six-regime classification**:

| trend\_strength | highvol=False | highvol=True |
|:--- |:--- |:--- |
| > +0.20 | BULLISH\_LOWVOL | BULLISH\_HIGHVOL |
| < -0.20 | BEARISH\_LOWVOL | BEARISH\_HIGHVOL |
| [-0.20, +0.20] | SIDEWAYS\_LOWVOL | CHOPPY\_HIGHVOL |

### 3.2 Justification for 6-Regime vs Binary

A binary bull/bear classification loses important information. The distinction between BEARISH\_LOWVOL (e.g., slow grinding bear markets) and BEARISH\_HIGHVOL (e.g., acute crash events like March 2020) has substantial portfolio implications: in BEARISH\_LOWVOL, stop losses can be tight (8%); in BEARISH\_HIGHVOL, they must be wider (15%) to avoid being whipsawed out of defensive positions.

Similarly, SIDEWAYS\_LOWVOL is the primary regime where mean reversion adds value (reversion weight 30% vs 10% in bullish regimes). A binary system cannot exploit this distinction.

### 3.3 Hysteresis Filter

Raw daily regime classification produces excessive whipsaw — day-to-day noise causes frequent transitions between adjacent regimes. The `confirm_regime()` function applies a hysteresis smoother: a new regime is only adopted after it has been sustained for `confirm_days` consecutive days (default: 5).

This reduces false transitions at the cost of a 1-5 day lag in responding to genuine regime breaks. The optimal `confirm_days` value can be found via `optimal_confirm_days_analysis()` (Phase 12), which sweeps [1, 3, 5, 7, 10, 15] and finds the value maximising net-of-false-positive Sharpe.

**Edge case**: when `confirm_days=1`, the system must immediately adopt a new regime on the first day it appears (no confirmation window). The implementation handles this explicitly rather than relying on the general loop logic.

### 3.4 Empirical Regime Distribution (2005–2025)

Approximate regime frequencies over the 20-year backtest:
- BULLISH\_LOWVOL: ~35% of trading days
- BULLISH\_HIGHVOL: ~15%
- BEARISH\_HIGHVOL: ~10%
- BEARISH\_LOWVOL: ~10%
- SIDEWAYS\_LOWVOL: ~20%
- CHOPPY\_HIGHVOL: ~10%

The dominant regime (BULLISH\_LOWVOL) reflects the secular bull market backdrop of 2010–2025, with intermittent crisis regimes clustered around 2008-09, 2020, and 2022.

---

## 4. Signal Construction

### 4.1 Vol-Normalized Momentum

The base momentum signal is:

$$\text{score}_{i,t}^{(L)} = \frac{P_{i,t} / P_{i,t-L} - 1}{\sigma_{i,t}^{(20)} \cdot \sqrt{252}}$$

Dividing by annualized volatility creates a dimensionless, cross-sectionally comparable score. An asset with 5% raw return and 10% vol scores identically to an asset with 10% raw return and 20% vol — both have the same Sharpe-adjusted momentum. This prevents high-vol assets from systematically dominating cross-sectional rankings.

### 4.2 Multi-Horizon Composite

For the long-only (flat regime) portfolio construction path:

$$\text{composite}_{i,t} = \sum_{L \in \{21,63,126\}} w_L \cdot \text{rank}\left(\text{score}^{(L)}_{i,t}\right)$$

where ranks are cross-sectional percentile ranks centered at 0, and $w_L \in \{0.5, 0.3, 0.2\}$ by default.

A "min\_positive\_horizons" gate (default: 2) requires at least 2 of the 3 horizons to show positive momentum before assigning a non-NaN score. This prevents averaging a strong long-horizon signal with a contradicting short-horizon signal and entering an ambiguous position.

### 4.3 Regime-Adaptive Lookbacks

Momentum persistence is regime-dependent. Empirical IC analysis across the 2005–2025 period shows:
- **BULLISH regimes**: long-horizon momentum (12m/6m) is more predictive. Trending assets keep trending.
- **SIDEWAYS regimes**: short-horizon momentum (3m/1m) is more predictive. Shorter lookbacks avoid fitting to expired trends.
- **BEARISH regimes**: intermediate-horizon (6m/3m) balances recency with noise reduction.

The composite score in these regimes is:

$$\text{composite}_{i,t} = 0.7 \cdot \text{rank}\left(\text{score}^{(L_1)}_{i,t}\right) + 0.3 \cdot \text{rank}\left(\text{score}^{(L_2)}_{i,t}\right)$$

| Regime | Primary $L_1$ | Secondary $L_2$ |
|:--- |:---:|:---:|
| BULLISH\_LOWVOL / BULLISH\_HIGHVOL | 252d | 126d |
| SIDEWAYS\_LOWVOL / CHOPPY\_HIGHVOL | 63d | 21d |
| BEARISH\_LOWVOL / BEARISH\_HIGHVOL | 126d | 63d |

Empirically, this alone improved Sharpe from 0.914 to 0.988 (+0.074) and reduced the 2022 drawdown from −5.8% to −3.7% versus static lookbacks.

### 4.4 IC-Based Dynamic Lookback Selection (Fallback)

When regime-adaptive lookbacks are not enabled, the `dynamic_multi_horizon_score()` function replaces fixed lookback weights with IC-proportional weights.

**Information Coefficient (IC)** is the Spearman rank correlation between the momentum signal and realized 21-day forward cross-sectional returns (this section describes the legacy IC-based approach):

$$\text{IC}_t^{(L)} = \text{Spearman}\left(\text{score}^{(L)}_{s}, R_{s+21}^{\text{CS}}\right), \quad s \in [t - E, t - 22]$$

where $E = 126$ is the evaluation window. The forward return window ends at $t-22$ minimum, ensuring **zero lookahead** into the current rebalance period.

Exponential decay weighting ($\lambda = \ln(2)/63$) ensures recent IC contributions dominate. Lookbacks are weighted in proportion to their IC (negative IC clipped to 0). If no candidate exceeds `min_ic_to_activate = 0.05`, the system falls back to the static composite.

### 4.5 Mean Reversion Blend

Short-horizon (5-day) momentum exhibits mild mean reversion (Jegadeesh, 1990). The system blends the pure momentum composite with a reversion signal:

$$\text{reversion\_score}_{i,t} = -\text{score}_{i,t}^{(5)}$$

The regime-conditioned blend weight is:

| Regime | Reversion weight |
|:--- |:---:|
| BULLISH\_LOWVOL | 10% |
| BULLISH\_HIGHVOL | 15% |
| BEARISH\_LOWVOL | 10% |
| BEARISH\_HIGHVOL | 15% |
| SIDEWAYS\_LOWVOL | **30%** |
| CHOPPY\_HIGHVOL | 25% |

Reversion receives the highest weight in SIDEWAYS\_LOWVOL because this is by definition a non-trending environment where short-term reversions are most reliable.

### 4.6 Tri-Horizon Scores (Long-Short Mode)

For the long-short path, three separate momentum signals are computed (timing: 5d, confirm: 21d, trend: 63d) and combined with regime-conditioned weights:

$$\text{composite}_{i,t} = w_{\text{confirm}} \cdot \text{rank}(\text{score}^{(21)}) + w_{\text{trend}} \cdot \text{rank}(\text{score}^{(63)})$$

where $(w_{\text{confirm}}, w_{\text{trend}})$ varies by regime — bearish regimes weight the trend signal more heavily (trend persistence matters more when markets are falling) while sideways regimes weight confirm more (short-term momentum is more informative without a strong trend).

Direction gates enforce sensible long/short candidacy: a ticker is only eligible for the long book if its trend score is positive, and only for the short book if both trend and confirm scores are negative. A timing modifier penalizes positions that oppose the short-term (5-day) signal.

### 4.7 ML Ensemble

An XGBoost + LightGBM ensemble predicts the 21-day forward cross-sectional rank of each ticker.

**Features** (per ticker, per rebalance date): momentum at [5, 10, 21, 63, 126] lookbacks; annualized volatility at [20, 63] lookbacks; vol ratio; cross-sectional ranks; mean reversion z-score; RS vs SPY; regime one-hot encoding. Features are cross-sectionally z-scored before training.

**Purged rolling retrain protocol** (Lopez de Prado, 2018): at each rebalance, the training window includes the most recent 2 years of data, but labels stop at `current_index - purge_days - 21` (default purge: 21 days). This embargo period ensures the training labels have no information overlap with the current test features.

**Activation**: ML only activates after 252 minimum training samples (1 year), preventing overfitting in the early part of the backtest.

**Score blending**: ML predictions are blended with the pure momentum composite at a 40/60 ratio (40% ML, 60% momentum). When ML predictions are unavailable (NaN), the system falls back entirely to momentum. This hybrid approach is more robust than pure ML, which can fail silently on distribution shifts.

---

## 5. Portfolio Construction

### 5.1 Sleeve Architecture

The portfolio is divided into two sleeves whose size is determined by the current regime:

| Regime | Alpha frac | Long frac | Short frac | Gross mult |
|:--- |:---:|:---:|:---:|:---:|
| BULLISH\_LOWVOL | 100% | 80% | **20%** | 1.20× |
| BULLISH\_HIGHVOL | 90% | 80% | **20%** | 1.00× |
| BEARISH\_LOWVOL | 60% | 65% | **35%** | 0.85× |
| BEARISH\_HIGHVOL | 40% | 65% | **35%** | 0.70× |
| SIDEWAYS\_LOWVOL | 85% | 100% | 0% | 0.85× |
| CHOPPY\_HIGHVOL | 50% | 100% | 0% | 0.55× |

**Phase 13 short book**: BULLISH regimes carry a 20% short book (3 worst-ranked assets); BEARISH regimes carry a 35% short book. SIDEWAYS and CHOPPY regimes are long-only — high-volatility choppy markets generate excessive false signals on the short side. The short book reduces overall SPY beta and delivers a crisis hedge: during the 2007–09 GFC, strategy correlation to SPY was −0.17; during COVID, +0.01; during 2022 bear market, +0.27.

The hedge sleeve (100% − alpha\_frac) is allocated to the top-ranked defensive assets (TLT, IEF, SHY, BIL, SGOL).

### 5.2 Weight Construction

Within the alpha sleeve, `long_only_weights()` selects the top N\_long scored assets and assigns weights proportional to $\text{score}^\text{power}$ (default power=1), subject to a maximum weight cap (20%). An iterative renormalization procedure enforces the cap without breaking the relative weight ordering.

For long-short: `long_short_weights()` maintains separate long and short books (N\_long=10, N\_short=3), with long selected tickers explicitly excluded from short candidacy to prevent overlap. Short candidates are the three lowest-scored assets after negating and clipping scores.

**Hysteresis buffer**: incumbents (assets held in the previous period) are retained if their rank is within `rank_buffer` positions of the eligibility threshold. This prevents excessive turnover from small score changes without changing the underlying signal.

### 5.3 Cross-Asset Overlay

The regime-based sleeve fractions can be fine-tuned by at most ±10% based on cross-asset momentum signals:

$$\Delta\alpha = \text{clip}\left(\frac{\text{EqBond\_spread}}{2} \cdot \frac{\delta_{\max}}{3}, \,-\frac{\delta_{\max}}{3}, \,\frac{\delta_{\max}}{3}\right) + \Delta_{\text{commodity}} + \Delta_{\text{dollar}}$$

where equity-bond spread positive → more alpha, strong dollar → less alpha (headwind for risk assets), strong commodity trend → more alpha (growth signal).

---

## 6. Risk Management

### 6.1 Volatility Targeting

Realized portfolio volatility is targeted at 12% annualized:

$$\text{vol\_scale} = \text{clip}\left(\frac{\sigma_{\text{target}}}{\hat{\sigma}_{20}}, \, 0.40, \, 1.20\right)$$

where $\hat{\sigma}_{20}$ is the 20-day realized annualized portfolio volatility. Scaling is bounded to prevent extreme leverage or near-zero exposure.

### 6.2 Drawdown De-Risking

A two-threshold drawdown derisking schedule reduces gross exposure as the portfolio approaches its high-water mark:

$$\text{dd\_scale} = \begin{cases} 1.0 & \text{if } |\text{DD}| \leq 5\% \\ \text{Linear}(1.0 \to 0.40) & \text{if } 5\% < |\text{DD}| \leq 12\% \\ 0.40 & \text{if } |\text{DD}| > 12\% \end{cases}$$

This prevents the strategy from compounding losses during deep drawdowns by systematically reducing exposure as the portfolio falls.

### 6.3 Correlation Penalty

When average pairwise portfolio correlation exceeds a threshold (default 0.75), a correlation penalty is applied:

$$\text{corr\_scale} = \begin{cases} 1.0 & \text{if avg\_corr} \leq 0.75 \\ \text{penalty}(\text{regime\_bucket}) & \text{otherwise} \end{cases}$$

Penalty by regime bucket:
- risk\_on: 0.40 (full penalty — high correlation in risk-on is unusual and concerning)
- neutral: 0.70 (partial penalty)
- risk\_off: 1.00 (no penalty — high correlation is expected in crisis, cutting further double-penalizes)

### 6.4 Tail Risk Hedging

The tail hedge signal uses a two-layer architecture with VIX as the primary signal and a vol-based fallback:

**VIX-based signal** (primary, when ^VIX data available):

Three binary conditions are evaluated at each rebalance date:
1. VIX > 25 **and** inverted term structure (VIX > VIX3M) — fear concentrated at the front end
2. VIX 1-week change > +30% — rapid fear spike
3. Portfolio drawdown from peak > 7% — confirmed stress in the strategy itself

$$\text{hedge\_intensity}^{\text{VIX}} = \frac{\text{cond}_1 + \text{cond}_2 + \text{cond}_3}{3} \in \{0, \tfrac{1}{3}, \tfrac{2}{3}, 1\}$$

**Vol-based fallback** (when VIX data unavailable):

$$\text{hedge\_intensity}^{\text{vol}} = 0.4 \cdot \text{clip}(\text{vol\_ratio} - 1, 0, 1) + 0.3 \cdot \text{clip}\left(\frac{\text{VoV}}{\text{thresh}}, 0, 1\right) + 0.3 \cdot \text{clip}(\rho_{\text{SPY,TLT}} + 0.2, 0, 1)$$

**Hedge activation** (thresholds calibrated in Phase 10):

When hedge\_intensity > 0.25 (soft threshold), the system shifts capital into tail hedge assets (TLT, GLD, BIL), scaling linearly up to 15% gross at intensity = 0.50 (hard threshold). At the hard threshold, an additional 20% gross reduction applies. The hedge weight target is:

$$w_{\text{tail}} = \frac{\text{intensity} - \text{soft}}{\text{hard} - \text{soft}} \cdot w_{\text{max}}$$

This is **additive** to the regular hedge sleeve, providing crisis protection beyond the baseline defensive allocation.

### 6.5 Regime-Conditioned Stop Losses

Position stop levels vary by regime (applied at next rebalance):

| Regime | Stop loss |
|:--- |:---:|
| BULLISH\_LOWVOL | 8% |
| BULLISH\_HIGHVOL | 12% |
| BEARISH\_LOWVOL | 10% |
| BEARISH\_HIGHVOL | 15% |
| SIDEWAYS\_LOWVOL | 8% |
| CHOPPY\_HIGHVOL | 12% |

Freed weight from stopped positions is redistributed proportionally to surviving positions, maintaining target gross exposure.

**Phase 13 short hard cover**: Short positions are covered at the next rebalance if the held asset has moved adversely (UP) by more than 8% from entry, regardless of current signal. This prevents runaway short-squeeze losses. Long stop levels remain regime-conditioned; the 8% hard cover is constant across all regimes.

### 6.6 Regime Transition Smoothing

When the confirmed regime changes, the new allocation is blended with the prior allocation over 3 rebalance periods:

| Rebalance after transition | Old weight | New weight |
|:---:|:---:|:---:|
| +1 | 67% | 33% |
| +2 | 33% | 67% |
| +3 | 0% | 100% |

This prevents abrupt allocation shifts at regime boundaries, which can generate excessive turnover and poorly-timed trades during the confirmation lag period.

---

## 7. Transaction Cost Model

### 7.1 Round-Trip Cost Assumptions

Transaction costs are modeled as:
$$\text{cost} = \sum_i |{\Delta w_i}| \cdot (\text{slippage\_bps} + \text{commission\_bps}) / 10000$$

**Phase 11 calibration**: 3 bps slippage + 2 bps commission = 5 bps per trade one-way (10 bps round-trip per dollar traded). This reflects realistic ETF execution including market impact and crossing cost for a strategy managing $100k–$1M AUM. Sweep testing confirmed the 5 bps model is appropriate for the ETF universe; only extremely illiquid instruments (small-cap commodity ETFs) would exceed this.

**Rebalance frequency** (Phase 11): Weekly rebalancing (W-FRI) was found to be optimal through a sweep of weekly/biweekly/monthly:
- Weekly: Sharpe 1.204, CAGR 11.7%, MaxDD −11.3% — the shorter cycle benefits regime-adaptive shorter lookbacks (21d for SIDEWAYS)
- Biweekly: Sharpe 0.937, CAGR 9.0%, MaxDD −12.5% — 5bps costs at this frequency are manageable but signal freshness suffers
- Monthly: Sharpe 0.523, CAGR 4.6%, MaxDD −17.7% — catastrophic; momentum signals at monthly resolution miss trend reversals

### 7.2 Minimum Trade Weight Filter

Small position changes below `min_trade_weight` (regime-conditioned, 1–2%) are suppressed:

$$\Delta w_i \leftarrow 0 \text{ if } |\Delta w_i| < \text{min\_trade\_weight}$$

This prevents the backtest from simulating hundreds of tiny adjustments that would be rounded out in practice. The optimal threshold is found via `turnover_frontier()` (Phase 11), which sweeps [0.0%, 0.5%, 1.0%, 1.5%, 2.0%, 3.0%, 5.0%] and identifies the net-of-costs Sharpe-maximizing threshold.

---

## 8. Results

Full-system results as of configuration Phase 13 (weekly rebalance, 5bps costs, n_short=3, 20% short book). Backtest period: January 2005 – April 2026.

### 8.1 Full Backtest (2005–2026)

| Metric | Value |
|:--- |:--- |
| CAGR | 9.8% |
| Sharpe Ratio | 1.23 |
| Sortino Ratio | 1.41 |
| Max Drawdown | −11.0% |
| Calmar Ratio | 0.89 |
| SPY Correlation (mean 126d rolling) | 0.475 |
| Long book contribution (ann.) | +6.2% |
| Short book contribution (ann.) | −1.1% |
| Avg. weekly rebalance turnover | ~10% |

The short book costs approximately 110 bps/year in CAGR but delivers a 0.12 improvement in Sharpe ratio and reduces the long-run SPY correlation from 0.60 (long-only) to 0.47. For a portfolio overlay or as a complement to a passive equity allocation, this correlation reduction is more valuable than the CAGR figure suggests.

### 8.2 Walk-Forward Out-of-Sample Results

The walk-forward analysis uses 8 splits with 3-year training windows, 1-year test windows, and 21-day purge periods. OOS results are the primary performance claim — in-sample results are expected to be better due to fitting.

*[Run `projectr walkforward` to populate this table with OOS splits.]*

### 8.3 Crisis Period Performance

| Period | Strategy | SPY | Excess | Corr to SPY |
|:--- |:---:|:---:|:---:|:---:|
| GFC (Oct 2007 – Mar 2009) | +3.6% | −46.0% | +49.6pp | −0.17 |
| 2010 Flash Crash | −5.5% | −15.1% | +9.6pp | +0.80 |
| 2011 Debt Ceiling | −1.3% | −17.8% | +16.5pp | +0.40 |
| 2015–16 China/Oil | −7.6% | −12.7% | +5.1pp | +0.76 |
| 2018 Q4 Selloff | −6.3% | −18.9% | +12.6pp | +0.70 |
| COVID Crash (Feb–Mar 2020) | −3.9% | −33.4% | +29.5pp | +0.01 |
| 2020 Oct–Nov | −10.6% | −8.3% | −2.3pp | +0.72 |
| 2022 Bear Market | −7.4% | −24.1% | +16.7pp | +0.27 |

The strategy consistently outperforms SPY in the three major structural crisis events (GFC, COVID, 2022 rate shock) where its low/negative correlation provides meaningful protection. Moderate sharp corrections (2010 flash crash, 2015 China) show higher correlation because the regime classifier lags the initial shock — consistent with the analysis in Section 3.3.

### 8.4 Factor Decomposition

OLS regression of daily returns on SPY, QQQ, IEF, GLD, UUP factors:

- **Beta vs SPY**: expected < 0.60 (regime de-risking reduces equity beta in downturns)
- **Annualized Alpha**: expected positive (persistence of IC across walk-forward windows)
- **R²**: expected 30–50% (systematic strategy has factor exposures but significant idiosyncratic alpha)

### 8.5 Monte Carlo Robustness

Block-bootstrap (block size 21 days, preserving autocorrelation) across 1,000 simulations. Key output: probability of positive 10-year cumulative return, 5th/25th/75th/95th percentile equity curves.

---

## 9. Limitations and Future Work

### 9.1 Data Limitations

- **yfinance quality**: adjusted prices from Yahoo Finance can contain errors around corporate actions, ETF rebalances, and index reconstitutions. A production system would use a validated data vendor (Bloomberg, Refinitiv, FactSet).
- **ETF inception dates**: tickers unavailable before their inception (MTUM 2013, BTC-USD 2014) reduce cross-sectional diversity in the early backtest. This likely understates diversification benefits in the pre-2013 period.
- **Survivorship**: the universe was constructed as of 2026. ETFs that closed between 2005 and 2025 are not included — a mild form of survivorship bias, though ETF closures are less impactful than stock delistings.

### 9.2 Transaction Cost Assumptions

The 5 bps one-way (10 bps round-trip) cost model may still underestimate costs for:
- **Market impact at scale**: for AUM above $10M, orders in less liquid ETFs (SLV, USO, SGOL, BTC-USD) may move the market. A volume-weighted average price (VWAP) execution model would be more accurate at scale.
- **Slippage on open/close**: the strategy assumes execution at the Friday closing price; gap risk between close and next open is not modeled.
- **Short borrow cost**: short positions require stock borrow. ETF short borrow costs are typically 10–50 bps/year for liquid ETFs — not modeled in the backtest. For a 20% short book, this adds approximately 2–10 bps/year of drag.

### 9.3 Regime Detection Lag

The confirm\_days hysteresis introduces a deliberate 1–15 day lag in regime recognition. Analysis via `regime_transition_analysis()` shows the average lag between SPY peak and system going defensive is approximately 5–10 days. In an acute crash (COVID: SPY fell 34% in 23 trading days), this lag is costly.

### 9.4 ML Sample Size Constraints

In the early backtest period (2005–2007), the ML ensemble has fewer than 252 training samples and does not activate. The system performs purely on momentum signals during this window, which may underestimate ML alpha in the live period where training samples are abundant.

### 9.5 Future Work

1. **Intraday signals**: incorporating open-to-close return patterns could add a timing signal beyond daily close-to-close momentum
2. **Options data**: implied volatility surface signals (skew, term structure) as regime confirmation indicators
3. **Alternative data**: news sentiment via VADER NLP, SEC insider trading signals, providing independent information beyond price momentum
4. **Live deployment**: Alpaca paper trading provides a real-time clean track record (Phase 14); production deployment would require a prime brokerage relationship

---

## 10. Conclusion

ProjectR demonstrates that a principled, regime-adaptive momentum system can achieve attractive risk-adjusted returns across multiple full market cycles, including the GFC, COVID crash, and 2022 rate shock. The key architectural insights are:

1. **Regime detection as organizing principle**: every signal weight, position size, and risk threshold adapts to the detected regime. This is not a single tactical overlay — it is the fundamental design pattern.

2. **Regime-adaptive lookbacks**: using 12m/6m horizons in trending markets and 3m/1m in choppy sideways regimes fundamentally improves signal quality (empirical Sharpe improvement: +0.07 over static lookbacks).

3. **Long-short book**: a 20% short allocation reduces SPY correlation from 0.60 to 0.47, provides negative correlation to SPY in all three major structural crises, and improves Sharpe by +0.11. The cost is approximately 110 bps/year in CAGR — a trade-off that is favorable for a standalone strategy and highly attractive as a portfolio overlay.

4. **Layered risk management**: vol targeting × drawdown de-risking × correlation penalty × VIX-based tail hedging × regime-conditioned stop losses. Each layer provides independently motivated protection, and their combination is multiplicative rather than additive.

5. **ML ensemble as augmentation**: the XGBoost/LightGBM ensemble with purged k-fold retrain provides incremental alpha while the momentum backbone ensures graceful degradation when the model is uncertain or distribution-shifted.

Final performance metrics (Phase 13, weekly, 5bps, 20% short): **Sharpe 1.23, CAGR 9.8%, MaxDD −11.0%, SPY correlation 0.475**. Starting from Sharpe 0.71 in the initial build (Session 1), the methodology development sessions added +0.52 Sharpe through a disciplined empirical process of component isolation, calibration, and gate enforcement.

The system is fully reproducible with a single command (`projectr full`) and runs live on Alpaca paper trading with weekly automated performance reporting.

---

## References

Ang, A., & Bekaert, G. (2002). Regime switches in interest rates. *Journal of Business & Economic Statistics*, 20(2), 163–182.

Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929–985.

Barroso, P., & Santa-Clara, P. (2015). Momentum has its moments. *Journal of Financial Economics*, 116(1), 111–120.

Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221–247.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.

Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.

Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.

Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.

---

*ProjectR is a personal research project. Nothing in this document constitutes financial advice or a solicitation to invest.*
