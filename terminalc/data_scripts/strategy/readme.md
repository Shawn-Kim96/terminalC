# Indicator Strategy Rulebook

This catalog captures a lightweight replica of the Investing.com style “technical summary” for crypto pairs.  
It stores the textual heuristics and thresholds that describe each indicator-driven strategy.  
Indicator computation and signal evaluation live in `src/data/core/indicator_signals.py`, which consumes the OHLCV tables and applies the same rules programmatically.

## Implemented Indicator Rules

### Oscillators & Momentum (stored in catalog, evaluated in `indicator_signals`)

| Key | Indicator | Buy Trigger | Sell Trigger | Neutral / Notes |
| --- | --- | --- | --- | --- |
| `rsi_14` | RSI (14) | RSI ≤ 30 → oversold bias | RSI ≥ 70 → overbought bias | 30 < RSI < 70 treated as neutral; missing values mark `unknown`. |
| `stoch_d_9_6` | Stochastic %D (9,6) | %D ≤ 20 → oversold | %D ≥ 80 → overbought | Mid-zone = neutral; unavailable values set `unknown`. |
| `stoch_rsi_14` | Stochastic RSI (14) | Value ≤ 20 → oversold RSI | Value ≥ 80 → overbought RSI | Mid-band defaults to neutral. |
| `macd_hist` | MACD Histogram (12,26,9) | Histogram > 0.05 → bullish momentum | Histogram < -0.05 → bearish momentum | |hist| ≤ 0.05 considered neutral. |
| `ultimate_osc` | Ultimate Oscillator (7,14,28) | Reading ≥ 55 → buy pressure | Reading ≤ 45 → sell pressure | Between 45–55 stays neutral. |
| `roc_12` | Rate of Change (12) | ROC > 0 → upside momentum | ROC < 0 → downside momentum | Exactly zero becomes neutral. |
| `cci_14` | Commodity Channel Index (14) | CCI ≥ +100 → strong upside | CCI ≤ -100 → strong downside | |CCI| < 100 treated as neutral. |
| `willr_14` | Williams %R (14) | %R ≤ -80 → oversold | %R ≥ -20 → overbought | Between thresholds remains neutral. |

### Trend Strength & Bias

| Key | Indicator | Buy Trigger | Sell Trigger | Neutral / Notes |
| --- | --- | --- | --- | --- |
| `ema12_vs_ema26` | EMA Spread (12 vs 26) | EMA12 − EMA26 > 0 → bullish alignment | EMA12 − EMA26 < 0 → bearish alignment | Spread near zero remains neutral. |
| `adx_14` | ADX (14) with ±DI | ADX ≥ 20 & +DI > -DI → bullish trend | ADX ≥ 20 & -DI > +DI → bearish trend | ADX < 20 or balanced ±DI stays neutral. |
| `highs_lows_14` | 14-period Range Bias | Close skewed to range highs | Close skewed to range lows | Close within ±0.10% of mid-range stays neutral. |
| `bull_bear_power_13` | Bull/Bear Power (13) | Bulls dominate by ≥0.10% of price | Bears dominate by ≥0.10% of price (negative value) | Within ±0.10% remains neutral. |

### Volatility

| Key | Indicator | Interpretation |
| --- | --- | --- |
| `atr_14` | ATR (14) | Reason field reports high (≥3% of price) or low (≤1%) volatility; signal stays `neutral` because the metric is directional-agnostic. |

### Moving Averages

All moving-average rules share one logic: if price sits ≥0.10 % above the average → `buy`, ≤0.10 % below → `sell`, otherwise `neutral`. Missing inputs mark `unknown`.

- Simple MAs: `sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_100`, `sma_200`
- Exponential MAs: `ema_5`, `ema_10`, `ema_20`, `ema_50`, `ema_100`, `ema_200`

The reason string quantifies the percentage gap so the LLM can cite how far price has diverged.

## Generated Artifacts (implemented in `src/data/core/indicator_signals.py`)

- **Per-indicator signals**: `generate_indicator_signals` returns the latest signal per `(asset_id, symbol, timeframe, indicator_key)` with accompanying reasoning and UTC evaluation timestamp.
- **Aggregated summary**: `summarize_indicator_signals` counts how many indicators vote `buy`, `sell`, `neutral`, or `unknown` per timeframe and resolves an overall signal (`buy`, `sell`, `neutral`, or `mixed` when ties occur).
- **Persistence helpers**: `save_indicator_artifacts` writes two pickle files under `data/processed/`:
  - `final_indicator_signals.pickle`
  - `final_indicator_summary.pickle`

You can inspect the rulebook via `indicator_rulebook()` or update thresholds by editing the evaluator functions in `indicator_signals.py`. Adding a new indicator only requires defining an `IndicatorRule` with the appropriate input columns and evaluator logic.***
