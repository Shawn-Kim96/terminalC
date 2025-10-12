# Data Layer Overview

The `src/data` package produces all artefacts that flow into the DuckDB warehouse under `data/database/market.duckdb`.  
Processing is performed via `src/data/main.py`, which:

- loads raw OHLCV CSVs,
- enriches them with indicators defined in `core/technical_indicators.py`,
- detects RSI-based divergences,
- derives Investing.com–style indicator signals (`core/indicator_signals.py`),
- builds the strategy catalog (`strategy/strategy_catalog.py`), and
- saves each result as a pickle in `data/processed/`.

The ingest script `database/injest_to_duckdb.py` simply reads those pickles and recreates the DuckDB tables.

## Processed Datasets

| File | Description |
| --- | --- |
| `final_candle.pickle` | Enriched OHLCV records for every asset/timeframe, including all technical indicators and feature columns. |
| `final_divergence.pickle` | Output from the RSI divergence detector with metadata on signal strength. |
| `final_indicator_signals.pickle` | Latest indicator verdicts (`buy`/`sell`/`neutral`/`unknown`) per asset, timeframe, and indicator. |
| `final_indicator_summary.pickle` | Aggregated counts of indicator votes per asset/timeframe plus the resolved consensus signal. |
| `final_indicator_rules.pickle` (optional) | Serialized rulebook describing each indicator and its required inputs; generated if not already present. |
| `final_strategies.pickle` | Textual catalog of the Investing.com–style strategy heuristics that LLM prompts can reference directly. |

## Core Tables

| Table | Columns | Description |
| --- | --- | --- |
| `assets` | `asset_id`, `symbol`, `name` | Static lookup for supported cryptocurrencies. |
| `candles` | `asset_id`, `coin`, `timeframe`, `ts`, `open`, `high`, `low`, `close`, `volume`,<br>`rsi`, `ema_12`, `ema_26`, `macd`, `macd_signal`, `macd_hist`,<br>`bb_middle`, `bb_upper`, `bb_lower`, `willr`, `atr`, `atr_14`,<br>`plus_di_14`, `minus_di_14`, `adx`, `adx_14`, `cci`, `cci_14`,<br>`stoch_k_9`, `stoch_d_9_6`, `stoch_rsi_14`, `ultimate_osc`, `roc_12`,<br>`ema_13`, `bull_power_13`, `bear_power_13`, `bull_bear_power_13`, `highs_lows_14`,<br>`sma_5`, `ema_5`, `sma_10`, `ema_10`, `sma_20`, `ema_20`,<br>`sma_50`, `ema_50`, `sma_100`, `ema_100`, `sma_200`, `ema_200`,<br>`peak_high_high`, `peak_high_close`, `peak_low_low`, `peak_low_close`, `ts_int` | Master OHLCV fact table with all derived indicators, ready for analytics or LLM retrieval. |
| `divergence` | `asset_id`, `timeframe`, `start_datetime`, `end_datetime`, `entry_datetime`, `entry_price`, `previous_peak_datetime`, `divergence`, `price_change`, `rsi_change`, `strength_score` | Strict RSI divergence signals emitted by `DivergenceDetector`. |

## Strategy Tables

| Table | Columns | Description |
| --- | --- | --- |
| `indicator_rules` | `indicator_key`, `indicator_name`, `description`, `required_columns`, `timeframes` | Rulebook describing which candle columns each indicator uses and how the signal should be interpreted. |
| `indicator_signals` | `asset_id`, `symbol`, `timeframe`, `indicator_key`, `indicator_name`, `indicator_value`, `signal`, `reason`, `evaluated_at` | Latest indicator verdicts (buy/sell/neutral/unknown) with rationale text for each asset/timeframe. |
| `indicator_signal_summary` | `asset_id`, `symbol`, `timeframe`, `evaluated_at`, `buy_count`, `sell_count`, `neutral_count`, `unknown_count`, `total_indicators`, `overall_signal`, `dominant_ratio` | Aggregated indicator votes and the resulting consensus signal. |
| `strategies` | `strategy_id`, `indicator_key`, `name`, `signal_type`, `buy_condition`, `sell_condition`, `neutral_condition`, `notes`, `timeframes`, `tags`, `confidence_level`, `source`, `created_at`, `last_updated` | Catalog of Investing.com–style textual strategies that can be surfaced to an LLM to explain each indicator rule. |

## Web & Hype Tables (Planned)

The current pipeline does not yet populate news or social “hype” tables. The intended schema mirrors the original design:

| Table | Purpose |
| --- | --- |
| `sources` | Metadata about each news or social feed (name, base URL, content type). |
| `news_items` | Scraped articles with timestamps, titles, snippets, full text, and relevance tags. |
| `hype_signals` | Mapping from social/news events to affected assets, including sentiment or impact scores. |

These tables can be added to the workflow once the scrapers and extractors are integrated.***
