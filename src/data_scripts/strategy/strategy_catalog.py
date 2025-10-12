from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_BASE_STRATEGIES = [
    {
        "indicator_key": "rsi_14",
        "name": "RSI (14) Oversold/Overbought",
        "signal_type": "oscillator",
        "buy_condition": "RSI ≤ 30 indicates oversold pressure; consider accumulation.",
        "sell_condition": "RSI ≥ 70 indicates overbought pressure; consider trimming.",
        "neutral_condition": "30 < RSI < 70 keeps the stance neutral.",
        "notes": "Classic Relative Strength Index interpretation used by Investing.com.",
        "tags": "momentum,mean_reversion",
    },
    {
        "indicator_key": "stoch_d_9_6",
        "name": "Stochastic %D (9,6)",
        "signal_type": "oscillator",
        "buy_condition": "%D ≤ 20 suggests the market is oversold.",
        "sell_condition": "%D ≥ 80 signals an overbought condition.",
        "neutral_condition": "%D between 20 and 80 indicates a balanced state.",
        "notes": "Slow stochastic signal line aligns with Investing.com guidance.",
        "tags": "momentum,fast",
    },
    {
        "indicator_key": "stoch_rsi_14",
        "name": "Stochastic RSI (14)",
        "signal_type": "oscillator",
        "buy_condition": "StochRSI ≤ 20 highlights RSI-based oversold momentum.",
        "sell_condition": "StochRSI ≥ 80 highlights overbought momentum.",
        "neutral_condition": "Mid-band values prompt a neutral outlook.",
        "notes": "Secondary oscillator confirming RSI extremes.",
        "tags": "momentum,volatility",
    },
    {
        "indicator_key": "macd_hist",
        "name": "MACD Histogram (12,26,9)",
        "signal_type": "momentum",
        "buy_condition": "Histogram > 0.05 implies bullish momentum.",
        "sell_condition": "Histogram < -0.05 implies bearish momentum.",
        "neutral_condition": "|histogram| ≤ 0.05 signals equilibrium.",
        "notes": "Histogram sign tracks MACD vs signal line divergence.",
        "tags": "momentum,trending",
    },
    {
        "indicator_key": "ema12_vs_ema26",
        "name": "EMA 12/26 Crossover",
        "signal_type": "trend",
        "buy_condition": "EMA12 above EMA26 points to bullish alignment.",
        "sell_condition": "EMA12 below EMA26 points to bearish alignment.",
        "neutral_condition": "Spread near zero yields no directional edge.",
        "notes": "Short/medium EMA crossover tracked per Investing.com summary.",
        "tags": "trend,ema",
    },
    {
        "indicator_key": "adx_14",
        "name": "ADX (14) Trend Strength",
        "signal_type": "trend_strength",
        "buy_condition": "ADX ≥ 20 with +DI > -DI supports a bullish trend.",
        "sell_condition": "ADX ≥ 20 with -DI > +DI supports a bearish trend.",
        "neutral_condition": "ADX < 20 keeps bias neutral or ranging.",
        "notes": "Directional Movement Index combined with ADX threshold.",
        "tags": "trend_strength,dmi",
    },
    {
        "indicator_key": "atr_14",
        "name": "ATR (14) Volatility Regime",
        "signal_type": "volatility",
        "buy_condition": "—",
        "sell_condition": "—",
        "neutral_condition": "ATR relative to price is used for context, not directional bias.",
        "notes": "Reports high (≥3%) or low (≤1%) volatility in the signal reason.",
        "tags": "volatility,risk",
    },
    {
        "indicator_key": "willr_14",
        "name": "Williams %R (14)",
        "signal_type": "oscillator",
        "buy_condition": "%R ≤ -80 indicates oversold territory.",
        "sell_condition": "%R ≥ -20 indicates overbought territory.",
        "neutral_condition": "%R between -80 and -20 retains neutrality.",
        "notes": "Fast oscillator complementing RSI/Stochastic.",
        "tags": "momentum,mean_reversion",
    },
    {
        "indicator_key": "cci_14",
        "name": "CCI (14)",
        "signal_type": "oscillator",
        "buy_condition": "CCI ≥ +100 signals upside momentum.",
        "sell_condition": "CCI ≤ -100 signals downside momentum.",
        "neutral_condition": "|CCI| < 100 keeps stance neutral.",
        "notes": "Commodity Channel Index short lookback variant.",
        "tags": "momentum,volatility",
    },
    {
        "indicator_key": "highs_lows_14",
        "name": "Highs/Lows Bias (14)",
        "signal_type": "range_position",
        "buy_condition": "Close skewed toward range highs (> +0.10%).",
        "sell_condition": "Close skewed toward range lows (< -0.10%).",
        "neutral_condition": "Close near midpoint (±0.10%) implies balance.",
        "notes": "Mirrors Investing.com highs/lows table output.",
        "tags": "range,breakout",
    },
    {
        "indicator_key": "ultimate_osc",
        "name": "Ultimate Oscillator",
        "signal_type": "oscillator",
        "buy_condition": "Reading ≥ 55 indicates bullish pressure.",
        "sell_condition": "Reading ≤ 45 indicates bearish pressure.",
        "neutral_condition": "Values between 45 and 55 remain neutral.",
        "notes": "Multi-timeframe oscillator spanning 7/14/28 periods.",
        "tags": "momentum,oscillator",
    },
    {
        "indicator_key": "roc_12",
        "name": "Rate of Change (12)",
        "signal_type": "momentum",
        "buy_condition": "ROC > 0 denotes upward momentum.",
        "sell_condition": "ROC < 0 denotes downward momentum.",
        "neutral_condition": "ROC ≈ 0 indicates flat momentum.",
        "notes": "Price rate-of-change used in Investing.com summary panel.",
        "tags": "momentum",
    },
    {
        "indicator_key": "bull_bear_power_13",
        "name": "Bull/Bear Power (13)",
        "signal_type": "trend_strength",
        "buy_condition": "Positive deviation ≥ 0.10% of price favours bulls.",
        "sell_condition": "Negative deviation ≤ -0.10% of price favours bears.",
        "neutral_condition": "Deviation within ±0.10% treated as neutral.",
        "notes": "Measures distance between high/low and EMA13.",
        "tags": "trend_strength,volume_proxy",
    },
]


_MA_PERIODS = [5, 10, 20, 50, 100, 200]

for period in _MA_PERIODS:
    _BASE_STRATEGIES.append(
        {
            "indicator_key": f"sma_{period}",
            "name": f"SMA {period}",
            "signal_type": "moving_average",
            "buy_condition": f"Close ≥ SMA{period} + 0.10% suggests bullish bias.",
            "sell_condition": f"Close ≤ SMA{period} - 0.10% suggests bearish bias.",
            "neutral_condition": f"Close within ±0.10% of SMA{period} => neutral.",
            "notes": f"Simple moving average over {period} periods.",
            "tags": "trend,sma",
        }
    )
    _BASE_STRATEGIES.append(
        {
            "indicator_key": f"ema_{period}",
            "name": f"EMA {period}",
            "signal_type": "moving_average",
            "buy_condition": f"Close ≥ EMA{period} + 0.10% suggests bullish bias.",
            "sell_condition": f"Close ≤ EMA{period} - 0.10% suggests bearish bias.",
            "neutral_condition": f"Close within ±0.10% of EMA{period} => neutral.",
            "notes": f"Exponential moving average over {period} periods.",
            "tags": "trend,ema",
        }
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_strategy_catalog() -> pd.DataFrame:
    records = []
    created = "2025-01-22T00:00:00Z"
    updated = "2025-01-22T00:00:00Z"
    for idx, payload in enumerate(_BASE_STRATEGIES, start=1):
        row = dict(payload)
        row["strategy_id"] = idx
        row.setdefault("timeframes", "all")
        row.setdefault("confidence_level", 0.6)
        row.setdefault("source", "investing.com technical summary heuristics")
        row.setdefault("created_at", created)
        row.setdefault("last_updated", updated)
        records.append(row)

    columns = [
        "strategy_id",
        "indicator_key",
        "name",
        "signal_type",
        "buy_condition",
        "sell_condition",
        "neutral_condition",
        "notes",
        "timeframes",
        "tags",
        "confidence_level",
        "source",
        "created_at",
        "last_updated",
    ]

    return pd.DataFrame.from_records(records, columns=columns)


def save_strategy_catalog(path: str | Path | None = None) -> Path:
    df = build_strategy_catalog()
    if path is None:
        path = _project_root() / "data" / "processed" / "final_strategies.pickle"
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return path


def load_strategy_catalog(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = _project_root() / "data" / "processed" / "final_strategies.pickle"
    else:
        path = Path(path)
    if path.exists():
        return pd.read_pickle(path)
    return build_strategy_catalog()


def as_records() -> Iterable[dict]:
    return build_strategy_catalog().to_dict(orient="records")


__all__ = [
    "build_strategy_catalog",
    "save_strategy_catalog",
    "load_strategy_catalog",
    "as_records",
]
