from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "add_rsi",
    "add_technical_indicators",
    "add_return_features",
    "ensure_datetime_index",
    "preprocess_peaks_strict",
]


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Return Wilder-style exponential moving average."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Append RSI column using price changes."""
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean().replace(0, np.finfo(float).eps)

    rs = avg_gain / avg_loss
    out["rsi"] = 100 - (100 / (1 + rs))
    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute core technical indicator set used across the project."""
    out = add_rsi(df.copy(), period=14)

    # EMA and MACD family
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Bollinger Bands
    out["bb_middle"] = out["close"].rolling(window=20).mean()
    std = out["close"].rolling(window=20).std()
    out["bb_upper"] = out["bb_middle"] + (std * 2)
    out["bb_lower"] = out["bb_middle"] - (std * 2)

    # Williams %R
    highest_high = out["high"].rolling(window=14).max()
    lowest_low = out["low"].rolling(window=14).min()
    out["willr"] = -100 * ((highest_high - out["close"]) / (highest_high - lowest_low))

    # ATR + ADX
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = _wilder_smooth(true_range, 14)
    out["atr"] = atr
    out["atr_14"] = atr

    up_move = out["high"].diff()
    down_move = out["low"].shift(1) - out["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (_wilder_smooth(pd.Series(plus_dm, index=out.index), 14) / atr.replace(0, np.nan))
    minus_di = 100 * (_wilder_smooth(pd.Series(minus_dm, index=out.index), 14) / atr.replace(0, np.nan))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = _wilder_smooth(dx, 14)
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di
    out["adx"] = adx
    out["adx_14"] = adx

    # Commodity Channel Index (20 + 14 variant)
    typical_price = (out["high"] + out["low"] + out["close"]) / 3
    for period, column in ((20, "cci"), (14, "cci_14")):
        mean_tp = typical_price.rolling(window=period).mean()
        mean_dev = (typical_price - mean_tp).abs().rolling(window=period).mean().replace(0, np.nan)
        out[column] = (typical_price - mean_tp) / (0.015 * mean_dev)

    # Stochastic Oscillator (9, 6)
    lowest_low_k = out["low"].rolling(window=9, min_periods=9).min()
    highest_high_k = out["high"].rolling(window=9, min_periods=9).max()
    denom = (highest_high_k - lowest_low_k).replace(0, np.nan)
    stoch_k = ((out["close"] - lowest_low_k) / denom) * 100
    out["stoch_k_9"] = stoch_k
    out["stoch_d_9_6"] = stoch_k.rolling(window=6, min_periods=6).mean()

    # Stochastic RSI (14)
    rsi_min = out["rsi"].rolling(window=14, min_periods=14).min()
    rsi_max = out["rsi"].rolling(window=14, min_periods=14).max()
    stoch_rsi_den = (rsi_max - rsi_min).replace(0, np.nan)
    out["stoch_rsi_14"] = ((out["rsi"] - rsi_min) / stoch_rsi_den) * 100

    # Ultimate Oscillator (7, 14, 28)
    prev_close = out["close"].shift(1)
    min_low_close = pd.concat([out["low"], prev_close], axis=1).min(axis=1)
    max_high_close = pd.concat([out["high"], prev_close], axis=1).max(axis=1)
    buying_pressure = out["close"] - min_low_close
    true_range_uo = max_high_close - min_low_close

    def _uo_avg(series: pd.Series, window: int) -> pd.Series:
        series_sum = series.rolling(window=window, min_periods=window).sum()
        tr_sum = true_range_uo.rolling(window=window, min_periods=window).sum().replace(0, np.nan)
        return series_sum / tr_sum

    ultimate = 100 * (4 * _uo_avg(buying_pressure, 7) + 2 * _uo_avg(buying_pressure, 14) + _uo_avg(buying_pressure, 28)) / 7
    out["ultimate_osc"] = ultimate

    # Rate of Change (12)
    out["roc_12"] = out["close"].pct_change(periods=12) * 100

    # Bull / Bear Power (13)
    ema_13 = out["close"].ewm(span=13, adjust=False).mean()
    out["ema_13"] = ema_13
    out["bull_power_13"] = out["high"] - ema_13
    out["bear_power_13"] = out["low"] - ema_13
    out["bull_bear_power_13"] = out["bull_power_13"] + out["bear_power_13"]

    # Highs/Lows bias (14)
    highest_high_14 = out["high"].rolling(window=14, min_periods=14).max()
    lowest_low_14 = out["low"].rolling(window=14, min_periods=14).min()
    out["highs_lows_14"] = out["close"] - (highest_high_14 + lowest_low_14) / 2

    # Moving averages
    for period in (5, 10, 20, 50, 100, 200):
        out[f"sma_{period}"] = out["close"].rolling(window=period, min_periods=period).mean()
        out[f"ema_{period}"] = out["close"].ewm(span=period, adjust=False).mean()

    return out


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append return, volatility, and volume features."""
    out = df.copy()
    out["return_1"] = out["close"].pct_change()
    out["return_5"] = out["close"].pct_change(5)
    out["return_10"] = out["close"].pct_change(10)
    out["volatility_5"] = out["return_1"].rolling(window=5).std()
    out["volatility_10"] = out["return_1"].rolling(window=10).std()
    out["volume_change"] = out["volume"].pct_change()
    out["volume_rolling_mean"] = out["volume"].rolling(window=5).mean()
    return out


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `datetime` column into index if needed."""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], format="mixed")
    if out.index.dtype == "int64":
        out.set_index("datetime", inplace=True)
    return out


def preprocess_peaks_strict(
    df: pd.DataFrame,
    window: int = 8,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    min_prom_atr: float | None = None,
) -> pd.DataFrame:
    """Identify strict local peaks/lows optionally filtered by ATR prominence."""
    out = df.copy()
    for col in (high_col, low_col, close_col):
        if col not in out.columns:
            raise KeyError(f"Missing column: {col}")

    def _future_agg(series: pd.Series, how: str) -> pd.Series:
        shifted = series.shift(-1)
        roll = shifted.iloc[::-1].rolling(window, min_periods=window)
        agg = roll.max() if how == "max" else roll.min()
        return agg.iloc[::-1]

    def _strict_local_max(series: pd.Series) -> pd.Series:
        past = series.shift(1).rolling(window, min_periods=window).max()
        future = _future_agg(series, "max")
        return ((series > past) & (series > future)).fillna(False)

    def _strict_local_min(series: pd.Series) -> pd.Series:
        past = series.shift(1).rolling(window, min_periods=window).min()
        future = _future_agg(series, "min")
        return ((series < past) & (series < future)).fillna(False)

    out["peak_high_high"] = _strict_local_max(out[high_col].astype(float))
    out["peak_high_close"] = _strict_local_max(out[close_col].astype(float))
    out["peak_low_low"] = _strict_local_min(out[low_col].astype(float))
    out["peak_low_close"] = _strict_local_min(out[close_col].astype(float))

    if min_prom_atr is None:
        return out

    atr = _wilder_smooth(
        pd.concat(
            [
                (out[high_col] - out[low_col]).abs(),
                (out[high_col] - out[close_col].shift()).abs(),
                (out[low_col] - out[close_col].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1),
        14,
    )

    eps = 1e-12

    def _filter(series: pd.Series, flags: pd.Series, is_max: bool) -> pd.Series:
        idx = np.flatnonzero(flags.to_numpy())
        if len(idx) == 0:
            return flags
        arr = series.to_numpy(dtype=float)
        atr_arr = atr.to_numpy(dtype=float)
        keep = np.zeros(len(series), dtype=bool)
        n = len(series)
        for i in idx:
            left = max(0, i - window)
            right = min(n, i + window + 1)
            if is_max:
                base = max(arr[left:i].min(initial=arr[i]), arr[i + 1 : right].min(initial=arr[i]))
                prominence = arr[i] - base
            else:
                base = min(arr[left:i].max(initial=arr[i]), arr[i + 1 : right].max(initial=arr[i]))
                prominence = base - arr[i]
            if prominence >= (min_prom_atr * max(atr_arr[i], eps)):
                keep[i] = True
        return pd.Series(keep, index=series.index)

    out["peak_high_high"] = _filter(out[high_col], out["peak_high_high"], True)
    out["peak_high_close"] = _filter(out[close_col], out["peak_high_close"], True)
    out["peak_low_low"] = _filter(out[low_col], out["peak_low_low"], False)
    out["peak_low_close"] = _filter(out[close_col], out["peak_low_close"], False)

    return out
