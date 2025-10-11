import pandas as pd
import numpy as np
from scipy.signal import find_peaks



def calculate_rsi(df, period=14):
    """
    Calculate RSI without using TA-Lib
    """
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def calculate_technical_indicators(df):
    """
    Calculate technical indicators without using TA-Lib
    """
    # Calculate RSI
    df = calculate_rsi(df, period=14)
    
    # Calculate EMA
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std * 2)
    df['bb_lower'] = df['bb_middle'] - (std * 2)
    
    # Calculate Williams %R
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['willr'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    
    # Calculate ATR (simplified)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Simplified ADX and CCI (these are more complex)
    # For now, we'll use placeholder calculations
    df['adx'] = df['close'].rolling(window=14).std() * 10  # Simplified
    
    # Simplified CCI
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    mean_tp = typical_price.rolling(window=20).mean()
    mean_deviation = (typical_price - mean_tp).abs().rolling(window=20).mean()
    df['cci'] = (typical_price - mean_tp) / (0.015 * mean_deviation)
    
    return df


def generate_features(df):
    df = df.copy()

    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)

    df['volatility_5'] = df['return_1'].rolling(window=5).std()
    df['volatility_10'] = df['return_1'].rolling(window=10).std()

    df['volume_change'] = df['volume'].pct_change()
    df['volume_rolling_mean'] = df['volume'].rolling(window=5).mean()

    return df


def change_index_to_datetime(df):
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    if df.index.dtype == 'int64':
        df.set_index('datetime', inplace=True)
    
    return df


def _future_agg(s: pd.Series, w: int, how: str):
    s1 = s.shift(-1)
    roll = s1.iloc[::-1].rolling(w, min_periods=w)
    out = (roll.max() if how == "max" else roll.min()).iloc[::-1]
    return out


def _strict_local_max(s: pd.Series, w: int) -> pd.Series:
    s = s.astype(float)
    past_max   = s.shift(1).rolling(w, min_periods=w).max()
    future_max = _future_agg(s, w, "max")
    return ((s > past_max) & (s > future_max)).fillna(False)


def _strict_local_min(s: pd.Series, w: int) -> pd.Series:
    s = s.astype(float)
    past_min   = s.shift(1).rolling(w, min_periods=w).min()
    future_min = _future_agg(s, w, "min")
    return ((s < past_min) & (s < future_min)).fillna(False)


def _atr14(df, high="high", low="low", close="close", n=14):
    h, l, c = df[high].astype(float), df[low].astype(float), df[close].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    return atr.replace(0, np.nan).bfill()


def preprocess_peaks_strict(
    df: pd.DataFrame, 
    w: int = 8,
    high_col="high", 
    low_col="low", 
    close_col="close",
    min_prom_atr: float | None = None
) -> pd.DataFrame:
    out = df.copy()
    for c in (high_col, low_col, close_col):
        if c not in out.columns:
            raise KeyError(f"missing column: {c}")

    out["peak_high_high"]  = _strict_local_max(out[high_col],  w)
    out["peak_high_close"] = _strict_local_max(out[close_col], w)
    out["peak_low_low"]    = _strict_local_min(out[low_col],   w)
    out["peak_low_close"]  = _strict_local_min(out[close_col], w)

    if min_prom_atr is not None:
        atr = _atr14(out, high_col, low_col, close_col)
        eps = 1e-12

        def keep_prom_max(s, flags):
            idx = np.flatnonzero(flags.to_numpy())
            if len(idx) == 0: return flags
            arr, a = s.to_numpy(float), atr.to_numpy(float)
            keep = np.zeros(len(s), bool)
            n = len(s)
            for i in idx:
                L, R = max(0, i-w), min(n, i+w+1)
                base = max(arr[L:i].min(initial=arr[i]),
                           arr[i+1:R].min(initial=arr[i]))
                prom = arr[i] - base
                if prom >= (min_prom_atr * max(a[i], eps)):
                    keep[i] = True
            return pd.Series(keep, index=s.index)

        def keep_prom_min(s, flags):
            idx = np.flatnonzero(flags.to_numpy())
            if len(idx) == 0: return flags
            arr, a = s.to_numpy(float), atr.to_numpy(float)
            keep = np.zeros(len(s), bool)
            n = len(s)
            for i in idx:
                L, R = max(0, i-w), min(n, i+w+1)
                base = min(arr[L:i].max(initial=arr[i]),
                           arr[i+1:R].max(initial=arr[i]))
                prom = base - arr[i]
                if prom >= (min_prom_atr * max(a[i], eps)):
                    keep[i] = True
            return pd.Series(keep, index=s.index)

        out["peak_high_high"]  = keep_prom_max(out[high_col],  out["peak_high_high"])
        out["peak_high_close"] = keep_prom_max(out[close_col], out["peak_high_close"])
        out["peak_low_low"]    = keep_prom_min(out[low_col],   out["peak_low_low"])
        out["peak_low_close"]  = keep_prom_min(out[close_col], out["peak_low_close"])

    return out