from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


SignalEvaluator = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True, slots=True)
class IndicatorRule:
    key: str
    name: str
    description: str
    required_columns: Sequence[str]
    evaluator: SignalEvaluator
    timeframes: Sequence[str] | None = None

    def applies_to_timeframe(self, timeframe: str) -> bool:
        return self.timeframes is None or timeframe in self.timeframes


def _init_result(
    df: pd.DataFrame, values: pd.Series, neutral_reason: str
) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    result["indicator_value"] = values
    result["signal"] = pd.Series("neutral", index=df.index, dtype="object")
    result["reason"] = pd.Series(neutral_reason, index=df.index, dtype="object")
    return result


def _evaluate_rsi(df: pd.DataFrame) -> pd.DataFrame:
    values = df["rsi"].astype(float)
    result = _init_result(
        df, values, "RSI between 30 and 70 suggests balanced momentum (Neutral)."
    )
    unknown = values.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "RSI unavailable; cannot derive signal."
    valid = ~unknown

    buy_mask = valid & (values <= 30)
    sell_mask = valid & (values >= 70)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "RSI at or below 30 implies oversold conditions (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "RSI at or above 70 implies overbought conditions (Sell bias)."
    )
    return result


def _evaluate_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    values = df["stoch_d_9_6"].astype(float)
    result = _init_result(
        df,
        values,
        "Stochastic %D between 20 and 80 signals balanced momentum (Neutral).",
    )
    unknown = values.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "Stochastic oscillator unavailable."
    valid = ~unknown

    buy_mask = valid & (values <= 20)
    sell_mask = valid & (values >= 80)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "Stochastic %D at or below 20 indicates oversold momentum (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "Stochastic %D at or above 80 indicates overbought momentum (Sell bias)."
    )
    return result


def _evaluate_stoch_rsi(df: pd.DataFrame) -> pd.DataFrame:
    values = df["stoch_rsi_14"].astype(float)
    result = _init_result(
        df,
        values,
        "Stochastic RSI between 20 and 80 shows neutral RSI momentum.",
    )
    unknown = values.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "Stochastic RSI unavailable."
    valid = ~unknown

    buy_mask = valid & (values <= 20)
    sell_mask = valid & (values >= 80)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "Stochastic RSI ≤ 20 highlights RSI oversold momentum (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "Stochastic RSI ≥ 80 highlights RSI overbought momentum (Sell bias)."
    )
    return result


def _evaluate_macd_hist(df: pd.DataFrame) -> pd.DataFrame:
    values = df["macd_hist"].astype(float)
    result = _init_result(
        df,
        values,
        "MACD histogram near zero indicates momentum equilibrium (Neutral).",
    )
    unknown = values.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "MACD histogram unavailable."
    valid = ~unknown

    buy_mask = valid & (values > 0.05)
    sell_mask = valid & (values < -0.05)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "Positive MACD histogram shows bullish momentum (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "Negative MACD histogram shows bearish momentum (Sell bias)."
    )
    return result


def _evaluate_ema_spread(df: pd.DataFrame) -> pd.DataFrame:
    fast = df["ema_12"].astype(float)
    slow = df["ema_26"].astype(float)
    spread = fast - slow
    result = _init_result(
        df,
        spread,
        "Fast and slow EMAs are converging; limited directional bias (Neutral).",
    )
    unknown = fast.isna() | slow.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "EMA values unavailable."
    valid = ~unknown

    buy_mask = valid & (spread > 0)
    sell_mask = valid & (spread < 0)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "EMA12 above EMA26 signals bullish trend alignment (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "EMA12 below EMA26 signals bearish trend alignment (Sell bias)."
    )
    return result


def _evaluate_williams_r(df: pd.DataFrame) -> pd.DataFrame:
    values = df["willr"].astype(float)
    result = _init_result(
        df,
        values,
        "Williams %R between -80 and -20 implies balanced momentum (Neutral).",
    )
    unknown = values.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "Williams %R unavailable."
    valid = ~unknown

    buy_mask = valid & (values <= -80)
    sell_mask = valid & (values >= -20)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "Williams %R below -80 suggests oversold territory (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "Williams %R above -20 suggests overbought territory (Sell bias)."
    )
    return result


def _evaluate_cci(df: pd.DataFrame) -> pd.DataFrame:
    values = df["cci_14"].astype(float)
    result = _init_result(
        df,
        values,
        "CCI within -100 to +100 indicates range-bound conditions (Neutral).",
    )
    unknown = values.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "CCI unavailable."
    valid = ~unknown

    buy_mask = valid & (values >= 100)
    sell_mask = valid & (values <= -100)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = "CCI above +100 signals strong upside momentum."
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = "CCI below -100 signals strong downside momentum."
    return result


def _evaluate_adx(df: pd.DataFrame) -> pd.DataFrame:
    adx = df["adx_14"].astype(float)
    plus_di = df["plus_di_14"].astype(float)
    minus_di = df["minus_di_14"].astype(float)

    result = _init_result(
        df,
        adx,
        "ADX below 20 or balanced DI values imply weak trend conviction (Neutral).",
    )
    unknown = adx.isna() | plus_di.isna() | minus_di.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "ADX/DI inputs unavailable."
    valid = ~unknown

    strong_trend = valid & (adx >= 20)
    diff = plus_di - minus_di
    buy_mask = strong_trend & (diff > 0.5)
    sell_mask = strong_trend & (diff < -0.5)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "ADX strong and +DI dominates -DI → bullish trend persistence (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "ADX strong and -DI dominates +DI → bearish trend persistence (Sell bias)."
    )
    return result


def _evaluate_atr(df: pd.DataFrame) -> pd.DataFrame:
    atr = df["atr_14"].astype(float)
    close = df["close"].astype(float)
    result = _init_result(
        df,
        atr,
        "ATR within usual bounds; volatility appears typical (Neutral).",
    )
    unknown = atr.isna() | close.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "ATR or price unavailable."
    valid = ~unknown

    ratio = pd.Series(np.nan, index=df.index)
    ratio.loc[valid] = (
        atr.loc[valid] / close.loc[valid].replace(0, np.nan)
    ) * 100

    high_vol = valid & (ratio >= 3)
    low_vol = valid & (ratio <= 1)

    result.loc[high_vol, "reason"] = ratio.loc[high_vol].map(
        lambda pct: f"ATR at {pct:.2f}% of price indicates high volatility (Informational)."
    )
    result.loc[low_vol, "reason"] = ratio.loc[low_vol].map(
        lambda pct: f"ATR at {pct:.2f}% of price indicates low volatility (Informational)."
    )
    return result


def _evaluate_highs_lows(df: pd.DataFrame) -> pd.DataFrame:
    values = df["highs_lows_14"].astype(float)
    close = df["close"].astype(float)
    result = _init_result(
        df,
        values,
        "Close near the 14-period range midpoint (Neutral).",
    )
    unknown = values.isna() | close.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "High/low range unavailable."
    valid = ~unknown

    tolerance = close.abs() * 0.001
    buy_mask = valid & (values >= tolerance)
    sell_mask = valid & (values <= -tolerance)
    neutral_mask = valid & ~(buy_mask | sell_mask)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "Close skewed toward 14-period highs (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "Close skewed toward 14-period lows (Sell bias)."
    )
    result.loc[neutral_mask, "reason"] = (
        "Close within ±0.10% of the 14-period range midpoint (Neutral)."
    )
    return result


def _evaluate_ultimate_osc(df: pd.DataFrame) -> pd.DataFrame:
    values = df["ultimate_osc"].astype(float)
    result = _init_result(
        df,
        values,
        "Ultimate Oscillator in the mid-band suggests balanced pressure (Neutral).",
    )
    unknown = values.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "Ultimate Oscillator unavailable."
    valid = ~unknown

    buy_mask = valid & (values >= 55)
    sell_mask = valid & (values <= 45)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = (
        "Ultimate Oscillator above 55 indicates bullish pressure (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = (
        "Ultimate Oscillator below 45 indicates bearish pressure (Sell bias)."
    )
    return result


def _evaluate_roc(df: pd.DataFrame) -> pd.DataFrame:
    values = df["roc_12"].astype(float)
    result = _init_result(
        df,
        values,
        "ROC near zero signals flat momentum (Neutral).",
    )
    unknown = values.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "Rate of Change unavailable."
    valid = ~unknown

    buy_mask = valid & (values > 0)
    sell_mask = valid & (values < 0)

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = "Positive ROC shows upside momentum (Buy bias)."
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = "Negative ROC shows downside momentum (Sell bias)."
    return result


def _evaluate_bull_bear_power(df: pd.DataFrame) -> pd.DataFrame:
    values = df["bull_bear_power_13"].astype(float)
    close = df["close"].astype(float)
    result = _init_result(
        df,
        values,
        "Bull/Bear Power near zero implies balance between buyers and sellers (Neutral).",
    )
    unknown = values.isna() | close.isna()
    result.loc[unknown, "signal"] = "unknown"
    result.loc[unknown, "reason"] = "Bull/Bear Power unavailable."
    valid = ~unknown

    tolerance = close.abs() * 0.001
    buy_mask = valid & (values >= tolerance)
    sell_mask = valid & (values <= -tolerance)
    neutral_mask = valid & ~(buy_mask | sell_mask)

    pct_delta = pd.Series(np.nan, index=df.index)
    pct_delta.loc[valid] = (
        values.loc[valid] / close.loc[valid].replace(0, np.nan)
    ) * 100

    result.loc[buy_mask, "signal"] = "buy"
    result.loc[buy_mask, "reason"] = pct_delta.loc[buy_mask].map(
        lambda pct: f"Bulls dominate by {pct:.2f}% of price (Buy bias)."
    )
    result.loc[sell_mask, "signal"] = "sell"
    result.loc[sell_mask, "reason"] = pct_delta.loc[sell_mask].map(
        lambda pct: f"Bears dominate by {abs(pct):.2f}% of price (Sell bias)."
    )
    result.loc[neutral_mask, "reason"] = (
        "Bull/Bear power within ±0.10% of price (Neutral)."
    )
    return result


def _make_ma_evaluator(column: str, label: str) -> SignalEvaluator:
    def _evaluate(df: pd.DataFrame) -> pd.DataFrame:
        ma_values = df[column].astype(float)
        close = df["close"].astype(float)
        result = _init_result(
            df,
            ma_values,
            f"Close near {label}; awaiting decisive move (Neutral).",
        )
        unknown = ma_values.isna() | close.isna()
        result.loc[unknown, "signal"] = "unknown"
        result.loc[unknown, "reason"] = f"{label} or price unavailable."
        valid = ~unknown

        delta = close - ma_values
        tolerance = close.abs() * 0.001
        buy_mask = valid & (delta >= tolerance)
        sell_mask = valid & (delta <= -tolerance)
        neutral_mask = valid & ~(buy_mask | sell_mask)

        pct_diff = pd.Series(np.nan, index=df.index)
        pct_diff.loc[valid] = (
            delta.loc[valid] / close.loc[valid].replace(0, np.nan)
        ) * 100

        result.loc[buy_mask, "signal"] = "buy"
        result.loc[buy_mask, "reason"] = pct_diff.loc[buy_mask].map(
            lambda pct: f"Close above {label} by {pct:.2f}% (Buy bias)."
        )
        result.loc[sell_mask, "signal"] = "sell"
        result.loc[sell_mask, "reason"] = pct_diff.loc[sell_mask].map(
            lambda pct: f"Close below {label} by {abs(pct):.2f}% (Sell bias)."
        )
        result.loc[neutral_mask, "reason"] = pct_diff.loc[neutral_mask].map(
            lambda pct: f"Close within {abs(pct):.2f}% of {label} (Neutral)."
        )
        return result

    return _evaluate


def _build_rules() -> tuple[IndicatorRule, ...]:
    rules: list[IndicatorRule] = [
        IndicatorRule(
            key="rsi_14",
            name="RSI (14)",
            description="Standard RSI oversold (<30) / overbought (>70) thresholds.",
            required_columns=("rsi",),
            evaluator=_evaluate_rsi,
        ),
        IndicatorRule(
            key="stoch_d_9_6",
            name="Stochastic %D (9,6)",
            description="Stochastic oscillator slow line thresholds 20/80.",
            required_columns=("stoch_d_9_6",),
            evaluator=_evaluate_stochastic,
        ),
        IndicatorRule(
            key="stoch_rsi_14",
            name="Stochastic RSI (14)",
            description="Stochastic RSI highlighting RSI-based extremes.",
            required_columns=("stoch_rsi_14",),
            evaluator=_evaluate_stoch_rsi,
        ),
        IndicatorRule(
            key="macd_hist",
            name="MACD Histogram (12,26,9)",
            description="MACD histogram sign captures directional momentum shifts.",
            required_columns=("macd_hist",),
            evaluator=_evaluate_macd_hist,
        ),
        IndicatorRule(
            key="ema12_vs_ema26",
            name="EMA 12/26 Crossover",
            description="Positive spread (EMA12 > EMA26) signals bullish trend, negative spread bearish.",
            required_columns=("ema_12", "ema_26"),
            evaluator=_evaluate_ema_spread,
        ),
        IndicatorRule(
            key="adx_14",
            name="ADX (14)",
            description="ADX with DI comparison identifies strong bullish/bearish trends.",
            required_columns=("adx_14", "plus_di_14", "minus_di_14"),
            evaluator=_evaluate_adx,
        ),
        IndicatorRule(
            key="atr_14",
            name="ATR (14)",
            description="Average True Range relative to price flags volatility regime.",
            required_columns=("atr_14", "close"),
            evaluator=_evaluate_atr,
        ),
        IndicatorRule(
            key="willr_14",
            name="Williams %R (14)",
            description="Overbought above -20 (Sell), oversold below -80 (Buy).",
            required_columns=("willr",),
            evaluator=_evaluate_williams_r,
        ),
        IndicatorRule(
            key="cci_14",
            name="CCI (14)",
            description="Commodity Channel Index ±100 extremes signal momentum exhaustion.",
            required_columns=("cci_14",),
            evaluator=_evaluate_cci,
        ),
        IndicatorRule(
            key="highs_lows_14",
            name="Highs/Lows (14)",
            description="Position of close within 14-period range for breakout/mean-reversion bias.",
            required_columns=("highs_lows_14", "close"),
            evaluator=_evaluate_highs_lows,
        ),
        IndicatorRule(
            key="ultimate_osc",
            name="Ultimate Oscillator (7,14,28)",
            description="Weighted multi-timeframe oscillator; 45/55 thresholds.",
            required_columns=("ultimate_osc",),
            evaluator=_evaluate_ultimate_osc,
        ),
        IndicatorRule(
            key="roc_12",
            name="Rate of Change (12)",
            description="12-period ROC > 0 bullish momentum, < 0 bearish momentum.",
            required_columns=("roc_12",),
            evaluator=_evaluate_roc,
        ),
        IndicatorRule(
            key="bull_bear_power_13",
            name="Bull/Bear Power (13)",
            description="High/low distance from EMA13 gauges buyer vs seller control.",
            required_columns=("bull_bear_power_13", "close"),
            evaluator=_evaluate_bull_bear_power,
        ),
    ]

    ma_periods = [5, 10, 20, 50, 100, 200]
    for period in ma_periods:
        sma_column = f"sma_{period}"
        ema_column = f"ema_{period}"
        rules.append(
            IndicatorRule(
                key=sma_column,
                name=f"SMA ({period})",
                description=f"Price relative to {period}-period simple moving average.",
                required_columns=("close", sma_column),
                evaluator=_make_ma_evaluator(sma_column, f"SMA {period}"),
            )
        )
        rules.append(
            IndicatorRule(
                key=ema_column,
                name=f"EMA ({period})",
                description=f"Price relative to {period}-period exponential moving average.",
                required_columns=("close", ema_column),
                evaluator=_make_ma_evaluator(ema_column, f"EMA {period}"),
            )
        )

    return tuple(rules)


_INDICATOR_RULES: tuple[IndicatorRule, ...] = _build_rules()

_SIGNAL_COLUMNS = (
    "asset_id",
    "symbol",
    "timeframe",
    "indicator_key",
    "indicator_name",
    "indicator_value",
    "signal",
    "reason",
    "evaluated_at",
)


def indicator_rulebook() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule in _INDICATOR_RULES:
        rows.append(
            {
                "indicator_key": rule.key,
                "indicator_name": rule.name,
                "description": rule.description,
                "required_columns": ",".join(rule.required_columns),
                "timeframes": ",".join(rule.timeframes) if rule.timeframes else "all",
            }
        )
    return pd.DataFrame(rows)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _latest_candles(
    candle_df: pd.DataFrame, *, timeframes: Sequence[str] | None
) -> pd.DataFrame:
    if timeframes is not None:
        candle_df = candle_df[candle_df["timeframe"].isin(timeframes)]

    if candle_df.empty:
        return candle_df

    if "ts_int" in candle_df.columns:
        idx = (
            candle_df.groupby(["asset_id", "coin", "timeframe"])["ts_int"]
            .idxmax()
            .to_list()
        )
    else:
        idx = (
            candle_df.groupby(["asset_id", "coin", "timeframe"])["ts"]
            .idxmax()
            .to_list()
        )

    latest = candle_df.loc[idx].copy()
    latest.sort_values(["asset_id", "coin", "timeframe"], inplace=True)
    latest.reset_index(drop=True, inplace=True)
    return latest


def generate_indicator_signals(
    candle_df: pd.DataFrame | None = None,
    *,
    timeframes: Sequence[str] | None = None,
) -> pd.DataFrame:
    if candle_df is None:
        candle_path = _project_root() / "data" / "processed" / "final_candle.pickle"
        candle_df = pd.read_pickle(candle_path)

    latest = _latest_candles(candle_df, timeframes=timeframes)
    if latest.empty:
        return pd.DataFrame(columns=_SIGNAL_COLUMNS)

    evaluation_time = pd.Timestamp.utcnow().isoformat()
    frames: list[pd.DataFrame] = []

    for rule in _INDICATOR_RULES:
        frame = latest
        if rule.timeframes is not None:
            frame = frame[frame["timeframe"].isin(rule.timeframes)]
        if frame.empty:
            continue

        missing = [col for col in rule.required_columns if col not in frame.columns]
        if missing:
            continue

        result = rule.evaluator(frame)
        result = result.reindex(frame.index)

        combined = pd.DataFrame(
            {
                "asset_id": frame["asset_id"].values,
                "symbol": frame["coin"].values,
                "timeframe": frame["timeframe"].values,
                "indicator_key": rule.key,
                "indicator_name": rule.name,
                "indicator_value": result["indicator_value"].values,
                "signal": result["signal"].values,
                "reason": result["reason"].values,
                "evaluated_at": evaluation_time,
            }
        )
        frames.append(combined)

    if not frames:
        return pd.DataFrame(columns=_SIGNAL_COLUMNS)

    output = pd.concat(frames, ignore_index=True)
    output = output.reindex(columns=_SIGNAL_COLUMNS)
    return output


def summarize_indicator_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    if signals_df.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "symbol",
                "timeframe",
                "evaluated_at",
                "buy_count",
                "sell_count",
                "neutral_count",
                "unknown_count",
                "total_indicators",
                "overall_signal",
                "dominant_ratio",
            ]
        )

    grouped = (
        signals_df.groupby(["asset_id", "symbol", "timeframe", "evaluated_at"])["signal"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    for column in ("buy", "sell", "neutral", "unknown"):
        if column not in grouped.columns:
            grouped[column] = 0

    grouped.rename(
        columns={
            "buy": "buy_count",
            "sell": "sell_count",
            "neutral": "neutral_count",
            "unknown": "unknown_count",
        },
        inplace=True,
    )

    grouped["total_indicators"] = (
        grouped["buy_count"]
        + grouped["sell_count"]
        + grouped["neutral_count"]
        + grouped["unknown_count"]
    )

    def _resolve_signal(row: pd.Series) -> str:
        ranked = {
            "buy": int(row["buy_count"]),
            "sell": int(row["sell_count"]),
            "neutral": int(row["neutral_count"]),
        }
        max_signal = max(ranked, key=ranked.get)
        max_value = ranked[max_signal]
        winners = [signal for signal, value in ranked.items() if value == max_value]
        if max_value == 0:
            return "unknown"
        if len(winners) > 1:
            return "mixed"
        return max_signal

    grouped["overall_signal"] = grouped.apply(_resolve_signal, axis=1)
    grouped["dominant_ratio"] = grouped.apply(
        lambda row: (
            0
            if row["total_indicators"] == 0
            else max(
                row["buy_count"],
                row["sell_count"],
                row["neutral_count"],
            )
            / row["total_indicators"]
        ),
        axis=1,
    )

    columns = [
        "asset_id",
        "symbol",
        "timeframe",
        "evaluated_at",
        "buy_count",
        "sell_count",
        "neutral_count",
        "unknown_count",
        "total_indicators",
        "overall_signal",
        "dominant_ratio",
    ]
    return grouped.reindex(columns=columns)


def save_indicator_artifacts(
    *,
    signals_path: Path | None = None,
    summary_path: Path | None = None,
    timeframes: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    project_root = _project_root()
    if signals_path is None:
        signals_path = project_root / "data" / "processed" / "final_indicator_signals.pickle"
    else:
        signals_path = Path(signals_path)

    if summary_path is None:
        summary_path = project_root / "data" / "processed" / "final_indicator_summary.pickle"
    else:
        summary_path = Path(summary_path)

    signals = generate_indicator_signals(timeframes=timeframes)
    summary = summarize_indicator_signals(signals)

    signals_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    signals.to_pickle(signals_path)
    summary.to_pickle(summary_path)

    return signals_path, summary_path


def load_indicator_signals(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = _project_root() / "data" / "processed" / "final_indicator_signals.pickle"
    else:
        path = Path(path)
    if not path.exists():
        save_indicator_artifacts(signals_path=path)
    return pd.read_pickle(path)


def load_indicator_summary(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = _project_root() / "data" / "processed" / "final_indicator_summary.pickle"
    else:
        path = Path(path)
    if not path.exists():
        save_indicator_artifacts(summary_path=path)
    return pd.read_pickle(path)


__all__ = [
    "IndicatorRule",
    "indicator_rulebook",
    "generate_indicator_signals",
    "summarize_indicator_signals",
    "save_indicator_artifacts",
    "load_indicator_signals",
    "load_indicator_summary",
]
