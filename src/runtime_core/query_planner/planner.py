"""Translate intents into executable query plans aligned with DuckDB schema."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from src.runtime_core.models.runtime_models import Intent, IntentSlots, QueryPlan, QuerySpec


class QueryOrchestrator:
    """Build query specs for candles / indicators / news / strategy tables."""

    def __init__(
        self,
        candles_table: str = "candles",
        indicator_summary_table: str = "indicator_signal_summary",
        indicator_signals_table: str = "indicator_signals",
        divergence_table: str = "divergence",
        news_table: str = "news_articles",
        indicator_rules_table: str = "indicator_rules",
        strategies_table: str = "strategies",
    ) -> None:
        self._candles_table = candles_table
        self._indicator_summary_table = indicator_summary_table
        self._indicator_signals_table = indicator_signals_table
        self._divergence_table = divergence_table
        self._news_table = news_table
        self._indicator_rules_table = indicator_rules_table
        self._strategies_table = strategies_table

        self._candles_columns = [
            "asset_id",
            "coin",
            "timeframe",
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
        ]
        self._indicator_summary_columns = [
            "asset_id",
            "symbol",
            "timeframe",
            "evaluated_at",
            "buy_count",
            "sell_count",
            "neutral_count",
            "overall_signal",
            "dominant_ratio",
        ]
        self._indicator_signal_columns = [
            "asset_id",
            "symbol",
            "timeframe",
            "indicator_key",
            "indicator_name",
            "indicator_value",
            "signal",
            "reason",
            "evaluated_at",
        ]
        self._divergence_columns = [
            "asset_id",
            "timeframe",
            "start_datetime",
            "end_datetime",
            "entry_datetime",
            "entry_price",
            "divergence",
            "price_change",
            "rsi_change",
            "strength_score",
        ]
        self._news_columns = [
            "article_id",
            "published_at",
            "source",
            "title",
            "sentiment",
            "categories",
            "tags",
        ]
        self._indicator_rule_columns = [
            "indicator_key",
            "indicator_name",
            "description",
            "required_columns",
            "timeframes",
        ]
        self._strategy_columns = [
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
            "last_updated",
        ]
        self._limits = {
            "candles": 400,
            "indicator_summary": 200,
            "indicator_signals": 200,
            "divergence": 200,
            "news": 120,
            "indicator_rules": 200,
            "strategies": 200,
        }

        self._intent_routes: Mapping[
            str,
            Callable[[Intent, IntentSlots, Mapping[str, Any]], tuple[list[QuerySpec], list[str]]],
        ] = {
            "market_price": self._plan_market,
            "trend_analysis": self._plan_market,
            "volatility_check": self._plan_market,
            "signal_scan": self._plan_market,
            "generic_query": self._plan_market,
            "strategy_info": self._plan_strategy,
            "news_context": self._plan_news,
            "multi_context": self._plan_multi,
        }

    # ------------------------------------------------------------------
    def build_plan(self, intent: Intent) -> QueryPlan:
        slots = intent.slots or IntentSlots()
        base_context = self._build_base_context(slots)
        planner = self._intent_routes.get(intent.name, self._plan_market)
        specs, steps = planner(intent, slots, base_context)
        if not specs:
            specs = [self._build_candles_spec(base_context, metrics=("price",))]
            steps = ["Review candles data for baseline context."]
        return QueryPlan(intent=intent, specs=tuple(specs), plan_steps=tuple(steps))

    # ------------------------------------------------------------------
    # Planner implementations
    # ------------------------------------------------------------------
    def _plan_market(
        self,
        intent: Intent,
        slots: IntentSlots,
        base_context: Mapping[str, Any],
    ) -> tuple[list[QuerySpec], list[str]]:
        
        metrics = slots.metrics or ("price",)
        specs: list[QuerySpec] = []
        steps: list[str] = []

        if any(metric in {"price", "volume", "volatility"} for metric in metrics):
            specs.append(self._build_candles_spec(base_context, metrics))
            steps.append("Summarize price/volume action from candles data.")

        if "signal" in metrics or intent.name == "signal_scan":
            specs.append(self._build_indicator_summary_spec(base_context))
            steps.append("Inspect indicator_signal_summary to understand technical bias.")

        if "signal_detail" in metrics:
            specs.append(self._build_indicator_signals_spec(base_context))
            steps.append("Review raw indicator_signals for rule-level justifications.")

        if "divergence" in metrics:
            specs.append(self._build_divergence_spec(base_context))
            steps.append("Scan divergence table for momentum shift setups.")

        if "news_sentiment" in metrics:
            specs.append(self._build_news_spec(base_context))
            steps.append("Blend in news sentiment/context for the requested assets.")

        return specs, steps

    def _plan_strategy(
        self,
        intent: Intent,
        slots: IntentSlots,
        base_context: Mapping[str, Any],
    ) -> tuple[list[QuerySpec], list[str]]:
        rule_filters: dict[str, Any] = {}
        timeframe = base_context.get("timeframe")
        if timeframe:
            rule_filters["timeframes"] = timeframe
        targets = list(slots.strategy_topics) if slots.strategy_topics else []
        if base_context.get("symbols"):
            targets.extend(base_context["symbols"])
        if targets:
            rule_filters["indicator_name"] = sorted(set(targets))

        strategies_filters: dict[str, Any] = {}
        if timeframe:
            strategies_filters["timeframes"] = timeframe
        if targets:
            strategies_filters["tags"] = sorted(set(targets))

        rule_spec = QuerySpec(
            table=self._indicator_rules_table,
            columns=tuple(self._indicator_rule_columns),
            filters=rule_filters,
            limit=self._limits["indicator_rules"],
        )
        strategies_spec = QuerySpec(
            table=self._strategies_table,
            columns=tuple(self._strategy_columns),
            filters=strategies_filters,
            limit=self._limits["strategies"],
        )
        steps = [
            "Enumerate indicator_rules and custom strategies matching requested topics/timeframes.",
            "Compare rule logic against strategies table to suggest actionable playbooks.",
        ]
        return [rule_spec, strategies_spec], steps

    def _plan_news(
        self,
        intent: Intent,
        slots: IntentSlots,
        base_context: Mapping[str, Any],
    ) -> tuple[list[QuerySpec], list[str]]:
        return [self._build_news_spec(base_context)], [
            "Aggregate news articles for the requested window.",
            "Summarize sentiment distribution and key headlines.",
        ]

    def _plan_multi(
        self,
        intent: Intent,
        slots: IntentSlots,
        base_context: Mapping[str, Any],
    ) -> tuple[list[QuerySpec], list[str]]:
        specs = [
            self._build_candles_spec(base_context, metrics=("price", "volume")),
            self._build_indicator_summary_spec(base_context),
            self._build_indicator_signals_spec(base_context),
            self._build_divergence_spec(base_context),
            self._build_news_spec(base_context),
        ]
        steps = [
            "Step 1: Pull candles to understand trend/volatility.",
            "Step 2: Inspect indicator_signal_summary for signals.",
            "Step 3: Drill into indicator_signals for rule-level context.",
            "Step 4: Check divergence table for possible momentum shifts.",
            "Step 5: Review news to contextualize the technical read.",
        ]
        return specs, steps

    # ------------------------------------------------------------------
    # Spec builders
    # ------------------------------------------------------------------
    def _build_candles_spec(self, base_context: Mapping[str, Any], metrics: Sequence[str]) -> QuerySpec:
        filters: dict[str, Any] = {}
        symbols = base_context.get("symbols")
        if symbols:
            filters["coin"] = symbols
        start, end = base_context.get("time_window", (None, None))
        if not start and not end:
            start, end = self._default_time_window(days=30)
        if start:
            filters["ts_start"] = start
        if end:
            filters["ts_end"] = end
        timeframe = base_context.get("timeframe") or "1d"
        filters["timeframe"] = timeframe

        columns = list(self._candles_columns)
        if "volatility" in metrics and "atr" not in columns:
            columns.append("atr")
        return QuerySpec(
            table=self._candles_table,
            columns=tuple(columns),
            filters=filters,
            limit=self._limits["candles"],
        )

    def _build_indicator_summary_spec(self, base_context: Mapping[str, Any]) -> QuerySpec:
        timeframe = base_context.get("timeframe", "1d")
        filters: dict[str, Any] = {"timeframe": timeframe}
        symbols = base_context.get("symbols")
        if symbols:
            filters["symbol"] = symbols
        start, end = base_context.get("time_window", (None, None))
        if not start and not end:
            start, end = self._default_time_window(days=30)
        if start:
            filters["evaluated_at_start"] = start
        if end:
            filters["evaluated_at_end"] = end
        return QuerySpec(
            table=self._indicator_summary_table,
            columns=tuple(self._indicator_summary_columns),
            filters=filters,
            limit=self._limits["indicator_summary"],
        )

    def _build_indicator_signals_spec(self, base_context: Mapping[str, Any]) -> QuerySpec:
        filters: dict[str, Any] = {}
        symbols = base_context.get("symbols")
        if symbols:
            filters["symbol"] = symbols
        timeframe = base_context.get("timeframe")
        if timeframe:
            filters["timeframe"] = timeframe
        start, end = base_context.get("time_window", (None, None))
        if not start and not end:
            start, end = self._default_time_window(days=30)
        if start:
            filters["evaluated_at_start"] = start
        if end:
            filters["evaluated_at_end"] = end
        return QuerySpec(
            table=self._indicator_signals_table,
            columns=tuple(self._indicator_signal_columns),
            filters=filters,
            limit=self._limits["indicator_signals"],
        )

    def _build_divergence_spec(self, base_context: Mapping[str, Any]) -> QuerySpec:
        filters: dict[str, Any] = {}
        timeframe = base_context.get("timeframe")
        if timeframe:
            filters["timeframe"] = timeframe
        return QuerySpec(
            table=self._divergence_table,
            columns=tuple(self._divergence_columns),
            filters=filters,
            limit=self._limits["divergence"],
        )

    def _build_news_spec(self, base_context: Mapping[str, Any]) -> QuerySpec:
        filters: dict[str, Any] = {}
        symbols = base_context.get("symbols")
        if symbols:
            filters["tags"] = symbols
        start, end = base_context.get("time_window", (None, None))
        if not start and not end:
            start, end = self._default_news_window()
        if start:
            filters["published_start"] = start
        if end:
            filters["published_end"] = end
        return QuerySpec(
            table=self._news_table,
            columns=tuple(self._news_columns),
            filters=filters,
            limit=self._limits["news"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_base_context(self, slots: IntentSlots) -> dict[str, Any]:
        context: dict[str, Any] = {}
        if slots.asset_scope.symbols:
            context["symbols"] = list(slots.asset_scope.symbols)

        time_window = self._resolve_time_window(slots.time_scope)
        if time_window != (None, None):
            context["time_window"] = time_window

        if slots.timeframe:
            context["timeframe"] = slots.timeframe

        return context

    @staticmethod
    def _resolve_time_window(time_scope) -> tuple[str | None, str | None]:
        if time_scope.start_date:
            end = time_scope.end_date or time_scope.start_date
            return time_scope.start_date, end
        if time_scope.relative:
            today = datetime.now(timezone.utc).date()
            mapping = {
                "today": (today, today),
                "yesterday": (today - timedelta(days=1), today - timedelta(days=1)),
                "this_week": (today - timedelta(days=today.weekday()), today),
                "this_month": (today.replace(day=1), today),
                "last_7d": (today - timedelta(days=7), today),
                "last_30d": (today - timedelta(days=30), today),
                "last_90d": (today - timedelta(days=90), today),
                "last_365d": (today - timedelta(days=365), today),
            }
            window = mapping.get(time_scope.relative)
            if window:
                start, end = window
                return start.isoformat(), end.isoformat()
        return (None, None)

    @staticmethod
    def _default_news_window() -> tuple[str, str]:
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=7)
        return start.isoformat(), today.isoformat()

    @staticmethod
    def _default_time_window(days: int) -> tuple[str, str]:
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=days)
        return start.isoformat(), today.isoformat()
