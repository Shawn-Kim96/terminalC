"""Rule-based input analyzer that maps prompts to intents + slots."""
from __future__ import annotations

import re
from typing import Any, Mapping
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from src.runtime_core.models.runtime_models import AssetScope, Intent, IntentSlots, NewsFilter, TimeScope


class InputAnalyzer:
    """Parse natural language prompts into intents and structured slots."""

    INTENT_MARKET = "market_price"
    INTENT_STRATEGY = "strategy_info"
    INTENT_NEWS = "news_context"
    INTENT_MULTI = "multi_context"

    _MONTH_LOOKUP = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    _DATE_PATTERNS = {
        "ymd": re.compile(r"(20\d{2})[./\s-](\d{1,2})[./\s-](\d{1,2})"),
        "numeric": re.compile(r"(\d{1,2})[./\s-](\d{1,2})[./\s-](20\d{2})"),
        "month_name_first": re.compile(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)"
            r"[.\s-]*(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s*(20\d{2})"
        ),
        "day_first_month_name": re.compile(
            r"(\d{1,2})(?:st|nd|rd|th)?[.\s-]*"
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)"
            r"[.,\s-]*(20\d{2})"
        ),
    }

    _RELATIVE_WINDOWS = {
        "today": "today",
        "yesterday": "yesterday",
        "this week": "this_week",
        "this month": "this_month",
        "last week": "last_7d",
        "past week": "last_7d",
        "last month": "last_30d",
        "past month": "last_30d",
        "last quarter": "last_90d",
        "past quarter": "last_90d",
        "last year": "last_365d",
    }

    _MARKET_KEYWORDS = {
        "price",
        "trend",
        "volume",
        "volatility",
        "candle",
        "candlestick",
        "trading",
        "attractive",
        "increase",
        "decrease",
        "divergence",
        "signal",
        "return",
        "returns",
        "performance",
    }

    _STRATEGY_KEYWORDS = {
        "strategy",
        "strategies",
        "startegy",
        "startegies",
        "rule",
        "entry",
        "exit",
        "indicator",
        "setup",
        "playbook",
        "technical",
        "trading plan",
    }

    _NEWS_KEYWORDS = {
        "news",
        "headline",
        "article",
        "report",
        "sentiment",
        "narrative",
    }

    _METRIC_KEYWORDS = {
        "price": {"price", "close", "trend", "candle", "candlestick", "return", "returns", "performance"},
        "volume": {"volume", "liquidity", "flow"},
        "volatility": {"volatility", "vol", "risk"},
        "signal": {"signal", "indicator", "overall_signal"},
        "news_sentiment": {"sentiment", "headline", "news"},
    }

    _STRATEGY_TOPICS = {
        "divergence": {"divergence", "diverge"},
        "momentum": {"momentum", "trend-follow"},
        "mean_reversion": {"mean reversion", "oscillator"},
        "breakout": {"breakout", "break-out"},
        "volume_profile": {"volume profile", "vwap"},
    }

    _NEWS_CATEGORIES = {
        "regulation": {"sec", "regulation", "policy", "law"},
        "partnership": {"partnership", "alliance", "collaboration"},
        "security": {"hack", "security", "breach"},
        "macro": {"macro", "inflation", "fed", "rate"},
    }

    _SENTIMENT_KEYWORDS = {
        "positive": {"bullish", "positive", "optimistic"},
        "negative": {"bearish", "negative", "pessimistic"},
    }

    _GLOBAL_SCOPE_MARKERS = {
        "all coins",
        "overall market",
        "entire market",
        "whole market",
        "crypto market",
        "general market",
        "crypto",
        "coin",
        "coins"
    }

    _PROMPT_CHAIN_MARKERS = {"overall", "combined", "holistic", "together"}

    _SYMBOL_ALIASES = {
        "btc": "BTC",
        "bitcoin": "BTC",
        "eth": "ETH",
        "ether": "ETH",
        "ethereum": "ETH",
        "sol": "SOL",
        "solana": "SOL",
        "xrp": "XRP",
        "doge": "DOGE",
        "dogecoin": "DOGE",
        "ada": "ADA",
        "cardano": "ADA",
        "link": "DOT",
        "chainlink": "DOT",
        "avax": "AVAX",
        "xlm": "XLM",
        "lumens": "XLM",
        "stellar": "XLM",
        "hbar": "HBAR",
        "hedera": "HBAR",
        "apt": "APT",
        "aptos": "APT",
        "ondo": "ONDO",
        "sui": "SUI"
    }

    def __init__(self, fallback_intent: str = "generic_query", threshold: float = 0.4) -> None:
        self._fallback_intent = fallback_intent
        self._threshold = threshold
        self._known_symbols = set(self._SYMBOL_ALIASES.values())

    def analyze(self, prompt: str, context: Mapping[str, Any] | None = None) -> Intent:
        context = dict(context or {})
        normalized = prompt.lower()
        slots = self._extract_slots(prompt, normalized)
        intent_name, confidence = self._decide_intent(normalized, slots)
        parameters: dict[str, Any] = {"raw": prompt}
        parameters.update(context)
        filters = self._build_filters(slots)
        if filters:
            parameters["filters"] = filters
        parameters["metrics"] = slots.metrics
        parameters["prompt_flags"] = slots.prompt_flags
        return Intent(name=intent_name, confidence=confidence, parameters=parameters, slots=slots)

    def _extract_slots(self, prompt: str, normalized: str) -> IntentSlots:
        asset_scope = self._extract_assets(normalized)
        time_scope = self._extract_time_scope(normalized)
        metrics = self._extract_metrics(normalized)
        strategy_topics = self._extract_strategy_topics(normalized)
        news_filters = self._extract_news_filters(normalized)
        prompt_flags = self._derive_prompt_flags(normalized, metrics, news_filters)
        return IntentSlots(
            asset_scope=asset_scope,
            time_scope=time_scope,
            metrics=metrics,
            strategy_topics=strategy_topics,
            news_filters=news_filters,
            prompt_flags=prompt_flags,
        )

    def _extract_assets(self, normalized: str) -> AssetScope:
        mentions = set()
        symbols = set()

        for alias, symbol in self._SYMBOL_ALIASES.items():
            if alias in normalized:
                mentions.add(alias)
                symbols.add(symbol)
        
        scope = "specific_asset" if symbols else "all_assets"
        for marker in self._GLOBAL_SCOPE_MARKERS:
            if marker in normalized:
                scope = "all_assets"
                break

        return AssetScope(scope=scope, symbols=tuple(sorted(symbols)), raw_mentions=tuple(sorted(mentions)))

    def _extract_time_scope(self, normalized: str) -> TimeScope:
        dates = self._collect_dates(normalized)
        if dates:
            start = dates[0]
            end = dates[1] if len(dates) > 1 else dates[0]
            return TimeScope(start_date=start, end_date=end, raw_text="explicit_dates")

        for phrase, label in self._RELATIVE_WINDOWS.items():
            if phrase in normalized:
                return TimeScope(relative=label, raw_text=phrase)
        return TimeScope()

    def _collect_dates(self, text: str) -> list[str]:
        dates: list[str] = []
        seen = set()

        for match in self._DATE_PATTERNS["ymd"].finditer(text):
            year, month, day = match.groups()
            iso = self._normalize_date(int(year), int(month), int(day))
            if iso not in seen:
                dates.append(iso)
                seen.add(iso)

        for match in self._DATE_PATTERNS["numeric"].finditer(text):
            first, second, year = match.groups()
            month, day = self._disambiguate_numeric(int(first), int(second))
            iso = self._normalize_date(int(year), month, day)
            if iso not in seen:
                dates.append(iso)
                seen.add(iso)

        for match in self._DATE_PATTERNS["month_name_first"].finditer(text):
            month_name, day, year = match.groups()
            month = self._MONTH_LOOKUP.get(month_name.lower())
            if not month:
                continue
            iso = self._normalize_date(int(year), month, int(day))
            if iso not in seen:
                dates.append(iso)
                seen.add(iso)

        for match in self._DATE_PATTERNS["day_first_month_name"].finditer(text):
            day, month_name, year = match.groups()
            month = self._MONTH_LOOKUP.get(month_name.lower())
            if not month:
                continue
            iso = self._normalize_date(int(year), month, int(day))
            if iso not in seen:
                dates.append(iso)
                seen.add(iso)

        return dates

    @staticmethod
    def _disambiguate_numeric(first: int, second: int) -> tuple[int, int]:
        """Best-effort detection for MM/DD/YYYY vs DD/MM/YYYY layouts."""
        if first > 12 and second <= 12:
            return second, first  # day-month-year
        if second > 12 and first <= 12:
            return first, second  # month-day-year
        if first > 12 and second > 12:
            # Both invalid months; clamp to bounds while keeping order.
            return min(first, 12), min(second, 31)
        # Ambiguous 01/02 -> assume first is month.
        return max(1, min(first, 12)), max(1, min(second, 31))

    @staticmethod
    def _normalize_date(year: int, month: int, day: int) -> str:
        month = min(max(month, 1), 12)
        day = min(max(day, 1), 31)
        return f"{year:04d}-{month:02d}-{day:02d}"

    def _extract_metrics(self, normalized: str) -> tuple[str, ...]:
        metrics = {name for name, keywords in self._METRIC_KEYWORDS.items() if any(word in normalized for word in keywords)}
        return tuple(sorted(metrics))

    def _extract_strategy_topics(self, normalized: str) -> tuple[str, ...]:
        topics = {name for name, keywords in self._STRATEGY_TOPICS.items() if any(word in normalized for word in keywords)}
        return tuple(sorted(topics))

    def _extract_news_filters(self, normalized: str) -> NewsFilter:
        categories = {name for name, keywords in self._NEWS_CATEGORIES.items() if any(word in normalized for word in keywords)}
        sentiments = {name for name, keywords in self._SENTIMENT_KEYWORDS.items() if any(word in normalized for word in keywords)}
        return NewsFilter(categories=tuple(sorted(categories)), sentiments=tuple(sorted(sentiments)))

    def _derive_prompt_flags(
        self,
        normalized: str,
        metrics: tuple[str, ...],
        news_filters: NewsFilter,
    ) -> tuple[str, ...]:
        flags: list[str] = []
        if ("news" in normalized or news_filters.categories) and {"price", "volume", "signal"}.intersection(metrics):
            flags.append("needs_prompt_chaining")
        for marker in self._PROMPT_CHAIN_MARKERS:
            if marker in normalized:
                flags.append("needs_prompt_chaining")
                break
        return tuple(sorted(set(flags)))

    def _decide_intent(self, normalized: str, slots: IntentSlots) -> tuple[str, float]:
        """Basic intent selection: pick the domain with the most keyword hits."""

        votes = []
        if self._contains_any(normalized, self._MARKET_KEYWORDS):
            votes.append(self.INTENT_MARKET)
        if self._contains_any(normalized, self._STRATEGY_KEYWORDS) or slots.strategy_topics:
            votes.append(self.INTENT_STRATEGY)
        if self._contains_any(normalized, self._NEWS_KEYWORDS) or slots.news_filters.categories or slots.news_filters.sentiments:
            votes.append(self.INTENT_NEWS)

        if not votes:
            return self._fallback_intent, self._threshold

        intents = list(dict.fromkeys(votes))  # preserve order, remove duplicates

        if (
            self.INTENT_STRATEGY in intents
            and self.INTENT_MARKET in intents
            and not slots.metrics
            and not slots.asset_scope.symbols
            and not (slots.time_scope.start_date or slots.time_scope.relative)
        ):
            intents = [intent for intent in intents if intent != self.INTENT_MARKET]

        if len(intents) > 1:
            return self.INTENT_MULTI, 0.7

        intent = intents[0]
        confidence = 0.6
        if slots.asset_scope.symbols:
            confidence += 0.1
        if slots.time_scope.start_date or slots.time_scope.relative:
            confidence += 0.1
        if slots.metrics:
            confidence += 0.05

        return intent, min(confidence, 0.9)

    def _build_filters(self, slots: IntentSlots) -> Mapping[str, Any]:
        filters: dict[str, Any] = {}
        if slots.asset_scope.symbols:
            filters["symbol"] = list(slots.asset_scope.symbols)
        if slots.time_scope.start_date:
            filters["start_date"] = slots.time_scope.start_date
        if slots.time_scope.end_date:
            filters["end_date"] = slots.time_scope.end_date
        elif slots.time_scope.relative:
            filters["relative_range"] = slots.time_scope.relative
        return filters

    @staticmethod
    def _contains_any(text: str, keywords: set[str]) -> bool:
        return any(keyword in text for keyword in keywords)
