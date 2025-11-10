"""Typed objects shared across runtime components."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


@dataclass(slots=True)
class AssetScope:
    scope: Literal["specific_asset", "all_assets"] = "all_assets"
    symbols: tuple[str, ...] = field(default_factory=tuple)
    raw_mentions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class TimeScope:
    start_date: str | None = None
    end_date: str | None = None
    relative: str | None = None
    raw_text: str | None = None


@dataclass(slots=True)
class NewsFilter:
    categories: tuple[str, ...] = field(default_factory=tuple)
    sentiments: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class IntentSlots:
    asset_scope: AssetScope = field(default_factory=AssetScope)
    time_scope: TimeScope = field(default_factory=TimeScope)
    metrics: tuple[str, ...] = field(default_factory=tuple)
    strategy_topics: tuple[str, ...] = field(default_factory=tuple)
    news_filters: NewsFilter = field(default_factory=NewsFilter)
    prompt_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class Intent:
    name: str
    confidence: float
    parameters: Mapping[str, Any] = field(default_factory=dict)
    slots: IntentSlots | None = None


@dataclass(slots=True)
class QuerySpec:
    """Normalized representation of a data request."""

    table: str
    columns: Sequence[str]
    filters: Mapping[str, Any]
    limit: int | None = None


@dataclass(slots=True)
class QueryPlan:
    """A higher-level plan that may produce multiple QuerySpec objects."""

    intent: Intent
    specs: Sequence[QuerySpec]
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class DataSnapshot:
    """Materialized query output metadata."""

    spec: QuerySpec
    row_count: int
    payload: Any
    cache_key: str | None = None


@dataclass(slots=True)
class PromptPayload:
    """Input body sent to the large language model."""

    template_id: str
    instructions: str
    context_blocks: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResult:
    response_text: str
    model_name: str
    total_tokens: int | None = None
    cached: bool = False


@dataclass(slots=True)
class CacheRecord:
    """Metadata describing cached artifacts on disk."""

    key: str
    path: Path
    created_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "AssetScope",
    "TimeScope",
    "NewsFilter",
    "IntentSlots",
    "Intent",
    "QuerySpec",
    "QueryPlan",
    "DataSnapshot",
    "PromptPayload",
    "LLMResult",
    "CacheRecord",
]
