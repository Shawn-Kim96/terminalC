"""Composable prompt builder."""
from __future__ import annotations
from typing import Mapping, Sequence
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from terminalc.runtime_core.models.runtime_models import DataSnapshot, PromptPayload, QueryPlan

_MARKET_PRICE_TEMPLATE = """You are a crypto market data assistant. Answer the user's question using only the provided data.

User Question:
{instruction}

Plan Outline:
{plan_outline}

Data Evidence Pack:
{context}

Instructions:
- Answer directly with the requested values (price, volume, etc.)
- Cite specific numbers from the data tables
- If the data is missing, state that explicitly
- Keep the response concise (1-2 sentences)
"""

_STRATEGY_INFO_TEMPLATE = """You are a crypto technical analysis assistant. Analyze the market based on the technical indicators in the data.

User Question:
{instruction}

Available Data:
{context}

Instructions:
- Provide a brief market assessment (bullish/bearish/neutral)
- Reference specific indicators from the data (RSI, MACD, volume, moving averages, etc.)
- Support your assessment with 2-3 key data points
- Keep the response actionable and concise
"""

_NEWS_CONTEXT_TEMPLATE = """You are a crypto news summarization assistant. Summarize the news content based on the provided articles.

User Question:
{instruction}

Available Data:
{context}

Instructions:
- Summarize the key themes or headlines from the news articles
- Mention sentiment if available in the data
- Keep the summary concise and factual
- If no relevant news exists, state that clearly
"""

_MULTI_CONTEXT_TEMPLATE = """You are a comprehensive crypto market analyst. Answer the user's question by combining insights from price data, technical indicators, and news.

User Question:
{instruction}

Available Data:
{context}

Instructions:
- Integrate insights from multiple data sources (price, indicators, news)
- Provide a balanced analysis citing specific data points
- Highlight any correlations or notable patterns
- Keep the response structured and actionable
"""

_DEFAULT_TEMPLATE = """You are a crypto market data analyst assistant. Answer the user's question using only the data provided below.

Operating Principles:
1. Use ONLY the data in the tables below - cite specific values, timestamps, and symbols.
2. If data is missing, say so explicitly. Never make up or guess values.
3. For simple data queries, provide direct answers with exact values.
4. For analysis questions, support your answer with specific indicators from the data.
5. Keep responses concise and grounded in observable data.

User Question:
{instruction}

Available Data:
{context}

Instructions:
- Answer the question directly and concisely
- Always cite specific values from the data tables
- If the question asks for analysis, reference relevant indicators
- If the question is a simple lookup, just provide the requested value(s)
"""


class PromptBuilder:
    def __init__(self, templates: Mapping[str, str] | None = None) -> None:
        if templates:
            self._templates = dict(templates)
        else:
            self._templates = {
                "market_default": _DEFAULT_TEMPLATE,
                "market_price": _MARKET_PRICE_TEMPLATE,
                "strategy_info": _STRATEGY_INFO_TEMPLATE,
                "news_context": _NEWS_CONTEXT_TEMPLATE,
                "multi_context": _MULTI_CONTEXT_TEMPLATE,
            }

    def build(
        self,
        plan: QueryPlan,
        snapshots: Sequence[DataSnapshot],
        instruction: str,
        template_id: str | None = None,
        extra_metadata: Mapping[str, str] | None = None,
    ) -> PromptPayload:
        # Auto-select template based on intent if not explicitly provided
        if template_id is None:
            template_id = plan.intent.name if plan.intent.name in self._templates else "market_default"

        template = self._templates.get(template_id)
        if not template:
            raise KeyError(f"Unknown template: {template_id}")

        context_blocks = tuple(self._format_snapshot(snapshot) for snapshot in snapshots)
        plan_outline = self._build_plan_outline(plan)
        payload_text = template.format(
            instruction=instruction,
            context="\n".join(context_blocks),
            plan_outline=plan_outline,
        )
        metadata = {
            "intent": plan.intent.name,
            "template_id": template_id,
            "cache_keys": [snap.cache_key for snap in snapshots],
            "user_instruction": instruction,
        }
        metadata.update(extra_metadata or {})

        return PromptPayload(
            template_id=template_id,
            instructions=payload_text,
            context_blocks=context_blocks,
            metadata=metadata,
        )

    @staticmethod
    def _format_snapshot(snapshot: DataSnapshot) -> str:
        df = PromptBuilder._prepare_dataframe(snapshot.payload)
        preview = df.head(10)
        return f"# {snapshot.spec.table} (rows={snapshot.row_count})\n{preview.to_markdown(index=False)}"

    @staticmethod
    def _prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        for column in df.columns:
            series = df[column]
            if is_datetime64_any_dtype(series):
                try:
                    if getattr(series.dt, "tz", None) is not None:
                        df[column] = series.dt.tz_convert("UTC")
                    else:
                        df[column] = series.dt.tz_localize("UTC")
                except (TypeError, ValueError):
                    continue
        if {"open", "close"}.issubset(df.columns):
            import numpy as np
            returns = df["close"] - df["open"]
            df["return_abs"] = returns
            df["return_pct"] = (returns / df["open"]) * 100
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    @staticmethod
    def _build_plan_outline(plan: QueryPlan) -> str:
        if not plan.plan_steps:
            return "1. Review the retrieved context tables in order."
        lines = []
        for idx, step in enumerate(plan.plan_steps, start=1):
            lines.append(f"{idx}. {step}")
        return "\n".join(lines)
