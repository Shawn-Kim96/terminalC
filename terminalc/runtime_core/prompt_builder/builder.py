"""Composable prompt builder."""
from __future__ import annotations
from typing import Mapping, Sequence
import pandas as pd
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from terminalc.runtime_core.models.runtime_models import DataSnapshot, PromptPayload, QueryPlan

_DEFAULT_TEMPLATE = """System Role:
You are a senior crypto market strategist specializing in technical indicators, on-chain signals, and quantitative market structure.

Operating Principles:
1. Treat the supplied context tables as ground truth—cite concrete metrics, timestamps, or symbols from them.
2. If information is missing, say so explicitly and request the missing metric instead of hallucinating.
3. Tie every claim to observable data (e.g., RSI, MACD, volume trends) and keep the narrative actionable.
4. Highlight risks or confidence levels when the data is mixed or inconclusive.

User Instruction:
{instruction}

Data Evidence Pack:
{context}

Response Requirements:
- Start with a concise thesis sentence (bullish/bearish/neutral + timeframe).
- Support the thesis with 2-3 bullet points referencing exact indicators/rows from the context.
- Close with a short recommendation or watch-list item grounded in the data.
"""


class PromptBuilder:
    def __init__(self, templates: Mapping[str, str] | None = None) -> None:
        self._templates = dict(templates) if templates else {"market_default": _DEFAULT_TEMPLATE}

    def build(
        self,
        plan: QueryPlan,
        snapshots: Sequence[DataSnapshot],
        instruction: str,
        template_id: str = "market_default",
        extra_metadata: Mapping[str, str] | None = None,
    ) -> PromptPayload:
        template = self._templates.get(template_id)
        if not template:
            raise KeyError(f"Unknown template: {template_id}")

        context_blocks = tuple(self._format_snapshot(snapshot) for snapshot in snapshots)
        payload_text = template.format(instruction=instruction, context="\n".join(context_blocks))
        metadata = {
            "intent": plan.intent.name,
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
        df: pd.DataFrame = snapshot.payload
        preview = df.head(10)
        return f"# {snapshot.spec.table} (rows={snapshot.row_count})\n{preview.to_markdown(index=False)}"
