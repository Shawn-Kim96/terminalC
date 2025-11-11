"""Secondary pass that forces the LLM to self-reflect before finalizing."""
from __future__ import annotations

from typing import Protocol

from terminalc.runtime_core.models.runtime_models import LLMResult, PromptPayload


class _SupportsGenerate(Protocol):
    def generate(self, payload: PromptPayload) -> LLMResult: ...


_DEFAULT_REFLECTION_TEMPLATE = """You already drafted the answer below:

<<<DRAFT>>>
{answer}
<<<END DRAFT>>>

Re-read the user instruction carefully:
{instruction}

Re-read the structured context that backs the analysis:
{context}

Self-reflection steps:
1. Verify every claim against the context. Flag any hallucinated metric, price, or timeframe.
2. Ensure the reasoning cites concrete indicators (RSI, MACD, ATR, news sentiment, etc.).
3. If the draft is incomplete or speculative, revise it so each statement traces back to the data.
4. Respond directly to the instruction and keep the tone professional and concise.

Return the final answer after reflection. Do not mention this review process explicitly.
"""


class SelfReflectionProcessor:
    def __init__(self, template: str | None = None) -> None:
        self._template = template or _DEFAULT_REFLECTION_TEMPLATE

    def refine(
        self,
        llm_client: _SupportsGenerate,
        payload: PromptPayload,
        draft_result: LLMResult,
    ) -> LLMResult:
        context_text = "\n".join(payload.context_blocks)
        user_instruction = payload.metadata.get("user_instruction", "")
        reflection_instruction = self._template.format(
            instruction=user_instruction,
            context=context_text,
            answer=draft_result.response_text,
        )
        reflection_payload = PromptPayload(
            template_id=f"{payload.template_id}__self_reflection",
            instructions=reflection_instruction,
            context_blocks=payload.context_blocks,
            metadata={
                **payload.metadata,
                "self_reflection": True,
            },
        )
        return llm_client.generate(reflection_payload)


__all__ = ["SelfReflectionProcessor"]
