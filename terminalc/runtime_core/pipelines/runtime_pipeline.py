"""High-level orchestration for the runtime system."""
from __future__ import annotations

from typing import Literal, Protocol, Sequence
import os
import sys
from pathlib import Path

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from terminalc.runtime_core.cache.prompt_cache import PromptCache
from terminalc.runtime_core.cache.query_cache import QueryCache
from terminalc.runtime_core.config import RuntimeConfig, load_runtime_config
from terminalc.runtime_core.data_access.duckdb_client import DuckDBClient
from terminalc.runtime_core.input_parser.analyzer import InputAnalyzer
from terminalc.runtime_core.models.runtime_models import DataSnapshot, LLMResult, PromptPayload, QueryPlan
from terminalc.runtime_core.pipelines.llm_client import (
    HuggingFaceInferenceClient,
    LocalTransformersClient,
)
from terminalc.runtime_core.processor.post_processor import LLMPostProcessor
from terminalc.runtime_core.processor.self_reflection import SelfReflectionProcessor
from terminalc.runtime_core.processor.pre_processor import PromptSecurityGuard
from terminalc.runtime_core.prompt_builder.builder import PromptBuilder
from terminalc.runtime_core.query_planner.planner import QueryOrchestrator

ModelType = Literal["large", "small"] | None


class LLMClientProtocol(Protocol):
    def generate(self, payload: PromptPayload) -> LLMResult:  # pragma: no cover - interface definition
        ...


class RuntimePipeline:
    def __init__(
        self,
        model_type: ModelType = "large",
        llm_client: LLMClientProtocol | None = None,
        config: RuntimeConfig | None = None,
        analyzer: InputAnalyzer | None = None,
        planner: QueryOrchestrator | None = None,
        prompt_builder: PromptBuilder | None = None,
        post_processor: LLMPostProcessor | None = None,
        prompt_guard: PromptSecurityGuard | None = None,
        self_reflector: SelfReflectionProcessor | None = None,
    ) -> None:
        self._config = config or load_runtime_config()
        self._model_type = model_type
        if llm_client is not None:
            self._llm_client = llm_client
        elif model_type is not None:
            self._llm_client = self._build_llm_client(model_type)
        else:
            self._llm_client = None

        self._analyzer = analyzer or InputAnalyzer()
        self._planner = planner or QueryOrchestrator()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._post_processor = post_processor or LLMPostProcessor()
        self._prompt_guard = prompt_guard or PromptSecurityGuard()
        self._self_reflector = self_reflector or SelfReflectionProcessor()
        self._duckdb = DuckDBClient(self._config.duckdb)
        self._query_cache = QueryCache(self._config.cache.query_cache_dir)
        self._prompt_cache = PromptCache(self._config.cache.prompt_cache_dir)

    def run(
        self,
        prompt: str,
        instruction: str | None = None,
        template_id: str = "market_default",
        return_payload: bool = False,
        build_only: bool = False,
    ) -> LLMResult | tuple[LLMResult, PromptPayload] | PromptPayload:
        if self._prompt_guard:
            notice = self._prompt_guard.enforce(prompt)
            if notice:
                return LLMResult(response_text=notice, model_name="prompt_security_guard", total_tokens=None)

        if self._llm_client is None and not build_only:
            raise RuntimeError("LLM client is not initialized. Provide model_type or llm_client.")

        intent = self._analyzer.analyze(prompt)
        plan = self._planner.build_plan(intent)
        instruction = instruction or prompt
        snapshots = self._query_execution(plan)
        payload = self._prompt_builder.build(plan, snapshots, instruction, template_id)
        prompt_key = self._prompt_cache.build_key(payload)

        if build_only:
            return payload if not return_payload else (None, payload)

        cached = self._prompt_cache.get(prompt_key)
        if cached:
            return (cached, payload) if return_payload else cached

        llm_result = self._llm_client.generate(payload)
        if self._self_reflector:
            try:
                llm_result = self._self_reflector.refine(self._llm_client, payload, llm_result)
            except Exception:
                pass
        processed = self._post_processor.process(llm_result.response_text, llm_result.model_name, llm_result.total_tokens)
        self._prompt_cache.store(prompt_key, processed)
        return (processed, payload) if return_payload else processed

    def build_prompt(
        self,
        prompt: str,
        instruction: str | None = None,
        template_id: str = "market_default",
    ) -> PromptPayload:
        if self._prompt_guard:
            notice = self._prompt_guard.enforce(prompt)
            if notice:
                # If blocked, return a minimal payload explaining the block
                return PromptPayload(
                    template_id=template_id,
                    instructions=notice,
                    context_blocks=(),
                    metadata={"intent": "blocked", "user_instruction": prompt},
                )

        intent = self._analyzer.analyze(prompt)
        plan = self._planner.build_plan(intent)
        instruction = instruction or prompt
        snapshots = self._query_execution(plan)
        payload = self._prompt_builder.build(plan, snapshots, instruction, template_id)
        return payload

    def _query_execution(self, plan: QueryPlan) -> Sequence[DataSnapshot]:
        snapshots: list[DataSnapshot] = []
        for spec in plan.specs:
            _, _, cache_key = self._duckdb.compile(spec)
            cached_df = self._query_cache.get(cache_key)
            if cached_df is not None:
                snapshots.append(
                    DataSnapshot(spec=spec, row_count=len(cached_df), payload=cached_df, cache_key=cache_key)
                )
                continue
            snapshot = self._duckdb.execute(spec)
            self._query_cache.store(snapshot)
            snapshots.append(snapshot)
        return snapshots

    def _build_llm_client(self, model_type: Literal["large", "small"]) -> LLMClientProtocol:
        models_cfg = self._config.models
        if model_type == "large":
            endpoint = models_cfg.large_model_endpoint
            if not endpoint:
                raise ValueError("LARGE_MODEL_ENDPOINT is not configured.")
            return HuggingFaceInferenceClient(endpoint, api_token=models_cfg.huggingface_token)

        if model_type == "small":
            endpoint = models_cfg.small_model_endpoint
            if not endpoint:
                raise ValueError("SMALL_MODEL_ENDPOINT is not configured.")
            local_root = models_cfg.local_model_dir
            local_root_path = local_root if isinstance(local_root, Path) else Path(local_root)
            candidate = local_root_path / endpoint.replace("/", os.sep)
            model_path = candidate if candidate.exists() else local_root_path / endpoint
            model_path = model_path.resolve()
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Local model path '{model_path}' does not exist. "
                    "Download the model or update SMALL_MODEL_ENDPOINT."
                )
            return LocalTransformersClient(str(model_path))

        raise ValueError(f"Unsupported model_type '{model_type}'.")
