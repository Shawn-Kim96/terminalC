"""High-level orchestration for the runtime system."""
from __future__ import annotations
from typing import Protocol, Sequence
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from src.runtime_core.cache.prompt_cache import PromptCache
from src.runtime_core.cache.query_cache import QueryCache
from src.runtime_core.config import RuntimeConfig, load_runtime_config
from src.runtime_core.data_access.duckdb_client import DuckDBClient
from src.runtime_core.input_parser.analyzer import InputAnalyzer
from src.runtime_core.models.runtime_models import DataSnapshot, LLMResult, PromptPayload, QueryPlan
from src.runtime_core.postprocess.processor import LLMPostProcessor
from src.runtime_core.prompt_builder.builder import PromptBuilder
from src.runtime_core.query_planner.planner import QueryOrchestrator


class LLMClientProtocol(Protocol):
    def generate(self, payload: PromptPayload) -> LLMResult:  # pragma: no cover - interface definition
        ...


class RuntimePipeline:
    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: RuntimeConfig | None = None,
        analyzer: InputAnalyzer | None = None,
        planner: QueryOrchestrator | None = None,
        prompt_builder: PromptBuilder | None = None,
        post_processor: LLMPostProcessor | None = None,
    ) -> None:
        self._config = config or load_runtime_config()
        self._llm_client = llm_client
        self._analyzer = analyzer or InputAnalyzer()
        self._planner = planner or QueryOrchestrator()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._post_processor = post_processor or LLMPostProcessor()
        self._duckdb = DuckDBClient(self._config.duckdb)
        self._query_cache = QueryCache(self._config.cache.query_cache_dir)
        self._prompt_cache = PromptCache(self._config.cache.prompt_cache_dir)

    def run(self, prompt: str, instruction: str | None = None, template_id: str = "market_default") -> LLMResult:
        intent = self._analyzer.analyze(prompt)
        plan = self._planner.build_plan(intent)
        instruction = instruction or prompt
        snapshots = self._query_execution(plan)
        payload = self._prompt_builder.build(plan, snapshots, instruction, template_id)
        prompt_key = self._prompt_cache.build_key(payload)

        cached = self._prompt_cache.get(prompt_key)
        if cached:
            return cached

        llm_result = self._llm_client.generate(payload)
        processed = self._post_processor.process(llm_result.response_text, llm_result.model_name, llm_result.total_tokens)
        self._prompt_cache.store(prompt_key, processed)
        return processed

    def _query_execution(self, plan: QueryPlan) -> Sequence[DataSnapshot]:
        snapshots: list[DataSnapshot] = []
        for spec in plan.specs:
            # spec = table / columns / filters / limit
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
