# runtime_core

Core runtime for the prompt-driven query + LLM pipeline.

## Layout & Notable Components

- `config.py`: shared runtime configuration (DuckDB path, cache dirs, model info).
- `models/`: dataclasses used by all components.
- `input_parser/`: lightweight intent classifier placeholder.
- `query_planner/`: rule-based plan generator that produces `QuerySpec` objects.
- `data_access/`: DuckDB adapter that materializes queries and emits snapshots.
- `prompt_builder/`: persona-heavy templates that enforce “data-first, no hallucination” responses.
- `processor/pre_processor.py`: prompt security guard (regex + heuristics) that blocks risky inputs before the LLM call.
- `processor/post_processor.py`: response validator + outbound security scanner.
- `processor/self_reflection.py`: optional second-pass prompt that forces the LLM to critique/revise its own draft.
- `cache/`: query and prompt cache implementations (pickle/JSON on disk).
- `pipelines/`: orchestration pipeline + HF API/local client implementations.
- `model_registry/`: utilities for downloading local (small) models.

## Quick start

```bash
# 1) Download a small model locally
poetry run python -m terminalc.runtime_core.model_registry.downloader sentence-transformers/all-MiniLM-L6-v2

# 2) Export LLM API token (large model)
export HUGGINGFACEHUB_API_TOKEN=<<token>>
```

```python
# 3) Instantiate the runtime pipeline in Python
from terminalc.runtime_core.config import load_runtime_config
from terminalc.runtime_core.pipelines.runtime_pipeline import RuntimePipeline

cfg = load_runtime_config()

# Hosted model declared in LARGE_MODEL_ENDPOINT.
# The pipeline will: prompt-scan -> plan -> build persona prompt -> LLM draft ->
# self-reflection pass -> post-process (validation + security) -> return result.
pipeline = RuntimePipeline(model_type="large", config=cfg)
result = pipeline.run("Show me the BTC trend this week")
print(result.response_text)

# Switch to the locally downloaded SMALL_MODEL_ENDPOINT
local_pipeline = RuntimePipeline(model_type="small", config=cfg)
```

Cache artifacts land in `data/cache/{query,prompt}` while downloaded small models live under `models/local/` by default. Tune directories via the `TERMINALC_*` environment variables documented in `config.py`.

## End-to-End Flow

1. **Prompt intake & guardrails**  
   User text enters `RuntimePipeline.run()`. `PromptSecurityGuard` scans for prompt-injection/secret-leak attempts and immediately returns a security notice if something is suspicious.

2. **Intent analysis & planning**  
   `InputAnalyzer` derives an intent, and `QueryOrchestrator` produces one or more `QuerySpec`s describing which DuckDB tables/columns to pull plus filter constraints.

3. **Data fetch & context building**  
   `DuckDBClient` executes each spec (with query caching). Snapshots feed into `PromptBuilder`, which produces the persona-driven instructions + markdown table previews.

4. **Primary LLM draft**  
   The configured LLM client (`HuggingFaceInferenceClient` via API or `LocalTransformersClient`) generates an initial answer from the prompt payload.

5. **Self-reflection pass (optional)**  
   `SelfReflectionProcessor` re-prompts the same LLM to critique and revise the draft using the same context/instruction. Failures fall back to the original draft.

6. **Post-processing & security scan**  
   `LLMPostProcessor` trims/validates the response and runs the outbound security scanner. If it flags anything, the user sees the security notice instead.

7. **Caching & return**  
   Successful responses land in the prompt cache (keyed by prompt payload) for reuse, and the final `LLMResult` is returned to the caller.

### Tuning Guards and Reflection

- To adjust prompt/response scanning, pass custom `SecurityScanner`/`PromptSecurityGuard`/`LLMPostProcessor` instances into `RuntimePipeline`.
- To disable the self-reflection step (e.g., for latency-sensitive paths), pass `self_reflector=None` when instantiating the pipeline.
- The default prompt template lives in `prompt_builder/builder.py`; customize it if you need a different persona or response format.
