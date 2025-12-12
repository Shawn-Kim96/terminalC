# Terminal C

Terminal C is the crypto research assistant I built for CMPE 259. The goal is simple: answer realistic analyst questions with hard data, cite exactly where every number came from, and keep the entire system portable so anyone on the team can run it locally or in Colab.

## How the repo is organized

- `terminalc/data_scripts/` – end-to-end data builders. They fetch OHLCV candles, compute all technical indicators, derive Investing.com style signals, assemble the strategy catalog, scrape CoinDesk headlines, and load everything into DuckDB.
- `terminalc/runtime_core/` – the production pipeline. Inside you’ll find the intent parser, query planner, DuckDB adapter, prompt builder, cache layers, LLM clients, security guards, and now a lightweight CoinDesk web-search tool.
- `notebooks/` – scratchpads I used while validating the pipeline (data sanity checks, pipeline dry-runs, final demo notebooks).
- `models/` – the LoRA adapter checkpoints for the distilled small model.
- `results/` – CSVs from earlier model comparisons plus any benchmark reports you generate with the new tooling.
- `report/` – the official proposal, abstract, and final report submitted to the course staff.

## Runtime pipeline at a glance

1. **Prompt intake & guardrails** – `PromptSecurityGuard` blocks prompt-injection attempts before anything reaches the tools.
2. **Intent analysis** – `InputAnalyzer` tags the question as market/strategy/news/multi, extracts symbols, windows, metrics, and flags if we need prompt chaining or a live news search.
3. **Query planning** – `QueryOrchestrator` turns those slots into one or more `QuerySpec`s for DuckDB tables (candles, indicator summaries, divergences, strategies, news). Every column and filter is validated against the schema.
4. **Data execution & caching** – `DuckDBClient` runs the specs with `read_only=True`, and the query cache stores each dataframe on disk so repeat calls are instant.
5. **Live CoinDesk search** – if the analyzer flagged the prompt as “needs live news” (e.g., “latest BTC headlines”), `CoinDeskWebSearch` hits the RSS feed in real time and produces a synthetic snapshot that joins the rest of the evidence pack.
6. **Prompt construction** – `PromptBuilder` stitches together the persona instructions (“Operating Principles”) and Markdown previews of each snapshot.
7. **LLM generation + reflection** – pick either the large Hugging Face router endpoint or the local LoRA student, generate an answer, then optionally run the self-reflection pass to catch sloppy math.
8. **Post-processing & prompt caching** – the outbound security scanner re-checks the response before caching it on disk. Cache hits return immediately without re-calling the model.

Every stage is dependency-injected so it’s easy to swap components (e.g., run without reflection for latency tests, or plug in a different model client).

## Key features

- **Two-model support** – Meta Llama 3.3 70B over the Hugging Face router plus a distilled Llama 3.2 3B LoRA adapter for offline demos.
- **Deterministic planning** – schema-aware plans mean every question produces the same SQL, so it’s trivial to audit why the assistant said something.
- **Prompt/query caching** – hashed SQL snapshots on disk and SHA-based prompt caches bring repeat latency down to milliseconds.
- **Security guardrails** – the same scanner runs before and after the LLM so prompt-injection, PII leaks, or destructive SQL never touch the warehouse.
- **Live CoinDesk search** – when a user explicitly asks for the latest headlines, the runtime fetches the RSS feed on the fly and merges those results with DuckDB data. No more stale news summaries.
- **Benchmark tooling** – `terminalc/runtime_core/benchmarks/model_benchmark.py` measures average latency per model and runs a few simple accuracy heuristics by checking whether the final answer cites the ground-truth values pulled from DuckDB.

## Getting started

1. Install dependencies (`poetry install` or your preferred workflow) and export `HUGGINGFACE_TOKEN`, `LARGE_MODEL_ENDPOINT`, and `SMALL_MODEL_ENDPOINT` as needed.
2. Build the DuckDB snapshot by running `terminalc/data_scripts/main.py` (fetch candles/news, compute indicators, ingest into DuckDB).
3. Run the runtimeƒ pipeline from Python:
   ```python
   from terminalc.runtime_core.pipelines.runtime_pipeline import RuntimePipeline
   pipeline = RuntimePipeline(model_type="small")  # or "large"
   print(pipeline.run("What was the closing price of BTC on Oct 15, 2025?").response_text)
   print(pipeline.run("What is the most recent news about BTC?").response_text)  # triggers live CoinDesk search
   ```
4. Need a Colab-friendly path? Set the `TERMINALC_*` environment variables documented in `terminalc/runtime_core/config.py` so the pipeline knows where to find the DuckDB file, caches, and local models.

## Measuring things

- **Latency & simple accuracy** – run `poetry run python -m terminalc.runtime_core.benchmarks.model_benchmark --model-type large` (or `small`). The script will time each prompt, compare the LLM answer against DuckDB ground-truth values, and save a JSON report under `results/benchmarks/`.
- **Token cost comparisons** – use the CSVs under `results/` as a template, or tweak the benchmark script to log token usage via the Hugging Face API response metadata.
- **Custom prompts** – drop your own list into the benchmark script or notebooks and re-run. Everything is deterministic so you can regenerate charts or tables for the final report quickly.

## What’s next

The current stack already satisfies the project requirements, but there’s plenty of room for follow-up work (hook up a second news source, add SERP-based web search, broaden the benchmark suite, or deploy the runtime as a simple FastAPI service). The README will stay up-to-date as those experiments land.
