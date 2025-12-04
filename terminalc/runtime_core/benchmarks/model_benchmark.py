"""Utility to benchmark runtime latency and simple accuracy checks."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from terminalc.runtime_core.config import load_runtime_config
from terminalc.runtime_core.data_access.duckdb_client import DuckDBClient
from terminalc.runtime_core.models.runtime_models import QuerySpec
from terminalc.runtime_core.pipelines.runtime_pipeline import RuntimePipeline


def _numeric_tokens(value: float) -> list[str]:
    integer = int(round(value))
    tokens = {str(integer), f"{integer:,}"}
    return list(tokens)


@dataclass(slots=True)
class BenchmarkCase:
    """Single benchmark prompt + reference query."""

    prompt: str
    spec: QuerySpec
    expectation: Callable[[pd.DataFrame], list[str]]
    description: str


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        prompt="What was the closing price of BTC on Oct 15, 2025?",
        description="BTC closing price on 2025-10-15",
        spec=QuerySpec(
            table="candles",
            columns=("coin", "close"),
            filters={
                "coin": ["BTC"],
                "timeframe": "1d",
                "ts_start": "2025-10-15",
                "ts_end": "2025-10-15",
            },
            limit=1,
        ),
        expectation=lambda df: _numeric_tokens(df.iloc[0]["close"]) if not df.empty else [],
    ),
    BenchmarkCase(
        prompt="Show me the trading volume for ETH on Nov 1, 2025.",
        description="ETH volume on 2025-11-01",
        spec=QuerySpec(
            table="candles",
            columns=("coin", "volume"),
            filters={
                "coin": ["ETH"],
                "timeframe": "1d",
                "ts_start": "2025-11-01",
                "ts_end": "2025-11-01",
            },
            limit=1,
        ),
        expectation=lambda df: _numeric_tokens(df.iloc[0]["volume"]) if not df.empty else [],
    ),
    BenchmarkCase(
        prompt="Which asset had the highest high on Oct 20, 2025: SOL or ADA?",
        description="Compare SOL vs ADA highs on 2025-10-20",
        spec=QuerySpec(
            table="candles",
            columns=("coin", "high"),
            filters={
                "coin": ["SOL", "ADA"],
                "timeframe": "1d",
                "ts_start": "2025-10-20",
                "ts_end": "2025-10-20",
            },
            limit=10,
        ),
        expectation=lambda df: _winner_tokens(df, column="high"),
    ),
    BenchmarkCase(
        prompt="What was the RSI value for BTC on Oct 15, 2025?",
        description="BTC RSI on 2025-10-15",
        spec=QuerySpec(
            table="candles",
            columns=("coin", "rsi"),
            filters={
                "coin": ["BTC"],
                "timeframe": "1d",
                "ts_start": "2025-10-15",
                "ts_end": "2025-10-15",
            },
            limit=1,
        ),
        expectation=lambda df: _decimal_tokens(df.iloc[0]["rsi"]) if not df.empty else [],
    ),
)


def _winner_tokens(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df:
        return []
    winner = df.sort_values(column, ascending=False).iloc[0]["coin"]
    return [winner, winner.lower()]


def _decimal_tokens(value: float) -> list[str]:
    rounded = round(float(value), 2)
    as_str = f"{rounded:.2f}"
    return [as_str, as_str.rstrip("0").rstrip(".")]


def run_case(
    pipeline: RuntimePipeline,
    duckdb_client: DuckDBClient,
    case: BenchmarkCase,
) -> dict[str, object]:
    start = time.perf_counter()
    result = pipeline.run(case.prompt)
    latency = time.perf_counter() - start

    reference = duckdb_client.execute(case.spec).payload
    expected_tokens = case.expectation(reference)
    response_text = result.response_text.lower()
    hits = sum(1 for token in expected_tokens if token.lower() in response_text)
    accuracy = hits / len(expected_tokens) if expected_tokens else 1.0

    return {
        "prompt": case.prompt,
        "description": case.description,
        "latency_sec": latency,
        "expected_tokens": expected_tokens,
        "accuracy": accuracy,
        "response_sample": result.response_text[:2000],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runtime latency and simple accuracy heuristics.")
    parser.add_argument("--model-type", choices=("large", "small"), default="large", help="RuntimePipeline model type.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the JSON results (defaults to results/benchmarks/<model>_metrics.json).",
    )
    args = parser.parse_args()

    config = load_runtime_config()
    pipeline = RuntimePipeline(model_type=args.model_type, config=config)
    duckdb_client = DuckDBClient(config.duckdb)

    case_reports = [run_case(pipeline, duckdb_client, case) for case in CASES]
    latencies = [entry["latency_sec"] for entry in case_reports]
    accuracies = [entry["accuracy"] for entry in case_reports]

    summary = {
        "model_type": args.model_type,
        "cases": case_reports,
        "average_latency_sec": statistics.mean(latencies) if latencies else None,
        "average_accuracy": statistics.mean(accuracies) if accuracies else None,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("results/benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.model_type}_metrics.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Saved benchmark report to {output_path}")


if __name__ == "__main__":
    main()
