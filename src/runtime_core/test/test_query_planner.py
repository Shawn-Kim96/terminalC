"""Quick CLI to inspect QueryOrchestrator outputs for sample prompts."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import List
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)
from src.runtime_core.query_planner.planner import QueryOrchestrator
from src.runtime_core.input_parser.analyzer import InputAnalyzer

DEFAULT_PROMPTS = [
    "Which assets delivered the highest and lowest returns at Oct 15, 2025?",
    "What coins are attractive to buy in terms of technical indicators?",
    "Summarize BTC news this week and tell me if the sentiment is positive.",
    "What momentum strategy rules are available for ETH right now?",
]

def serialize_query_plan(query_plan) -> dict:
    return {
        "specs": [
            {
                "table": spec.table,
                "columns": list(spec.columns),
                "filters": spec.filters,
                "limit": spec.limit,
            }
            for spec in query_plan.specs
        ],
        "plan_steps": list(query_plan.plan_steps),
    }


def test_query_planner(prompts: List[str] = None) -> None:
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    analyzer = InputAnalyzer()
    qo = QueryOrchestrator()
    for prompt in prompts:
        intent = analyzer.analyze(prompt)
        query_plan = qo.build_plan(intent)
        print(f"\n=== Prompt ===\n{prompt}")
        print("--- Query Plan ---")
        print(json.dumps(serialize_query_plan(query_plan), indent=4))
