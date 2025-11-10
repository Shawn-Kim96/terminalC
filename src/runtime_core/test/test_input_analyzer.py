"""Quick CLI to inspect InputAnalyzer outputs for sample prompts."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)
from src.runtime_core.input_parser.analyzer import InputAnalyzer

DEFAULT_PROMPTS = [
    "Which assets delivered the highest and lowest returns at Oct 15, 2025?",
    "What coins are attractive to buy in terms of technical indicators?",
    "Summarize BTC news this week and tell me if the sentiment is positive.",
    "What momentum strategy rules are available for ETH right now?",
]

def serialize_intent(intent) -> dict:
    payload = {
        "name": intent.name,
        "confidence": round(intent.confidence, 4),
        "parameters": intent.parameters,
    }
    if intent.slots:
        payload["slots"] = asdict(intent.slots)
    else:
        payload["slots"] = None
    return payload


def test_input_analyzer(prompts: List[str] = None) -> None:
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    
    analyzer = InputAnalyzer()
    for prompt in prompts:
        intent = analyzer.analyze(prompt)
        print(f"\n=== Prompt ===\n{prompt}")
        print("--- Intent ---")
        print(json.dumps(serialize_intent(intent), indent=4))
