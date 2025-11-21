"""Generate synthetic training dataset using the Large Model pipeline."""
import json
import os
import sys
import argparse
from pathlib import Path
from typing import List

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from terminalc.runtime_core.pipelines.runtime_pipeline import RuntimePipeline
from terminalc.runtime_core.config import load_runtime_config

# Seed prompts covering various intents
SEED_PROMPTS = [
    # Market Data
    "What is the price of BTC?",
    "Current volume for ETH?",
    "How is SOL performing today?",
    "Show me the 24h change for ADA.",
    "What is the high and low for XRP yesterday?",
    "Is DOGE up or down this week?",
    "Give me the latest candle data for AVAX.",
    "What's the market cap trend for DOT?",
    "Price action for LINK over the last 3 days.",
    "Did MATIC break its resistance?",

    # Technical Indicators
    "What is the RSI for BTC?",
    "Is ETH overbought or oversold?",
    "Show me the MACD signal for SOL.",
    "Are there any bullish divergences on ADA?",
    "Check for bearish signals on XRP daily chart.",
    "What does the Bollinger Band say about DOGE?",
    "Is the 50-day MA above the 200-day MA for AVAX?",
    "Any golden cross on DOT?",
    "What is the ATR for LINK?",
    "Is momentum positive for MATIC?",

    # News & Sentiment
    "Summarize the latest news for Bitcoin.",
    "What is the sentiment around Ethereum right now?",
    "Any major headlines for Solana?",
    "Why is Cardano moving today?",
    "News impacting Ripple recently.",
    "Is there any FUD around Dogecoin?",
    "What are analysts saying about Avalanche?",
    "Latest regulatory news for Polkadot.",
    "Chainlink partnership news.",
    "Polygon network updates.",

    # Combined / Complex
    "Analyze BTC using both price and RSI.",
    "What is the outlook for ETH based on news and technicals?",
    "Is SOL a buy based on recent momentum and volume?",
    "Compare the performance of ADA and XRP this week.",
    "Find me a coin with low RSI and positive news.",
    "Which asset has the highest volatility today?",
    "Is the trend for DOGE supported by volume?",
    "What are the risks for AVAX right now?",
    "Give me a technical summary for DOT.",
    "Explain the recent price drop in MATIC."
]

def generate_dataset(output_file: str, num_samples: int = None):
    """Run prompts through the pipeline and save prompt-completion pairs."""
    print(f"Initializing pipeline with Large Model...")
    rp = RuntimePipeline(model_type="large")
    
    prompts = SEED_PROMPTS
    if num_samples:
        prompts = prompts[:num_samples]

    data = []
    print(f"Generating data for {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Processing: {prompt}")
        try:
            # Run pipeline
            result, payload = rp.run(prompt, return_payload=True)
            
            # Create training example
            example = {
                "prompt": payload.instructions,
                "completion": result.response_text,
                "model": result.model_name
            }
            data.append(example)
            
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")

    # Save to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
            
    print(f"Saved {len(data)} examples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--output", type=str, default="data/training_data.jsonl", help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to generate")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.samples)
