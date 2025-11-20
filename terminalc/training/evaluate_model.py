"""Evaluate the fine-tuned small model."""
import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

TEST_PROMPTS = [
    "What is the current price of BTC?",
    "Is ETH showing bullish signals?",
    "Summarize the latest news for Solana.",
    "What is the RSI for ADA?",
    "Compare BTC and ETH performance."
]

def evaluate(base_model_name: str, adapter_path: str):
    print(f"Loading base model: {base_model_name}")
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("\n=== Evaluation Results ===")
    for prompt in TEST_PROMPTS:
        input_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant part if possible, or print whole thing
        print(f"\nPrompt: {prompt}")
        print("-" * 20)
        print(response.replace(input_text, "").strip())
        print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate small model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name")
    parser.add_argument("--adapter", type=str, default="models/small_model_lora", help="Path to LoRA adapter")
    args = parser.parse_args()

    evaluate(args.base_model, args.adapter)
