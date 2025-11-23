"""Fine-tune a small model using LoRA and the generated dataset."""
import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath(".").split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

# Stream logs promptly (useful for sbatch)
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def train_model(
    base_model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    hf_token: str | None = None,
    local_files_only: bool = True,
    max_length: int = 256,
    gradient_accumulation_steps: int = 1,
):
    print(f"Loading model: {base_model_name}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=token,
        padding_side="right",
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        token=token,
        local_files_only=local_files_only,
        device_map="auto" if device != "cpu" else None,
    )
    if device == "cpu":
        model.to(device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Dataset
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Format function
    def format_prompts(examples):
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Simple chat format
            text = f"<|user|>\n{prompt}\n<|assistant|>\n{completion}"
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format(type="torch")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_strategy="epoch",  # log once per epoch for clean logs
        save_strategy="epoch",
        fp16=(device == "cuda"),
        bf16=False,
        use_mps_device=(device == "mps"),
        report_to="none",
        disable_tqdm=False,  # keep progress bar visible in logs
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train small model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="data/model/meta-llama/llama-3.1-8B-instruct",
        help="Base model name or local path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training_data.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/small_model_lora",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HF token for gated/private models (or set HF_TOKEN/HUGGINGFACEHUB_API_TOKEN)",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force loading from local files/cache only (use --no-local-files-only to allow downloads)",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Tokenization max length")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    train_model(
        args.base_model,
        args.data,
        args.output,
        args.epochs,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum,
    )
