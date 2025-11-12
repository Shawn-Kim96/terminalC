"""Concrete LLM clients that satisfy the RuntimePipeline protocol."""
from __future__ import annotations
from pathlib import Path
import os
from typing import Any, Sequence
import requests
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from terminalc.runtime_core.models.runtime_models import LLMResult, PromptPayload


class HuggingFaceInferenceClient:
    """Wrapper around the Hugging Face router chat completions API."""

    def __init__(
        self,
        model_id: str,
        api_token: str | None = None,
        timeout: int = 60,
        max_new_tokens: int = 512,
        base_url: str = "https://router.huggingface.co/v1/chat/completions",
    ) -> None:
        self._endpoint = base_url
        self._model_id = model_id
        self._api_token = api_token or os.getenv("HUGGINGFACE_TOKEN")
        if not self._api_token:
            raise EnvironmentError("HUGGINGFACE_TOKEN is not set")
        self._timeout = timeout
        self._max_new_tokens = max_new_tokens

    def generate(self, payload: PromptPayload) -> LLMResult:
        headers = {"Authorization": f"Bearer {self._api_token}"}
        body = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": payload.instructions,
                }
            ],
            "max_tokens": self._max_new_tokens,
        }
        response = requests.post(
            self._endpoint,
            headers=headers,
            json=body,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content", "")
        else:
            text = data.get("error", {}).get("message", "")
        return LLMResult(response_text=text or "", model_name=self._model_id, total_tokens=data.get("usage", {}).get("total_tokens"))


class LocalTransformersClient:
    """Run inference using a locally downloaded Hugging Face model."""

    def __init__(
        self,
        model_path: str = "",
        device: int | None = None,
        max_new_tokens: int = 512,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Local model path '{path}' does not exist.")

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto")
        pipeline_device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=pipeline_device,
        )
        self._max_new_tokens = max_new_tokens
        self._generation_kwargs = generation_kwargs or {"do_sample": False, "temperature": 0.7}
        self._model_name = str(path)

    def generate(self, payload: PromptPayload) -> LLMResult:
        outputs: Sequence[dict[str, Any]] = self._generator(
            payload.instructions,
            max_new_tokens=self._max_new_tokens,
            pad_token_id=self._generator.tokenizer.pad_token_id,
            **self._generation_kwargs,
        )
        text = outputs[0].get("generated_text", "") if outputs else ""
        return LLMResult(response_text=text, model_name=self._model_name, total_tokens=None)
