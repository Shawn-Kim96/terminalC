"""Concrete LLM clients that satisfy the RuntimePipeline protocol."""
from __future__ import annotations
from pathlib import Path
import os
from typing import Any
import requests
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM

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

    def generate(
        self,
        payload: PromptPayload,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> LLMResult:
        headers = {"Authorization": f"Bearer {self._api_token}"}
        model_id = self._model_id if self._model_id.split(":")[-1] == "cheapest" else f"{self._model_id}:cheapest"
        extra = dict(generation_kwargs or {})
        if "max_tokens" in extra:
            max_tokens = extra.pop("max_tokens")
        elif "max_new_tokens" in extra:
            max_tokens = extra.pop("max_new_tokens")
        else:
            max_tokens = self._max_new_tokens
        body = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": payload.instructions,
                }
            ],
            "max_tokens": max_tokens,
        }
        body.update(extra)
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
        return LLMResult(response_text=text or "", model_name=model_id, total_tokens=data.get("usage", {}).get("total_tokens"))


class LocalTransformersClient:
    """Run inference using a locally downloaded Hugging Face model."""

    def __init__(
        self,
        model_path: str = "",
        adapter_path: str | None = None,
        device: str | int | torch.device | None = None,
        max_new_tokens: int = 512,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Local model path '{path}' does not exist.")
        self._adapter_path = Path(adapter_path).resolve() if adapter_path else None
        if self._adapter_path and not self._adapter_path.exists():
            raise FileNotFoundError(f"Adapter path '{self._adapter_path}' does not exist.")

        self._device, dtype = self._resolve_device(device)
        model_device_map: str | dict[str, str]
        if isinstance(self._device, str) and self._device.startswith("cuda:"):
            model_device_map = {"": self._device}
        else:
            model_device_map = self._device

        tokenizer_source = self._adapter_path or path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map=model_device_map)
        if self._adapter_path:
            try:
                base_model = PeftModel.from_pretrained(base_model, str(self._adapter_path))
            except Exception:
                base_model = AutoPeftModelForCausalLM.from_pretrained(
                    str(self._adapter_path),
                    torch_dtype=dtype,
                    device_map=model_device_map,
                )
        self._model = base_model.eval()
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._generation_kwargs = generation_kwargs or {"do_sample": False, "temperature": 0.7}
        self._model_name = str(path)
        self._pad_token_id = self._tokenizer.pad_token_id

    def generate(
        self,
        payload: PromptPayload,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> LLMResult:
        gen_kwargs = dict(self._generation_kwargs)
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)
        max_tokens = gen_kwargs.pop("max_new_tokens", gen_kwargs.pop("max_tokens", self._max_new_tokens))

        # Apply chat template if available (for instruct models like Llama 3.1)
        if hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": payload.instructions}]
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = payload.instructions

        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(self._device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self._pad_token_id,
                **gen_kwargs,
            )
        # Decode only the newly generated tokens so we don't echo the prompt
        generated = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        model_id = f"{self._model_name} (adapter={self._adapter_path})" if self._adapter_path else self._model_name
        return LLMResult(response_text=text, model_name=model_id, total_tokens=None)

    def _resolve_device(self, override: str | int | torch.device | None) -> tuple[str, torch.dtype]:
        """Map user/device availability to a concrete torch device and dtype."""
        # Normalize overrides: torch.device -> str, int -> cuda index (if possible)
        normalized: str | None = None
        if isinstance(override, torch.device):
            normalized = str(override)
        elif isinstance(override, str):
            normalized = override
        elif isinstance(override, int):
            normalized = f"cuda:{override}" if torch.cuda.is_available() else "cpu"

        if normalized:
            if normalized.startswith("cuda") and torch.cuda.is_available():
                return normalized, torch.float16
            if normalized == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps", torch.float16
            if normalized == "cpu":
                return "cpu", torch.float32

        if torch.cuda.is_available():
            return "cuda", torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32
