"""Concrete LLM clients that satisfy the RuntimePipeline protocol."""
from __future__ import annotations

import os
from typing import Any
import requests
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from src.runtime_core.models.runtime_models import LLMResult, PromptPayload


class HuggingFaceInferenceClient:
    """Thin wrapper around the Hugging Face text generation inference API."""

    def __init__(self, model_id: str, api_token: str | None = None, timeout: int = 60) -> None:
        self._endpoint = f"https://api-inference.huggingface.co/models/{model_id}"
        self._api_token = api_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self._api_token:
            raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN is not set")
        self._timeout = timeout

    def generate(self, payload: PromptPayload) -> LLMResult:
        headers = {"Authorization": f"Bearer {self._api_token}"}
        response = requests.post(
            self._endpoint,
            headers=headers,
            json={"inputs": payload.instructions, "parameters": {"max_new_tokens": 512}},
            timeout=self._timeout,
        )
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json()
        text = data[0].get("generated_text") if data else response.text
        return LLMResult(response_text=text or "", model_name=self._endpoint, total_tokens=None)
