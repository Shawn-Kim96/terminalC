"""Cache LLM prompts/responses on disk for reuse."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from src.runtime_core.models.runtime_models import CacheRecord, LLMResult, PromptPayload


class PromptCache:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    def build_key(self, payload: PromptPayload) -> str:
        digest_input = json.dumps(
            {
                "template_id": payload.template_id,
                "instructions": payload.instructions,
                "context_blocks": payload.context_blocks,
                "metadata": payload.metadata,
            },
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        return f"{payload.template_id}:{digest}"

    def get(self, key: str) -> Optional[LLMResult]:
        path = self._resolve_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return LLMResult(
            response_text=data["response_text"],
            model_name=data["model_name"],
            total_tokens=data.get("total_tokens"),
            cached=True,
        )

    def store(self, key: str, result: LLMResult) -> CacheRecord:
        path = self._resolve_path(key)
        payload = {
            "response_text": result.response_text,
            "model_name": result.model_name,
            "total_tokens": result.total_tokens,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        return CacheRecord(
            key=key,
            path=path,
            created_at=datetime.now(timezone.utc),
            metadata={"model": result.model_name},
        )

    def _resolve_path(self, key: str) -> Path:
        safe_key = key.replace("/", "_")
        return os.path.join(self._cache_dir, f"{safe_key}.pkl")
