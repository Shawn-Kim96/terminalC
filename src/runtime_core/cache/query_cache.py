"""Query result cache that stores pandas DataFrames on disk."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import os
import sys
import pandas as pd

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from src.runtime_core.models.runtime_models import CacheRecord, DataSnapshot


class QueryCache:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        path = self._resolve_path(cache_key)
        if not os.path.exists(path):
            return None
        return pd.read_pickle(path)

    def store(self, snapshot: DataSnapshot) -> CacheRecord:
        if not snapshot.cache_key:
            raise ValueError("Snapshot missing cache key")
        path = self._resolve_path(snapshot.cache_key)
        snapshot.payload.to_pickle(path)
        return CacheRecord(
            key=snapshot.cache_key,
            path=path,
            created_at=datetime.now(timezone.utc),
            metadata={"row_count": snapshot.row_count, "table": snapshot.spec.table},
        )

    def _resolve_path(self, cache_key: str) -> Path:
        return os.path.join(self._cache_dir, f"{cache_key}.pkl")


__all__ = ["QueryCache"]
