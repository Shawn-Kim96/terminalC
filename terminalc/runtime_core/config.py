"""Central configuration objects for the runtime pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class DuckDBConfig:
    """DuckDB connection information."""

    database_path: Path
    read_only: bool = True


@dataclass(slots=True)
class CacheConfig:
    """Where to persist cached artifacts."""

    base_dir: Path
    query_cache_dir: Path
    prompt_cache_dir: Path


@dataclass(slots=True)
class ModelConfig:
    """Large/small model configuration."""

    large_model_endpoint: str | None
    small_model_endpoint: str | None
    local_model_dir: Path
    huggingface_token: str | None = None


@dataclass(slots=True)
class RuntimeConfig:
    """Aggregated config consumed by the runtime pipeline."""

    duckdb: DuckDBConfig
    cache: CacheConfig
    models: ModelConfig


def load_runtime_config() -> RuntimeConfig:
    """Create a RuntimeConfig from environment variables and repo defaults."""

    data_dir = os.path.join(PROJECT_DIR, os.getenv("TERMINALC_DATA_SUB_DIR"))
    cache_base_dir = os.path.join(PROJECT_DIR, os.getenv("TERMINALC_CACHE_SUB_DIR"))
    models_dir = Path(os.path.join(PROJECT_DIR, os.getenv("TERMINALC_MODELS_SUB_DIR")))

    duckdb_config = DuckDBConfig(
        database_path=Path(
            os.path.join(PROJECT_DIR, os.getenv("TERMINALC_DUCKDB_SUB_PATH"))
        )
    )

    cache_config = CacheConfig(
        base_dir=os.path.join(PROJECT_DIR, os.getenv("TERMINALC_CACHE_SUB_DIR")),
        query_cache_dir=os.path.join(PROJECT_DIR, os.getenv("TERMINALC_CACHE_SUB_DIR"), "query"),
        prompt_cache_dir=os.path.join(PROJECT_DIR, os.getenv("TERMINALC_CACHE_SUB_DIR"), "prompt")
    )

    model_config = ModelConfig(
        large_model_endpoint=os.getenv("LARGE_MODEL_ENDPOINT"),
        small_model_endpoint=os.getenv("SMALL_MODEL_ENDPOINT", "sentence-transformers/all-MiniLM-L6-v2"),
        local_model_dir=models_dir.resolve(),
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
    )

    return RuntimeConfig(
        duckdb=duckdb_config,
        cache=cache_config,
        models=model_config,
    )


__all__ = [
    "DuckDBConfig",
    "CacheConfig",
    "ModelConfig",
    "RuntimeConfig",
    "load_runtime_config",
]
