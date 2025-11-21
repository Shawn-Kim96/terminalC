"""DuckDB access layer."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence
import duckdb
import pandas as pd
import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from terminalc.runtime_core.config import DuckDBConfig
from terminalc.runtime_core.data_access.schema import DUCKDB_TABLE_COLUMNS
from terminalc.runtime_core.models.runtime_models import DataSnapshot, QuerySpec


class DuckDBClient:
    _RANGE_FILTERS: Mapping[str, tuple[str, str]] = {
        "ts_start": ("ts", ">="),
        "ts_end": ("ts", "<="),
        "evaluated_at_start": ("evaluated_at", ">="),
        "evaluated_at_end": ("evaluated_at", "<="),
        "published_start": ("published_at", ">="),
        "published_end": ("published_at", "<="),
    }

    def __init__(self, config: DuckDBConfig) -> None:
        self._config = config

    def compile(self, spec: QuerySpec) -> tuple[str, Sequence[Any], str]:
        query, params, hash_bindings = self._build_query(spec)
        cache_key = self._hash_query(query, hash_bindings)
        return query, params, cache_key

    def execute(self, spec: QuerySpec) -> DataSnapshot:
        query, params, cache_key = self.compile(spec)
        with duckdb.connect(str(self._config.database_path), read_only=self._config.read_only) as conn:
            df: pd.DataFrame = conn.execute(query, params).fetch_df()
        return DataSnapshot(spec=spec, row_count=len(df), payload=df, cache_key=cache_key)

    def _build_query(self, spec: QuerySpec) -> tuple[str, Sequence[Any], Sequence[tuple[str, Any]]]:
        table_columns = DUCKDB_TABLE_COLUMNS.get(spec.table)
        if table_columns is None:
            raise ValueError(f"Unsupported table '{spec.table}'.")

        requested_columns: tuple[str, ...]
        if not spec.columns:
            requested_columns = table_columns
        elif len(spec.columns) == 1 and spec.columns[0] == "*":
            requested_columns = table_columns
        else:
            requested_columns = tuple(spec.columns)

        invalid_columns = [col for col in requested_columns if col not in table_columns]
        if invalid_columns:
            raise ValueError(
                f"Columns {invalid_columns} are not available on table '{spec.table}'."
            )

        valid_filter_keys = set(table_columns) | set(self._RANGE_FILTERS.keys())
        invalid_filters = [key for key in spec.filters if key not in valid_filter_keys]
        if invalid_filters:
            raise ValueError(
                f"Filters {invalid_filters} are not valid for table '{spec.table}'."
            )

        where_clause: list[str] = []
        param_values: list[Any] = []
        hash_bindings: list[tuple[str, Any]] = []
        param_counts: dict[str, int] = defaultdict(int)

        def next_param(base: str) -> str:
            sanitized = base.replace('"', "").replace(".", "_")
            sanitized = sanitized or "param"
            count = param_counts[sanitized]
            param_counts[sanitized] += 1
            return sanitized if count == 0 else f"{sanitized}_{count}"

        def bind(name: str, val: Any) -> str:
            param_values.append(val)
            hash_bindings.append((name, val))
            return "?"

        def filter_sort_key(key: str) -> tuple[int, str]:
            priority_map = {
                "timeframe": 0,
                "ts_start": 1,
                "ts_end": 2,
                "evaluated_at_start": 1,
                "evaluated_at_end": 2,
                "published_start": 1,
                "published_end": 2,
            }
            if key in priority_map:
                return priority_map[key], key
            if key.endswith("_start"):
                return 3, key
            if key.endswith("_end"):
                return 4, key
            return 5, key
        def is_like_value(val: Any) -> bool:
            return isinstance(val, str) and "%" in val

        for key in sorted(spec.filters.keys(), key=filter_sort_key):
            value = spec.filters[key]
            if value is None:
                continue

            if key in self._RANGE_FILTERS:
                column, operator = self._RANGE_FILTERS[key]
                param_name = next_param(key)
                placeholder = bind(param_name, value)
                where_clause.append(f"{column} {operator} {placeholder}")
                continue

            if isinstance(value, str) and is_like_value(value):
                param_name = next_param(key)
                placeholder = bind(param_name, value)
                where_clause.append(f"{key} ILIKE {placeholder}")
                continue

            if isinstance(value, (list, tuple, set)) and value:
                like_values = [v for v in value if is_like_value(v)]
                normal_values = [v for v in value if not is_like_value(v)]
                if like_values:
                    comparisons = []
                    for i, val in enumerate(like_values):
                        param_name = next_param(f"{key}_{i}")
                        placeholder = bind(param_name, val)
                        comparisons.append(f"{key} ILIKE {placeholder}")
                    like_clause = "(" + " OR ".join(comparisons) + ")"
                    if normal_values:
                        placeholders = []
                        for i, val in enumerate(normal_values):
                            holder = next_param(f"{key}_eq_{i}")
                            placeholders.append(bind(holder, val))
                        equality_clause = f"{key} IN ({', '.join(placeholders)})"
                        where_clause.append(f"({like_clause} OR {equality_clause})")
                    else:
                        where_clause.append(like_clause)
                    continue

            if isinstance(value, (list, tuple, set)):
                placeholders = []
                for i, val in enumerate(value):
                    holder = next_param(f"{key}_{i}")
                    placeholders.append(bind(holder, val))
                placeholder_sql = f"({', '.join(placeholders)})"
                where_clause.append(f"{key} IN {placeholder_sql}")
            else:
                param_name = next_param(key)
                placeholder = bind(param_name, value)
                where_clause.append(f"{key} = {placeholder}")

        where_sql = " AND ".join(where_clause)
        limit_sql = f" LIMIT {spec.limit}" if spec.limit else ""
        query = f"SELECT {', '.join(requested_columns)} FROM {spec.table}"
        if where_sql:
            query += f" WHERE {where_sql}"
        query += limit_sql
        return query, tuple(param_values), tuple(hash_bindings)

    @staticmethod
    def _hash_query(query: str, params: Sequence[tuple[str, Any]]) -> str:
        payload = json.dumps({"query": query, "params": list(params)}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
