"""
data_loader.py
--------------
Load data from CSV, API, or Postgres (Neon) and provide small helpers used across the project.

Design notes:
- Keep I/O in this module only (separation of concerns).
- Provide simple, explicit functions so notebooks and pipelines stay clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pandas as pd

# Optional imports (only needed for DB / API)
try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from sqlalchemy import create_engine
except Exception:  # pragma: no cover
    create_engine = None  # type: ignore


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """Load robot sensor data from a local CSV."""
    return pd.read_csv(csv_path)


def load_from_api(api_endpoint: str, timeout_s: int = 30) -> pd.DataFrame:
    """
    Load JSON data from an API endpoint into a DataFrame.

    Expected API response:
      - Either a list[dict] records, or a dict with a "data" field containing list[dict].
    """
    if requests is None:
        raise ImportError("requests is required for API loading. Install with: pip install requests")

    resp = requests.get(api_endpoint, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]

    if not isinstance(payload, list):
        raise ValueError("API response must be a list of records or a dict containing a 'data' list.")

    return pd.DataFrame(payload)


def save_to_db(df: pd.DataFrame, db_conn_str: str, table_name: str, if_exists: str = "replace") -> None:
    """Save a DataFrame to Postgres (Neon) using SQLAlchemy."""
    if create_engine is None:
        raise ImportError("sqlalchemy is required for DB operations. Install with: pip install sqlalchemy")
    engine = create_engine(db_conn_str)
    # index=False so the dataframe index isn't stored as a column
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)


def load_from_db(db_conn_str: str, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Read a table from Postgres (Neon) into a DataFrame."""
    if create_engine is None:
        raise ImportError("sqlalchemy is required for DB operations. Install with: pip install sqlalchemy")
    engine = create_engine(db_conn_str)
    q = f"SELECT * FROM {table_name}"
    if limit is not None:
        q += f" LIMIT {int(limit)}"
    return pd.read_sql(q, engine)


def make_multi_robot(df: pd.DataFrame, robot_ids: Optional[List[str]] = None, seed: int = 42) -> pd.DataFrame:
    """
    Requirement helper: create at least 3 "robot" sources using one dataset.

    This duplicates the dataset 3 times with small scaling + noise and adds a 'robot_id' column.
    Useful when you only have one robot export file.
    """
    import numpy as np

    if robot_ids is None:
        robot_ids = ["R1", "R2", "R3"]

    rng = np.random.default_rng(seed)

    numeric_cols = [c for c in df.columns if c.startswith("Axis #")]
    out = []
    for i, rid in enumerate(robot_ids):
        dfi = df.copy()
        # Small per-robot scale factor (e.g., 0.98, 1.00, 1.02)
        scale = 1.0 + (i - 1) * 0.02
        noise = rng.normal(loc=0.0, scale=0.01, size=(len(dfi), len(numeric_cols)))
        dfi[numeric_cols] = dfi[numeric_cols].astype(float) * scale + noise
        dfi["robot_id"] = rid
        out.append(dfi)

    return pd.concat(out, ignore_index=True)
