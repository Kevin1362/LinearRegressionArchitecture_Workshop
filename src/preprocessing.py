"""
preprocessing.py
----------------
Cleaning and feature engineering helpers.

This module should NOT do any file/network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def clean_robot_current_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the robot current export:
    - Parse Time as datetime
    - Coerce axis columns to numeric
    - Handle missing values by median fill
    """
    out = df.copy()

    if "Time" in out.columns:
        out["Time"] = pd.to_datetime(out["Time"], errors="coerce", utc=True)

    axis_cols = [c for c in out.columns if c.startswith("Axis #")]
    for c in axis_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # fill numeric NaNs with median; if an entire column is missing, fall back to 0.0
    for c in axis_cols:
        if out[c].isna().all():
            med = 0.0
        else:
            med = float(out[c].median())
        out[c] = out[c].fillna(med)


    # drop rows without a timestamp if Time exists
    if "Time" in out.columns:
        out = out.dropna(subset=["Time"])

    return out


def add_days_until_failure_target(df: pd.DataFrame, time_col: str = "Time", target_col: str = "days_until_failure") -> pd.DataFrame:
    """
    Create a regression target from the time series.

    We define "failure" as the last timestamp in the dataset (proxy for "end of run").
    Target is: (failure_time - current_time) in days.

    This gives a continuous value suitable for univariate linear regression, and supports the
    "alert 2 weeks before failure" idea in Session 2.
    """
    out = df.copy()
    if time_col not in out.columns:
        raise ValueError(f"Missing '{time_col}' column. Can't compute target.")

    from pandas.api.types import is_datetime64_any_dtype

    if not is_datetime64_any_dtype(out[time_col]):
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)

    failure_time = out[time_col].max()
    delta = (failure_time - out[time_col]).dt.total_seconds() / (60 * 60 * 24)
    out[target_col] = delta.astype(float)
    return out


def select_feature_target(df: pd.DataFrame, feature_col: str, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    if feature_col not in df.columns:
        raise ValueError(f"Feature '{feature_col}' not found. Available: {list(df.columns)[:20]} ...")
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found. Available: {list(df.columns)[:20]} ...")
    X = df[[feature_col]].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_univariate(X_train: np.ndarray, X_test: np.ndarray):
    """
    Standardize a single feature using training mean/std.
    Returns standardized arrays + (mean, std) for reproducibility.
    """
    mu = float(X_train.mean())
    sigma = float(X_train.std()) if float(X_train.std()) > 0 else 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma
