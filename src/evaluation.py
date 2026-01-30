"""
evaluation.py
-------------
Metrics, plotting, and experiment tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import csv
import os
from datetime import datetime

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple R² implementation to avoid extra dependencies inside this module.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)


def plot_regression_line(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, title: str, save_path: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    # Sort for a clean line
    order = np.argsort(X[:, 0])
    Xs = X[order, 0]
    ys = y[order]
    yps = y_pred[order]

    plt.figure()
    plt.scatter(Xs, ys, s=8)
    plt.plot(Xs, yps)
    plt.title(title)
    plt.xlabel("Feature (standardized if enabled)")
    plt.ylabel("Days until failure (proxy)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def append_results_csv(
    results_csv_path: str,
    row: Dict[str, Any],
    fieldnames: Optional[list] = None,
) -> None:
    """
    Append a row to experiments/results.csv. Create file + header if missing.
    """
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

    if fieldnames is None:
        fieldnames = list(row.keys())

    file_exists = os.path.exists(results_csv_path)
    with open(results_csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
