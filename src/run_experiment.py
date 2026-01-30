"""
run_experiment.py
-----------------
Single entrypoint for running an experiment end-to-end using a YAML config.

Usage:
  python -m src.run_experiment --config configs/experiment_config.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime

import yaml
import numpy as np

from .data_loader import load_from_csv, load_from_api, load_from_db, make_multi_robot
from .preprocessing import clean_robot_current_df, add_days_until_failure_target, select_feature_target, train_test_split, standardize_univariate
from .model import ScratchLinearRegression, fit_sklearn_linear_regression
from .evaluation import rmse, mae, r2_score, plot_regression_line, append_results_csv


def _load_data(cfg: dict):
    src = cfg["data"]["source"]
    if src == "csv":
        return load_from_csv(cfg["data"]["csv_path"])
    if src == "api":
        return load_from_api(cfg["data"]["api_endpoint"])
    if src == "db":
        return load_from_db(cfg["data"]["db_conn_str"], cfg["data"]["db_table"], limit=cfg["data"].get("db_limit"))
    raise ValueError("data.source must be one of: csv, api, db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = _load_data(cfg)

    # Optional: make 3 robots if needed
    if cfg["data"].get("make_multi_robot", True):
        df = make_multi_robot(df, robot_ids=cfg["data"].get("robot_ids", ["R1", "R2", "R3"]), seed=int(cfg["experiment"].get("seed", 42)))

    # Clean + create target
    df = clean_robot_current_df(df)
    df = add_days_until_failure_target(df, target_col=cfg["data"]["target_col"])

    # Select feature/target
    X, y = select_feature_target(df, cfg["data"]["feature_col"], cfg["data"]["target_col"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(cfg["training"]["test_size"]),
        seed=int(cfg["experiment"].get("seed", 42)),
    )

    # Standardize
    if cfg["training"].get("standardize", True):
        X_train, X_test, mu, sigma = standardize_univariate(X_train, X_test)
    else:
        mu, sigma = 0.0, 1.0

    # Train scratch
    scratch = ScratchLinearRegression(
        learning_rate=float(cfg["training"]["learning_rate"]),
        n_iters=int(cfg["training"]["iterations"]),
    ).fit(X_train, y_train)
    y_pred_scratch = scratch.predict(X_test)

    # Train sklearn
    sk = fit_sklearn_linear_regression(X_train, y_train)
    y_pred_sk = sk.predict(X_test)

    # Metrics
    metrics = {
        "scratch_rmse": rmse(y_test, y_pred_scratch),
        "scratch_mae": mae(y_test, y_pred_scratch),
        "scratch_r2": r2_score(y_test, y_pred_scratch),
        "sklearn_rmse": rmse(y_test, y_pred_sk),
        "sklearn_mae": mae(y_test, y_pred_sk),
        "sklearn_r2": r2_score(y_test, y_pred_sk),
    }

    # Save plots
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_regression_line(X_test, y_test, y_pred_scratch, f"Scratch LR ({cfg['data']['feature_col']} → {cfg['data']['target_col']})",
                         save_path=f"experiments/plot_scratch_{ts}.png")
    plot_regression_line(X_test, y_test, y_pred_sk, f"Sklearn LR ({cfg['data']['feature_col']} → {cfg['data']['target_col']})",
                         save_path=f"experiments/plot_sklearn_{ts}.png")

    # Track experiments
    results_path = cfg["experiment"]["results_csv"]
    base_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_source": cfg["data"]["source"],
        "feature": cfg["data"]["feature_col"],
        "target": cfg["data"]["target_col"],
        "standardize": bool(cfg["training"].get("standardize", True)),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "iterations": int(cfg["training"]["iterations"]),
        "test_size": float(cfg["training"]["test_size"]),
        "seed": int(cfg["experiment"].get("seed", 42)),
        "mu": float(mu),
        "sigma": float(sigma),
    }

    append_results_csv(results_path, {**base_row, "model": "scratch", "rmse": metrics["scratch_rmse"], "mae": metrics["scratch_mae"], "r2": metrics["scratch_r2"]})
    append_results_csv(results_path, {**base_row, "model": "sklearn", "rmse": metrics["sklearn_rmse"], "mae": metrics["sklearn_mae"], "r2": metrics["sklearn_r2"]})

    print("✅ Run complete.")
    print("Metrics:", metrics)

    # Session 2: "2 weeks before failure" alert demo (simple rule)
    # If predicted days_until_failure <= 14 -> issue alert.
    # We'll show how many test points would trigger it.
    alerts = int(np.sum(y_pred_sk <= 14.0))
    print(f"🔔 Alert check (sklearn): {alerts} / {len(y_pred_sk)} test points predict failure within 14 days.")


if __name__ == "__main__":
    main()
