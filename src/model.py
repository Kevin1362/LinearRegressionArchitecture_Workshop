"""
model.py
--------
Univariate linear regression models.
- Scratch implementation (gradient descent)
- scikit-learn implementation for comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np


@dataclass
class ScratchLinearRegression:
    learning_rate: float = 0.01
    n_iters: int = 2000

    w: float = 0.0
    b: float = 0.0
    history: Optional[list] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchLinearRegression":
        """
        Fit w, b using gradient descent.
        X: shape (n, 1)
        y: shape (n,)
        """
        X = X.reshape(-1, 1).astype(float)
        y = y.astype(float)
        n = len(X)

        self.w = 0.0
        self.b = 0.0
        self.history = []

        for _ in range(int(self.n_iters)):
            y_pred = (self.w * X[:, 0]) + self.b
            # gradients for MSE
            dw = (2 / n) * np.sum((y_pred - y) * X[:, 0])
            db = (2 / n) * np.sum(y_pred - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            mse = float(np.mean((y_pred - y) ** 2))
            self.history.append(mse)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, 1).astype(float)
        return (self.w * X[:, 0]) + self.b


def fit_sklearn_linear_regression(X_train: np.ndarray, y_train: np.ndarray):
    """Fit scikit-learn LinearRegression (univariate)."""
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr
