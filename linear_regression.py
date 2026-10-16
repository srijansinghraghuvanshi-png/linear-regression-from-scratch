"""Linear regression with one feature, implemented using NumPy only.

The model learns y = weight * x + bias by gradient descent.
No machine-learning library is used.
"""
from __future__ import annotations

import numpy as np


class LinearRegression:
    """Simple univariate linear regression trained with gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1_000):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weight = 0.0
        self.bias = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _validate_x_y(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if x.size == 0 or y.size == 0:
            raise ValueError("x and y must not be empty")
        if x.size != y.size:
            raise ValueError("x and y must contain the same number of samples")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("x and y must contain only finite numbers")

        return x, y

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target values for one-dimensional input features."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite numbers")
        return self.weight * x + self.bias

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Learn weight and bias by minimizing mean squared error."""
        x, y = self._validate_x_y(x, y)
        n_samples = x.size
        self.weight = 0.0
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            predictions = self.predict(x)
            errors = predictions - y

            # Derivatives of MSE with respect to weight and bias.
            dw = (2.0 / n_samples) * np.sum(errors * x)
            db = (2.0 / n_samples) * np.sum(errors)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            updated_errors = self.predict(x) - y
            self.loss_history.append(float(np.mean(updated_errors ** 2)))

        return self

    def mean_squared_error(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return mean squared prediction error for supplied examples."""
        x, y = self._validate_x_y(x, y)
        return float(np.mean((self.predict(x) - y) ** 2))
