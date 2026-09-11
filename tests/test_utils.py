"""
Unit tests for common utilities and regression metrics.
"""

import numpy as np
import pytest
from src.utils import calculate_regression_metrics, get_project_root


def test_project_root():
    """Verify project root resolution."""
    root = get_project_root()
    assert root.exists()
    assert (root / "src").exists()
    assert (root / "notebooks").exists()


def test_calculate_regression_metrics_perfect():
    """Verify metrics on identical predictions."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    metrics = calculate_regression_metrics(y_true, y_pred)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["mape"] == 0.0


def test_calculate_regression_metrics_imperfect():
    """Verify metrics on known discrepancies."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    metrics = calculate_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == round((2.0 + 2.0 + 3.0) / 3.0, 4)
    assert metrics["r2"] > 0.9
    assert metrics["rmse"] > 0.0
