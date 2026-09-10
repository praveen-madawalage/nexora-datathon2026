"""
Unit tests for model training, evaluation metrics, tolerance bands, and serialization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models import (
    evaluate_duration_tolerance,
    evaluate_regression,
    load_model,
    save_model,
    train_linear_baseline,
)


@pytest.fixture
def synthetic_regression_data():
    """Create a synthetic dataset for quick model smoke tests."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "distance_miles": np.random.uniform(0.5, 15.0, n),
        "pickup_hour": np.random.randint(0, 24, n),
        "is_airport_trip": np.random.choice([0, 1], n),
        "rate_class_id": np.random.choice([1, 2, 3], n),
    })
    y = 3.0 + 2.5 * X["distance_miles"] + 0.5 * X["pickup_hour"] + np.random.normal(0, 0.5, n)
    return X, y


def test_linear_baseline_shape(synthetic_regression_data):
    """Verify linear baseline trains and predicts matching output shape."""
    X, y = synthetic_regression_data
    model = train_linear_baseline(X, y)
    preds = model.predict(X[model.numeric_cols_])
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))


def test_evaluate_regression_perfect():
    """Verify regression metrics on identical true and predicted vectors."""
    y_true = np.array([5.0, 12.5, 25.0, 50.0, 70.0])
    y_pred = np.array([5.0, 12.5, 25.0, 50.0, 70.0])
    metrics = evaluate_regression(y_true, y_pred, label="PerfectTest")

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["rmsle"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["model_label"] == "PerfectTest"


def test_duration_tolerance_known():
    """Verify tolerance band percentages on known residual distributions."""
    # 5 samples with known errors: 1.0, 2.5, 4.0, 7.5, 12.0 min
    y_true = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    y_pred = np.array([11.0, 17.5, 24.0, 32.5, 42.0])  # errors: 1.0, 2.5, 4.0, 7.5, 12.0
    
    # <= 3.0 min: 1.0, 2.5 -> 2 out of 5 = 40.0%
    # <= 5.0 min: 1.0, 2.5, 4.0 -> 3 out of 5 = 60.0%
    # <= 10.0 min: 1.0, 2.5, 4.0, 7.5 -> 4 out of 5 = 80.0%
    # <= 15.0 min: all 5 -> 100.0%
    metrics = evaluate_duration_tolerance(y_true, y_pred)

    assert metrics["pct_within_3min"] == 40.0
    assert metrics["pct_within_5min"] == 60.0
    assert metrics["pct_within_10min"] == 80.0
    assert metrics["pct_within_15min"] == 100.0
    assert metrics["median_abs_err_min"] == 4.0


def test_model_save_and_load(tmp_path, synthetic_regression_data):
    """Verify serialization and deserialization preserves model predictions."""
    X, y = synthetic_regression_data
    model = train_linear_baseline(X, y)
    original_preds = model.predict(X[model.numeric_cols_])

    save_path = tmp_path / "test_model.pkl"
    meta = {"task": "fare", "version": "1.0", "r2": 0.95}
    save_model(model, save_path, metadata=meta)

    loaded_model, loaded_meta = load_model(save_path)
    loaded_preds = loaded_model.predict(X[loaded_model.numeric_cols_])

    np.testing.assert_allclose(original_preds, loaded_preds)
    assert loaded_meta["task"] == "fare"
    assert loaded_meta["version"] == "1.0"
