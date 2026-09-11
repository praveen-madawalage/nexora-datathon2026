"""
Unit tests for src/demand.py (Task 3.1: The Fleet Dispatcher).
"""

import numpy as np
import pandas as pd
import pytest

from src.demand import (
    DEMAND_FEATURE_COLS,
    build_regular_hourly_grid,
    evaluate_demand_predictions,
    generate_dispatch_recommendations,
    prepare_demand_features,
)


def test_build_regular_hourly_grid_gap_fill():
    # 2 zones, sparse hours
    data = [
        {"origin_loc_id": 1, "hour_bucket": pd.Timestamp("2026-01-01 00:00:00"), "trip_count": 10},
        {"origin_loc_id": 1, "hour_bucket": pd.Timestamp("2026-01-01 03:00:00"), "trip_count": 15},
        {"origin_loc_id": 2, "hour_bucket": pd.Timestamp("2026-01-01 01:00:00"), "trip_count": 5},
    ]
    df = pd.DataFrame(data)
    grid = build_regular_hourly_grid(
        df,
        start_time=pd.Timestamp("2026-01-01 00:00:00"),
        end_time=pd.Timestamp("2026-01-01 03:00:00"),
    )

    # Each zone must have 4 hours (00, 01, 02, 03) -> total 8 rows
    assert len(grid) == 8
    # Missing hours must have 0 count
    missing_z1_h1 = grid[(grid["origin_loc_id"] == 1) & (grid["hour_bucket"] == pd.Timestamp("2026-01-01 01:00:00"))]
    assert missing_z1_h1["trip_count"].values[0] == 0
    # Original hours preserved
    orig_z1_h0 = grid[(grid["origin_loc_id"] == 1) & (grid["hour_bucket"] == pd.Timestamp("2026-01-01 00:00:00"))]
    assert orig_z1_h0["trip_count"].values[0] == 10


def test_prepare_demand_features_leakage_and_structure():
    # Generate 200 consecutive hours for 1 zone
    hours = pd.date_range("2026-01-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame(
        {
            "origin_loc_id": 100,
            "hour_bucket": hours,
            "trip_count": np.arange(200),
        }
    )

    featured = prepare_demand_features(df, lags=[1, 2, 24], roll_windows=[6, 24])

    for col in ["hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos", "lag_1", "lag_2", "lag_24", "roll_mean_6h", "roll_mean_24h"]:
        assert col in featured.columns

    # Verify lag_1 is strictly t-1
    assert featured.loc[1, "lag_1"] == featured.loc[0, "trip_count"]
    # Verify roll_mean_6h uses strictly past observations (shifted by 1)
    # At index 6, past 6 counts are [0, 1, 2, 3, 4, 5], mean is 2.5
    assert np.isclose(featured.loc[6, "roll_mean_6h"], 2.5)


def test_evaluate_demand_predictions_metrics():
    y_true = np.array([100.0, 200.0, 150.0, 50.0])
    y_pred = np.array([100.0, 200.0, 150.0, 50.0])

    perfect = evaluate_demand_predictions(y_true, y_pred)
    assert perfect["rmse"] == 0.0
    assert perfect["mae"] == 0.0
    assert perfect["r2"] == 1.0
    assert perfect["wape"] == 0.0

    # Test with known difference
    y_noisy = np.array([110.0, 190.0, 160.0, 40.0])
    noisy = evaluate_demand_predictions(y_true, y_noisy)
    assert noisy["mae"] == 10.0
    assert noisy["rmse"] == 10.0
    # WAPE = sum(|10, -10, 10, -10|) / 500 = 40 / 500 = 8.0%
    assert noisy["wape"] == 8.0


def test_generate_dispatch_recommendations():
    forecast_df = pd.DataFrame(
        {
            "origin_loc_id": [1, 2, 3],
            "hour_bucket": [pd.Timestamp("2026-03-01 08:00:00")] * 3,
            "step_ahead": [1, 1, 1],
            "predicted_demand": [10.0, 50.0, 250.0],
            "actual_demand": [12.0, 48.0, 260.0],
        }
    )

    recs = generate_dispatch_recommendations(forecast_df, threshold_percentile=70.0)
    assert len(recs) == 3
    assert "dispatch_action" in recs.columns
    assert "recommended_staging_vehicles" in recs.columns
    # Zone 3 with 250 trips should have highest vehicle staging recommendation
    top_rec = recs.iloc[0]
    assert top_rec["origin_loc_id"] == 3
    assert top_rec["recommended_staging_vehicles"] > 100
