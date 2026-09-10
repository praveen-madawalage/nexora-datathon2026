"""
Unit tests for feature extraction and modeling matrix assemblers.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    extract_temporal_features,
    extract_spatial_features,
    prepare_upfront_pricing_features,
    prepare_trip_duration_features,
)


@pytest.fixture
def sample_trip_df():
    """Create a synthetic dataframe matching the clean schema."""
    return pd.DataFrame({
        "pickup_timestamp": [
            "2025-05-12 08:30:00",  # Monday morning rush hour
            "2025-05-17 23:45:00",  # Saturday late night
            "2025-05-14 14:00:00",  # Wednesday mid-day
        ],
        "dropoff_timestamp": [
            "2025-05-12 08:50:00",
            "2025-05-18 00:15:00",
            "2025-05-14 14:20:00",
        ],
        "distance_miles": [3.2, 8.5, 2.1],
        "origin_loc_id": [132, 236, 142],   # 132 is JFK airport
        "dest_loc_id": [236, 138, 142],     # 138 is LaGuardia, 142 is same zone
        "rate_class_id": [2, 1, 1],
        "rider_count": [1, 2, 1],
        "base_fare": [70.0, 32.5, 12.0],
        "trip_duration_minutes": [20.0, 30.0, 20.0],
    })


def test_extract_temporal_features(sample_trip_df):
    """Verify calendar, cyclical, and operational period feature calculations."""
    temporal = extract_temporal_features(sample_trip_df)
    
    assert "pickup_hour" in temporal.columns
    assert "pickup_dayofweek" in temporal.columns
    assert "is_weekend" in temporal.columns
    assert "is_rush_hour" in temporal.columns
    assert "is_late_night" in temporal.columns
    assert "hour_sin" in temporal.columns
    assert "hour_cos" in temporal.columns
    
    # Monday 08:30 is rush hour, not weekend
    assert temporal.loc[0, "is_weekend"] == 0
    assert temporal.loc[0, "is_rush_hour"] == 1
    assert temporal.loc[0, "is_late_night"] == 0
    
    # Saturday 23:45 is weekend and late night, not rush hour
    assert temporal.loc[1, "is_weekend"] == 1
    assert temporal.loc[1, "is_rush_hour"] == 0
    assert temporal.loc[1, "is_late_night"] == 1


def test_extract_spatial_features(sample_trip_df):
    """Verify airport classification and same-zone flags."""
    spatial = extract_spatial_features(sample_trip_df)
    
    assert "is_same_zone" in spatial.columns
    assert "is_airport_pickup" in spatial.columns
    assert "is_airport_dropoff" in spatial.columns
    assert "is_airport_trip" in spatial.columns
    
    # Row 0: origin 132 (JFK) -> airport pickup and airport trip
    assert spatial.loc[0, "is_airport_pickup"] == 1
    assert spatial.loc[0, "is_airport_dropoff"] == 0
    assert spatial.loc[0, "is_airport_trip"] == 1
    assert spatial.loc[0, "is_same_zone"] == 0
    
    # Row 1: dest 138 (LGA) -> airport dropoff
    assert spatial.loc[1, "is_airport_pickup"] == 0
    assert spatial.loc[1, "is_airport_dropoff"] == 1
    assert spatial.loc[1, "is_airport_trip"] == 1
    
    # Row 2: 142 -> 142 is same zone, not airport
    assert spatial.loc[2, "is_same_zone"] == 1
    assert spatial.loc[2, "is_airport_trip"] == 0


def test_prepare_upfront_pricing_features(sample_trip_df):
    """Verify upfront fare pricing feature matrix assembler."""
    features_with_target = prepare_upfront_pricing_features(sample_trip_df, include_target=True)
    assert "base_fare" in features_with_target.columns
    assert "distance_miles" in features_with_target.columns
    assert "pickup_hour" in features_with_target.columns
    assert "is_airport_trip" in features_with_target.columns
    assert features_with_target["base_fare"].dtype == np.float32

    features_no_target = prepare_upfront_pricing_features(sample_trip_df, include_target=False)
    assert "base_fare" not in features_no_target.columns
    assert "distance_miles" in features_no_target.columns


def test_prepare_trip_duration_features(sample_trip_df):
    """Verify on-time arrival trip duration feature matrix assembler (Task 2.2)."""
    features_with_target = prepare_trip_duration_features(sample_trip_df, include_target=True)
    assert "trip_duration_minutes" in features_with_target.columns
    assert "distance_miles" in features_with_target.columns
    assert "pickup_hour" in features_with_target.columns
    assert "is_airport_trip" in features_with_target.columns
    assert features_with_target["trip_duration_minutes"].dtype == np.float32

    features_no_target = prepare_trip_duration_features(sample_trip_df, include_target=False)
    assert "trip_duration_minutes" not in features_no_target.columns
    assert "distance_miles" in features_no_target.columns
    # Verify no post-trip leakage features present
    assert "trip_speed_mph" not in features_no_target.columns
    assert "dropoff_timestamp" not in features_no_target.columns
