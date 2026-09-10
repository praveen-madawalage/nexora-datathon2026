"""
Urban Flow Analytics - Feature Engineering Module

Extracts spatial, temporal, cyclical, and interaction features from cleaned
taxi records for predictive modeling of base fare and trip duration.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def extract_temporal_features(df: pd.DataFrame, timestamp_col: str = "pickup_timestamp") -> pd.DataFrame:
    """
    Extract calendar, cyclical, and operational period features from a timestamp series.
    """
    ts = pd.to_datetime(df[timestamp_col])
    features = pd.DataFrame(index=df.index)

    features["pickup_hour"] = ts.dt.hour.astype(np.int8)
    features["pickup_dayofweek"] = ts.dt.dayofweek.astype(np.int8)
    features["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)
    features["pickup_month"] = ts.dt.month.astype(np.int8)
    features["pickup_day"] = ts.dt.day.astype(np.int8)

    # Cyclical hour encoding
    features["hour_sin"] = np.sin(2 * np.pi * features["pickup_hour"] / 24.0).astype(np.float32)
    features["hour_cos"] = np.cos(2 * np.pi * features["pickup_hour"] / 24.0).astype(np.float32)

    # Cyclical day-of-week encoding
    features["dow_sin"] = np.sin(2 * np.pi * features["pickup_dayofweek"] / 7.0).astype(np.float32)
    features["dow_cos"] = np.cos(2 * np.pi * features["pickup_dayofweek"] / 7.0).astype(np.float32)

    # Rush hour flags (weekday 07:00-10:00 and 16:00-20:00)
    features["is_rush_hour"] = (
        (~features["is_weekend"].astype(bool)) &
        ((features["pickup_hour"].between(7, 10)) | (features["pickup_hour"].between(16, 20)))
    ).astype(np.int8)

    # Late night flag (23:00 to 05:00)
    features["is_late_night"] = (
        (features["pickup_hour"] >= 23) | (features["pickup_hour"] < 5)
    ).astype(np.int8)

    return features


def extract_spatial_features(
    df: pd.DataFrame,
    origin_col: str = "origin_loc_id",
    dest_col: str = "dest_loc_id",
    airport_zone_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract spatial routing indicators and airport classification features.
    """
    airports = airport_zone_ids or [132, 138]  # 132: JFK, 138: LaGuardia
    features = pd.DataFrame(index=df.index)

    features["is_same_zone"] = (df[origin_col] == df[dest_col]).astype(np.int8)
    features["is_airport_pickup"] = df[origin_col].isin(airports).astype(np.int8)
    features["is_airport_dropoff"] = df[dest_col].isin(airports).astype(np.int8)
    features["is_airport_trip"] = (features["is_airport_pickup"] | features["is_airport_dropoff"]).astype(np.int8)

    return features


def prepare_upfront_pricing_features(
    df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Assemble feature matrix available strictly BEFORE the start of a trip
    for real-time upfront fare pricing.
    """
    temporal = extract_temporal_features(df, "pickup_timestamp")
    spatial = extract_spatial_features(df, "origin_loc_id", "dest_loc_id")

    feature_cols = [
        "distance_miles",
        "origin_loc_id",
        "dest_loc_id",
        "rate_class_id",
        "rider_count",
    ]
    available_cols = [col for col in feature_cols if col in df.columns]

    base = df[available_cols].copy()
    result = pd.concat([base, temporal, spatial], axis=1)

    if include_target and "base_fare" in df.columns:
        result["base_fare"] = df["base_fare"].astype(np.float32)

    return result


def prepare_trip_duration_features(
    df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Assemble feature matrix available strictly BEFORE the start of a trip
    for real-time trip duration estimation (Task 2.2).
    """
    temporal = extract_temporal_features(df, "pickup_timestamp")
    spatial = extract_spatial_features(df, "origin_loc_id", "dest_loc_id")

    feature_cols = [
        "distance_miles",
        "origin_loc_id",
        "dest_loc_id",
        "rate_class_id",
        "rider_count",
    ]
    available_cols = [col for col in feature_cols if col in df.columns]

    base = df[available_cols].copy()
    result = pd.concat([base, temporal, spatial], axis=1)

    if include_target and "trip_duration_minutes" in df.columns:
        result["trip_duration_minutes"] = df["trip_duration_minutes"].astype(np.float32)

    return result

