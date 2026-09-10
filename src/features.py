"""
Urban Flow Analytics - Feature Engineering Module

Extracts spatial, temporal, cyclical, and interaction features from cleaned
taxi records for predictive modeling of base fare and trip duration.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from src.utils import REFERENCE_DATA_DIR


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
    features["is_ewr_trip"] = ((df[origin_col] == 1) | (df[dest_col] == 1)).astype(np.int8)

    return features


def enrich_with_zone_metadata(
    df: pd.DataFrame,
    zone_df: Optional[pd.DataFrame] = None,
    origin_col: str = "origin_loc_id",
    dest_col: str = "dest_loc_id",
) -> pd.DataFrame:
    """
    Enrich trip records with borough-level metadata and interborough travel indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trip records containing origin and destination location IDs.
    zone_df : pd.DataFrame, optional
        Zone lookup table. If None, loaded from reference directory if present.
    origin_col : str, default 'origin_loc_id'
        Name of origin zone ID column.
    dest_col : str, default 'dest_loc_id'
        Name of destination zone ID column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with spatial indicators: is_interborough, is_manhattan_intra,
        is_ewr_trip, origin_borough, dest_borough.
    """
    if zone_df is None:
        ref_path = REFERENCE_DATA_DIR / "Urban_Flow_Analytics_Zone_Dataset.csv"
        if ref_path.exists():
            zone_df = pd.read_csv(ref_path)

    features = pd.DataFrame(index=df.index)

    if zone_df is not None and origin_col in df.columns and dest_col in df.columns:
        loc_to_borough = dict(zip(zone_df["loc_id"], zone_df["borough_name"]))
        loc_to_service = dict(zip(zone_df["loc_id"], zone_df["service_zone"]))

        origin_borough = df[origin_col].map(loc_to_borough).fillna("Unknown")
        dest_borough = df[dest_col].map(loc_to_borough).fillna("Unknown")
        origin_service = df[origin_col].map(loc_to_service).fillna("Unknown")
        dest_service = df[dest_col].map(loc_to_service).fillna("Unknown")

        features["origin_borough"] = origin_borough.astype(str)
        features["dest_borough"] = dest_borough.astype(str)
        features["origin_service_zone"] = origin_service.astype(str)
        features["dest_service_zone"] = dest_service.astype(str)

        features["is_interborough"] = (
            (origin_borough != dest_borough) & (origin_borough != "Unknown") & (dest_borough != "Unknown")
        ).astype(np.int8)
        features["is_manhattan_intra"] = (
            (origin_borough == "Manhattan") & (dest_borough == "Manhattan")
        ).astype(np.int8)
        features["is_ewr_trip"] = ((df[origin_col] == 1) | (df[dest_col] == 1)).astype(np.int8)
    else:
        features["is_interborough"] = pd.Series(0, index=df.index, dtype=np.int8)
        features["is_manhattan_intra"] = pd.Series(0, index=df.index, dtype=np.int8)
        if origin_col in df.columns and dest_col in df.columns:
            features["is_ewr_trip"] = ((df[origin_col] == 1) | (df[dest_col] == 1)).astype(np.int8)
        else:
            features["is_ewr_trip"] = pd.Series(0, index=df.index, dtype=np.int8)

    return features


def prepare_upfront_pricing_features(
    df: pd.DataFrame,
    zone_df: Optional[pd.DataFrame] = None,
    include_target: bool = True,
    include_borough_categories: bool = False,
) -> pd.DataFrame:
    """
    Assemble feature matrix available strictly BEFORE the start of a trip
    for real-time upfront fare pricing (Task 2.1).
    """
    temporal = extract_temporal_features(df, "pickup_timestamp")
    spatial = extract_spatial_features(df, "origin_loc_id", "dest_loc_id")
    zone_meta = enrich_with_zone_metadata(df, zone_df=zone_df, origin_col="origin_loc_id", dest_col="dest_loc_id")

    feature_cols = [
        "distance_miles",
        "origin_loc_id",
        "dest_loc_id",
        "rate_class_id",
        "rider_count",
        "provider_code",
        "fare_settlement_method",
    ]
    available_cols = [col for col in feature_cols if col in df.columns]
    base = df[available_cols].copy()

    engineered = pd.DataFrame(index=df.index)
    if "distance_miles" in df.columns:
        engineered["log_distance"] = np.log1p(np.maximum(0.0, df["distance_miles"].values)).astype(np.float32)

    if "rate_class_id" in df.columns:
        rc = df["rate_class_id"]
        engineered["is_standard_rate"] = (rc == 1).astype(np.int8)
        engineered["is_jfk_flat_rate"] = (rc == 2).astype(np.int8)
        engineered["is_newark_rate"] = (rc == 3).astype(np.int8)
        engineered["is_negotiated_rate"] = (rc == 5).astype(np.int8)

    zone_cols_to_use = ["is_interborough", "is_manhattan_intra"]
    if include_borough_categories and "origin_borough" in zone_meta.columns:
        zone_cols_to_use.extend(["origin_borough", "dest_borough"])

    result = pd.concat([base, engineered, temporal, spatial, zone_meta[zone_cols_to_use]], axis=1)

    if include_target and "base_fare" in df.columns:
        result["base_fare"] = df["base_fare"].astype(np.float32)

    return result


def prepare_trip_duration_features(
    df: pd.DataFrame,
    zone_df: Optional[pd.DataFrame] = None,
    include_target: bool = True,
    include_borough_categories: bool = False,
) -> pd.DataFrame:
    """
    Assemble feature matrix available strictly BEFORE the start of a trip
    for real-time trip duration estimation (Task 2.2).
    """
    temporal = extract_temporal_features(df, "pickup_timestamp")
    spatial = extract_spatial_features(df, "origin_loc_id", "dest_loc_id")
    zone_meta = enrich_with_zone_metadata(df, zone_df=zone_df, origin_col="origin_loc_id", dest_col="dest_loc_id")

    feature_cols = [
        "distance_miles",
        "origin_loc_id",
        "dest_loc_id",
        "rate_class_id",
        "rider_count",
    ]
    available_cols = [col for col in feature_cols if col in df.columns]
    base = df[available_cols].copy()

    engineered = pd.DataFrame(index=df.index)
    if "distance_miles" in df.columns:
        engineered["log_distance"] = np.log1p(np.maximum(0.0, df["distance_miles"].values)).astype(np.float32)

    if "rate_class_id" in df.columns:
        rc = df["rate_class_id"]
        engineered["is_standard_rate"] = (rc == 1).astype(np.int8)
        engineered["is_jfk_flat_rate"] = (rc == 2).astype(np.int8)
        engineered["is_newark_rate"] = (rc == 3).astype(np.int8)
        engineered["is_negotiated_rate"] = (rc == 5).astype(np.int8)

    zone_cols_to_use = ["is_interborough", "is_manhattan_intra"]
    if include_borough_categories and "origin_borough" in zone_meta.columns:
        zone_cols_to_use.extend(["origin_borough", "dest_borough"])

    result = pd.concat([base, engineered, temporal, spatial, zone_meta[zone_cols_to_use]], axis=1)

    if include_target and "trip_duration_minutes" in df.columns:
        result["trip_duration_minutes"] = df["trip_duration_minutes"].astype(np.float32)

    return result

