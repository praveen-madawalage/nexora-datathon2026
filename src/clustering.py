"""
Spatial-Temporal Hotspot and Origin-Destination Flow Clustering Engine.

This module powers Task 3.2 (Hotspot & Origin-Destination Flow Clustering):
- Zone-level temporal movement profiling across 4 operational time-of-day segments.
- K-Means spatial-temporal clustering identifying functional mobility archetypes.
- Elbow and silhouette diagnostics for defensible cluster selection.
- High-throughput DuckDB origin-destination flow matrix extraction.
- Morning-to-night movement shift and directional reversal quantification for city planners.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import REFERENCE_DATA_DIR

logger = logging.getLogger(__name__)

CLUSTER_FEATURE_COLS = [
    "pu_morning_share",
    "pu_midday_share",
    "pu_evening_share",
    "pu_night_share",
    "net_flow_ratio",
    "avg_distance",
    "avg_duration",
]

ARCHETYPE_LABELS = {
    0: "Commercial & Business Core",
    1: "Commuter Exporter Hub",
    2: "Nightlife & Entertainment District",
    3: "Residential High-Attraction Inflow",
    4: "Outer Borough Long-Haul Peripheral",
}


def extract_zone_temporal_profiles(
    parquet_path: Union[str, Path],
    reference_csv_path: Optional[Union[str, Path]] = None,
    min_pickups: int = 50,
) -> pd.DataFrame:
    """
    Aggregates trip records per taxi zone to profile movement patterns across
    diurnal time-of-day segments (Morning, Midday, Evening, Night).

    Parameters
    ----------
    parquet_path : str or Path
        Path to processed trip Parquet file.
    reference_csv_path : str or Path, optional
        Path to Zone Reference CSV. If None, uses canonical reference directory.
    min_pickups : int, default 50
        Minimum pickup count threshold to exclude inactive or non-operational zones.

    Returns
    -------
    pd.DataFrame
        Zone profile DataFrame containing volume shares, net flow ratio,
        average distance/duration, and geographical metadata.
    """
    con = duckdb.connect()
    p_str = str(Path(parquet_path).resolve()).replace("\\", "/")

    query = f"""
    WITH pickups AS (
        SELECT 
            origin_loc_id as loc_id,
            count(*) as total_pickups,
            count(*) FILTER (WHERE extract('hour' from pickup_timestamp) BETWEEN 6 AND 9) as pu_morning,
            count(*) FILTER (WHERE extract('hour' from pickup_timestamp) BETWEEN 10 AND 15) as pu_midday,
            count(*) FILTER (WHERE extract('hour' from pickup_timestamp) BETWEEN 16 AND 19) as pu_evening,
            count(*) FILTER (WHERE extract('hour' from pickup_timestamp) NOT BETWEEN 6 AND 19) as pu_night,
            avg(distance_miles) as avg_distance,
            avg(trip_duration_minutes) as avg_duration
        FROM parquet_scan('{p_str}')
        GROUP BY origin_loc_id
    ),
    dropoffs AS (
        SELECT 
            dest_loc_id as loc_id,
            count(*) as total_dropoffs,
            count(*) FILTER (WHERE extract('hour' from dropoff_timestamp) BETWEEN 6 AND 9) as do_morning,
            count(*) FILTER (WHERE extract('hour' from dropoff_timestamp) BETWEEN 10 AND 15) as do_midday,
            count(*) FILTER (WHERE extract('hour' from dropoff_timestamp) BETWEEN 16 AND 19) as do_evening,
            count(*) FILTER (WHERE extract('hour' from dropoff_timestamp) NOT BETWEEN 6 AND 19) as do_night
        FROM parquet_scan('{p_str}')
        GROUP BY dest_loc_id
    )
    SELECT 
        p.loc_id,
        p.total_pickups,
        COALESCE(d.total_dropoffs, 0) as total_dropoffs,
        round(p.pu_morning * 1.0 / NULLIF(p.total_pickups, 0), 4) as pu_morning_share,
        round(p.pu_midday * 1.0 / NULLIF(p.total_pickups, 0), 4) as pu_midday_share,
        round(p.pu_evening * 1.0 / NULLIF(p.total_pickups, 0), 4) as pu_evening_share,
        round(p.pu_night * 1.0 / NULLIF(p.total_pickups, 0), 4) as pu_night_share,
        round((p.total_pickups - COALESCE(d.total_dropoffs, 0)) * 1.0 / NULLIF(p.total_pickups + COALESCE(d.total_dropoffs, 0), 0), 4) as net_flow_ratio,
        round(p.avg_distance, 2) as avg_distance,
        round(p.avg_duration, 2) as avg_duration
    FROM pickups p
    LEFT JOIN dropoffs d ON p.loc_id = d.loc_id
    WHERE p.total_pickups >= {min_pickups}
    ORDER BY p.total_pickups DESC
    """
    df = con.execute(query).df()
    con.close()

    if reference_csv_path is None:
        ref_path = REFERENCE_DATA_DIR / "Urban_Flow_Analytics_Zone_Dataset.csv"
    else:
        ref_path = Path(reference_csv_path)

    if ref_path.exists():
        ref_df = pd.read_csv(ref_path)
        df = pd.merge(df, ref_df, on="loc_id", how="left")
    else:
        logger.warning("Zone reference metadata not found at %s", ref_path)
        df["zone_name"] = "Zone " + df["loc_id"].astype(str)
        df["borough_name"] = "Unknown"
        df["service_zone"] = "Unknown"

    return df


def find_optimal_k_kmeans(
    df_profiles: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    k_range: range = range(2, 9),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluates inertia and silhouette scores over a range of cluster counts k
    to establish the optimal number of functional mobility archetypes.

    Parameters
    ----------
    df_profiles : pd.DataFrame
        Zone profiles table.
    feature_cols : list of str, optional
        Features to include in clustering. Defaults to CLUSTER_FEATURE_COLS.
    k_range : range, default range(2, 9)
        Range of k values to test.
    random_state : int, default 42
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Columns: k, inertia, silhouette_score.
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURE_COLS

    X = df_profiles[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0
        results.append(
            {
                "k": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette_score": round(sil, 4),
            }
        )

    return pd.DataFrame(results)


def cluster_zone_hotspots(
    df_profiles: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_clusters: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits K-Means clustering on standardized zone movement features,
    assigning each taxi zone to an operational mobility archetype.

    Parameters
    ----------
    df_profiles : pd.DataFrame
        Zone profiles DataFrame.
    feature_cols : list of str, optional
        Feature columns for clustering.
    n_clusters : int, default 5
        Number of clusters to extract.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    df_clustered : pd.DataFrame
        Input dataframe enriched with cluster_id and archetype_name.
    centroid_summary : pd.DataFrame
        Cluster centers transformed back to original feature scale.
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURE_COLS

    df_out = df_profiles.copy()
    X = df_out[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    cluster_labels = km.fit_predict(X_scaled)
    df_out["cluster_id"] = cluster_labels

    # Map archetype names dynamically based on cluster characteristics
    centroids_unscaled = scaler.inverse_transform(km.cluster_centers_)
    centroid_df = pd.DataFrame(centroids_unscaled, columns=feature_cols)
    centroid_df["cluster_id"] = range(n_clusters)

    # Dynamic labeling based on centroid profiles
    archetype_map: dict[int, str] = {}
    for cid in range(n_clusters):
        row = centroid_df.loc[cid]
        if row["pu_night_share"] >= 0.40:
            label = "Nightlife & Entertainment District"
        elif row["avg_distance"] >= 8.0:
            label = "Outer Borough Long-Haul Peripheral"
        elif row["pu_morning_share"] >= 0.25 and row["net_flow_ratio"] < -0.15:
            label = "Commuter Exporter Hub"
        elif row["net_flow_ratio"] < -0.30:
            label = "Residential Inflow / Attractor"
        else:
            label = "Commercial & High-Density Core"
        archetype_map[cid] = label

    df_out["archetype_name"] = df_out["cluster_id"].map(archetype_map)
    centroid_df["archetype_name"] = centroid_df["cluster_id"].map(archetype_map)

    return df_out, centroid_df


def extract_od_flows(
    parquet_path: Union[str, Path],
    time_segment: Optional[str] = None,
    top_n: int = 50,
    reference_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Extracts origin-destination flow matrices categorized by time segment.

    Parameters
    ----------
    parquet_path : str or Path
        Path to trip records Parquet file.
    time_segment : str, optional
        Filter to 'Morning_Peak', 'Midday', 'Evening_Peak', or 'Night'.
        If None, returns flows across all segments.
    top_n : int, default 50
        Top N origin-destination corridors to retrieve.
    reference_csv_path : str or Path, optional
        Path to Zone Reference CSV.

    Returns
    -------
    pd.DataFrame
        Columns: origin_loc_id, dest_loc_id, time_segment, trip_volume,
        avg_distance, avg_duration, origin_name, dest_name, origin_borough, dest_borough.
    """
    con = duckdb.connect()
    p_str = str(Path(parquet_path).resolve()).replace("\\", "/")

    segment_clause = ""
    if time_segment is not None:
        segment_clause = f"HAVING time_segment = '{time_segment}'"

    query = f"""
    WITH trips AS (
        SELECT 
            origin_loc_id,
            dest_loc_id,
            CASE 
                WHEN extract('hour' from pickup_timestamp) BETWEEN 6 AND 9 THEN 'Morning_Peak'
                WHEN extract('hour' from pickup_timestamp) BETWEEN 10 AND 15 THEN 'Midday'
                WHEN extract('hour' from pickup_timestamp) BETWEEN 16 AND 19 THEN 'Evening_Peak'
                ELSE 'Night'
            END as time_segment,
            distance_miles,
            trip_duration_minutes
        FROM parquet_scan('{p_str}')
    )
    SELECT 
        origin_loc_id,
        dest_loc_id,
        time_segment,
        count(*) as trip_volume,
        round(avg(distance_miles), 2) as avg_distance,
        round(avg(trip_duration_minutes), 2) as avg_duration
    FROM trips
    GROUP BY origin_loc_id, dest_loc_id, time_segment
    {segment_clause}
    ORDER BY trip_volume DESC
    LIMIT {top_n}
    """
    df = con.execute(query).df()
    con.close()

    if reference_csv_path is None:
        ref_path = REFERENCE_DATA_DIR / "Urban_Flow_Analytics_Zone_Dataset.csv"
    else:
        ref_path = Path(reference_csv_path)

    if ref_path.exists():
        ref_df = pd.read_csv(ref_path)
        name_map = ref_df.set_index("loc_id")["zone_name"].to_dict()
        boro_map = ref_df.set_index("loc_id")["borough_name"].to_dict()
        df["origin_name"] = df["origin_loc_id"].map(name_map).fillna("Zone " + df["origin_loc_id"].astype(str))
        df["dest_name"] = df["dest_loc_id"].map(name_map).fillna("Zone " + df["dest_loc_id"].astype(str))
        df["origin_borough"] = df["origin_loc_id"].map(boro_map).fillna("Unknown")
        df["dest_borough"] = df["dest_loc_id"].map(boro_map).fillna("Unknown")
    else:
        df["origin_name"] = "Zone " + df["origin_loc_id"].astype(str)
        df["dest_name"] = "Zone " + df["dest_loc_id"].astype(str)
        df["origin_borough"] = "Unknown"
        df["dest_borough"] = "Unknown"

    df["corridor"] = df["origin_name"] + " -> " + df["dest_name"]
    return df


def summarize_temporal_movement_shifts(
    morning_flows: pd.DataFrame,
    night_flows: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compares travel movement between Morning Peak work hours and Late Night.

    Returns
    -------
    dict
        Key findings, corridor shift summaries, and quantitative metrics for city planners.
    """
    m_top_corridor = morning_flows.iloc[0]["corridor"] if len(morning_flows) > 0 else "N/A"
    m_top_vol = int(morning_flows.iloc[0]["trip_volume"]) if len(morning_flows) > 0 else 0
    m_avg_dist = float(morning_flows["avg_distance"].mean()) if len(morning_flows) > 0 else 0.0

    n_top_corridor = night_flows.iloc[0]["corridor"] if len(night_flows) > 0 else "N/A"
    n_top_vol = int(night_flows.iloc[0]["trip_volume"]) if len(night_flows) > 0 else 0
    n_avg_dist = float(night_flows["avg_distance"].mean()) if len(night_flows) > 0 else 0.0

    return {
        "morning_top_corridor": m_top_corridor,
        "morning_top_corridor_volume": m_top_vol,
        "morning_avg_corridor_distance": round(m_avg_dist, 2),
        "night_top_corridor": n_top_corridor,
        "night_top_corridor_volume": n_top_vol,
        "night_avg_corridor_distance": round(n_avg_dist, 2),
        "distance_shift_pct": round(((n_avg_dist - m_avg_dist) / m_avg_dist) * 100.0, 1) if m_avg_dist > 0 else 0.0,
    }
