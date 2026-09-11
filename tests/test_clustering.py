"""
Unit tests for src/clustering.py (Task 3.2: Hotspot & OD Flow Clustering).
"""

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    CLUSTER_FEATURE_COLS,
    cluster_zone_hotspots,
    find_optimal_k_kmeans,
    summarize_temporal_movement_shifts,
)


@pytest.fixture
def sample_profiles_df():
    np.random.seed(42)
    n = 20
    data = {
        "loc_id": list(range(1, n + 1)),
        "total_pickups": np.random.randint(500, 10000, size=n),
        "total_dropoffs": np.random.randint(500, 10000, size=n),
        "pu_morning_share": np.random.uniform(0.1, 0.4, size=n),
        "pu_midday_share": np.random.uniform(0.2, 0.4, size=n),
        "pu_evening_share": np.random.uniform(0.2, 0.4, size=n),
        "pu_night_share": np.random.uniform(0.1, 0.5, size=n),
        "net_flow_ratio": np.random.uniform(-0.5, 0.5, size=n),
        "avg_distance": np.random.uniform(1.0, 10.0, size=n),
        "avg_duration": np.random.uniform(5.0, 30.0, size=n),
        "zone_name": [f"Zone {i}" for i in range(1, n + 1)],
        "borough_name": ["Manhattan"] * 10 + ["Queens"] * 5 + ["Brooklyn"] * 5,
        "service_zone": ["Yellow Zone"] * 20,
    }
    return pd.DataFrame(data)


def test_find_optimal_k_kmeans(sample_profiles_df):
    k_res = find_optimal_k_kmeans(sample_profiles_df, k_range=range(2, 5))
    assert len(k_res) == 3
    assert set(k_res.columns) == {"k", "inertia", "silhouette_score"}
    assert k_res["k"].tolist() == [2, 3, 4]
    assert (k_res["inertia"] > 0).all()


def test_cluster_zone_hotspots(sample_profiles_df):
    clustered, centroids = cluster_zone_hotspots(sample_profiles_df, n_clusters=3)
    assert "cluster_id" in clustered.columns
    assert "archetype_name" in clustered.columns
    assert clustered["cluster_id"].nunique() <= 3
    assert len(centroids) == 3
    assert "archetype_name" in centroids.columns


def test_summarize_temporal_movement_shifts():
    morning = pd.DataFrame(
        [
            {
                "corridor": "Zone A -> Zone B",
                "trip_volume": 1000,
                "avg_distance": 2.5,
                "avg_duration": 15.0,
            }
        ]
    )
    night = pd.DataFrame(
        [
            {
                "corridor": "Zone B -> Zone C",
                "trip_volume": 1200,
                "avg_distance": 3.5,
                "avg_duration": 18.0,
            }
        ]
    )
    summary = summarize_temporal_movement_shifts(morning, night)
    assert summary["morning_top_corridor"] == "Zone A -> Zone B"
    assert summary["night_top_corridor"] == "Zone B -> Zone C"
    assert summary["morning_top_corridor_volume"] == 1000
    assert summary["night_top_corridor_volume"] == 1200
    assert summary["distance_shift_pct"] == 40.0
