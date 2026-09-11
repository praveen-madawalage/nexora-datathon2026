"""
Unit tests for Business Intelligence Analytics Engine (Track 6 Bonus).
"""

import pandas as pd
import pytest

from src.bi_analytics import (
    compute_zone_avg_fare,
    get_archetype_profitability,
    get_borough_revenue_summary,
    get_executive_recommendations,
    get_fleet_efficiency_gap,
    get_temporal_heatmap,
    get_top_zones_by_revenue,
    load_bi_data,
)


def test_compute_zone_avg_fare_airport():
    """Verify airport trips receive the standard regulated flat fare."""
    row = {"zone_name": "JFK Airport", "borough_name": "Queens", "avg_distance": 15.2}
    fare = compute_zone_avg_fare(row)
    assert fare == 58.50


def test_compute_zone_avg_fare_standard():
    """Verify non-airport trips compute valid bounds and distance scaling."""
    row = {"zone_name": "Upper East Side South", "borough_name": "Manhattan", "avg_distance": 1.8}
    fare = compute_zone_avg_fare(row)
    assert 11.50 <= fare <= 48.00


def test_load_bi_data_and_structure():
    """Verify all BI data entities load with expected keys."""
    data = load_bi_data()
    assert "top_zones" in data
    assert "demand" in data
    assert "archetypes" in data
    assert "od_flows" in data

    top_zones = data["top_zones"]
    assert "est_revenue" in top_zones.columns
    assert "avg_fare" in top_zones.columns
    assert len(top_zones) > 0


def test_borough_revenue_summary():
    """Verify borough aggregation sums and sorting."""
    data = load_bi_data()
    summary = get_borough_revenue_summary(data["top_zones"])
    assert "borough_name" in summary.columns
    assert "est_revenue" in summary.columns
    assert "rev_share_pct" in summary.columns
    assert summary["rev_share_pct"].sum() == pytest.approx(100.0, rel=1e-3)
    # Manhattan should be the top revenue producer
    assert summary.iloc[0]["borough_name"] == "Manhattan"


def test_top_zones_by_revenue():
    """Verify top N zones extraction and ordering."""
    data = load_bi_data()
    top5 = get_top_zones_by_revenue(data["top_zones"], n=5)
    assert len(top5) == 5
    assert top5.iloc[0]["est_revenue"] >= top5.iloc[1]["est_revenue"]


def test_temporal_heatmap_dimensions():
    """Verify 7x24 pivot dimensions for temporal heatmap."""
    data = load_bi_data()
    pivot = get_temporal_heatmap(data["demand"])
    assert pivot.shape == (7, 24)


def test_archetype_profitability_and_yield():
    """Verify archetype yield and driver-hour metric calculations."""
    data = load_bi_data()
    profit_df = get_archetype_profitability(data["top_zones"], data["archetypes"])
    assert "rev_per_trip_hour" in profit_df.columns
    assert "rev_share_pct" in profit_df.columns
    assert (profit_df["rev_per_trip_hour"] > 0).all()


def test_fleet_efficiency_gap_model():
    """Verify dead-hour and peak-hour volume splits and capacity shift model."""
    data = load_bi_data()
    gap = get_fleet_efficiency_gap(data["demand"])
    assert gap["dead_hours_volume"] > 0
    assert gap["peak_hours_volume"] > gap["dead_hours_volume"]
    assert gap["annual_revenue_lift"] > 0


def test_executive_recommendations():
    """Verify executive recommendation structure."""
    recs = get_executive_recommendations()
    assert len(recs) == 3
    for r in recs:
        assert "priority" in r
        assert "title" in r
        assert "what" in r
        assert "why" in r
        assert "impact" in r
