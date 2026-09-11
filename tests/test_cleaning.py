"""
Unit tests for data cleaning logic and configuration.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.cleaning import CleaningConfig, build_cleaning_sql
from src.utils import PROJECT_ROOT, RAW_DATA_DIR


def test_cleaning_config_defaults():
    """Verify canonical cleaning thresholds match the verified audit contract."""
    cfg = CleaningConfig()
    assert cfg.min_valid_timestamp == "2020-01-01"
    assert cfg.nominal_start_timestamp == "2025-04-01"
    assert cfg.nominal_end_timestamp == "2026-04-01"
    assert cfg.max_duration_minutes == 1440.0
    assert cfg.max_speed_mph == 100.0
    assert cfg.stuck_meter_duration_minutes == 60.0
    assert cfg.stuck_meter_speed_mph == 1.0
    assert cfg.max_distance_miles == 100.0
    assert cfg.default_sample_size == 100_000


def test_build_cleaning_sql_structure():
    """Verify the generated SQL query defines canonical CTEs and view."""
    cfg = CleaningConfig()
    raw_glob = RAW_DATA_DIR / "Urban_Flow_Analytics_Taxi_Dataset_*.csv"
    zone_file = RAW_DATA_DIR / "reference" / "Urban_Flow_Analytics_Zone_Dataset.csv"
    
    sql = build_cleaning_sql(raw_glob, zone_file, cfg)
    assert "CREATE OR REPLACE VIEW clean_trips AS" in sql
    assert "is_exact_duplicate" in sql
    assert "is_corrupt_timestamp" in sql
    assert "is_inverted_timestamp" in sql
    assert "is_zero_duration" in sql
    assert "is_speed_over_100" in sql
    assert "is_stuck_meter" in sql
    assert "is_extreme_distance" in sql
    assert "is_negative_fare" in sql
    assert "is_standard_zero_distance_drop" in sql
    assert "trip_duration_minutes" in sql
    assert "trip_speed_mph" in sql
    assert "split" in sql
