"""
Integration tests for processed dataset splits.
"""

from pathlib import Path
import duckdb
import pytest

from src.utils import PROCESSED_DATA_DIR


def test_split_files_exist():
    """Verify all Parquet and sample CSV files were created."""
    expected_files = [
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "train_sample.csv",
        "val_sample.csv",
        "test_sample.csv",
        "split_summary.csv",
        "cleaning_summary.csv",
    ]
    for filename in expected_files:
        p = PROCESSED_DATA_DIR / filename
        assert p.exists(), f"Missing expected output file: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"


def test_split_row_counts_and_integrity():
    """Verify exact row counts and absence of nulls in key modeling fields."""
    con = duckdb.connect()

    for split_name, expected_count in [
        ("train", 37_459_831),
        ("val", 3_221_802),
        ("test", 3_777_555),
    ]:
        parquet_path = PROCESSED_DATA_DIR / f"{split_name}.parquet"
        sql = f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(*) FILTER (WHERE base_fare IS NULL OR base_fare < 0) AS invalid_fares,
            COUNT(*) FILTER (WHERE trip_duration_minutes IS NULL OR trip_duration_minutes <= 0) AS invalid_durations,
            COUNT(*) FILTER (WHERE origin_loc_id IS NULL OR dest_loc_id IS NULL) AS null_zones
        FROM read_parquet('{parquet_path.as_posix()}')
        """
        row = con.execute(sql).fetchdf().iloc[0]
        assert int(row["row_count"]) == expected_count, f"{split_name} count mismatch"
        assert int(row["invalid_fares"]) == 0, f"{split_name} has invalid fares"
        assert int(row["invalid_durations"]) == 0, f"{split_name} has invalid durations"
        assert int(row["null_zones"]) == 0, f"{split_name} has null zone IDs"


def test_temporal_non_overlap():
    """Verify strict chronological ordering and zero data leakage across splits."""
    con = duckdb.connect()

    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    train_max = con.execute(f"SELECT MAX(pickup_timestamp) FROM read_parquet('{train_path.as_posix()}')").fetchone()[0]
    val_min = con.execute(f"SELECT MIN(pickup_timestamp) FROM read_parquet('{val_path.as_posix()}')").fetchone()[0]
    val_max = con.execute(f"SELECT MAX(pickup_timestamp) FROM read_parquet('{val_path.as_posix()}')").fetchone()[0]
    test_min = con.execute(f"SELECT MIN(pickup_timestamp) FROM read_parquet('{test_path.as_posix()}')").fetchone()[0]

    assert train_max < val_min, f"Temporal leakage: Train max {train_max} >= Val min {val_min}"
    assert val_max < test_min, f"Temporal leakage: Val max {val_max} >= Test min {test_min}"
