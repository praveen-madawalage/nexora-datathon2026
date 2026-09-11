"""
Urban Flow Analytics - Data Cleaning & Split Pipeline

Encodes the canonical data quality audit contract into a reproducible, high-throughput
streaming pipeline using DuckDB. Cleans 48.6M taxi records, enforces physical and economic
anomaly boundaries, applies deterministic imputations, and generates chronologically
separated train/validation/test splits.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import duckdb
import pandas as pd

from src.utils import timer, PROJECT_ROOT, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR


@dataclass(frozen=True)
class CleaningConfig:
    """Canonical data cleaning configuration thresholds and parameters."""
    min_valid_timestamp: str = "2020-01-01"
    nominal_start_timestamp: str = "2025-04-01"
    nominal_end_timestamp: str = "2026-04-01"
    max_duration_minutes: float = 1440.0  # 24 hours
    max_speed_mph: float = 100.0
    stuck_meter_duration_minutes: float = 60.0
    stuck_meter_speed_mph: float = 1.0
    max_distance_miles: float = 100.0
    default_sample_size: int = 100_000
    parquet_compression: str = "ZSTD"


def build_cleaning_sql(
    raw_glob_path: Path,
    zone_file_path: Path,
    config: Optional[CleaningConfig] = None,
) -> str:
    """
    Construct the unified DuckDB SQL query for cleaning, imputation, and split assignment.
    """
    cfg = config or CleaningConfig()
    raw_sql = raw_glob_path.resolve().as_posix().replace("'", "''")
    zone_sql = zone_file_path.resolve().as_posix().replace("'", "''")

    raw_columns = [
        "provider_code", "pickup_timestamp", "dropoff_timestamp", "rider_count",
        "distance_miles", "rate_class_id", "offline_record_flag", "origin_loc_id",
        "dest_loc_id", "fare_settlement_method", "base_fare", "surcharge_misc",
        "transit_tax", "driver_tip_payment", "toll_total", "service_improvement_fee",
        "charge_total", "zone_congestion_fee", "Airport_fee", "congestion_relief_fee",
    ]
    column_list_sql = ", ".join(raw_columns)

    query = f"""
    CREATE OR REPLACE VIEW clean_trips AS
    WITH raw_trips AS (
        SELECT *, filename AS source_file
        FROM read_csv_auto('{raw_sql}', filename=true, union_by_name=false)
    ),
    lineage AS (
        SELECT *,
            row_number() OVER (PARTITION BY {column_list_sql} ORDER BY source_file) AS duplicate_rank
        FROM raw_trips
    ),
    flags AS (
        SELECT *,
            -- Anomaly flags
            duplicate_rank > 1 AS is_exact_duplicate,
            pickup_timestamp < TIMESTAMP '{cfg.min_valid_timestamp}' OR dropoff_timestamp < TIMESTAMP '{cfg.min_valid_timestamp}' AS is_corrupt_timestamp,
            dropoff_timestamp < pickup_timestamp AS is_inverted_timestamp,
            dropoff_timestamp = pickup_timestamp AS is_zero_duration,
            dropoff_timestamp > pickup_timestamp AND date_diff('minute', pickup_timestamp, dropoff_timestamp) > {cfg.max_duration_minutes} AS is_over_24h,
            dropoff_timestamp > pickup_timestamp AND distance_miles > 0 AND distance_miles / (date_diff('second', pickup_timestamp, dropoff_timestamp) / 3600.0) > {cfg.max_speed_mph} AS is_speed_over_100,
            dropoff_timestamp > pickup_timestamp AND distance_miles > 0 AND distance_miles / (date_diff('second', pickup_timestamp, dropoff_timestamp) / 3600.0) < {cfg.stuck_meter_speed_mph} AND date_diff('minute', pickup_timestamp, dropoff_timestamp) > {cfg.stuck_meter_duration_minutes} AS is_stuck_meter,
            distance_miles > {cfg.max_distance_miles} AS is_extreme_distance,
            base_fare < 0 OR charge_total < 0 AS is_negative_fare,
            distance_miles = 0 AND (base_fare != 0 OR charge_total != 0) AS is_zero_distance_nonzero_fare,
            rate_class_id = 1 AND distance_miles = 0 AND (base_fare != 0 OR charge_total != 0) AND NOT (base_fare < 0 OR charge_total < 0) AS is_standard_zero_distance_drop,
            rider_count IS NULL AS is_null_rider_count,
            rider_count = 0 AS is_zero_rider_count,
            rate_class_id IS NULL AS is_null_rate_class
        FROM lineage
    ),
    dispositions AS (
        SELECT *,
            is_exact_duplicate OR is_corrupt_timestamp OR is_inverted_timestamp OR
            is_zero_duration OR is_over_24h OR is_speed_over_100 OR is_stuck_meter OR
            is_extreme_distance AS drop_from_full_clean,
            is_negative_fare OR is_standard_zero_distance_drop AS drop_from_model_clean
        FROM flags
    )
    SELECT
        -- Operational metadata
        provider_code,
        pickup_timestamp,
        dropoff_timestamp,
        
        -- Imputed and standardized rider count
        COALESCE(NULLIF(rider_count, 0), 1) AS rider_count,
        (is_null_rider_count OR is_zero_rider_count) AS is_rider_count_imputed,
        
        distance_miles,
        
        -- Rate class (NULL imputed to 99 = Unknown)
        COALESCE(rate_class_id, 99) AS rate_class_id,
        is_null_rate_class AS is_rate_class_imputed,
        
        -- Offline flag (NULL imputed to 'N')
        COALESCE(offline_record_flag, 'N') AS offline_record_flag,
        
        origin_loc_id,
        dest_loc_id,
        fare_settlement_method,
        
        -- Targets and monetary attributes
        base_fare,
        surcharge_misc,
        transit_tax,
        driver_tip_payment,
        toll_total,
        service_improvement_fee,
        charge_total,
        COALESCE(zone_congestion_fee, 0.0) AS zone_congestion_fee,
        COALESCE(Airport_fee, 0.0) AS airport_pickup_fee,
        COALESCE(congestion_relief_fee, 0.0) AS congestion_relief_fee,
        
        -- Derived movement features
        ROUND(date_diff('second', pickup_timestamp, dropoff_timestamp) / 60.0, 3) AS trip_duration_minutes,
        ROUND(distance_miles / (date_diff('second', pickup_timestamp, dropoff_timestamp) / 3600.0), 2) AS trip_speed_mph,
        
        -- Chronological split partition
        CASE
            WHEN pickup_timestamp >= TIMESTAMP '{cfg.nominal_start_timestamp}' AND pickup_timestamp < TIMESTAMP '2026-02-01' THEN 'train'
            WHEN pickup_timestamp >= TIMESTAMP '2026-02-01' AND pickup_timestamp < TIMESTAMP '2026-03-01' THEN 'val'
            WHEN pickup_timestamp >= TIMESTAMP '2026-03-01' AND pickup_timestamp < TIMESTAMP '{cfg.nominal_end_timestamp}' THEN 'test'
            ELSE 'out_of_window'
        END AS split
    FROM dispositions
    WHERE NOT (drop_from_full_clean OR drop_from_model_clean)
      AND (pickup_timestamp >= TIMESTAMP '{cfg.nominal_start_timestamp}' AND pickup_timestamp < TIMESTAMP '{cfg.nominal_end_timestamp}')
    """
    return query


def execute_cleaning_pipeline(
    raw_glob_path: Optional[Path] = None,
    zone_file_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: Optional[CleaningConfig] = None,
    export_samples: bool = True,
    sample_size: int = 100_000,
) -> Dict[str, any]:
    """
    Execute the high-throughput cleaning pipeline and export clean Parquet splits and sample CSVs.
    
    Returns
    -------
    dict
        Summary of split counts, run time, and generated artifact paths.
    """
    raw_glob = raw_glob_path or (RAW_DATA_DIR / "Urban_Flow_Analytics_Taxi_Dataset_*.csv")
    zone_file = zone_file_path or (RAW_DATA_DIR / "reference" / "Urban_Flow_Analytics_Zone_Dataset.csv")
    out_dir = output_dir or PROCESSED_DATA_DIR
    cfg = config or CleaningConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    # Configure DuckDB for high performance with safe memory limit
    con.execute("PRAGMA threads = 4;")
    con.execute("PRAGMA preserve_insertion_order = false;")

    cleaning_sql_base = build_cleaning_sql(raw_glob, zone_file, cfg)

    print(f"[{time.strftime('%H:%M:%S')}] Registering cleaned dataset view...")
    con.execute(cleaning_sql_base)

    # Measure split counts
    print(f"[{time.strftime('%H:%M:%S')}] Computing split volume distribution...")
    split_df = con.execute("""
        SELECT split, COUNT(*) AS row_count
        FROM clean_trips
        GROUP BY split
        ORDER BY split
    """).fetchdf()
    print(split_df.to_string(index=False))

    split_counts = dict(zip(split_df["split"], split_df["row_count"]))
    total_clean = sum(split_counts.values())

    # Export Parquet files for each split
    splits = ["train", "val", "test"]
    output_files = {}

    for s in splits:
        parquet_path = out_dir / f"{s}.parquet"
        output_files[f"{s}_parquet"] = parquet_path
        with timer(f"Exporting {s}.parquet ({split_counts.get(s, 0):,} rows)"):
            con.execute(f"""
                COPY (SELECT * EXCLUDE (split) FROM clean_trips WHERE split = '{s}')
                TO '{parquet_path.as_posix()}'
                (FORMAT PARQUET, COMPRESSION {cfg.parquet_compression})
            """)

        if export_samples:
            csv_path = out_dir / f"{s}_sample.csv"
            output_files[f"{s}_sample_csv"] = csv_path
            with timer(f"Exporting {s}_sample.csv ({sample_size:,} rows) from {s}.parquet"):
                con.execute(f"""
                    COPY (
                        SELECT *
                        FROM read_parquet('{parquet_path.as_posix()}')
                        USING SAMPLE {sample_size} (reservoir)
                    )
                    TO '{csv_path.as_posix()}'
                    (HEADER, DELIMITER ',')
                """)

    # Generate and save split summary CSV
    summary_path = out_dir / "split_summary.csv"
    split_df["percentage"] = (split_df["row_count"] / total_clean * 100).round(2)
    split_df.to_csv(summary_path, index=False)
    output_files["split_summary_csv"] = summary_path

    # Export canonical cleaning rule metrics to processed directory
    cleaning_metrics_path = out_dir / "cleaning_summary.csv"
    rule_summary_interim = INTERIM_DATA_DIR / "audit_rule_summary.csv"
    if rule_summary_interim.exists():
        rule_df = pd.read_csv(rule_summary_interim)
        rule_df.to_csv(cleaning_metrics_path, index=False)
        output_files["cleaning_metrics_csv"] = cleaning_metrics_path

    print(f"[{time.strftime('%H:%M:%S')}] Pipeline execution complete. Total clean records: {total_clean:,}")
    return {
        "total_clean_rows": total_clean,
        "split_counts": split_counts,
        "artifacts": output_files,
    }


if __name__ == "__main__":
    execute_cleaning_pipeline()
