"""
Generate lightweight precomputed assets for Streamlit Community Cloud deployment.
Produces:
- app/data/zone_lookup.csv
- app/data/demand_summary.csv
- app/data/clustering_summary.json
"""

import json
import os
import shutil
from pathlib import Path
import pandas as pd
import duckdb

def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "app" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Zone lookup
    ref_file = root / "data" / "raw" / "reference" / "Urban_Flow_Analytics_Zone_Dataset.csv"
    zone_df = pd.read_csv(ref_file)
    zone_df.to_csv(data_dir / "zone_lookup.csv", index=False)
    print(f"Saved zone lookup ({len(zone_df)} zones)")
    
    # 2. Demand summary: Top 10 zones hourly aggregate + test forecast metrics
    test_parquet = root / "data" / "processed" / "test.parquet"
    train_parquet = root / "data" / "processed" / "train.parquet"
    
    # Aggregate top 10 zones by volume from duckdb
    con = duckdb.connect()
    top_zones_query = f"""
        SELECT origin_loc_id, count(*) as cnt 
        FROM read_parquet('{test_parquet}')
        GROUP BY origin_loc_id ORDER BY cnt DESC LIMIT 10
    """
    top_zones = [r[0] for r in con.execute(top_zones_query).fetchall()]
    print(f"Top zones for demand: {top_zones}")
    
    # Aggregate hourly counts for top zones across March 2026 test set
    hourly_query = f"""
        SELECT 
            origin_loc_id,
            date_trunc('hour', pickup_timestamp) as pickup_hour,
            count(*) as actual_demand
        FROM read_parquet('{test_parquet}')
        WHERE origin_loc_id IN ({','.join(map(str, top_zones))})
        GROUP BY origin_loc_id, date_trunc('hour', pickup_timestamp)
        ORDER BY origin_loc_id, pickup_hour
    """
    hourly_df = con.execute(hourly_query).df()
    
    # Merge zone name
    hourly_df = hourly_df.merge(zone_df[['loc_id', 'zone_name', 'borough_name']], left_on='origin_loc_id', right_on='loc_id', how='left')
    hourly_df.to_csv(data_dir / "demand_summary.csv", index=False)
    print(f"Saved demand summary ({len(hourly_df)} rows)")
    
    # 3. Clustering summary: Archetypes, zone archetypes, and top OD corridors
    from src.clustering import extract_zone_temporal_profiles, cluster_zone_hotspots, extract_od_flows
    
    profiles = extract_zone_temporal_profiles(str(train_parquet), reference_csv_path=str(ref_file))
    clustered, centroids = cluster_zone_hotspots(profiles, n_clusters=5, random_state=42)
    od_flows = extract_od_flows(str(train_parquet), top_n=50, reference_csv_path=str(ref_file))
    
    clustering_data = {
        "archetypes": centroids.to_dict(orient="records"),
        "top_zones": clustered[['loc_id', 'zone_name', 'borough_name', 'total_pickups', 'cluster_id', 'archetype_name', 'avg_distance', 'net_flow_ratio']].head(100).to_dict(orient="records"),
        "cluster_counts": clustered['archetype_name'].value_counts().to_dict(),
        "od_flows": od_flows.head(50).to_dict(orient="records")
    }
    
    with open(data_dir / "clustering_summary.json", "w") as f:
        json.dump(clustering_data, f, indent=2, default=str)
    print("Saved clustering summary JSON")

if __name__ == "__main__":
    main()
