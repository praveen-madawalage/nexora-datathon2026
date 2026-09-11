"""Save K-Means clustering model as pkl for submission."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import joblib
import duckdb
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_DIR = Path('data/processed')

print('Building zone feature profiles from parquet data...')
con = duckdb.connect()
train_path = str((DATA_DIR / 'train.parquet').resolve()).replace('\\', '/')
test_path  = str((DATA_DIR / 'test.parquet').resolve()).replace('\\', '/')
val_path   = str((DATA_DIR / 'val.parquet').resolve()).replace('\\', '/')

zone_csv = str(Path('data/raw/Urban_Flow_Analytics_Zone_Dataset.csv').resolve()).replace('\\', '/')

query = """
SELECT
    t.origin_loc_id AS loc_id,
    COUNT(*) AS total_pickups,
    AVG(t.distance_miles) AS avg_distance,
    AVG(t.trip_duration_minutes) AS avg_duration,
    SUM(CASE WHEN HOUR(t.pickup_timestamp) BETWEEN 6 AND 9 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS morning_share,
    SUM(CASE WHEN HOUR(t.pickup_timestamp) BETWEEN 20 AND 23 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS night_share,
    SUM(CASE WHEN HOUR(t.pickup_timestamp) BETWEEN 11 AND 15 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS midday_share
FROM (
    SELECT * FROM parquet_scan('""" + train_path + """')
    UNION ALL
    SELECT * FROM parquet_scan('""" + val_path + """')
    UNION ALL
    SELECT * FROM parquet_scan('""" + test_path + """')
) t
GROUP BY t.origin_loc_id
HAVING COUNT(*) > 100
"""
df_zones = con.execute(query).df()
con.close()
print('Zone profiles shape:', df_zones.shape)

feature_cols = ['avg_distance', 'avg_duration', 'morning_share', 'night_share', 'midday_share']
X = df_zones[feature_cols].fillna(0).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, labels)
print('Silhouette score (k=5):', round(sil, 4))

bundle = {
    'kmeans_model': kmeans,
    'scaler': scaler,
    'metadata': {
        'task': 'zone_mobility_archetype_clustering',
        'model_type': 'KMeans_k5',
        'n_clusters': 5,
        'feature_cols': feature_cols,
        'n_zones': int(df_zones.shape[0]),
        'silhouette_score': round(float(sil), 4),
        'version': '1.0.0'
    },
    'timestamp': datetime.now().isoformat()
}

out = Path('models/clustering_kmeans.pkl')
joblib.dump(bundle, out)
size_kb = round(out.stat().st_size / 1024, 1)
print('Saved -> models/clustering_kmeans.pkl  (' + str(size_kb) + ' KB)')
