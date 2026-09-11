"""Save demand LightGBM model as pkl for submission."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from src.demand import (
    extract_hourly_demand, build_regular_hourly_grid,
    prepare_demand_features, train_demand_lightgbm,
    evaluate_demand_predictions, DEMAND_FEATURE_COLS
)

DATA_DIR = Path('data/processed')
demand_df = pd.read_csv('app/data/demand_summary.csv', parse_dates=['pickup_hour'])
zone_ids = sorted(demand_df['origin_loc_id'].unique().tolist())

df_tr = extract_hourly_demand(DATA_DIR / 'train.parquet', zone_ids=zone_ids)
df_va = extract_hourly_demand(DATA_DIR / 'val.parquet', zone_ids=zone_ids)
df_te = extract_hourly_demand(DATA_DIR / 'test.parquet', zone_ids=zone_ids)

df_grid = build_regular_hourly_grid(pd.concat([df_tr, df_va, df_te], ignore_index=True))
df_feat = prepare_demand_features(df_grid).dropna().reset_index(drop=True)

train_mask = df_feat['hour_bucket'] <= '2026-01-31 23:59:59'
val_mask = (df_feat['hour_bucket'] >= '2026-02-01') & (df_feat['hour_bucket'] <= '2026-02-28 23:59:59')
test_mask = df_feat['hour_bucket'] >= '2026-03-01'

X_train = df_feat.loc[train_mask, DEMAND_FEATURE_COLS]
y_train = df_feat.loc[train_mask, 'trip_count']
X_val   = df_feat.loc[val_mask, DEMAND_FEATURE_COLS]
y_val   = df_feat.loc[val_mask, 'trip_count']
X_test  = df_feat.loc[test_mask, DEMAND_FEATURE_COLS]
y_test  = df_feat.loc[test_mask, 'trip_count']

print('Training demand model...')
lgb_model, importance_df = train_demand_lightgbm(
    X_train, y_train, X_val, y_val,
    categorical_features=['origin_loc_id'],
    num_boost_round=400, early_stopping_rounds=30
)

y_pred = lgb_model.predict(X_test)
metrics = evaluate_demand_predictions(y_test, y_pred, label='LightGBM Test')
r2_val = round(metrics['r2'], 4)
mae_val = round(metrics['mae'], 2)
rmse_val = round(metrics['rmse'], 2)
print('R2=' + str(r2_val) + '  MAE=' + str(mae_val) + '  RMSE=' + str(rmse_val))

bundle = {
    'model': lgb_model,
    'metadata': {
        'task': 'zone_hourly_demand_forecasting',
        'model_type': 'LightGBM_Booster_Autoregressive',
        'target_zones': zone_ids,
        'features': DEMAND_FEATURE_COLS,
        'categorical_features': ['origin_loc_id'],
        'horizon_hours': '1-72',
        'test_metrics': {'rmse': rmse_val, 'mae': mae_val, 'r2': r2_val},
        'version': '1.0.0'
    },
    'timestamp': datetime.now().isoformat()
}

out = Path('models/demand_lgbm.pkl')
joblib.dump(bundle, out)
size_kb = round(out.stat().st_size / 1024)
print('Saved -> models/demand_lgbm.pkl  (' + str(size_kb) + ' KB)')
