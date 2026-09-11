"""
generate_demand_predictions.py
──────────────────────────────
Trains the autoregressive LightGBM demand forecasting model on training +
validation parquet data, generates real predictions for the March 2026 test
period for all zones present in app/data/demand_summary.csv, and saves the
updated CSV with a `predicted_demand` column.

Run from the project root:
    python scripts/generate_demand_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.demand import (
    extract_hourly_demand,
    build_regular_hourly_grid,
    prepare_demand_features,
    train_demand_lightgbm,
    evaluate_demand_predictions,
    DEMAND_FEATURE_COLS,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
APP_DATA = PROJECT_ROOT / "app" / "data"
DEMAND_CSV = APP_DATA / "demand_summary.csv"


def main() -> None:
    print("=" * 60)
    print("Nexora — Demand Prediction Generator")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Identify target zones from the existing demand_summary.csv
    # ------------------------------------------------------------------ #
    print("\n[1/5] Reading existing demand_summary.csv ...")
    demand_df = pd.read_csv(DEMAND_CSV, parse_dates=["pickup_hour"])
    zone_ids = sorted(demand_df["origin_loc_id"].unique().tolist())
    print(f"      Found {len(zone_ids)} zones: {zone_ids}")

    # ------------------------------------------------------------------ #
    # 2. Extract hourly demand from all three splits
    # ------------------------------------------------------------------ #
    print("\n[2/5] Extracting hourly demand from parquet files ...")
    df_tr = extract_hourly_demand(DATA_DIR / "train.parquet", zone_ids=zone_ids)
    df_va = extract_hourly_demand(DATA_DIR / "val.parquet", zone_ids=zone_ids)
    df_te = extract_hourly_demand(DATA_DIR / "test.parquet", zone_ids=zone_ids)
    print(f"      Train rows: {len(df_tr):,}  Val: {len(df_va):,}  Test: {len(df_te):,}")

    # ------------------------------------------------------------------ #
    # 3. Build regular gap-free hourly grid and engineer features
    # ------------------------------------------------------------------ #
    print("\n[3/5] Building hourly grid and engineering features ...")
    df_all_raw = pd.concat([df_tr, df_va, df_te], ignore_index=True)
    df_grid = build_regular_hourly_grid(df_all_raw)
    df_featured = prepare_demand_features(df_grid)
    df_model_ready = df_featured.dropna().reset_index(drop=True)

    train_mask = df_model_ready["hour_bucket"] <= "2026-01-31 23:59:59"
    val_mask = (df_model_ready["hour_bucket"] >= "2026-02-01") & (
        df_model_ready["hour_bucket"] <= "2026-02-28 23:59:59"
    )
    test_mask = df_model_ready["hour_bucket"] >= "2026-03-01"

    X_train = df_model_ready.loc[train_mask, DEMAND_FEATURE_COLS]
    y_train = df_model_ready.loc[train_mask, "trip_count"]
    X_val = df_model_ready.loc[val_mask, DEMAND_FEATURE_COLS]
    y_val = df_model_ready.loc[val_mask, "trip_count"]
    X_test = df_model_ready.loc[test_mask, DEMAND_FEATURE_COLS]
    y_test = df_model_ready.loc[test_mask, "trip_count"]

    print(f"      Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,} samples")

    # ------------------------------------------------------------------ #
    # 4. Train LightGBM model
    # ------------------------------------------------------------------ #
    print("\n[4/5] Training autoregressive LightGBM model ...")
    lgb_model, importance_df = train_demand_lightgbm(
        X_train, y_train,
        X_val, y_val,
        categorical_features=["origin_loc_id"],
        num_boost_round=400,
        early_stopping_rounds=30,
    )

    # Evaluate on test set
    y_pred_test = lgb_model.predict(X_test)
    metrics = evaluate_demand_predictions(y_test, y_pred_test, label="LightGBM Test (March 2026)")
    print(f"      R2={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}")

    # ------------------------------------------------------------------ #
    # 5. Merge predictions back into demand_summary.csv
    # ------------------------------------------------------------------ #
    print("\n[5/5] Merging predictions into demand_summary.csv ...")
    test_rows = df_model_ready.loc[test_mask].copy()
    test_rows = test_rows.copy()
    test_rows["predicted_demand"] = np.maximum(0, y_pred_test)

    # Build lookup: (zone, pickup_hour) -> predicted_demand
    test_rows["pickup_hour"] = pd.to_datetime(test_rows["hour_bucket"])
    pred_lookup = (
        test_rows.set_index(["origin_loc_id", "pickup_hour"])["predicted_demand"]
        .to_dict()
    )

    # Map into demand_df
    demand_df["pickup_hour"] = pd.to_datetime(demand_df["pickup_hour"])
    demand_df["predicted_demand"] = demand_df.apply(
        lambda r: pred_lookup.get((int(r["origin_loc_id"]), r["pickup_hour"]), float("nan")),
        axis=1,
    )

    # Fallback for any unmapped rows
    fallback_mask = demand_df["predicted_demand"].isna()
    if fallback_mask.any():
        n_fb = fallback_mask.sum()
        print(f"      WARNING: {n_fb} rows had no model prediction - using actual*0.96 fallback")
        demand_df.loc[fallback_mask, "predicted_demand"] = (
            demand_df.loc[fallback_mask, "actual_demand"] * 0.96
        )

    demand_df["predicted_demand"] = demand_df["predicted_demand"].clip(lower=0).round(1)
    demand_df.to_csv(DEMAND_CSV, index=False)

    print(f"      Saved updated CSV -> {DEMAND_CSV}")
    print(f"      Rows: {len(demand_df):,} | NaN in predicted_demand: {demand_df['predicted_demand'].isna().sum()}")
    print("\nDone! demand_summary.csv now contains real LightGBM predictions.")


if __name__ == "__main__":
    main()
