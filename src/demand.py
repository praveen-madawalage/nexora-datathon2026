"""
Demand Analytics and Forecasting Engine for Urban Flow Analytics.

This module powers Task 3.1 (The Fleet Dispatcher):
- High-throughput hourly demand aggregation per taxi zone via DuckDB.
- Feature engineering: calendar cyclical indicators, autoregressive lags, and historical rolling windows.
- Production LightGBM regressor optimized for time-series demand forecasting with early stopping.
- Recursive multi-step forecasting engine predicting 24 to 72 hours ahead.
- Operational dispatch recommendations to optimize taxi fleet pre-positioning and minimize passenger wait times.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_LAGS = [1, 2, 3, 24, 48, 168]
DEFAULT_ROLL_WINDOWS = [6, 24, 168]

DEMAND_FEATURE_COLS = [
    "origin_loc_id",
    "hour",
    "dayofweek",
    "is_weekend",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_rush_hour",
    "is_late_night",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_168",
    "roll_mean_6h",
    "roll_mean_24h",
    "roll_std_24h",
    "roll_mean_168h",
]


def get_top_demand_zones(
    parquet_path: Union[str, Path],
    top_n: int = 10,
) -> List[int]:
    """
    Identifies the top N taxi pickup zones ranked by cumulative trip volume.

    Parameters
    ----------
    parquet_path : str or Path
        Path to processed trip Parquet file.
    top_n : int, default 10
        Number of top zones to return.

    Returns
    -------
    list of int
        List of origin_loc_id integers.
    """
    con = duckdb.connect()
    p_str = str(Path(parquet_path).resolve()).replace("\\", "/")
    query = f"""
        SELECT origin_loc_id, count(*) as trip_count
        FROM parquet_scan('{p_str}')
        GROUP BY origin_loc_id
        ORDER BY trip_count DESC
        LIMIT {top_n}
    """
    df = con.execute(query).df()
    con.close()
    return df["origin_loc_id"].astype(int).tolist()


def extract_hourly_demand(
    parquet_path: Union[str, Path],
    zone_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extracts hourly aggregated trip volumes for specified zones using DuckDB.

    Parameters
    ----------
    parquet_path : str or Path
        Path to processed trip Parquet file.
    zone_ids : list of int, optional
        Filter to these origin_loc_id values. If None, extracts all zones.

    Returns
    -------
    pd.DataFrame
        Columns: origin_loc_id, hour_bucket (timestamp), trip_count (int).
    """
    con = duckdb.connect()
    p_str = str(Path(parquet_path).resolve()).replace("\\", "/")
    where_clause = ""
    if zone_ids is not None and len(zone_ids) > 0:
        zone_list_str = ", ".join(map(str, zone_ids))
        where_clause = f"WHERE origin_loc_id IN ({zone_list_str})"

    query = f"""
        SELECT 
            origin_loc_id,
            date_trunc('hour', pickup_timestamp) as hour_bucket,
            count(*) as trip_count
        FROM parquet_scan('{p_str}')
        {where_clause}
        GROUP BY origin_loc_id, date_trunc('hour', pickup_timestamp)
        ORDER BY origin_loc_id, hour_bucket
    """
    df = con.execute(query).df()
    con.close()
    df["origin_loc_id"] = df["origin_loc_id"].astype(int)
    df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])
    df["trip_count"] = df["trip_count"].astype(int)
    return df


def build_regular_hourly_grid(
    df: pd.DataFrame,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Ensures a continuous, gap-free hourly time series for each zone.
    Missing hours are filled with 0 trips.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with origin_loc_id, hour_bucket, trip_count.
    start_time : pd.Timestamp, optional
        Grid start boundary. Defaults to min(df['hour_bucket']).
    end_time : pd.Timestamp, optional
        Grid end boundary. Defaults to max(df['hour_bucket']).

    Returns
    -------
    pd.DataFrame
        Complete hourly grid per zone, sorted by origin_loc_id and hour_bucket.
    """
    if df.empty:
        return df

    zones = sorted(df["origin_loc_id"].unique())
    t_min = start_time if start_time is not None else df["hour_bucket"].min()
    t_max = end_time if end_time is not None else df["hour_bucket"].max()

    all_hours = pd.date_range(t_min, t_max, freq="h")
    grid = (
        pd.MultiIndex.from_product(
            [zones, all_hours], names=["origin_loc_id", "hour_bucket"]
        )
        .to_frame()
        .reset_index(drop=True)
    )

    merged = pd.merge(grid, df, on=["origin_loc_id", "hour_bucket"], how="left")
    merged["trip_count"] = merged["trip_count"].fillna(0).astype(int)
    merged = merged.sort_values(["origin_loc_id", "hour_bucket"]).reset_index(drop=True)
    return merged


def prepare_demand_features(
    df: pd.DataFrame,
    lags: List[int] = DEFAULT_LAGS,
    roll_windows: List[int] = DEFAULT_ROLL_WINDOWS,
) -> pd.DataFrame:
    """
    Constructs comprehensive calendar, autoregressive, and rolling features.

    Features generated:
    - Calendar & cyclical: hour, dayofweek, is_weekend, month, hour_sin, hour_cos, dow_sin, dow_cos.
    - Operational flags: is_rush_hour (7-9, 16-19 weekdays), is_late_night (22-4).
    - Autoregressive lags: t-1, t-2, t-3, t-24 (yesterday), t-48, t-168 (last week same hour).
    - Rolling window historical aggregations: roll_mean_6h, roll_mean_24h, roll_std_24h, roll_mean_168h.

    Zero Leakage:
    Rolling calculations are shifted by 1 hour (strictly using t-1 and prior),
    ensuring that the target trip_count at time t is never leaked.

    Parameters
    ----------
    df : pd.DataFrame
        Continuous regular hourly DataFrame.
    lags : list of int, optional
        Lags in hours.
    roll_windows : list of int, optional
        Rolling windows in hours.

    Returns
    -------
    pd.DataFrame
        Feature matrix including engineered columns and target trip_count.
    """
    out = df.copy().sort_values(["origin_loc_id", "hour_bucket"]).reset_index(drop=True)

    # Calendar features
    out["hour"] = out["hour_bucket"].dt.hour
    out["dayofweek"] = out["hour_bucket"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["month"] = out["hour_bucket"].dt.month

    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)

    # Operational indicators
    out["is_rush_hour"] = (
        (out["hour"].isin([7, 8, 9, 16, 17, 18, 19])) & (out["is_weekend"] == 0)
    ).astype(int)
    out["is_late_night"] = (out["hour"].isin([22, 23, 0, 1, 2, 3, 4])).astype(int)

    # Autoregressive lags per zone
    for l in lags:
        out[f"lag_{l}"] = out.groupby("origin_loc_id")["trip_count"].shift(l)

    # Rolling statistics per zone (shifted by 1 to prevent data leakage)
    out["roll_mean_6h"] = out.groupby("origin_loc_id")["trip_count"].transform(
        lambda s: s.shift(1).rolling(6).mean()
    )
    out["roll_mean_24h"] = out.groupby("origin_loc_id")["trip_count"].transform(
        lambda s: s.shift(1).rolling(24).mean()
    )
    out["roll_std_24h"] = out.groupby("origin_loc_id")["trip_count"].transform(
        lambda s: s.shift(1).rolling(24).std()
    )
    out["roll_mean_168h"] = out.groupby("origin_loc_id")["trip_count"].transform(
        lambda s: s.shift(1).rolling(168).mean()
    )

    return out


def train_demand_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: Optional[List[str]] = None,
    params: Optional[dict[str, Any]] = None,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 40,
) -> Tuple[lgb.Booster, pd.DataFrame]:
    """
    Trains a LightGBM regressor for demand forecasting with early stopping.

    Parameters
    ----------
    X_train, y_train : train features and target
    X_val, y_val : validation features and target
    categorical_features : list of categorical feature names
    params : hyperparameter overrides
    num_boost_round : maximum boosting rounds
    early_stopping_rounds : patience for early stopping

    Returns
    -------
    model : lgb.Booster
    importance_df : pd.DataFrame with Gain and Split importances
    """
    if categorical_features is None:
        categorical_features = ["origin_loc_id"] if "origin_loc_id" in X_train.columns else []

    lgb_params: dict[str, Any] = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        lgb_params.update(params)

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    return booster, importance_df


def evaluate_demand_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    label: str = "Demand Model",
) -> dict[str, Any]:
    """
    Computes standard and operational metrics for time-series demand predictions.

    Metrics:
    - RMSE (trips/hr)
    - MAE (trips/hr)
    - R² Score
    - WAPE (Weighted Absolute Percentage Error, %): sum(|y - y_hat|) / sum(y) * 100
    - MAPE (Mean Absolute Percentage Error, % for non-zero entries)

    Returns
    -------
    dict
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_pred, dtype=float), a_min=0.0, a_max=None)

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0

    total_actual = float(np.sum(yt))
    wape = float(np.sum(np.abs(yt - yp)) / total_actual * 100.0) if total_actual > 0 else 0.0

    nonzero_mask = yt > 0
    mape = (
        float(np.mean(np.abs((yt[nonzero_mask] - yp[nonzero_mask]) / yt[nonzero_mask])) * 100.0)
        if np.any(nonzero_mask)
        else 0.0
    )

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "wape": round(wape, 2),
        "mape": round(mape, 2),
        "model_label": label,
    }


def forecast_demand_recursive(
    model: lgb.Booster,
    history_df: pd.DataFrame,
    zone_id: int,
    forecast_start: pd.Timestamp,
    horizon_hours: int = 72,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Executes recursive multi-step forecasting for a specified zone.
    Generates minute/hour predictions for 24h to 72h operational horizons.

    Parameters
    ----------
    model : lgb.Booster
        Trained LightGBM model.
    history_df : pd.DataFrame
        Complete hourly DataFrame for the zone containing historical observations
        (must have at least 168 hours prior to forecast_start).
    zone_id : int
        origin_loc_id for forecasting.
    forecast_start : pd.Timestamp
        Timestamp of the first forecast hour (T+1).
    horizon_hours : int, default 72
        Number of consecutive hours to forecast ahead (e.g. 24, 48, 72).
    feature_cols : list of str, optional
        List of feature column names expected by the model.

    Returns
    -------
    pd.DataFrame
        Columns: origin_loc_id, hour_bucket, step_ahead, predicted_demand, actual_demand (if available).
    """
    if feature_cols is None:
        feature_cols = DEMAND_FEATURE_COLS

    df_z = history_df[history_df["origin_loc_id"] == zone_id].sort_values("hour_bucket").reset_index(drop=True)

    hist_data = df_z[df_z["hour_bucket"] < forecast_start]
    if len(hist_data) < 168:
        raise ValueError(
            f"Insufficient history for zone {zone_id}: {len(hist_data)} rows available, minimum 168 required."
        )

    series_counts = hist_data["trip_count"].tolist()
    series_timestamps = hist_data["hour_bucket"].tolist()

    future_data = df_z[(df_z["hour_bucket"] >= forecast_start) & (df_z["hour_bucket"] < forecast_start + pd.Timedelta(hours=horizon_hours))]
    actual_dict = dict(zip(future_data["hour_bucket"], future_data["trip_count"]))

    results = []

    for step in range(horizon_hours):
        t_next = forecast_start + pd.Timedelta(hours=step)
        h = t_next.hour
        dow = t_next.dayofweek
        is_wknd = int(dow >= 5)
        m = t_next.month

        row_feat = {
            "origin_loc_id": zone_id,
            "hour": h,
            "dayofweek": dow,
            "is_weekend": is_wknd,
            "month": m,
            "hour_sin": np.sin(2 * np.pi * h / 24.0),
            "hour_cos": np.cos(2 * np.pi * h / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "is_rush_hour": int((h in [7, 8, 9, 16, 17, 18, 19]) and (is_wknd == 0)),
            "is_late_night": int(h in [22, 23, 0, 1, 2, 3, 4]),
            "lag_1": series_counts[-1],
            "lag_2": series_counts[-2],
            "lag_3": series_counts[-3],
            "lag_24": series_counts[-24],
            "lag_48": series_counts[-48],
            "lag_168": series_counts[-168],
            "roll_mean_6h": float(np.mean(series_counts[-6:])),
            "roll_mean_24h": float(np.mean(series_counts[-24:])),
            "roll_std_24h": float(np.std(series_counts[-24:], ddof=1)) if len(series_counts) >= 24 else 0.0,
            "roll_mean_168h": float(np.mean(series_counts[-168:])),
        }

        X_step = pd.DataFrame([row_feat])[feature_cols]
        pred_val = max(0.0, float(model.predict(X_step)[0]))

        actual_val = actual_dict.get(t_next, np.nan)

        results.append(
            {
                "origin_loc_id": zone_id,
                "hour_bucket": t_next,
                "step_ahead": step + 1,
                "predicted_demand": round(pred_val, 1),
                "actual_demand": actual_val,
            }
        )

        # Autoregressive recursive append
        series_counts.append(pred_val)
        series_timestamps.append(t_next)

    return pd.DataFrame(results)


def generate_dispatch_recommendations(
    forecast_df: pd.DataFrame,
    zone_reference_df: Optional[pd.DataFrame] = None,
    threshold_percentile: float = 80.0,
) -> pd.DataFrame:
    """
    Translates forecasted demand into operational fleet dispatch actions.

    Identifies peak demand surge hours and recommends pre-positioning quotas
    for taxi fleet operators.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of forecast_demand_recursive or multi-zone forecast table.
    zone_reference_df : pd.DataFrame, optional
        Metadata table containing zone_name, borough_name, service_zone.
    threshold_percentile : float, default 80.0
        Percentile of predicted volume to trigger 'High Priority Surge' dispatch.

    Returns
    -------
    pd.DataFrame
        Operational dispatch schedule with zone names, dispatch windows,
        predicted hourly demand, and recommended fleet positioning level.
    """
    df = forecast_df.copy()
    if zone_reference_df is not None:
        ref_map = zone_reference_df.set_index("loc_id")["zone_name"].to_dict()
        boro_map = zone_reference_df.set_index("loc_id")["borough_name"].to_dict()
        df["zone_name"] = df["origin_loc_id"].map(ref_map).fillna("Unknown Zone")
        df["borough"] = df["origin_loc_id"].map(boro_map).fillna("Unknown")

    p_thresh = np.percentile(df["predicted_demand"], threshold_percentile)

    def classify_action(val: float) -> str:
        if val >= p_thresh * 1.25:
            return "URGENT PRE-POSITION (SURGE > +25%)"
        elif val >= p_thresh:
            return "HIGH DISPATCH PRIORITY"
        elif val >= p_thresh * 0.5:
            return "MODERATE BUFFER"
        else:
            return "ROUTINE PATROL"

    df["dispatch_action"] = df["predicted_demand"].apply(classify_action)
    df["recommended_staging_vehicles"] = np.ceil(df["predicted_demand"] * 0.85).astype(int)

    return df.sort_values(["predicted_demand"], ascending=False).reset_index(drop=True)
