"""
Urban Flow Analytics - Model Training & Evaluation Engine

Implements baseline linear regression, high-performance gradient boosting
(LightGBM and XGBoost), domain-specific evaluation metrics, tolerance bands,
and model serialization for Upfront Fare Pricing and On-Time Arrival Duration.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb

from src.features import (
    prepare_upfront_pricing_features,
    prepare_trip_duration_features,
)
from src.utils import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    calculate_regression_metrics,
    timer,
)


def evaluate_regression(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    label: str = "Model",
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Compute comprehensive regression metrics: RMSE, MAE, R2, RMSLE, and MAPE.
    """
    metrics = calculate_regression_metrics(y_true, y_pred, prefix=prefix)
    metrics["model_label"] = label
    return metrics


def evaluate_duration_tolerance(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Calculate passenger-centric duration tolerance metrics for Task 2.2.
    
    Computes the percentage of trip duration predictions that fall within
    strict operational bands (+/-3 min, +/-5 min, +/-10 min, +/-15 min),
    along with median absolute error and 90th percentile absolute error.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    valid_mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid_mask]
    yp = yp[valid_mask]

    if len(yt) == 0:
        return {}

    abs_errors = np.abs(yp - yt)
    n = len(abs_errors)

    pct_3min = float(np.sum(abs_errors <= 3.0) / n * 100.0)
    pct_5min = float(np.sum(abs_errors <= 5.0) / n * 100.0)
    pct_10min = float(np.sum(abs_errors <= 10.0) / n * 100.0)
    pct_15min = float(np.sum(abs_errors <= 15.0) / n * 100.0)
    median_abs_err = float(np.median(abs_errors))
    p90_abs_err = float(np.percentile(abs_errors, 90))

    return {
        f"{prefix}pct_within_3min": round(pct_3min, 2),
        f"{prefix}pct_within_5min": round(pct_5min, 2),
        f"{prefix}pct_within_10min": round(pct_10min, 2),
        f"{prefix}pct_within_15min": round(pct_15min, 2),
        f"{prefix}median_abs_err_min": round(median_abs_err, 2),
        f"{prefix}p90_abs_err_min": round(p90_abs_err, 2),
    }


def train_linear_baseline(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    alpha: float = 1.0,
) -> Ridge:
    """
    Train a fast Ridge linear regression benchmark.
    Uses strictly numeric features to establish an interpretable baseline.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X_train[numeric_cols].fillna(0.0)

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_num, y_train)
    model.numeric_cols_ = numeric_cols
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    categorical_features: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 0,
) -> Tuple[lgb.Booster, pd.DataFrame, Dict[str, float]]:
    """
    Train a LightGBM regressor with early stopping and native categorical support.
    
    Returns
    -------
    booster : lgb.Booster
        Trained model.
    importance_df : pd.DataFrame
        DataFrame with feature importance (gain and split count).
    best_score : dict
        Validation metrics at best iteration.
    """
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    X_tr = X_train.copy()
    cat_cols = categorical_features or [
        "origin_loc_id",
        "dest_loc_id",
        "rate_class_id",
        "provider_code",
        "fare_settlement_method",
    ]
    cat_cols = [c for c in cat_cols if c in X_tr.columns]
    for c in cat_cols:
        if X_tr[c].dtype == object:
            X_tr[c] = X_tr[c].astype("category")

    dtrain = lgb.Dataset(X_tr, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_sets = [dtrain]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        X_v = X_val.copy()
        for c in cat_cols:
            if c in X_v.columns and X_v[c].dtype == object:
                X_v[c] = X_v[c].astype("category")
        dval = lgb.Dataset(X_v, label=y_val, reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)
        valid_sets.append(dval)
        valid_names.append("val")

    callbacks = []
    if early_stopping_rounds and X_val is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
    if verbose_eval > 0:
        callbacks.append(lgb.log_evaluation(period=verbose_eval))
    else:
        callbacks.append(lgb.log_evaluation(period=0))

    booster = lgb.train(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    importance_df = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    best_score = booster.best_score
    return booster, importance_df, best_score


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> Tuple[xgb.Booster, pd.DataFrame]:
    """
    Train an XGBoost regressor for benchmark comparison.
    Converts categorical columns to integer codes to support DMatrix.
    """
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    X_tr = X_train.copy()
    for col in X_tr.select_dtypes(include=["object", "category"]).columns:
        X_tr[col] = X_tr[col].astype("category").cat.codes

    dtrain = xgb.DMatrix(X_tr, label=y_train)
    evals = [(dtrain, "train")]

    if X_val is not None and y_val is not None:
        X_v = X_val.copy()
        for col in X_v.select_dtypes(include=["object", "category"]).columns:
            X_v[col] = X_v[col].astype("category").cat.codes
        dval = xgb.DMatrix(X_v, label=y_val)
        evals.append((dval, "val"))

    booster = xgb.train(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
        verbose_eval=False,
    )

    scores = booster.get_score(importance_type="gain")
    importance_df = pd.DataFrame([
        {"feature": k, "importance_gain": v} for k, v in scores.items()
    ]).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    return booster, importance_df


def predict_lightgbm(
    booster: lgb.Booster,
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Generate predictions from LightGBM booster ensuring categorical alignment.
    """
    X_pred = X.copy()
    if booster.pandas_categorical:
        # Match only the columns that were recorded as pandas categoricals during training
        for col in X_pred.columns:
            if X_pred[col].dtype == object:
                X_pred[col] = X_pred[col].astype("category")
    return booster.predict(X_pred)


def predict_xgboost(
    booster: xgb.Booster,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions from XGBoost booster, encoding categories to integer codes.
    """
    X_pred = X.copy()
    for col in X_pred.select_dtypes(include=["object", "category"]).columns:
        X_pred[col] = X_pred[col].astype("category").cat.codes
    dmatrix = xgb.DMatrix(X_pred)
    return booster.predict(dmatrix)


def save_model(
    model: Any,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save trained model and associated metadata via joblib.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "metadata": metadata or {},
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    joblib.dump(payload, path)
    return path


def load_model(filepath: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
    """
    Load saved model bundle.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})
    return payload, {}


def load_modeling_data(
    split: str,
    task: str = "fare",
    sample_rows: Optional[int] = None,
    use_sample: bool = False,
    include_borough_categories: bool = False,
    reservoir_sampling: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load pre-split dataset from data/processed/ and assemble the pre-trip feature matrix.
    
    Parameters
    ----------
    split : {'train', 'val', 'test'}
        Dataset split name.
    task : {'fare', 'duration'}
        Target prediction task.
    sample_rows : int, optional
        Row limit if sampling.
    use_sample : bool, default False
        Whether to use fast sample CSV instead of Parquet.
    include_borough_categories : bool, default False
        Whether to retain categorical borough strings.
    reservoir_sampling : bool, default True
        If sample_rows is specified, whether to use uniform reservoir sampling
        across the whole split rather than top N rows.
        
    Returns
    -------
    X : pd.DataFrame
        Pre-trip feature matrix.
    y : pd.Series
        Target vector.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    if task == "fare":
        cols = [
            "provider_code",
            "pickup_timestamp",
            "distance_miles",
            "rate_class_id",
            "rider_count",
            "origin_loc_id",
            "dest_loc_id",
            "fare_settlement_method",
            "base_fare",
        ]
    elif task == "duration":
        cols = [
            "pickup_timestamp",
            "distance_miles",
            "rate_class_id",
            "rider_count",
            "origin_loc_id",
            "dest_loc_id",
            "trip_duration_minutes",
        ]
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'fare' or 'duration'.")

    if use_sample:
        sample_file = PROCESSED_DATA_DIR / f"{split}_sample.csv"
        df = pd.read_csv(sample_file, usecols=cols, nrows=sample_rows)
    else:
        parquet_file = PROCESSED_DATA_DIR / f"{split}.parquet"
        con = duckdb.connect()
        cols_str = ", ".join(cols)
        if sample_rows:
            if reservoir_sampling:
                sampling_clause = f"USING SAMPLE {sample_rows} (reservoir, 42)"
            else:
                sampling_clause = f"LIMIT {sample_rows}"
        else:
            sampling_clause = ""
        sql = f"SELECT {cols_str} FROM read_parquet('{parquet_file.as_posix()}') {sampling_clause}"
        df = con.execute(sql).fetchdf()

    if task == "fare":
        features = prepare_upfront_pricing_features(
            df,
            include_target=True,
            include_borough_categories=include_borough_categories,
        )
        X = features.drop(columns=["base_fare"])
        y = features["base_fare"]
    else:
        features = prepare_trip_duration_features(
            df,
            include_target=True,
            include_borough_categories=include_borough_categories,
        )
        X = features.drop(columns=["trip_duration_minutes"])
        y = features["trip_duration_minutes"]

    cat_cols = [
        "origin_loc_id",
        "dest_loc_id",
        "rate_class_id",
        "provider_code",
        "fare_settlement_method",
        "origin_borough",
        "dest_borough",
    ]
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return X, y
