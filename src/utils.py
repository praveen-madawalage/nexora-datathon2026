"""
Urban Flow Analytics - Common Utilities

Provides path management, performance profiling, and evaluation metrics
for the Urban Flow Analytics data processing and modeling workflows.
"""

from contextlib import contextmanager
from pathlib import Path
import time
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Project Directory Paths
# ----------------------------------------------------------------------

def get_project_root() -> Path:
    """Resolve the canonical project root directory."""
    current = Path.cwd().resolve()
    if current.name in ("notebooks", "src", "tests"):
        return current.parent
    return current


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure runtime directories exist
for path in (INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR):
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Timing and Logging Helpers
# ----------------------------------------------------------------------

@contextmanager
def timer(description: str):
    """Context manager for measuring wall-clock execution time."""
    start_time = time.perf_counter()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START: {description}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END:   {description} - Elapsed: {elapsed:.2f}s ({elapsed / 60:.2f}m)")


# ----------------------------------------------------------------------
# Model Evaluation Metrics
# ----------------------------------------------------------------------

def calculate_regression_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Calculate standard regression evaluation metrics: RMSE, MAE, R², and MAPE.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Estimated target values.
    prefix : str, optional
        Optional prefix for metric keys in the output dictionary.
        
    Returns
    -------
    dict
        Dictionary containing calculated metrics: RMSE, MAE, R2, and MAPE.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    # Valid mask to avoid NaN issues
    valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    yt = y_true_arr[valid_mask]
    yp = y_pred_arr[valid_mask]

    if len(yt) == 0:
        return {f"{prefix}rmse": np.nan, f"{prefix}mae": np.nan, f"{prefix}r2": np.nan, f"{prefix}mape": np.nan}

    error = yp - yt
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Nonzero mask for MAPE calculation
    nonzero_mask = yt > 0
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((yp[nonzero_mask] - yt[nonzero_mask]) / yt[nonzero_mask])) * 100.0)
    else:
        mape = np.nan

    return {
        f"{prefix}rmse": round(rmse, 4),
        f"{prefix}mae": round(mae, 4),
        f"{prefix}r2": round(r2, 4),
        f"{prefix}mape": round(mape, 2),
    }


# ----------------------------------------------------------------------
# Parquet / Dataset Loaders
# ----------------------------------------------------------------------

def load_split_dataset(
    split: str,
    columns: Optional[List[str]] = None,
    use_sample: bool = False,
    sample_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load train, val, or test split from data/processed/.
    
    Parameters
    ----------
    split : {'train', 'val', 'test'}
        Name of the dataset split.
    columns : list of str, optional
        Subset of columns to read.
    use_sample : bool, default False
        Whether to load the fast CSV sample instead of the full Parquet.
    sample_rows : int, optional
        If specified, limits the number of rows read.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    if use_sample:
        file_path = PROCESSED_DATA_DIR / f"{split}_sample.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Sample file not found: {file_path}")
        return pd.read_csv(file_path, usecols=columns, nrows=sample_rows)

    file_path = PROCESSED_DATA_DIR / f"{split}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Processed split file not found: {file_path}")

    import duckdb
    con = duckdb.connect()
    cols_clause = ", ".join(columns) if columns else "*"
    limit_clause = f" LIMIT {sample_rows}" if sample_rows else ""
    sql = f"SELECT {cols_clause} FROM read_parquet('{file_path.as_posix()}') {limit_clause}"
    return con.execute(sql).fetchdf()
