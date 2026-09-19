"""
Universal Datathon Rapid Triage and Implementation Generator.

Usage:
    python scripts/universal_triage.py <path_to_csv_or_parquet> [--target <col_name>]

This script is designed for 5-6 hour datathons:
1. Ingests any tabular dataset (CSV or Parquet) safely.
2. Identifies data types, missing values, anomalies, and column distributions.
3. Automatically classifies the problem: Regression, Binary/Multiclass Classification, Time Series, or Clustering.
4. Generates copy-paste baseline training code tailored to your dataset.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def load_dataset(file_path: Path, max_rows: int = 200_000) -> pd.DataFrame:
    """Load dataset with automatic format detection and smart sampling if large."""
    suffix = file_path.suffix.lower()
    print(f"\n[1] Loading dataset: {file_path.name} (File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB)")

    if suffix in (".csv", ".txt"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f) - 1
            
            if line_count > max_rows:
                print(f"    Notice: Large file ({line_count:,} rows). Reading first {max_rows:,} rows for rapid triage.")
                df = pd.read_csv(file_path, nrows=max_rows, low_memory=False)
            else:
                df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"    Standard read failed ({e}). Retrying with latin-1 encoding...")
            df = pd.read_csv(file_path, nrows=max_rows, encoding="latin-1", low_memory=False)

    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(file_path)
        if len(df) > max_rows:
            print(f"    Notice: Large parquet ({len(df):,} rows). Sampling first {max_rows:,} rows.")
            df = df.iloc[:max_rows].copy()
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .csv or .parquet")

    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"    Loaded shape: {df.shape[0]:,} rows x {df.shape[1]} columns ({mem_mb:.2f} MB in RAM)")
    return df


def audit_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Audit every column for type, nulls, cardinality, and semantic role."""
    records = []
    categorized = {
        "numeric_continuous": [],
        "numeric_discrete": [],
        "categorical_low": [],
        "categorical_high": [],
        "datetime_candidate": [],
        "id_or_constant": [],
    }

    for col in df.columns:
        s = df[col]
        null_count = int(s.isnull().sum())
        null_pct = round((null_count / len(df)) * 100.0, 2)
        n_unique = int(s.nunique(dropna=True))
        dtype_str = str(s.dtype)

        # Detect semantic role
        role = "unknown"
        sample_vals = [str(x) for x in s.dropna().head(3).tolist()]
        sample_str = ", ".join(sample_vals) if sample_vals else "None"

        # Check for datetime
        is_dt = False
        if "datetime" in dtype_str or "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
            try:
                pd.to_datetime(s.dropna().head(20))
                is_dt = True
                role = "DATETIME"
                categorized["datetime_candidate"].append(col)
            except Exception:
                pass

        if not is_dt:
            if n_unique <= 1:
                role = "CONSTANT (Drop)"
                categorized["id_or_constant"].append(col)
            elif n_unique == len(df) and ("id" in col.lower() or "key" in col.lower()):
                role = "UNIQUE ID (Drop/Key)"
                categorized["id_or_constant"].append(col)
            elif pd.api.types.is_numeric_dtype(s):
                if n_unique <= 10:
                    role = "NUMERIC DISCRETE"
                    categorized["numeric_discrete"].append(col)
                else:
                    role = "NUMERIC CONTINUOUS"
                    categorized["numeric_continuous"].append(col)
            else:
                if n_unique <= 50:
                    role = "CATEGORICAL (Low Cardinality)"
                    categorized["categorical_low"].append(col)
                else:
                    role = "CATEGORICAL (High Cardinality)"
                    categorized["categorical_high"].append(col)

        records.append({
            "Column": col,
            "Detected Role": role,
            "Null Count": null_count,
            "Null %": null_pct,
            "Unique Count": n_unique,
            "Sample Values": sample_str[:40],
        })

    audit_df = pd.DataFrame(records)
    return audit_df, categorized


def detect_problem_type(
    df: pd.DataFrame,
    categorized: Dict[str, List[str]],
    target_col: Optional[str] = None,
) -> Dict[str, str]:
    """Infer the machine learning task and recommend baseline algorithms and metrics."""
    result = {}
    target = target_col

    # Auto-guess target if not specified
    if not target:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ["target", "label", "price", "fare", "cost", "revenue", "demand", "churn", "is_", "has_", "class", "status", "outcome", "amount"])]
        if candidates:
            target = candidates[0]
            result["target_inferred"] = f"Auto-detected target candidate: '{target}' (override with --target <name>)"
        else:
            result["target_inferred"] = "No clear target column detected by name. Please specify via --target."

    if target and target in df.columns:
        s = df[target].dropna()
        n_unique = s.nunique()
        result["target_column"] = target
        result["target_nulls"] = f"{df[target].isnull().sum():,} ({df[target].isnull().mean() * 100:.2f}%)"

        if pd.api.types.is_numeric_dtype(s) and n_unique > 20:
            result["task"] = "REGRESSION"
            result["recommended_model"] = "LightGBM Regressor (objective='regression')"
            result["loss_function"] = "L2 (RMSE) for penalizing large errors, or L1 (MAE) for robust median errors"
            result["eval_metrics"] = "RMSE, MAE, R-squared, MAPE"
            result["baseline_benchmark"] = f"Dummy Mean (Mean = {s.mean():.4f}, Median = {s.median():.4f}, Std = {s.std():.4f})"

        elif n_unique == 2:
            result["task"] = "BINARY CLASSIFICATION"
            result["recommended_model"] = "LightGBM Classifier (objective='binary')"
            result["loss_function"] = "Binary Cross-Entropy (LogLoss)"
            result["eval_metrics"] = "ROC-AUC, Macro F1-score, Precision, Recall, Confusion Matrix"
            val_counts = s.value_counts(normalize=True).to_dict()
            dist_str = ", ".join([f"Class {k}: {v * 100:.1f}%" for k, v in val_counts.items()])
            result["class_distribution"] = dist_str
            if min(val_counts.values()) < 0.15:
                result["imbalance_warning"] = "Severe class imbalance detected! Use scale_pos_weight or SMOTE."

        elif 2 < n_unique <= 50:
            result["task"] = "MULTICLASS CLASSIFICATION"
            result["recommended_model"] = f"LightGBM Classifier (objective='multiclass', num_class={n_unique})"
            result["loss_function"] = "Multiclass Cross-Entropy (Multi-Logloss)"
            result["eval_metrics"] = "Macro F1-score, Multi-class LogLoss, Accuracy"
            result["num_classes"] = str(n_unique)

        else:
            result["task"] = "HIGH-CARDINALITY / COMPLEX"
            result["recommended_model"] = "LightGBM with target encoding or deep embeddings"
    else:
        if categorized["datetime_candidate"] and categorized["numeric_continuous"]:
            result["task"] = "TIME-SERIES FORECASTING"
            result["recommended_model"] = "Autoregressive LightGBM with lag features (t-1, t-24, t-168)"
            result["eval_metrics"] = "RMSE, MAE, WAPE, R-squared"
        else:
            result["task"] = "UNSUPERVISED / CLUSTERING"
            result["recommended_model"] = "StandardScaler + K-Means (test k=2..8 with Silhouette Score)"
            result["eval_metrics"] = "Silhouette Score, Inertia (Elbow Method)"

    return result


def print_triage_report(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    categorized: Dict[str, List[str]],
    task_info: Dict[str, str],
):
    """Print clean, executive-style diagnostic report to terminal."""
    print("\n" + "=" * 80)
    print("               NEXORA RAPID DATATHON TRIAGE REPORT")
    print("=" * 80)

    print("\n[2] DATASET SUMMARY:")
    print(f"    Total Records:    {len(df):,}")
    print(f"    Total Features:   {df.shape[1]}")
    print(f"    Duplicate Rows:   {df.duplicated().sum():,}")
    total_nulls = df.isnull().sum().sum()
    print(f"    Total Null Cells: {total_nulls:,} ({total_nulls / (df.shape[0] * df.shape[1]) * 100:.2f}% of all cells)")

    print("\n[3] PROBLEM TYPE CLASSIFICATION:")
    for k, v in task_info.items():
        print(f"    * {k.upper()}: {v}")

    print("\n[4] COLUMN BREAKDOWN BY DETECTED ROLE:")
    for cat_name, cols in categorized.items():
        if cols:
            col_list_str = ", ".join(cols[:8])
            if len(cols) > 8:
                col_list_str += f" ... (+{len(cols) - 8} more)"
            print(f"    * {cat_name.upper()} ({len(cols)} cols): {col_list_str}")

    high_null = audit_df[audit_df["Null %"] > 25.0]
    if not high_null.empty:
        print("\n[5] WARNING - HIGH NULL COLUMNS (>25% missing):")
        for _, row in high_null.iterrows():
            print(f"    ! Column '{row['Column']}': {row['Null %']}% missing ({row['Null Count']:,} rows). Suggestion: Add indicator flag or drop if uninformative.")

    print("\n[6] COLUMN DETAILS (Top 25 Columns):")
    print("-" * 80)
    print(f"{'Column Name':<26} | {'Detected Role':<20} | {'Null %':<8} | {'Unique':<8} | {'Sample'}")
    print("-" * 80)
    for _, r in audit_df.head(25).iterrows():
        print(f"{r['Column'][:25]:<26} | {r['Detected Role'][:20]:<20} | {str(r['Null %']) + '%':<8} | {r['Unique Count']:<8} | {r['Sample Values'][:20]}")
    if len(audit_df) > 25:
        print(f"    ... and {len(audit_df) - 25} more columns.")
    print("-" * 80)

    target = task_info.get("target_column", "YOUR_TARGET_COLUMN")
    task = task_info.get("task", "REGRESSION")

    print("\n[7] INSTANT BASELINE STARTER CODE (Copy and run in notebook):")
    print("=" * 80)
    code = f'''
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, roc_auc_score

# 1. Load Data
df = pd.read_csv("DATASET_PATH", low_memory=False)

# 2. Separate Target and Drop Constant/ID columns
target_col = "{target}"
drop_cols = {categorized.get("id_or_constant", [])} + [target_col]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df[target_col].copy()

# 3. Handle Missing Values
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna("missing").astype("category")

# 4. Chronological or Stratified Split (80% Train, 20% Val)
'''
    if "TIME" in task:
        code += '''
# Time-series split (Strict chronological non-overlap)
train_size = int(len(df) * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
'''
    elif "CLASSIFICATION" in task:
        code += '''
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
'''
    else:
        code += '''
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
'''

    if "CLASSIFICATION" in task:
        code += f'''
# 5. Train LightGBM Classifier
clf = lgb.LGBMClassifier(
    objective="{'binary' if 'BINARY' in task else 'multiclass'}",
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)

# 6. Evaluate
preds = clf.predict(X_val)
probs = clf.predict_proba(X_val)
print("F1 Score (Macro):", round(f1_score(y_val, preds, average='macro'), 4))
'''
    else:
        code += '''
# 5. Train LightGBM Regressor
reg = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
reg.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)

# 6. Evaluate
preds = reg.predict(X_val)
print("Validation RMSE:", round(np.sqrt(mean_squared_error(y_val, preds)), 4))
print("Validation MAE: ", round(mean_absolute_error(y_val, preds), 4))
print("Validation R²:  ", round(r2_score(y_val, preds), 4))
'''

    code += '''
# 7. Feature Importance
imp = pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_ if "clf" in locals() else reg.feature_importances_})
print(imp.sort_values(by="importance", ascending=False).head(10))
'''
    print(code)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Universal Datathon Rapid Triage Scanner")
    parser.add_argument("path", type=str, help="Path to CSV or Parquet file")
    parser.add_argument("--target", type=str, default=None, help="Target column name (optional)")
    parser.add_argument("--max_rows", type=int, default=200_000, help="Maximum rows to load for triage")
    args = parser.parse_args()

    file_path = Path(args.path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    df = load_dataset(file_path, max_rows=args.max_rows)
    audit_df, categorized = audit_columns(df)
    task_info = detect_problem_type(df, categorized, target_col=args.target)
    print_triage_report(df, audit_df, categorized, task_info)


if __name__ == "__main__":
    main()
