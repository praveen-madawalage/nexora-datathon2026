"""
Shared utilities, caching loaders, and custom glassmorphic styling
for the Nexora Urban Flow Analytics Streamlit application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "app" / "data"
MODELS_DIR = ROOT_DIR / "models"


def apply_custom_theme() -> None:
    """Inject ultra-premium modern glassmorphism CSS into the Streamlit session."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Background & container styling */
        .stApp {
            background: radial-gradient(circle at 10% 20%, #0d1527 0%, #080c14 90%);
            color: #E2E8F0;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(18, 26, 43, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(0, 210, 255, 0.3);
            transform: translateY(-2px);
        }

        /* Metric Hero Cards */
        .metric-hero {
            background: linear-gradient(135deg, rgba(14, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 14px;
            padding: 20px;
            text-align: center;
        }
        .metric-hero-val {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38BDF8, #818CF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-hero-lbl {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            margin-top: 4px;
        }
        .metric-hero-sub {
            font-size: 0.75rem;
            color: #10B981;
            margin-top: 6px;
        }

        /* Prediction Result Box */
        .result-box {
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid #0284C7;
            border-radius: 18px;
            padding: 28px;
            text-align: center;
            box-shadow: 0 12px 40px rgba(2, 132, 199, 0.25);
            animation: fadeIn 0.4s ease-in-out;
        }
        .result-amount {
            font-size: 3.4rem;
            font-weight: 800;
            color: #38BDF8;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.1;
        }
        .result-subtitle {
            font-size: 0.95rem;
            color: #CBD5E1;
            margin-top: 8px;
        }

        /* Badge chips */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-cyan { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
        .badge-emerald { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-purple { background: rgba(168, 85, 247, 0.15); color: #A855F7; border: 1px solid rgba(168, 85, 247, 0.3); }

        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            background-color: rgba(18, 26, 43, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            padding: 16px 20px !important;
            margin-bottom: 14px !important;
            color: #F1F5F9 !important;
        }
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div {
            color: #F1F5F9 !important;
        }

        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_zone_lookup() -> pd.DataFrame:
    """Load taxi zone dictionary with coordinates and borough labels."""
    csv_path = DATA_DIR / "zone_lookup.csv"
    if not csv_path.exists():
        csv_path = ROOT_DIR / "data" / "raw" / "reference" / "Urban_Flow_Analytics_Zone_Dataset.csv"
    df = pd.read_csv(csv_path)
    df["display_name"] = df["zone_name"] + " (" + df["borough_name"] + ")"
    return df


@st.cache_resource
def load_fare_model() -> Dict[str, Any]:
    """Load serialized LightGBM upfront fare bundle."""
    return joblib.load(MODELS_DIR / "fare_lgbm.pkl")


@st.cache_resource
def load_duration_model() -> Dict[str, Any]:
    """Load serialized LightGBM duration bundle."""
    return joblib.load(MODELS_DIR / "duration_lgbm.pkl")


@st.cache_data
def load_demand_summary() -> pd.DataFrame:
    """Load pre-aggregated hourly demand timeseries for top zones."""
    df = pd.read_csv(DATA_DIR / "demand_summary.csv")
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    return df


@st.cache_data
def load_clustering_summary(cache_version: str = "v2_251_canonical_zones") -> Dict[str, Any]:
    """Load pre-computed spatial-temporal clustering and OD corridor analytics."""
    with open(DATA_DIR / "clustering_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def predict_fare_amount(
    pickup_dt: str,
    origin_id: int,
    dest_id: int,
    distance_miles: float,
    rate_class_id: int = 1,
    rider_count: int = 1,
    provider_code: int = 1,
    fare_settlement_method: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """
    Generate real-time upfront base fare prediction with model confidence metadata.
    """
    from src.features import prepare_upfront_pricing_features

    bundle = load_fare_model()
    zone_df = load_zone_lookup()

    input_df = pd.DataFrame(
        [
            {
                "pickup_timestamp": str(pickup_dt),
                "origin_loc_id": int(origin_id),
                "dest_loc_id": int(dest_id),
                "distance_miles": float(distance_miles),
                "rate_class_id": int(rate_class_id),
                "rider_count": int(rider_count),
                "provider_code": int(provider_code),
                "fare_settlement_method": int(fare_settlement_method),
            }
        ]
    )

    X = prepare_upfront_pricing_features(input_df, zone_df=zone_df, include_target=False)
    feature_cols = bundle["metadata"]["features"]
    X = X[feature_cols].astype(np.float32)

    pred = float(bundle["model"].predict(X)[0])
    pred = max(2.50, pred)  # NYC standard base meter floor

    mae = bundle["metadata"]["test_metrics"]["mae"]
    return pred, {
        "lower_bound": max(2.50, pred - mae),
        "upper_bound": pred + mae,
        "mae": mae,
        "test_rmse": bundle["metadata"]["test_metrics"]["rmse"],
        "test_r2": bundle["metadata"]["test_metrics"]["r2"],
    }


def predict_trip_duration(
    pickup_dt: str,
    origin_id: int,
    dest_id: int,
    distance_miles: float,
    rate_class_id: int = 1,
    rider_count: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """
    Generate real-time on-time arrival duration prediction.
    """
    from src.features import prepare_trip_duration_features

    bundle = load_duration_model()
    zone_df = load_zone_lookup()

    input_df = pd.DataFrame(
        [
            {
                "pickup_timestamp": str(pickup_dt),
                "origin_loc_id": int(origin_id),
                "dest_loc_id": int(dest_id),
                "distance_miles": float(distance_miles),
                "rate_class_id": int(rate_class_id),
                "rider_count": int(rider_count),
            }
        ]
    )

    X = prepare_trip_duration_features(input_df, zone_df=zone_df, include_target=False)
    feature_cols = bundle["metadata"]["features"]
    X = X[feature_cols].astype(np.float32)

    pred = float(bundle["model"].predict(X)[0])
    pred = max(1.0, pred)  # Minimum trip duration 1 minute

    tol = bundle["metadata"]["test_tolerance"]
    med_err = tol["median_abs_err_min"]

    return pred, {
        "lower_bound": max(1.0, pred - med_err),
        "upper_bound": pred + med_err,
        "median_abs_err_min": med_err,
        "pct_within_5min": tol["pct_within_5min"],
        "test_rmse": bundle["metadata"]["test_metrics"]["rmse"],
        "test_r2": bundle["metadata"]["test_metrics"]["r2"],
    }
