<div align="center">

# Nexora - Urban Flow Analytics

### End-to-End Machine Learning Platform for NYC Taxi Operations

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-00897B)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://nexora-datathon2026git-7645anq4yj2baw5vpdzjyp.streamlit.app/)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-FDD835)](https://duckdb.org)
[![Tests](https://img.shields.io/badge/Tests-47%20passing-4CAF50)](./tests/)

[Live App](https://nexora-datathon2026git-7645anq4yj2baw5vpdzjyp.streamlit.app/) &nbsp;·&nbsp; [Final Notebook](./notebooks/Nexora_FinalNotebook.ipynb) &nbsp;·&nbsp; [Technical Report](./reports/Nexora_Technical_Report.md)

</div>

---

## Overview

Nexora is a **production-grade, end-to-end machine learning platform** built on **44.5 million cleaned NYC taxi trip records** (April 2025 to March 2026). It is fully deployed on Streamlit Cloud and covers six analytical modules.

| Module | Problem | Solution | Key Metric |
|---|---|---|---|
| **2.1** | Upfront Fare Pricing | LightGBM Regressor | MAE = **$3.61**, R2 = **0.802** |
| **2.2** | Trip Duration Estimation | LightGBM Regressor | **80.8%** within +/-5 min |
| **3.1** | Fleet Demand Forecasting | Autoregressive LightGBM | R2 = **0.941** (10 zones) |
| **3.2** | Hotspot Clustering | K-Means k=5 | Silhouette = **0.524** |
| **5** | AI Mobility Assistant | Gemini Flash + DuckDB NL-to-SQL | **~8ms** query latency |
| **6** | Executive BI Dashboard | Streamlit + Plotly Analytics | **+$5.2M** revenue insight |

---

## Architecture

```
Raw Taxi CSVs (44.5M rows)
        |
        v
DuckDB Cleaning Pipeline  -->  data/processed/  (44,459,188 rows, 0 NULLs)
        |
        |-->  train.parquet  (Apr 2025 - Jan 2026 | 37.5M rows | 84.3%)
        |-->  val.parquet    (Feb 2026             |  3.2M rows |  7.2%)
        +-->  test.parquet   (Mar 2026             |  3.8M rows |  8.5%)
                |
                |-->  LightGBM Fare Model      ->  models/fare_lgbm.pkl
                |-->  LightGBM Duration Model  ->  models/duration_lgbm.pkl
                |-->  LightGBM Demand Model    ->  models/demand_lgbm.pkl
                +-->  K-Means Clustering       ->  models/clustering_kmeans.pkl
                                |
                                v
                    Streamlit Cloud (6-page interactive app)
                    + AI Assistant (Gemini Flash + DuckDB)
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/praveen-madawalage/nexora-datathon2026.git
cd nexora-datathon2026

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App Locally

```bash
streamlit run streamlit_app.py
```

Or use the convenience script (Windows):
```bash
run_app.bat
```

The app opens at `http://localhost:8501`

### Live Deployment

**[https://nexora-datathon2026git-7645anq4yj2baw5vpdzjyp.streamlit.app/](https://nexora-datathon2026git-7645anq4yj2baw5vpdzjyp.streamlit.app/)**

---

## Project Structure

```
nexora-datathon2026/
|
|-- notebooks/
|   |-- 01_profiling.ipynb              # Data quality audit and anomaly profiling
|   |-- 02_cleaning.ipynb               # DuckDB cleaning pipeline
|   |-- 03_fare_prediction.ipynb        # Upfront fare model
|   |-- 04_duration_prediction.ipynb    # Duration estimator
|   |-- 06_demand_forecasting.ipynb     # Fleet demand forecaster
|   |-- 07_hotspot_clustering.ipynb     # Spatial clustering
|   +-- Nexora_FinalNotebook.ipynb      # Final consolidated notebook
|
|-- src/                                # Production Python modules
|   |-- cleaning.py                     # DuckDB cleaning and validation SQL
|   |-- features.py                     # Zero-leakage feature engineering
|   |-- models.py                       # Training, evaluation and serialization
|   |-- demand.py                       # Autoregressive demand forecasting
|   |-- clustering.py                   # K-Means zone archetype clustering
|   |-- ai_assistant.py                 # Gemini Flash NL-to-SQL and safety validator
|   |-- bi_analytics.py                 # BI KPIs and revenue analytics
|   |-- app_utils.py                    # Streamlit shared utilities and CSS theme
|   +-- utils.py                        # Logging, metrics, path helpers
|
|-- pages/                              # Streamlit multi-page app
|   |-- 1_Upfront_Fare_Pricing.py
|   |-- 2_Trip_Duration_Estimator.py
|   |-- 3_Fleet_Demand_Forecast.py
|   |-- 4_Spatial_Hotspot_Clusters.py
|   |-- 5_AI_Mobility_Assistant.py
|   +-- 6_BI_Dashboard.py
|
|-- models/                             # Trained model bundles (.pkl)
|   |-- fare_lgbm.pkl                   # LightGBM fare model (0.8 MB)
|   |-- duration_lgbm.pkl               # LightGBM duration model (0.9 MB)
|   |-- demand_lgbm.pkl                 # Autoregressive demand forecaster (1.1 MB)
|   +-- clustering_kmeans.pkl           # K-Means + StandardScaler (2.6 KB)
|
|-- data/
|   |-- raw/                            # Original source CSVs (not committed)
|   |-- interim/                        # Audit and profiling outputs
|   +-- processed/                      # Cleaned gold-tier splits
|       |-- train.parquet               # Apr 2025 - Jan 2026 (37.5M rows)
|       |-- val.parquet                 # Feb 2026 (3.2M rows)
|       |-- test.parquet                # Mar 2026 (3.8M rows)
|       |-- train_sample.csv            # 100k representative sample
|       |-- val_sample.csv              # 100k representative sample
|       +-- test_sample.csv             # 100k representative sample
|
|-- tests/                              # 47 automated pytest tests
|   |-- test_cleaning.py
|   |-- test_features.py
|   |-- test_models.py
|   |-- test_splits.py
|   |-- test_demand.py
|   |-- test_clustering.py
|   |-- test_bi_analytics.py
|   |-- test_ai_assistant.py
|   +-- test_ui_pages.py
|
|-- app/data/                           # Pre-computed dashboard assets
|   |-- demand_summary.csv              # Hourly demand with LightGBM predictions
|   |-- clustering_summary.json         # Zone archetype profiles
|   +-- zone_lookup.csv                 # 265 NYC zone reference table
|
|-- scripts/
|   |-- generate_demand_predictions.py  # Regenerate demand forecast predictions
|   |-- save_demand_model.py            # Re-export demand_lgbm.pkl
|   +-- save_clustering_model.py        # Re-export clustering_kmeans.pkl
|
|-- reports/
|   +-- Nexora_Technical_Report.md      # Full technical report
|
|-- streamlit_app.py                    # App home page and navigation hub
|-- requirements.txt                    # Python dependencies
+-- README.md                           # This file
```

---

## Using the Trained Models

All models are `joblib` bundles containing the model object, metadata, and a timestamp.

```python
import joblib

# Load any model
bundle = joblib.load("models/fare_lgbm.pkl")
model  = bundle["model"]       # LightGBM Booster
meta   = bundle["metadata"]    # task, features, metrics, version

# View required input features
print(meta["features"])

# Make a prediction
prediction = model.predict(X_features)
```

### Model Performance

| Model File | Algorithm | Trees | Test R2 | Test MAE |
|---|---|---|---|---|
| `fare_lgbm.pkl` | LightGBM | 131 | **0.802** | **$3.61** |
| `duration_lgbm.pkl` | LightGBM | 133 | **0.779** | **3.60 min** |
| `demand_lgbm.pkl` | LightGBM Autoregressive | ~400 | **0.941** | **21.3 trips/hr** |
| `clustering_kmeans.pkl` | K-Means k=5 | N/A | Silhouette: **0.524** | N/A |

### Demand Forecast - Per-Zone R2 (March 2026 Test Set)

| Zone | R2 | Zone | R2 |
|---|---|---|---|
| Midtown Center (161) | 0.965 | Upper East Side South (237) | 0.967 |
| Upper East Side North (236) | 0.966 | Midtown East (162) | 0.953 |
| Union Sq (234) | 0.954 | Times Sq/Theatre (230) | 0.947 |
| East Village (79) | 0.937 | JFK Airport (132) | 0.883 |

---

## AI Mobility Assistant - Setup

The AI assistant requires a **Google Gemini API key**.

```bash
# Option 1: Environment variable
export GEMINI_API_KEY="your-key-here"        # macOS/Linux
set GEMINI_API_KEY=your-key-here             # Windows

# Option 2: Streamlit secrets (create .streamlit/secrets.toml)
GEMINI_API_KEY = "your-key-here"

# Option 3: Enter directly in the app sidebar at runtime
```

**Example natural language queries:**
- *"Which borough generates the most taxi revenue?"*
- *"What is the peak demand hour on weekdays?"*
- *"Compare JFK airport trips vs Manhattan intra-trips by average fare"*
- *"Show me the top 10 zones by total pickup volume"*

**Security - Dual-layer SQL guardrails:**
1. Gemini is constrained via prompt engineering to output `SELECT`-only queries
2. Python AST + regex validator blocks all mutations (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `EXEC`) and system table access before DuckDB execution

---

## Running Tests

```bash
# Run all 47 tests
python -m pytest

# Verbose with short traceback
python -m pytest -v --tb=short

# Run a specific module
python -m pytest tests/test_ai_assistant.py -v
python -m pytest tests/test_demand.py -v
```

| Module | Tests | What It Covers |
|---|---|---|
| `test_cleaning.py` | 2 | SQL structure, cleaning config defaults |
| `test_features.py` | 6 | Temporal/spatial features, zero-leakage boundaries |
| `test_models.py` | 4 | Model shape, metrics, pkl save/load |
| `test_splits.py` | 3 | File existence, row counts, temporal non-overlap |
| `test_demand.py` | 4 | Grid building, feature engineering, dispatch recs |
| `test_clustering.py` | 3 | K-Means fitting, temporal shift summaries |
| `test_bi_analytics.py` | 9 | KPI calculations, fleet gap model |
| `test_ai_assistant.py` | 11 | SQL validation, chart detection, preset queries |
| `test_ui_pages.py` | 2 | Streamlit button callbacks and session state |
| **Total** | **47** | **All passing** |

---

## Data Splits

Strict chronological partitioning with **zero temporal leakage**:

```
max(train_timestamps) < min(val_timestamps) < min(test_timestamps)
```

| Split | Period | Rows | Share |
|---|---|---|---|
| **Train** | Apr 1, 2025 - Jan 31, 2026 | 37,459,831 | 84.3% |
| **Validation** | Feb 1, 2026 - Feb 28, 2026 | 3,221,802 | 7.2% |
| **Test (Hold-Out)** | Mar 1, 2026 - Mar 31, 2026 | 3,777,555 | 8.5% |

All splits share the same 24-column schema with 0 NULLs. 100k-row sample CSVs are provided in `data/processed/` for quick inspection without loading the full parquet files.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Data Engine** | DuckDB 0.10 - columnar in-memory SQL |
| **ML Framework** | LightGBM 4.x, scikit-learn |
| **App Framework** | Streamlit 1.x |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **AI / LLM** | Google Gemini Flash (gemini-2.0-flash) |
| **Serialization** | joblib |
| **Testing** | pytest (47 tests) |
| **Version Control** | Git + GitHub |
| **Deployment** | Streamlit Cloud |

---

## Setup Notes

- Raw source CSV files are excluded from version control (4.5 GB total). Only the cleaned parquet splits and sample CSVs are tracked.
- Model `.pkl` files for the fare, duration, demand, and clustering models are committed to the repository.
- To regenerate predictions or re-export a model, use the scripts in the `scripts/` directory.
- The AI assistant requires a Gemini API key at runtime; all other pages work without one.
