"""
Urban Flow Analytics — Executive Command Center (Home)
Team Nexora — Production Machine Learning & Spatial-Temporal Dispatch Suite
"""

import streamlit as st
import pandas as pd
from src.app_utils import apply_custom_theme, load_zone_lookup

st.set_page_config(
    page_title="Nexora | Urban Flow Analytics",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()

# Sidebar summary
with st.sidebar:
    st.markdown("### 🚕 **Team Nexora**")
    st.markdown("**Datathon 2026**")
    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.85rem; line-height: 1.6; color: #94A3B8;'>
        <b>Production ML Architecture</b><br>
        • <b>Dataset:</b> 44M trip records<br>
        • <b>Engine:</b> DuckDB + LightGBM<br>
        • <b>Zones:</b> 265 NYC Taxi Zones<br>
        • <b>Zero-Leakage:</b> Strictly pre-trip features<br>
        • <b>Deployment:</b> Streamlit Cloud Native
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.info("💡 Use the sidebar navigation above to explore each task module.")

# Hero Header
st.markdown(
    """
    <div style='margin-bottom: 24px;'>
        <div style='display: flex; gap: 8px; margin-bottom: 12px;'>
            <span class='badge badge-cyan'>PROD READY</span>
            <span class='badge badge-emerald'>ZERO LEAKAGE</span>
            <span class='badge badge-purple'>LIGHTGBM v4.7</span>
        </div>
        <h1 style='font-size: 2.8rem; font-weight: 800; margin: 0; background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 60%, #818CF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Nexora Urban Flow Analytics
        </h1>
        <p style='font-size: 1.15rem; color: #94A3B8; margin-top: 8px;'>
            Enterprise-grade taxi operations intelligence: real-time upfront fare pricing, precision arrival estimation, recursive 72-hour fleet demand forecasting, and diurnal spatial clustering.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4 KPI Hero Metric Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
        <div class='metric-hero'>
            <div class='metric-hero-val'>$3.61</div>
            <div class='metric-hero-lbl'>Task 2.1 • Fare MAE</div>
            <div class='metric-hero-sub'>R² = 0.8020 on Out-of-Time Test</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class='metric-hero'>
            <div class='metric-hero-val'>80.8%</div>
            <div class='metric-hero-lbl'>Task 2.2 • On-Time Arrival</div>
            <div class='metric-hero-sub'>Within ±5 min (Median err: 2.11m)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class='metric-hero'>
            <div class='metric-hero-val'>0.945</div>
            <div class='metric-hero-lbl'>Task 3.1 • Demand R²</div>
            <div class='metric-hero-sub'>72h Autoregressive Dispatch Forecast</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class='metric-hero'>
            <div class='metric-hero-val'>5</div>
            <div class='metric-hero-lbl'>Task 3.2 • Mobility Archetypes</div>
            <div class='metric-hero-sub'>K-Means across 265 Taxi Zones</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Main Grid: Two Feature Panels
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown(
        """
        <div class='glass-card'>
            <h3 style='margin-top: 0; color: #38BDF8;'>🚀 Operational Suite Overview</h3>
            <p style='color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;'>
                Team Nexora designed an end-to-end analytical pipeline handling over <b>44 million trip records</b> with zero feature leakage. The platform addresses city mobility challenges across passenger confidence, dispatcher efficiency, and urban planning.
            </p>
            <div style='margin-top: 16px;'>
                <div style='margin-bottom: 12px;'>
                    <b style='color: #38BDF8;'>1. Upfront Pricing Engine (Task 2.1)</b><br>
                    <span style='color: #94A3B8; font-size: 0.88rem;'>Eliminates meter anxiety by locking in base fares pre-trip using distance, cyclical calendar features, and borough routing graphs.</span>
                </div>
                <div style='margin-bottom: 12px;'>
                    <b style='color: #38BDF8;'>2. On-Time Arrival Estimator (Task 2.2)</b><br>
                    <span style='color: #94A3B8; font-size: 0.88rem;'>High-precision journey duration prediction beating traditional naive speed heuristics by <b>34.8 percentage points</b> in 5-minute tolerance.</span>
                </div>
                <div style='margin-bottom: 12px;'>
                    <b style='color: #38BDF8;'>3. Fleet Dispatcher (Task 3.1)</b><br>
                    <span style='color: #94A3B8; font-size: 0.88rem;'>Autoregressive lag modeling (t-1h, t-24h, t-168h) projecting zone-by-zone pickup volumes 24 to 72 hours into the future.</span>
                </div>
                <div>
                    <b style='color: #38BDF8;'>4. Spatial-Temporal Hotspot Clustering (Task 3.2)</b><br>
                    <span style='color: #94A3B8; font-size: 0.88rem;'>Discovers functional city archetypes and tracks diurnal OD corridor reversals from morning commute to late-night entertainment.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        """
        <div class='glass-card'>
            <h3 style='margin-top: 0; color: #818CF8;'>🛡️ Architecture & Integrity</h3>
            <table style='width: 100%; border-collapse: collapse; font-size: 0.88rem; color: #CBD5E1;'>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.08);'>
                    <td style='padding: 10px 0; color: #94A3B8;'>Data Ingestion</td>
                    <td style='padding: 10px 0; font-weight: 600;'>DuckDB Parquet Scan (zero memory bottleneck)</td>
                </tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.08);'>
                    <td style='padding: 10px 0; color: #94A3B8;'>Temporal Splits</td>
                    <td style='padding: 10px 0; font-weight: 600;'>Train (Apr '25–Jan '26) · Val (Feb '26) · Test (Mar '26)</td>
                </tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.08);'>
                    <td style='padding: 10px 0; color: #94A3B8;'>Forbidden Columns</td>
                    <td style='padding: 10px 0; font-weight: 600;'>Dropoff ts, duration, tips, tolls, total fare excluded</td>
                </tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.08);'>
                    <td style='padding: 10px 0; color: #94A3B8;'>Unit Test Suite</td>
                    <td style='padding: 10px 0; font-weight: 600; color: #10B981;'>25 / 25 Passing (100% test coverage)</td>
                </tr>
                <tr>
                    <td style='padding: 10px 0; color: #94A3B8;'>Inference Latency</td>
                    <td style='padding: 10px 0; font-weight: 600; color: #38BDF8;'>&lt; 5 ms per trip request</td>
                </tr>
            </table>
            <div style='margin-top: 20px; text-align: center;'>
                <span class='badge badge-cyan'>Ready for Evaluation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Quick Navigation Section
st.markdown("### 🧭 Interactive Application Modules")
b1, b2, b3, b4 = st.columns(4)

with b1:
    st.markdown(
        """
        <div class='glass-card' style='text-align: center; padding: 18px;'>
            <div style='font-size: 2rem;'>💰</div>
            <b style='color: #FFFFFF;'>Upfront Fare</b><br>
            <span style='font-size: 0.8rem; color: #94A3B8;'>Pre-trip guaranteed price lock</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Fare Engine →", key="nav_fare", use_container_width=True):
        st.switch_page("pages/1_💰_Upfront_Fare_Pricing.py")

with b2:
    st.markdown(
        """
        <div class='glass-card' style='text-align: center; padding: 18px;'>
            <div style='font-size: 2rem;'>⏱️</div>
            <b style='color: #FFFFFF;'>Arrival Estimator</b><br>
            <span style='font-size: 0.8rem; color: #94A3B8;'>Minute-precision trip duration</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Duration Engine →", key="nav_dur", use_container_width=True):
        st.switch_page("pages/2_⏱️_Trip_Duration_Estimator.py")

with b3:
    st.markdown(
        """
        <div class='glass-card' style='text-align: center; padding: 18px;'>
            <div style='font-size: 2rem;'>📈</div>
            <b style='color: #FFFFFF;'>Fleet Dispatcher</b><br>
            <span style='font-size: 0.8rem; color: #94A3B8;'>24-72h zone demand forecasting</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Dispatcher →", key="nav_demand", use_container_width=True):
        st.switch_page("pages/3_📈_Fleet_Demand_Forecast.py")

with b4:
    st.markdown(
        """
        <div class='glass-card' style='text-align: center; padding: 18px;'>
            <div style='font-size: 2rem;'>🗺️</div>
            <b style='color: #FFFFFF;'>Hotspot Clusters</b><br>
            <span style='font-size: 0.8rem; color: #94A3B8;'>Diurnal archetypes & OD flow</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Clustering →", key="nav_cluster", use_container_width=True):
        st.switch_page("pages/4_🗺️_Spatial_Hotspot_Clusters.py")
