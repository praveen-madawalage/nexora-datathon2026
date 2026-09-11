"""
Task 2.1: Upfront Base Fare Pricing Engine
Interactive real-time upfront fare calculation with zero post-trip leakage.
"""

from datetime import datetime, time
import streamlit as st
import pandas as pd
import plotly.express as px
from src.app_utils import apply_custom_theme, load_zone_lookup, predict_fare_amount, load_fare_model

st.set_page_config(page_title="Upfront Fare Engine | Nexora", page_icon="💰", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div style='margin-bottom: 20px;'>
        <span class='badge badge-cyan'>TASK 2.1</span>
        <span class='badge badge-emerald'>NO-SURPRISES UPFRONT PRICING</span>
        <h1 style='margin: 8px 0 4px 0; color: #FFFFFF;'>Upfront Base Fare Pricing Engine</h1>
        <p style='color: #94A3B8; font-size: 1rem;'>
            Guarantees a fixed trip fare prior to passenger boarding. Built strictly with pre-trip attributes to prevent feature leakage.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

zones_df = load_zone_lookup()
zone_names = zones_df["display_name"].tolist()
zone_ids = zones_df["loc_id"].tolist()
zone_map = dict(zip(zone_names, zone_ids))

col_form, col_result = st.columns([1.1, 0.9])

with col_form:
    st.markdown("<h4 style='color: #38BDF8; margin-top: 0;'>🚖 Trip Specification</h4>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Default to Upper East Side South (237)
        default_origin_idx = zone_ids.index(237) if 237 in zone_ids else 0
        origin_sel = st.selectbox("Pickup Location (Origin)", zone_names, index=default_origin_idx, key="fare_origin")
    with c2:
        # Default to Midtown Center (161)
        default_dest_idx = zone_ids.index(161) if 161 in zone_ids else 1
        dest_sel = st.selectbox("Dropoff Location (Destination)", zone_names, index=default_dest_idx, key="fare_dest")

    c3, c4 = st.columns(2)
    with c3:
        pickup_date = st.date_input("Pickup Date", datetime(2026, 3, 16), key="fare_date")
    with c4:
        pickup_time = st.time_input("Pickup Time", time(8, 30), key="fare_time")

    if "fare_dist_slider" not in st.session_state:
        st.session_state["fare_dist_slider"] = 2.4

    def set_fare_dist(val: float) -> None:
        st.session_state["fare_dist_slider"] = float(val)

    st.markdown("<b>Trip Distance (miles)</b>", unsafe_allow_html=True)
    preset_cols = st.columns(4)
    with preset_cols[0]:
        st.button("Short (1.2 mi)", on_click=set_fare_dist, args=(1.2,), key="btn_fare_short", use_container_width=True)
    with preset_cols[1]:
        st.button("Medium (3.5 mi)", on_click=set_fare_dist, args=(3.5,), key="btn_fare_med", use_container_width=True)
    with preset_cols[2]:
        st.button("JFK Run (15.2 mi)", on_click=set_fare_dist, args=(15.2,), key="btn_fare_jfk", use_container_width=True)
    with preset_cols[3]:
        st.button("Long Cross (24.0 mi)", on_click=set_fare_dist, args=(24.0,), key="btn_fare_cross", use_container_width=True)

    distance_miles = st.slider(
        "Distance Slider",
        min_value=0.1,
        max_value=40.0,
        step=0.1,
        key="fare_dist_slider",
        label_visibility="collapsed",
    )

    c5, c6 = st.columns(2)
    with c5:
        rate_class = st.selectbox(
            "Rate Structure",
            [
                (1, "Standard Metered Rate"),
                (2, "JFK Airport Flat Rate"),
                (3, "Newark Out-of-State Rate"),
                (5, "Negotiated / Corporate"),
            ],
            format_func=lambda x: x[1],
            key="fare_rate_class",
        )
    with c6:
        riders = st.number_input("Passenger Count", min_value=1, max_value=6, value=1, key="fare_riders")

    predict_clicked = st.button("⚡ Calculate Guaranteed Upfront Fare", type="primary", use_container_width=True, key="btn_calc_fare")
    if predict_clicked:
        st.toast("⚡ Guaranteed upfront fare locked & recalculated!", icon="💰")
        st.session_state["fare_calc_timestamp"] = datetime.now().strftime("%I:%M:%S %p")

# Calculation
origin_id = zone_map[origin_sel]
dest_id = zone_map[dest_sel]
pickup_dt_str = f"{pickup_date} {pickup_time}"

pred_fare, meta = predict_fare_amount(
    pickup_dt=pickup_dt_str,
    origin_id=origin_id,
    dest_id=dest_id,
    distance_miles=distance_miles,
    rate_class_id=rate_class[0],
    rider_count=riders,
)

calc_status_badge = (
    f"<span class='badge badge-emerald'>✓ Locked at {st.session_state['fare_calc_timestamp']}</span>"
    if "fare_calc_timestamp" in st.session_state
    else "<span class='badge badge-emerald'>✓ Live Synced</span>"
)

with col_result:
    st.markdown(
        f"""
        <div class='result-box'>
            <div style='font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94A3B8; font-weight: 700;'>
                Guaranteed Upfront Fare
            </div>
            <div class='result-amount'>${pred_fare:.2f}</div>
            <div class='result-subtitle'>
                Locked Pre-Trip Price • Confidence Band: <b>${meta['lower_bound']:.2f} – ${meta['upper_bound']:.2f}</b>
            </div>
            <div style='margin-top: 18px;'>
                <span class='badge badge-cyan'>{distance_miles:.1f} Miles</span>
                <span class='badge badge-emerald'>Rate Class {rate_class[0]}</span>
                <span class='badge badge-purple'>{'Rush Hour' if (pickup_time.hour in [7,8,9,16,17,18,19]) else 'Standard Traffic'}</span>
                {calc_status_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='glass-card' style='padding: 16px;'>
            <h5 style='margin: 0 0 10px 0; color: #38BDF8;'>Model Verification Benchmark</h5>
            <table style='width: 100%; font-size: 0.82rem; color: #CBD5E1;'>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; color: #94A3B8;'>
                    <td style='padding: 4px;'>Model</td>
                    <td>Val RMSE</td>
                    <td>Test RMSE</td>
                    <td>Test MAE</td>
                    <td>Test R²</td>
                </tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <td style='padding: 4px;'>Ridge Baseline</td>
                    <td>8.97</td>
                    <td>9.42</td>
                    <td>4.83</td>
                    <td>0.740</td>
                </tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <td style='padding: 4px;'>XGBoost Benchmark</td>
                    <td>7.55</td>
                    <td>8.32</td>
                    <td>3.70</td>
                    <td>0.815</td>
                </tr>
                <tr style='color: #10B981; font-weight: 700;'>
                    <td style='padding: 4px;'>★ LightGBM (Production)</td>
                    <td>7.29</td>
                    <td>8.14</td>
                    <td>3.61</td>
                    <td>0.802</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Feature Driver Insights
st.markdown("### 📊 Top Feature Drivers for Fare Prediction")
bundle = load_fare_model()
importance_gain = bundle["model"].feature_importance(importance_type="gain")
feature_names = bundle["metadata"]["features"]

feat_df = (
    pd.DataFrame({"Feature": feature_names, "ImportanceGain": importance_gain})
    .sort_values(by="ImportanceGain", ascending=False)
    .head(8)
)

fig = px.bar(
    feat_df,
    x="ImportanceGain",
    y="Feature",
    orientation="h",
    color="ImportanceGain",
    color_continuous_scale=["#0284C7", "#38BDF8", "#818CF8"],
)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=20, b=20),
    height=280,
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False,
)
st.plotly_chart(fig, use_container_width=True)
