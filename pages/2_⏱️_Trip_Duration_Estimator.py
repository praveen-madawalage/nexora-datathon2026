"""
Task 2.2: The On-Time Arrival Estimator
Predicts total trip time down to the minute with traffic-aware temporal encoding.
"""

from datetime import datetime, time, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
from src.app_utils import apply_custom_theme, load_zone_lookup, predict_trip_duration, load_duration_model

st.set_page_config(page_title="Arrival Estimator | Nexora", page_icon="⏱️", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div style='margin-bottom: 20px;'>
        <span class='badge badge-purple'>TASK 2.2</span>
        <span class='badge badge-emerald'>ON-TIME ARRIVAL ESTIMATOR</span>
        <h1 style='margin: 8px 0 4px 0; color: #FFFFFF;'>On-Time Arrival & Duration Estimator</h1>
        <p style='color: #94A3B8; font-size: 1rem;'>
            Predicts journey time down to the minute, incorporating rush hour seasonality, interborough bottlenecks, and bridge crossings.
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
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #818CF8; margin-top: 0;'>⏱️ Journey Details</h4>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        default_origin_idx = zone_ids.index(237) if 237 in zone_ids else 0
        origin_sel = st.selectbox("Departure Zone", zone_names, index=default_origin_idx)
    with c2:
        default_dest_idx = zone_ids.index(161) if 161 in zone_ids else 1
        dest_sel = st.selectbox("Destination Zone", zone_names, index=default_dest_idx)

    c3, c4 = st.columns(2)
    with c3:
        pickup_date = st.date_input("Departure Date", datetime(2026, 3, 16))
    with c4:
        pickup_time = st.time_input("Departure Time", time(17, 30))

    st.markdown("<b>Expected Distance (miles)</b>", unsafe_allow_html=True)
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("Short (1.5 mi)", use_container_width=True, key="d_short"):
            st.session_state["dur_dist_val"] = 1.5
    with preset_cols[1]:
        if st.button("Midtown (4.2 mi)", use_container_width=True, key="d_mid"):
            st.session_state["dur_dist_val"] = 4.2
    with preset_cols[2]:
        if st.button("JFK Expressway (16.0 mi)", use_container_width=True, key="d_jfk"):
            st.session_state["dur_dist_val"] = 16.0
    with preset_cols[3]:
        if st.button("Brooklyn Cross (8.5 mi)", use_container_width=True, key="d_cross"):
            st.session_state["dur_dist_val"] = 8.5

    current_dist = st.session_state.get("dur_dist_val", 2.8)
    distance_miles = st.slider("Distance", 0.1, 40.0, float(current_dist), 0.1, label_visibility="collapsed", key="dur_slider")

    c5, c6 = st.columns(2)
    with c5:
        rate_class = st.selectbox(
            "Rate Structure",
            [
                (1, "Standard Metered Rate"),
                (2, "JFK Flat Rate"),
                (3, "Newark Out-of-State"),
                (5, "Negotiated / Corporate"),
            ],
            format_func=lambda x: x[1],
            key="dur_rate",
        )
    with c6:
        riders = st.number_input("Riders", min_value=1, max_value=6, value=1, key="dur_riders")

    calc_clicked = st.button("⚡ Estimate Exact Arrival Time", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Calculation
origin_id = zone_map[origin_sel]
dest_id = zone_map[dest_sel]
pickup_dt_str = f"{pickup_date} {pickup_time}"

pred_mins, meta = predict_trip_duration(
    pickup_dt=pickup_dt_str,
    origin_id=origin_id,
    dest_id=dest_id,
    distance_miles=distance_miles,
    rate_class_id=rate_class[0],
    rider_count=riders,
)

mins_int = int(pred_mins)
secs_int = int((pred_mins - mins_int) * 60)

# Calculate expected arrival clock time
dt_start = datetime.combine(pickup_date, pickup_time)
dt_arrival = dt_start + timedelta(minutes=pred_mins)

with col_result:
    st.markdown(
        f"""
        <div class='result-box' style='border-color: #818CF8; box-shadow: 0 12px 40px rgba(129, 140, 248, 0.25);'>
            <div style='font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94A3B8; font-weight: 700;'>
                Predicted Trip Duration
            </div>
            <div class='result-amount' style='color: #A5B4FC;'>
                {mins_int}<span style='font-size: 1.8rem; font-weight: 600;'> min </span>{secs_int:02d}<span style='font-size: 1.8rem; font-weight: 600;'> sec</span>
            </div>
            <div class='result-subtitle'>
                Estimated Arrival at <b style='color: #38BDF8;'>{dt_arrival.strftime('%I:%M %p')}</b> • Tolerance: ±<b>{meta['median_abs_err_min']:.1f} min</b>
            </div>
            <div style='margin-top: 18px;'>
                <span class='badge badge-purple'><b>80.8%</b> within ±5 min</span>
                <span class='badge badge-cyan'><b>93.5%</b> within ±10 min</span>
                <span class='badge badge-emerald'>Median Err: 2.1m</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Segment Performance breakdown
    st.markdown(
        """
        <div class='glass-card' style='padding: 16px;'>
            <h5 style='margin: 0 0 10px 0; color: #818CF8;'>Operational Segment Reliability</h5>
            <table style='width: 100%; font-size: 0.82rem; color: #CBD5E1;'>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; color: #94A3B8;'>
                    <td style='padding: 4px;'>Segment</td>
                    <td>Within ±5 min</td>
                    <td>Median Error</td>
                    <td>RMSE</td>
                </tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <td style='padding: 4px;'>Manhattan Intra</td>
                    <td style='color: #10B981; font-weight: 600;'>82.4%</td>
                    <td>2.01 min</td>
                    <td>5.22 min</td>
                </tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <td style='padding: 4px;'>Rush Hour Traffic</td>
                    <td>74.1%</td>
                    <td>2.49 min</td>
                    <td>7.68 min</td>
                </tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <td style='padding: 4px;'>Airport Corridors</td>
                    <td>54.7%</td>
                    <td>4.41 min</td>
                    <td>12.30 min</td>
                </tr>
                <tr style='color: #38BDF8; font-weight: 700;'>
                    <td style='padding: 4px;'>★ Overall Test Split</td>
                    <td>80.8%</td>
                    <td>2.11 min</td>
                    <td>7.01 min</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Feature Driver Insights
st.markdown("### 📊 Top Feature Drivers for Trip Duration")
bundle = load_duration_model()
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
    color_continuous_scale=["#4338CA", "#6366F1", "#A5B4FC"],
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
