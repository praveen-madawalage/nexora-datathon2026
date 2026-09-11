"""
Task 3.2: Hotspot & Origin-Destination Flow Clustering
Reveals functional urban archetypes and diurnal travel pattern shifts from morning rush to late night.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.app_utils import apply_custom_theme, load_clustering_summary

st.set_page_config(page_title="Hotspot Clustering | Nexora", page_icon="🗺️", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div style='margin-bottom: 20px;'>
        <span class='badge badge-purple'>TASK 3.2</span>
        <span class='badge badge-cyan'>SPATIAL-TEMPORAL CLUSTERING</span>
        <h1 style='margin: 8px 0 4px 0; color: #FFFFFF;'>Hotspot & Origin-Destination Flow Clustering</h1>
        <p style='color: #94A3B8; font-size: 1rem;'>
            K-Means clustering identifying 5 functional mobility archetypes and tracking diurnal origin-destination corridor reversals across 44M trips.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

cluster_data = load_clustering_summary()
archetypes = cluster_data["archetypes"]
top_zones = pd.DataFrame(cluster_data["top_zones"])
cluster_counts = cluster_data["cluster_counts"]
od_flows = pd.DataFrame(cluster_data["od_flows"])

# Metric Hero Cards
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        """
        <div class='metric-hero'>
            <div class='metric-hero-val'>5</div>
            <div class='metric-hero-lbl'>Urban Archetypes</div>
            <div class='metric-hero-sub'>Optimal k via Elbow & Silhouette</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class='metric-hero'>
            <div class='metric-hero-val'>251</div>
            <div class='metric-hero-lbl'>Active Zones Profiled</div>
            <div class='metric-hero-sub'>Across All 5 NYC Boroughs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
        <div class='metric-hero'>
            <div class='metric-hero-val'>+119.7%</div>
            <div class='metric-hero-lbl'>Night Distance Shift</div>
            <div class='metric-hero-sub'>Longer leisure trips vs. Commute</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    top_corridor = od_flows.iloc[0]["corridor"] if "corridor" in od_flows.columns else "UES South → UES North"
    st.markdown(
        f"""
        <div class='metric-hero'>
            <div class='metric-hero-val' style='font-size: 1.4rem; padding-top: 10px;'>{top_corridor}</div>
            <div class='metric-hero-lbl'>#1 Corridor</div>
            <div class='metric-hero-sub'>Highest Volume Flow Pair</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Main Visual: Archetype distribution & Centroid characteristics
tab1, tab2, tab3 = st.tabs(["🏙️ Functional Archetypes", "🔄 Origin-Destination Flows", "📐 Cluster Diagnostics"])

with tab1:
    col_arch_chart, col_arch_detail = st.columns([1.1, 0.9])

    with col_arch_chart:
        count_df = pd.DataFrame(list(cluster_counts.items()), columns=["Archetype", "ZoneCount"]).sort_values(
            "ZoneCount", ascending=True
        )

        fig_bar = px.bar(
            count_df,
            x="ZoneCount",
            y="Archetype",
            orientation="h",
            color="Archetype",
            text="ZoneCount",
            color_discrete_sequence=["#38BDF8", "#818CF8", "#F472B6", "#34D399", "#FBBF24"],
            title="Taxi Zone Count by Functional Archetype",
        )
        fig_bar.update_traces(
            texttemplate="%{x} zones",
            textposition="outside",
            textfont=dict(size=12, color="#E2E8F0"),
            cliponaxis=False,
        )
        max_val = max(count_df["ZoneCount"]) if not count_df.empty else 100
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 26, 43, 0.5)",
            margin=dict(l=20, r=40, t=40, b=20),
            height=340,
            showlegend=False,
            xaxis=dict(title="Number of Zones", range=[0, max_val * 1.22]),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_arch_detail:
        c_core = cluster_counts.get("Commercial & High-Density Core", 53)
        c_commuter = cluster_counts.get("Commuter Exporter Hub", 64)
        c_nightlife = cluster_counts.get("Nightlife & Entertainment District", 23)
        c_residential = cluster_counts.get("Residential Inflow / Attractor", 69)
        c_peripheral = cluster_counts.get("Outer Borough Long-Haul Peripheral", 42)

        st.markdown(
            f"""
            <div class='glass-card' style='padding: 18px;'>
                <h4 style='color: #38BDF8; margin-top: 0;'>Urban Mobility Archetype Descriptions</h4>
                <div style='font-size: 0.85rem; line-height: 1.6; color: #CBD5E1;'>
                    <b>1. Commercial & High-Density Core ({c_core} zones):</b><br>
                    Balanced midday/evening activity; high trip volume; short intra-Manhattan hops (avg 1.8–2.3 mi).<br><br>
                    <b>2. Commuter Exporter Hub ({c_commuter} zones):</b><br>
                    Dominant morning outbound surge (06:00–09:00); workers traveling toward central business districts.<br><br>
                    <b>3. Nightlife & Entertainment District ({c_nightlife} zones):</b><br>
                    High volume post-21:00; concentrated in East Village, SoHo, Meatpacking.<br><br>
                    <b>4. Residential Inflow / Attractor ({c_residential} zones):</b><br>
                    Net dropoff destination in evenings; negative net flow ratio during work hours.<br><br>
                    <b>5. Outer Borough Long-Haul Peripheral ({c_peripheral} zones):</b><br>
                    Outer Queens/Bronx/Staten Island; long average distances (>8.0 mi) and airport connections.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Sample Taxi Zones Classified by Mobility Archetype")
    arch_filter = st.selectbox("Filter by Archetype", ["All Archetypes"] + list(cluster_counts.keys()))
    if arch_filter != "All Archetypes":
        filtered_zones = top_zones[top_zones["archetype_name"] == arch_filter]
    else:
        filtered_zones = top_zones

    st.dataframe(
        filtered_zones[["loc_id", "zone_name", "borough_name", "archetype_name", "total_pickups", "avg_distance", "net_flow_ratio"]].head(25),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.markdown("#### Top Origin-Destination Travel Corridors by Segment")

    segment = st.radio("Diurnal Operational Segment", ["All Segments", "Morning_Peak", "Midday", "Evening_Peak", "Night"], horizontal=True)

    if segment != "All Segments" and "time_segment" in od_flows.columns:
        display_od = od_flows[od_flows["time_segment"] == segment].head(20)
    else:
        display_od = od_flows.head(20)

    col_od_table, col_od_chart = st.columns([1.0, 1.0])

    with col_od_table:
        st.dataframe(
            display_od[["corridor", "time_segment", "trip_volume", "avg_distance", "avg_duration"]].head(15),
            use_container_width=True,
            hide_index=True,
        )

    with col_od_chart:
        top10_od = display_od.head(10).sort_values("trip_volume", ascending=True)
        max_od = max(top10_od["trip_volume"]) if not top10_od.empty else 1000
        fig_od = px.bar(
            top10_od,
            x="trip_volume",
            y="corridor",
            orientation="h",
            color="avg_duration",
            text="trip_volume",
            color_continuous_scale=["#38BDF8", "#818CF8", "#F43F5E"],
            title="Top Corridors Ranked by Trip Volume (Color = Avg Duration min)",
        )
        fig_od.update_traces(
            texttemplate="%{x:,.0f}",
            textposition="outside",
            textfont=dict(size=11, color="#E2E8F0"),
            cliponaxis=False,
        )
        fig_od.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 26, 43, 0.5)",
            margin=dict(l=20, r=40, t=40, b=20),
            height=380,
            yaxis=dict(title=""),
            xaxis=dict(title="Trip Count", range=[0, max_od * 1.2]),
        )
        st.plotly_chart(fig_od, use_container_width=True)

with tab3:
    st.markdown("#### K-Means Cluster Optimization Diagnostics")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown(
            """
            <div class='glass-card'>
                <h4 style='color: #38BDF8; margin-top: 0;'>Elbow & Silhouette Evaluation</h4>
                <table style='width: 100%; font-size: 0.85rem; color: #CBD5E1;'>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.1); color: #94A3B8;'>
                        <td style='padding: 6px;'>k (Clusters)</td>
                        <td>Inertia</td>
                        <td>Silhouette Score</td>
                        <td>Interpretation</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 6px;'>k = 2</td>
                        <td>1182.94</td>
                        <td>0.3248</td>
                        <td>Under-segmented (Core vs. Outer)</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 6px;'>k = 3</td>
                        <td>907.73</td>
                        <td>0.3311</td>
                        <td>Mathematical Silhouette Peak</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 6px;'>k = 4</td>
                        <td>767.35</td>
                        <td>0.2727</td>
                        <td>Elbow Transition</td>
                    </tr>
                    <tr style='color: #10B981; font-weight: 700; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 6px;'>★ k = 5</td>
                        <td>692.04</td>
                        <td>0.2525</td>
                        <td>Optimal Operational Archetypes</td>
                    </tr>
                    <tr>
                        <td style='padding: 6px;'>k = 6</td>
                        <td>635.32</td>
                        <td>0.2484</td>
                        <td>Diminishing Returns</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_d2:
        st.markdown(
            """
            <div class='glass-card'>
                <h4 style='color: #818CF8; margin-top: 0;'>Why k=5 was Selected for Production</h4>
                <p style='color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;'>
                    While <b>k=3</b> yielded the highest purely mathematical silhouette score, it grouped distinct operational behaviors together: airport runs and suburban outer zones were conflated, and nightlife districts were merged into the general commercial core.
                </p>
                <p style='color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;'>
                    <b>k=5</b> provided the inflection point where the reduction in inertia remained steep while unlocking clean, actionable segregation for city planners: Commuter hubs, Nightlife zones, Inflow residential districts, and Airport/Long-haul corridors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
