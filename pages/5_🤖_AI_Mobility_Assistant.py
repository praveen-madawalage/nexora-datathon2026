"""
AI Mobility Assistant (Track 5 Bonus Module)
Intelligent Question-Answering Agent translating natural language questions
to safe DuckDB SQL queries over taxi mobility datasets, delivering visual charts,
data tables, and executive summaries. Powered by Google Gemini Flash.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.ai_assistant import (
    FALLBACK_PRESETS,
    create_chart,
    detect_chart_type,
    execute_sql,
    generate_sql_query,
    load_duckdb_tables,
    narrate_result,
    validate_sql,
)
from src.app_utils import apply_custom_theme

st.set_page_config(
    page_title="AI Mobility Assistant | Nexora",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()

# In-memory database connection
conn = load_duckdb_tables()

# API Key resolution: Environment -> Secrets -> Default empty
env_key = os.environ.get("GEMINI_API_KEY", "")
if not env_key:
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            env_key = str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        env_key = ""

# Initialize session state for messages
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to the Nexora AI Mobility Assistant!**\n\n"
                "I translate your natural language questions into secure DuckDB SQL queries across "
                "our **hourly demand timeseries**, **265 taxi zones**, and **mobility cluster archetypes**.\n\n"
                "Try asking a question below, or click any of the **Quick Preset Questions** to explore!"
            ),
            "sql": None,
            "df": None,
            "chart_type": None,
            "exec_time_ms": 0.0,
        }
    ]

# Sidebar: Configuration & Schema Reference
with st.sidebar:
    st.markdown("### 🤖 **AI Assistant Config**")
    st.markdown("**Powered by Gemini Flash & DuckDB**")
    st.markdown("---")

    # API Key management
    user_key_input = st.text_input(
        "🔑 Gemini API Key (Optional Override):",
        value="",
        type="password",
        help="If not configured in Streamlit Secrets, enter your Google Gemini API key here.",
    )
    active_api_key = user_key_input.strip() if user_key_input.strip() else env_key

    if active_api_key:
        st.success("🟢 Gemini Flash API Active")
    else:
        st.warning("🟡 Using Offline Presets & Rules")
        with st.expander("ℹ️ How to set API Key"):
            st.markdown(
                """
                **Streamlit Cloud Secrets:**
                In App Settings → Secrets, add:
                ```toml
                GEMINI_API_KEY = "your-api-key"
                ```
                **Local Run:**
                ```powershell
                $env:GEMINI_API_KEY = "your-api-key"
                ```
                *Preset queries work even without an API key!*
                """
            )

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "Conversation cleared. How can I assist your fleet analysis today?",
                "sql": None,
                "df": None,
                "chart_type": None,
                "exec_time_ms": 0.0,
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📚 **Database Schema**")
    with st.expander("📊 `demand` (Hourly Pickups)", expanded=False):
        st.markdown(
            """
            - `origin_loc_id` (INT)
            - `pickup_hour` (TIMESTAMP)
            - `actual_demand` (INT)
            - `zone_name` (VARCHAR)
            - `borough_name` (VARCHAR)
            """
        )
    with st.expander("🗺️ `zones` (265 Taxi Zones)", expanded=False):
        st.markdown(
            """
            - `loc_id` (INT)
            - `borough_name` (VARCHAR)
            - `zone_name` (VARCHAR)
            - `service_zone` (VARCHAR)
            """
        )
    with st.expander("🏙️ `clusters` (Archetypes)", expanded=False):
        st.markdown(
            """
            - `loc_id` (INT)
            - `zone_name` (VARCHAR)
            - `borough_name` (VARCHAR)
            - `total_pickups` (INT)
            - `archetype_name` (VARCHAR)
            - `avg_distance` (FLOAT)
            - `net_flow_ratio` (FLOAT)
            """
        )

    st.markdown("---")
    st.markdown("### 🛡️ **Security Guardrails**")
    st.markdown(
        """
        <div style='font-size: 0.8rem; color: #94A3B8; line-height: 1.5;'>
        • <b>Zero-Mutation:</b> Blocks <code>DROP</code>, <code>DELETE</code>, <code>INSERT</code>, <code>UPDATE</code>, <code>ALTER</code>.<br>
        • <b>Sandbox:</b> DuckDB strictly in-memory; no filesystem writes.<br>
        • <b>Graceful Fallbacks:</b> Rejection of ambiguous / malformed prompts.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Page Header
st.markdown(
    """
    <div style='margin-bottom: 24px;'>
        <div style='display: flex; gap: 8px; margin-bottom: 10px;'>
            <span class='badge badge-cyan'>TRACK 5 BONUS</span>
            <span class='badge badge-emerald'>GEMINI FLASH</span>
            <span class='badge badge-purple'>DUCKDB IN-MEMORY</span>
            <span class='badge badge-cyan'>ZERO LEAKAGE</span>
        </div>
        <h1 style='font-size: 2.6rem; font-weight: 800; margin: 0; background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 60%, #818CF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            AI Mobility Assistant
        </h1>
        <p style='font-size: 1.1rem; color: #94A3B8; margin-top: 8px; line-height: 1.6;'>
            Natural language business intelligence for city planners and fleet dispatchers.
            Ask arbitrary questions about trip volume, hourly spikes, borough distribution, or mobility archetypes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Preset Question Chips (Quick select)
st.markdown("##### ⚡ **Quick Presets (Click to Ask):**")
preset_cols = st.columns(3)
preset_questions = list(FALLBACK_PRESETS.keys())

selected_preset: Optional[str] = None

for idx, q in enumerate(preset_questions):
    col = preset_cols[idx % 3]
    with col:
        if st.button(f"💬 {q}", key=f"btn_preset_{idx}", use_container_width=True):
            selected_preset = q

# Process Question Function
def process_user_query(question_text: str) -> None:
    # 1. Add user message
    st.session_state["chat_messages"].append(
        {
            "role": "user",
            "content": question_text,
            "sql": None,
            "df": None,
            "chart_type": None,
            "exec_time_ms": 0.0,
        }
    )

    # 2. Generate SQL
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["chat_messages"][-4:]
    ]

    with st.spinner("🤖 Translating question to safe DuckDB SQL via Gemini Flash..."):
        action, payload = generate_sql_query(
            question=question_text,
            api_key=active_api_key,
            chat_history=history,
        )

    # 3. Handle Responses
    if action == "ASK":
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": f"🤔 **Clarification needed:**\n\n{payload}",
                "sql": None,
                "df": None,
                "chart_type": None,
                "exec_time_ms": 0.0,
            }
        )
    elif action == "CANNOT_ANSWER":
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": f"⚠️ **Notice:**\n\n{payload}",
                "sql": None,
                "df": None,
                "chart_type": None,
                "exec_time_ms": 0.0,
            }
        )
    elif action == "ERROR":
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": f"❌ **Error generating query:**\n\n{payload}",
                "sql": None,
                "df": None,
                "chart_type": None,
                "exec_time_ms": 0.0,
            }
        )
    elif action == "SQL":
        sql_to_run = payload
        is_safe, reason = validate_sql(sql_to_run)
        if not is_safe:
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": f"🛡️ **Security Check Blocked Execution:**\n\n{reason}",
                    "sql": sql_to_run,
                    "df": None,
                    "chart_type": None,
                    "exec_time_ms": 0.0,
                }
            )
            return

        # Execute query
        t0 = time.perf_counter()
        try:
            res_df = execute_sql(conn, sql_to_run)
            exec_time = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": f"⚠️ **DuckDB Execution Error:**\n\n`{e}`\n\nPlease try rephrasing your question.",
                    "sql": sql_to_run,
                    "df": None,
                    "chart_type": None,
                    "exec_time_ms": 0.0,
                }
            )
            return

        # Detect chart format
        chart_format = detect_chart_type(res_df)

        # Generate narrative
        with st.spinner("✍️ Composing executive insight summary..."):
            narrative = narrate_result(
                question=question_text,
                sql=sql_to_run,
                df=res_df,
                api_key=active_api_key,
            )

        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": narrative,
                "sql": sql_to_run,
                "df": res_df,
                "chart_type": chart_format,
                "exec_time_ms": exec_time,
            }
        )


# Handle Preset Click
if selected_preset:
    process_user_query(selected_preset)
    st.rerun()

# Render Chat History
st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

for m in st.session_state["chat_messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # If there's an attached dataframe and chart
        df: Optional[pd.DataFrame] = m.get("df")
        chart_type = m.get("chart_type")
        sql = m.get("sql")
        exec_ms = m.get("exec_time_ms", 0.0)

        if df is not None and not df.empty:
            # 1. Visual Presentation
            if chart_type == "metric" and len(df) == 1 and len(df.columns) <= 2:
                col_name = df.columns[-1]
                val = df.iloc[0, -1]
                formatted_val = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}" if isinstance(val, int) else str(val)
                st.markdown(
                    f"""
                    <div class='glass-card' style='text-align: center; max-width: 320px; margin: 12px 0;'>
                        <div class='metric-hero-val' style='font-size: 2.4rem;'>{formatted_val}</div>
                        <div class='metric-hero-lbl'>{col_name.replace('_', ' ').title()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif chart_type in {"timeseries", "bar"}:
                fig = create_chart(df, chart_type)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

            # 2. Expandable Query and Raw Data Details
            with st.expander(f"🔍 View Generated SQL & Execution Details ({exec_ms:.1f} ms)"):
                st.markdown(f"**DuckDB Query Executed:**")
                st.code(sql, language="sql")
                st.markdown(f"**Query Results Table ({len(df)} rows):**")
                st.dataframe(df, use_container_width=True)
                st.caption(f"🛡️ Security status: Read-only SELECT validated • In-memory execution: {exec_ms:.2f} ms")
        elif sql:
            with st.expander("🔍 View Attempted SQL"):
                st.code(sql, language="sql")

# Chat input bar at bottom
prompt = st.chat_input("Ask any question about NYC taxi demand, zone trends, or archetypes...")
if prompt:
    process_user_query(prompt)
    st.rerun()
