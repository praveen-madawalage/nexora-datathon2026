"""
AI Mobility Assistant (Track 5 Bonus Module)
Translates natural language questions to DuckDB SQL queries over taxi mobility datasets,
validates queries against strict safety rules, executes in-memory, and provides visual charts
and plain-English executive summaries. Powered by Google Gemini Flash.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "app" / "data"

# Pre-defined schema context describing tables, types, and domain relationships
SCHEMA_CONTEXT = """
Database: DuckDB (in-memory)
Available Tables:

1. Table: demand
   Description: Hourly aggregated pickup demand for top high-volume zones in NYC (March 2026).
   Columns:
     - origin_loc_id (INTEGER): Taxi zone identifier (e.g. 79, 132, 237)
     - pickup_hour (TIMESTAMP): Timestamp of the observation hour (e.g. '2026-03-01 00:00:00')
     - actual_demand (INTEGER): Number of completed taxi pickups in that zone during that hour
     - zone_name (VARCHAR): Human-readable zone name (e.g. 'East Village', 'JFK Airport', 'Midtown Center')
     - borough_name (VARCHAR): Borough name ('Manhattan', 'Queens', etc.)

2. Table: zones
   Description: Reference directory of all 265 NYC taxi zones.
   Columns:
     - loc_id (INTEGER): Unique zone ID (1 to 265)
     - borough_name (VARCHAR): Borough name ('Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island', 'EWR')
     - zone_name (VARCHAR): Full zone title
     - service_zone (VARCHAR): Regulatory service zone ('Yellow Zone', 'Boro Zone', 'Airports', 'EWR')

3. Table: clusters
   Description: Spatial-temporal mobility clusters and archetype classification across NYC zones.
   Columns:
     - loc_id (INTEGER): Taxi zone ID
     - zone_name (VARCHAR): Name of the zone
     - borough_name (VARCHAR): Borough name
     - total_pickups (INTEGER): Overall observed trips in this zone
     - cluster_id (INTEGER): Cluster numeric ID (0 to 3)
     - archetype_name (VARCHAR): Mobility category name:
         * 'Commercial & High-Density Core'
         * 'Nightlife & Entertainment District'
         * 'Commuter Exporter Hub'
         * 'Residential Inflow / Attractor'
     - avg_distance (DOUBLE): Average trip length in miles
     - net_flow_ratio (DOUBLE): Net directional flow ratio (outflow vs inflow)
"""

# Forbidden SQL statements / operations to prevent injection or modification
FORBIDDEN_KEYWORDS = {
    "drop",
    "delete",
    "insert",
    "update",
    "alter",
    "create",
    "truncate",
    "replace",
    "merge",
    "grant",
    "revoke",
    "attach",
    "detach",
    "copy",
    "load",
    "install",
    "call",
    "execute",
    "exec",
    "pragma",
    "vacuum",
    "reindex",
    "upsert",
    "system",
    "export",
}

DISALLOWED_SOURCES = {
    "read_csv",
    "read_parquet",
    "read_json",
    "duckdb_",
    "information_schema",
    "sqlite_master",
    "glob",
    "httpfs",
}

# Curated benchmark / demo queries that always work offline or as quick chips
FALLBACK_PRESETS: Dict[str, Dict[str, Any]] = {
    "Which borough has the highest trip demand?": {
        "sql": "SELECT borough_name, SUM(actual_demand) AS total_demand FROM demand GROUP BY borough_name ORDER BY total_demand DESC;",
        "narrative": "Manhattan leads NYC taxi demand with over 1.12M trips in March 2026, followed by Queens with 148K trips.",
    },
    "What are the top 5 zones by pickup volume?": {
        "sql": "SELECT zone_name, borough_name, SUM(actual_demand) AS total_pickups FROM demand GROUP BY zone_name, borough_name ORDER BY total_pickups DESC LIMIT 5;",
        "narrative": "The top 5 zones are Upper East Side South, Midtown Center, Midtown East, Upper East Side North, and Union Square.",
    },
    "What is the peak demand hour of the day?": {
        "sql": "SELECT HOUR(pickup_hour) AS hour_of_day, SUM(actual_demand) AS total_demand FROM demand GROUP BY hour_of_day ORDER BY total_demand DESC LIMIT 5;",
        "narrative": "Demand peaks during the evening commute and dinner hours, with 9:00 PM (21:00) reaching the highest total volume at 93.5K trips.",
    },
    "Which mobility archetype generates the most trips?": {
        "sql": "SELECT archetype_name, COUNT(*) AS zone_count, SUM(total_pickups) AS total_trips FROM clusters GROUP BY archetype_name ORDER BY total_trips DESC;",
        "narrative": "Commercial & High-Density Core dominates trip generation with 28.1M trips across 50 zones, followed by Nightlife & Entertainment with 5.5M trips.",
    },
    "What archetype and metrics does JFK Airport have?": {
        "sql": "SELECT zone_name, archetype_name, avg_distance, net_flow_ratio, total_pickups FROM clusters WHERE zone_name ILIKE '%JFK%';",
        "narrative": "JFK Airport is categorized under the Commercial & High-Density Core archetype with a high average trip distance of 15.2 miles.",
    },
    "Which day of the week has the most trips?": {
        "sql": "SELECT DAYNAME(pickup_hour) AS day_of_week, SUM(actual_demand) AS total_demand FROM demand GROUP BY day_of_week ORDER BY total_demand DESC;",
        "narrative": "Tuesdays and Thursdays record the highest aggregate demand (212K and 195K trips), reflecting mid-week business travel patterns.",
    },
}

_CONNECTION_CACHE: Optional[duckdb.DuckDBPyConnection] = None


def load_duckdb_tables() -> duckdb.DuckDBPyConnection:
    """
    Initialize an in-memory DuckDB instance and register demand, zones, and clusters tables.
    Returns the active database connection.
    """
    global _CONNECTION_CACHE
    if _CONNECTION_CACHE is not None:
        return _CONNECTION_CACHE

    conn = duckdb.connect(":memory:")

    # 1. Demand table
    demand_csv = DATA_DIR / "demand_summary.csv"
    if demand_csv.exists():
        df_demand = pd.read_csv(demand_csv)
        df_demand["pickup_hour"] = pd.to_datetime(df_demand["pickup_hour"])
        conn.register("demand", df_demand)
    else:
        conn.execute("CREATE TABLE demand (origin_loc_id INT, pickup_hour TIMESTAMP, actual_demand INT, zone_name VARCHAR, borough_name VARCHAR);")

    # 2. Zones table
    zones_csv = DATA_DIR / "zone_lookup.csv"
    if zones_csv.exists():
        df_zones = pd.read_csv(zones_csv)
        conn.register("zones", df_zones)
    else:
        conn.execute("CREATE TABLE zones (loc_id INT, borough_name VARCHAR, zone_name VARCHAR, service_zone VARCHAR);")

    # 3. Clusters table
    clusters_json = DATA_DIR / "clustering_summary.json"
    if clusters_json.exists():
        with open(clusters_json, "r", encoding="utf-8") as f:
            cl_data = json.load(f)
        top_zones = cl_data.get("top_zones", [])
        df_clusters = pd.DataFrame(top_zones)
        conn.register("clusters", df_clusters)
    else:
        conn.execute("CREATE TABLE clusters (loc_id INT, zone_name VARCHAR, borough_name VARCHAR, total_pickups INT, cluster_id INT, archetype_name VARCHAR, avg_distance FLOAT, net_flow_ratio FLOAT);")

    _CONNECTION_CACHE = conn
    return conn


def clean_sql(raw_text: str) -> str:
    """
    Strips code fences (```sql ... ```), commentary, and surrounding whitespace.
    Preserves special protocol tokens like ASK: and CANNOT_ANSWER:.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # If it's a protocol prefix, pass it directly
    if text.startswith("ASK:") or text.startswith("CANNOT_ANSWER:"):
        return text

    # Remove markdown fences
    text = re.sub(r"^```(?:sql|duckdb)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # Strip single line comment headers if any
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    cleaned = "\n".join(lines).strip()

    return cleaned


def validate_sql(sql: str) -> Tuple[bool, str]:
    """
    Validates that a SQL string is strictly read-only and safe for execution.
    Returns (is_valid, error_reason).
    """
    if not sql or not sql.strip():
        return False, "Query is empty."

    # Strip SQL comments
    s = re.sub(r"--[^\n]*", " ", sql)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    s = s.strip().rstrip(";").strip()

    if not s:
        return False, "Query contains no executable statements."

    # Prevent multiple chained queries
    if ";" in s:
        return False, "Multiple SQL statements are not permitted."

    # Must start with SELECT or WITH (CTE)
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        return False, "Only read-only SELECT or WITH (CTE) queries are allowed."

    # Check for forbidden mutation and system administration keywords
    words = set(re.findall(r"\b[a-zA-Z_]\w*\b", s.lower()))
    found_forbidden = words.intersection(FORBIDDEN_KEYWORDS)
    if found_forbidden:
        return False, f"Forbidden keyword detected: {', '.join(sorted(found_forbidden))}"

    # Check for disallowed system tables or file read functions
    for disallowed in DISALLOWED_SOURCES:
        if disallowed in s.lower():
            return False, f"Disallowed table source or function detected: '{disallowed}'"

    return True, ""


def execute_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """
    Executes a validated SQL statement against DuckDB and returns a pandas DataFrame.
    """
    clean = sql.strip().rstrip(";")
    return conn.execute(clean).df()


def call_gemini(
    prompt: str,
    api_key: str,
    system_instruction: str = "",
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Calls the Google Gemini API using google.genai or fallback to google.generativeai.
    """
    if not api_key:
        raise ValueError("No Gemini API key provided.")

    # Try modern google.genai first
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config_kwargs: Dict[str, Any] = {"temperature": 0.1, "max_output_tokens": 1000}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        # Try specified model with fallbacks
        models_to_try = [model_name, "gemini-2.0-flash-exp", "gemini-1.5-flash"]
        last_err = None
        for m in models_to_try:
            try:
                resp = client.models.generate_content(model=m, contents=prompt, config=config)
                if resp and resp.text:
                    return resp.text.strip()
            except Exception as e:
                last_err = e
                continue

        if last_err:
            raise last_err
    except ImportError:
        pass

    # Fallback to legacy google.generativeai
    try:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=api_key)
        models_to_try = [model_name, "gemini-2.0-flash-exp", "gemini-1.5-flash"]
        for m in models_to_try:
            try:
                gen_model = legacy_genai.GenerativeModel(
                    model_name=m,
                    system_instruction=system_instruction if system_instruction else None,
                )
                resp = gen_model.generate_content(prompt)
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                continue
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}")

    raise RuntimeError("Unable to communicate with Gemini API. Check your API key and network connection.")


def generate_sql_query(
    question: str,
    api_key: str = "",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str]:
    """
    Translates a user question into SQL using Gemini Flash or fallback presets.
    Returns:
        (action, payload)
        - ("SQL", valid_sql_string)
        - ("ASK", clarifying_question)
        - ("CANNOT_ANSWER", explanation)
        - ("ERROR", error_description)
    """
    # Check offline fallback presets first if question closely matches
    q_lower = question.strip().lower().rstrip("?")
    for preset_q, data in FALLBACK_PRESETS.items():
        if preset_q.lower().rstrip("?") in q_lower or q_lower in preset_q.lower().rstrip("?"):
            return "SQL", data["sql"]

    # If no API key is set, return a friendly guidance message
    if not api_key:
        return (
            "CANNOT_ANSWER",
            "Gemini API key is not configured. Please add `GEMINI_API_KEY` to your Streamlit secrets or use one of the preset questions below.",
        )

    system_prompt = f"""
You are an expert DuckDB SQL analyst assistant for NYC taxi mobility operations.
You answer user questions by converting them into a single valid DuckDB SQL SELECT statement.

{SCHEMA_CONTEXT}

STRICT INSTRUCTIONS:
1. ONLY return a single valid DuckDB SQL query. No explanation, no markdown code block (no ```sql).
2. ONLY use SELECT or WITH (CTE) queries. Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER.
3. If the question is ambiguous or lacks necessary parameters to formulate a query, output:
   ASK: <your clarifying question>
4. If the question is gibberish, offensive, or completely unrelated to NYC taxi operations, output:
   CANNOT_ANSWER: <polite statement that this assistant only queries NYC taxi mobility data>
5. DuckDB tips:
   - For case-insensitive zone search: zone_name ILIKE '%keyword%'
   - For hour of day: HOUR(pickup_hour)
   - For day of week name: DAYNAME(pickup_hour)
   - For date: CAST(pickup_hour AS DATE)
   - For aggregations: Use SUM(actual_demand) AS total_demand
   - For ranking zones: Add LIMIT 10 or LIMIT 20 unless specified.
"""

    history_context = ""
    if chat_history:
        recent = chat_history[-3:]  # Last 3 turns
        hist_lines = []
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            hist_lines.append(f"{role.upper()}: {content}")
        history_context = "Recent Conversation:\n" + "\n".join(hist_lines) + "\n\n"

    prompt = f"{history_context}User Question: {question}\n\nDuckDB SQL Query:"

    try:
        raw_res = call_gemini(prompt=prompt, api_key=api_key, system_instruction=system_prompt)
        cleaned = clean_sql(raw_res)

        if cleaned.startswith("ASK:"):
            return "ASK", cleaned[4:].strip()
        if cleaned.startswith("CANNOT_ANSWER:"):
            return "CANNOT_ANSWER", cleaned[14:].strip()

        # Validate SQL
        is_safe, reason = validate_sql(cleaned)
        if not is_safe:
            return "ERROR", f"Safety validator blocked generated query: {reason}"

        return "SQL", cleaned

    except Exception as e:
        return "ERROR", f"Failed to generate query with Gemini: {e}"


def narrate_result(
    question: str,
    sql: str,
    df: pd.DataFrame,
    api_key: str = "",
) -> str:
    """
    Generates a concise 1-2 sentence plain-English executive summary of the query results.
    """
    if df is None or df.empty:
        return "The query returned no matching records in the current observation window."

    # Check preset match first
    q_lower = question.strip().lower().rstrip("?")
    for preset_q, data in FALLBACK_PRESETS.items():
        if preset_q.lower().rstrip("?") in q_lower or q_lower in preset_q.lower().rstrip("?"):
            return data["narrative"]

    # Rule-based fast narrative if no API key
    if not api_key:
        if "borough_name" in df.columns or "zone_name" in df.columns:
            name_col = "zone_name" if "zone_name" in df.columns else "borough_name"
            top_name = df.iloc[0][name_col]
            val_col = df.columns[-1]
            val = df.iloc[0, -1]
            if isinstance(val, (int, float)):
                return f"{top_name} leads the results with a {val_col.replace('_', ' ')} of {val:,.0f}."
            return f"Top result: {top_name} ({val})."
        elif len(df) == 1 and len(df.columns) <= 2:
            col = df.columns[-1]
            val = df.iloc[0, -1]
            if isinstance(val, float):
                return f"The computed {col.replace('_', ' ')} is {val:,.2f}."
            elif isinstance(val, int):
                return f"The computed {col.replace('_', ' ')} is {val:,}."
            return f"The result is {val}."
        return f"Retrieved {len(df)} records matching your query."

    # Gemini LLM narrative
    prompt = f"""
User Question: "{question}"
Executed SQL: "{sql}"
Result Preview (up to 5 rows):
{df.head(5).to_string(index=False)}

Provide a concise, professional 1 to 2 sentence executive summary of what this data shows.
State the top figures or key findings directly in plain English.
Do NOT mention SQL, tables, columns, or database jargon.
"""
    try:
        summary = call_gemini(
            prompt=prompt,
            api_key=api_key,
            system_instruction="You are a data communications executive explaining taxi metrics to city leadership.",
        )
        return summary.strip()
    except Exception:
        # Fallback to simple rule
        return f"Successfully retrieved {len(df)} rows answering your question."


def detect_chart_type(df: pd.DataFrame) -> str:
    """
    Determines the most intuitive visualization format for a given result DataFrame.
    Returns one of: 'empty', 'metric', 'timeseries', 'bar', 'table'.
    """
    if df is None or df.empty:
        return "empty"

    rows, cols = df.shape

    # Single number / metric
    if rows == 1 and cols <= 2:
        return "metric"

    col_names_lower = [c.lower() for c in df.columns]

    # Time-series (hour of day, date, timestamp)
    time_keywords = {"hour", "pickup_hour", "hour_of_day", "date", "day", "pickup_date", "hr"}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or c.lower() in time_keywords:
            # Check if there is also a numeric column
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                return "timeseries"

    # Bar chart for categorical rankings (<= 30 rows)
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1 and rows <= 30:
        return "bar"

    return "table"


def create_chart(
    df: pd.DataFrame,
    chart_type: str,
    title: str = "",
) -> Optional[go.Figure]:
    """
    Generates a dark-themed glassmorphic Plotly figure for the detected chart type.
    """
    if chart_type == "empty" or df.empty:
        return None

    # Common layout styles
    theme_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14, 23, 42, 0.6)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False),
    )

    if chart_type == "timeseries":
        # Identify x and y
        time_cols = [c for c in df.columns if c.lower() in {"hour", "pickup_hour", "hour_of_day", "date", "day", "hr"}]
        x_col = time_cols[0] if time_cols else df.columns[0]
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != x_col]
        y_col = numeric_cols[0] if numeric_cols else df.columns[-1]

        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            markers=True,
            title=title or f"{y_col.replace('_', ' ').title()} over {x_col.replace('_', ' ').title()}",
            color_discrete_sequence=["#38BDF8"],
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=7, color="#818CF8"))
        fig.update_layout(**theme_layout)
        return fig

    if chart_type == "bar":
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        x_col = cat_cols[0] if len(cat_cols) > 0 else df.columns[0]
        num_cols = df.select_dtypes(include=["number"]).columns
        y_col = num_cols[0] if len(num_cols) > 0 else df.columns[-1]

        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title or f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
            color=y_col,
            color_continuous_scale="Blues",
        )
        fig.update_layout(**theme_layout)
        fig.update_coloraxes(showscale=False)
        return fig

    return None
