"""
Unit tests for AI Mobility Assistant (Track 5 Bonus).
Validates SQL security sanitization, DuckDB in-memory loading,
preset query execution, chart format detection, and narration fallbacks.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.ai_assistant as ai


def test_validate_sql_valid_queries():
    """Verify that legitimate analytical read-only queries pass validation."""
    valid_queries = [
        "SELECT borough_name, SUM(actual_demand) FROM demand GROUP BY borough_name;",
        "SELECT * FROM zones WHERE borough_name = 'Manhattan' LIMIT 10",
        "SELECT zone_name, archetype_name FROM clusters WHERE avg_distance > 5.0",
        "WITH ranked AS (SELECT zone_name, SUM(actual_demand) AS d FROM demand GROUP BY zone_name) SELECT * FROM ranked ORDER BY d DESC LIMIT 5;",
        "SELECT HOUR(pickup_hour) AS hr, AVG(actual_demand) FROM demand WHERE zone_name ILIKE '%Village%' GROUP BY hr;",
    ]
    for q in valid_queries:
        is_valid, reason = ai.validate_sql(q)
        assert is_valid, f"Expected query to be valid: '{q}'. Reason: {reason}"


def test_validate_sql_forbidden_mutations():
    """Verify that dangerous write, DDL, or administrative queries are strictly blocked."""
    forbidden_queries = [
        "DROP TABLE demand;",
        "DELETE FROM zones WHERE loc_id = 1;",
        "INSERT INTO demand VALUES (1, '2026-03-01', 50, 'Test', 'Manhattan');",
        "UPDATE demand SET actual_demand = 0;",
        "ALTER TABLE clusters ADD COLUMN test INT;",
        "CREATE TABLE test (id INT);",
        "TRUNCATE TABLE demand;",
    ]
    for q in forbidden_queries:
        is_valid, reason = ai.validate_sql(q)
        assert not is_valid, f"Expected forbidden query to fail validation: '{q}'"
        assert "Only read-only" in reason or "Forbidden keyword" in reason


def test_validate_sql_multiple_statements():
    """Verify that query chaining via semicolon injection is rejected."""
    chained_query = "SELECT * FROM demand; DELETE FROM demand;"
    is_valid, reason = ai.validate_sql(chained_query)
    assert not is_valid
    assert "Multiple SQL statements" in reason


def test_validate_sql_disallowed_sources():
    """Verify that queries targeting system schemas or external file readers are blocked."""
    disallowed_queries = [
        "SELECT * FROM read_csv('secret.csv')",
        "SELECT * FROM read_parquet('data.parquet')",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM sqlite_master",
    ]
    for q in disallowed_queries:
        is_valid, reason = ai.validate_sql(q)
        assert not is_valid
        assert "Disallowed table source" in reason


def test_clean_sql():
    """Verify markdown code fences and comment headers are properly stripped."""
    raw_markdown = "```sql\nSELECT * FROM demand LIMIT 5;\n```"
    cleaned = ai.clean_sql(raw_markdown)
    assert cleaned == "SELECT * FROM demand LIMIT 5;"

    raw_duckdb = "```duckdb\n-- Comment\nSELECT count(*) FROM zones;\n```"
    cleaned_duckdb = ai.clean_sql(raw_duckdb)
    assert cleaned_duckdb == "SELECT count(*) FROM zones;"

    ask_prompt = "ASK: Which borough are you interested in?"
    assert ai.clean_sql(ask_prompt) == ask_prompt


def test_load_duckdb_tables_and_execute():
    """Verify DuckDB loads the three required tables and executes analytical queries."""
    conn = ai.load_duckdb_tables()
    assert conn is not None

    df_demand = ai.execute_sql(conn, "SELECT COUNT(*) AS total_rows FROM demand;")
    assert not df_demand.empty
    assert df_demand.iloc[0]["total_rows"] > 0

    df_zones = ai.execute_sql(conn, "SELECT COUNT(*) AS total_zones FROM zones;")
    assert not df_zones.empty
    assert df_zones.iloc[0]["total_zones"] >= 200

    df_clusters = ai.execute_sql(conn, "SELECT COUNT(*) AS total_clustered FROM clusters;")
    assert not df_clusters.empty
    assert df_clusters.iloc[0]["total_clustered"] > 0


def test_fallback_presets_execution():
    """Verify that every single preset query executes without error and yields results."""
    conn = ai.load_duckdb_tables()
    for question, data in ai.FALLBACK_PRESETS.items():
        sql = data["sql"]
        is_valid, reason = ai.validate_sql(sql)
        assert is_valid, f"Preset '{question}' has invalid SQL: {reason}"

        df = ai.execute_sql(conn, sql)
        assert not df.empty, f"Preset '{question}' returned empty DataFrame"
        assert len(df) > 0


def test_detect_chart_type():
    """Verify correct chart type deduction for various table structures."""
    # Empty
    assert ai.detect_chart_type(pd.DataFrame()) == "empty"

    # Single metric
    df_metric = pd.DataFrame([{"total_trips": 1250000}])
    assert ai.detect_chart_type(df_metric) == "metric"

    # Timeseries
    df_ts = pd.DataFrame([{"hour": 0, "demand": 100}, {"hour": 1, "demand": 120}])
    assert ai.detect_chart_type(df_ts) == "timeseries"

    # Bar chart
    df_bar = pd.DataFrame([
        {"borough": "Manhattan", "trips": 500},
        {"borough": "Queens", "trips": 200},
    ])
    assert ai.detect_chart_type(df_bar) == "bar"


def test_create_chart():
    """Verify Plotly figure generation for bar and timeseries."""
    df_bar = pd.DataFrame([{"borough": "Manhattan", "trips": 500}, {"borough": "Queens", "trips": 200}])
    fig_bar = ai.create_chart(df_bar, "bar")
    assert fig_bar is not None

    df_ts = pd.DataFrame([{"hour": 1, "trips": 10}, {"hour": 2, "trips": 25}])
    fig_ts = ai.create_chart(df_ts, "timeseries")
    assert fig_ts is not None


def test_generate_sql_query_preset_and_no_key():
    """Verify preset query matching and graceful degradation when API key is missing."""
    # Preset match
    action, sql = ai.generate_sql_query("Which borough has the highest trip demand?", api_key="")
    assert action == "SQL"
    assert "demand" in sql.lower()

    # Unknown question with no API key
    action_no_key, msg = ai.generate_sql_query("What is the weather tomorrow?", api_key="")
    assert action_no_key == "CANNOT_ANSWER"
    assert "API key" in msg


def test_call_gemini_uses_supported_model_fallbacks(monkeypatch):
    """Verify Gemini retries only models that support generateContent."""
    attempted_models = []

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            attempted_models.append(model)
            if model == "gemini-2.5-flash-lite":
                return type("Response", (), {"text": "SELECT 1"})()
            raise RuntimeError("model unavailable")

    class FakeClient:
        models = FakeModels()

    class FakeGenai:
        Client = lambda self, api_key: FakeClient()

    class FakeTypes:
        GenerateContentConfig = lambda self, **kwargs: kwargs

    monkeypatch.setitem(__import__("sys").modules, "google.genai", FakeGenai())
    monkeypatch.setitem(__import__("sys").modules, "google.genai.types", FakeTypes())
    monkeypatch.setattr("google.genai", FakeGenai(), raising=False)

    result = ai.call_gemini("question", "test-key")

    assert result == "SELECT 1"
    assert attempted_models == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    assert "gemini-1.5-flash" not in attempted_models


def test_narrate_result_fallbacks():
    """Verify narrative synthesis fallback under various data shapes."""
    # Empty
    narrative_empty = ai.narrate_result("test", "SELECT 1", pd.DataFrame())
    assert "no matching records" in narrative_empty

    # Preset
    preset_q = "Which borough has the highest trip demand?"
    narrative_preset = ai.narrate_result(preset_q, "SELECT ...", pd.DataFrame([{"a": 1}]))
    assert "Manhattan" in narrative_preset

    # Generic data without API key
    df_generic = pd.DataFrame([{"zone_name": "Midtown Center", "total_demand": 50000}])
    narrative_generic = ai.narrate_result("Top zone?", "SELECT ...", df_generic, api_key="")
    assert "Midtown Center" in narrative_generic
