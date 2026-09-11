"""
Automated unit tests for Streamlit UI button interactivity and session state bindings.
Validates preset distance buttons, slider synchronization, and calculation trigger feedback
for both Page 1 (Upfront Fare Pricing) and Page 2 (Trip Duration Estimator).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_fare_pricing_preset_buttons_update_slider():
    """Verify that preset distance buttons on Page 1 update the slider and recalculate."""
    page_path = str(ROOT / "pages" / "1_💰_Upfront_Fare_Pricing.py")
    at = AppTest.from_file(page_path).run()

    # Initial default is 2.4 miles
    assert at.slider[0].value == 2.4

    # Click Short (1.2 mi)
    at.button(key="btn_fare_short").click().run()
    assert at.slider[0].value == 1.2

    # Click JFK Run (15.2 mi)
    at.button(key="btn_fare_jfk").click().run()
    assert at.slider[0].value == 15.2

    # Click Calculate Upfront Fare
    at.button(key="btn_calc_fare").click().run()
    assert len(at.toast) == 1
    assert "fare_calc_timestamp" in at.session_state


def test_duration_estimator_preset_buttons_update_slider():
    """Verify that preset distance buttons on Page 2 update the slider and recalculate."""
    page_path = str(ROOT / "pages" / "2_⏱️_Trip_Duration_Estimator.py")
    at = AppTest.from_file(page_path).run()

    # Initial default is 2.8 miles
    assert at.slider[0].value == 2.8

    # Click Short (1.5 mi)
    at.button(key="d_short").click().run()
    assert at.slider[0].value == 1.5

    # Click Midtown (4.2 mi)
    at.button(key="d_mid").click().run()
    assert at.slider[0].value == 4.2

    # Click JFK Expressway (16.0 mi)
    at.button(key="d_jfk").click().run()
    assert at.slider[0].value == 16.0

    # Click Calculate Arrival Time
    at.button(key="btn_calc_duration").click().run()
    assert len(at.toast) == 1
    assert "dur_calc_timestamp" in at.session_state
