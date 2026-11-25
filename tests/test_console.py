"""Exercise the console's experiment lifecycle with Streamlit's real runner."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_experiment_submission_resets_cursor_and_retains_previous_run(monkeypatch):
    monkeypatch.delenv("LAB_ACCESS_CODE", raising=False)
    monkeypatch.delenv("LAB_REQUIRE_AUTH", raising=False)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    original = app.session_state["engine"]
    app.slider(key="replay_cursor").set_value(100).run()
    assert app.slider(key="replay_cursor").value == 100
    app.sidebar.selectbox[0].set_value("Occlusion Stress Test")
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert app.session_state["engine"].config.scenario == "Occlusion Stress Test"
    assert app.session_state["previous_run"] is original
    assert app.slider(key="replay_cursor").value == len(app.session_state["engine"].log)-1


@pytest.mark.parametrize("workspace", ["Perception Lab", "Model Evaluation", "Deployment", "Architecture"])
def test_workspaces_render(workspace, monkeypatch):
    monkeypatch.delenv("LAB_ACCESS_CODE", raising=False)
    monkeypatch.delenv("LAB_REQUIRE_AUTH", raising=False)
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.radio[0].set_value(workspace).run()
    assert not app.exception


def test_cloud_access_is_required_before_simulation(monkeypatch):
    monkeypatch.setenv("LAB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("LAB_ACCESS_CODE", "test-console-code")
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert "engine" not in app.session_state
    app.text_input[0].set_value("wrong")
    app.button[0].click().run()
    assert "engine" not in app.session_state
    app.text_input[0].set_value("test-console-code")
    app.button[0].click().run()
    assert not app.exception
    assert "engine" in app.session_state
