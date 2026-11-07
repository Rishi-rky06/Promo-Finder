"""Smoke tests: the Streamlit app runs, opens a committed demo run, and processes the sample from the UI."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


def test_app_loads_and_opens_demo_run():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    sel = at.sidebar.selectbox[0]
    demo = [o for o in sel.options if "synthetic_sample_nemotron" in o]
    assert demo, "committed demo run must be selectable on a fresh deployment"
    at.sidebar.selectbox[0].select(demo[0]).run()
    assert not at.exception
    assert any("Promo Finder" in t.value for t in at.title)


def test_app_processes_new_screenplays_with_heuristic(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("promo_finder.config.load_dotenv", lambda *a, **k: [])  # ignore a developer .env
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Process new screenplays").run()
    labels = at.sidebar.selectbox[0].options
    assert labels == ["heuristic (no model, no key)"], "only keyless providers may be offered when no key is configured"
    at.sidebar.text_input[0].set_value(str(ROOT / "samples" / "synthetic_show"))
    at.sidebar.text_input[1].set_value("ui_test_run")
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert (ROOT / "outputs" / "ui_test_run" / "results.json").exists()
