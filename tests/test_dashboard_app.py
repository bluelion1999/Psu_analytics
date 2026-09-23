from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import seed_dashboard_db
from psu.db import connect

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ["overview", "trends", "games", "players", "predictions"]
REAL_DB = ROOT / "data" / "psu.duckdb"


def _seed(tmp_path, monkeypatch, *, with_model=True):
    path = tmp_path / "psu.duckdb"
    con = connect(path)
    try:
        seed_dashboard_db(con, with_model=with_model)
    finally:
        con.close()
    if with_model:
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "game_model.json").write_text('{"sigma": 16.0}', encoding="utf-8")
    monkeypatch.setenv("PSU_DB_PATH", str(path))
    return path


def _run(view, season=None):
    at = AppTest.from_file(str(ROOT / "app" / "views" / f"{view}.py"), default_timeout=60)
    at.run()
    if season is not None:
        at.sidebar.selectbox[0].set_value(season).run()
    return at


@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders_for_the_latest_season(tmp_path, monkeypatch, view):
    _seed(tmp_path, monkeypatch)
    at = _run(view)  # default season 2027: one unplayed game
    assert not at.exception, at.exception
    assert at.title


@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders_for_a_season_in_progress(tmp_path, monkeypatch, view):
    _seed(tmp_path, monkeypatch)
    at = _run(view, season=2026)
    assert not at.exception, at.exception


def test_overview_shows_record_and_schedule(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = _run("overview", season=2025)
    assert [m.value for m in at.metric][:2] == ["2-0", "1-0"]
    assert len(at.dataframe) >= 1


def test_game_explorer_shows_win_probability_for_the_first_game(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = _run("games", season=2025)
    assert not at.exception
    assert at.selectbox[0].value == 101
    assert any("Win probability" in s.value for s in at.subheader)


def test_navigation_entry_point_runs(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("view", VIEWS)
def test_pages_explain_a_missing_database(tmp_path, monkeypatch, view):
    monkeypatch.setenv("PSU_DB_PATH", str(tmp_path / "missing.duckdb"))
    at = _run(view)
    assert not at.exception
    assert any("psu ingest" in i.value for i in at.info)


@pytest.mark.parametrize("view", VIEWS)
def test_pages_work_before_train_and_simulate(tmp_path, monkeypatch, view):
    _seed(tmp_path, monkeypatch, with_model=False)
    at = _run(view, season=2026)
    assert not at.exception, at.exception


def test_predictions_page_names_the_missing_commands(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, with_model=False)
    at = _run("predictions", season=2026)
    text = " ".join(i.value for i in at.info)
    assert "psu simulate" in text and "psu train" in text


def test_refresh_button_clears_cache(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = _run("overview")
    at.sidebar.button[0].click().run()
    assert not at.exception


def test_predictions_scopes_simulation_to_its_own_season(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = _run("predictions")  # default season 2027; the seeded simulation covers 2026 only
    assert not at.exception, at.exception
    assert any("covers 2026" in i.value for i in at.info)
    assert not at.metric


def test_season_picker_keeps_the_selection_across_pages(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = AppTest.from_file(str(ROOT / "app" / "views" / "overview.py"), default_timeout=60)
    at.session_state["_season"] = 2025
    at.run()
    assert at.sidebar.selectbox[0].value == 2025


def test_vegas_cell_is_blank_for_a_game_without_a_line(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, with_model=False)
    at = _run("overview", season=2026)
    assert not at.exception, at.exception
    table = at.dataframe[0].value
    assert (table["Vegas"] == "").all()
    assert not table.astype(str).apply(lambda c: c.str.contains("None")).any().any()


@pytest.mark.skipif(not REAL_DB.exists(), reason="needs data/psu.duckdb")
@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders_on_the_real_database(monkeypatch, view):
    monkeypatch.setenv("PSU_DB_PATH", str(REAL_DB))
    at = _run(view)
    assert not at.exception, at.exception
