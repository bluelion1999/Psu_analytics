import json

import pytest
from conftest import seed_league_db, synthetic_league

from psu.db import connect
from psu.simulate import (
    SUMMARY_COLUMNS,
    MissingModel,
    load_inputs,
    load_sigma,
    report_markdown,
    run_simulation,
    simulate_season,
    write_results,
)


def test_load_inputs_without_predictions_is_missing_model():
    con = connect(":memory:")
    with pytest.raises(MissingModel, match="psu train"):
        load_inputs(con, 2026)


def test_load_sigma_missing_file(tmp_path):
    with pytest.raises(MissingModel, match="psu train"):
        load_sigma(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "game_model.json").write_text(json.dumps({"sigma": 16.25}), encoding="utf-8")
    assert load_sigma(tmp_path) == 16.25


def test_load_sigma_missing_key_or_bad_json_is_missing_model(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "game_model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(MissingModel, match="psu train"):
        load_sigma(tmp_path)
    (tmp_path / "reports" / "game_model.json").write_text("not json", encoding="utf-8")
    with pytest.raises(MissingModel, match="psu train"):
        load_sigma(tmp_path)


def test_simulate_season_from_duckdb_matches_dataframe_run():
    con = connect(":memory:")
    seed_league_db(con)
    from_db = simulate_season(con, season=2026, sigma=16.0, team="A", n_sims=500, seed=3)
    games, upcoming = synthetic_league()
    direct = run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=500, seed=3)
    assert from_db.win_totals.equals(direct.win_totals)
    assert from_db.p_conf_champ == direct.p_conf_champ
    assert from_db.as_of == direct.as_of


def test_load_inputs_returns_every_prediction_of_the_season():
    con = connect(":memory:")
    seed_league_db(con)
    con.execute("UPDATE game_predictions SET split = 'in_sample' WHERE game_id = 5")
    _, predictions = load_inputs(con, 2026)
    assert (predictions["game_id"] == 5).any()


def test_write_results_replaces_tables_and_writes_report(tmp_path):
    con = connect(":memory:")
    games, upcoming = synthetic_league()
    result = run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=300)
    write_results(con, result, tmp_path)
    write_results(con, result, tmp_path)  # a second run replaces, never appends

    summary = con.execute("SELECT * FROM sim_team_summary").df()
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert len(summary) == 1 and summary.loc[0, "team"] == "A" and summary.loc[0, "n_sims"] == 300

    totals = con.execute("SELECT * FROM sim_win_totals").df()
    assert list(totals.columns) == ["season", "team", "wins", "prob"]
    assert len(totals) == 6 and totals["prob"].sum() == pytest.approx(1.0)

    conf = con.execute("SELECT * FROM sim_conference").df()
    assert list(conf.columns) == ["season", "team", "mean_conf_wins", "p_title_game", "p_conf_champ"]
    assert len(conf) == 4

    text = (tmp_path / "reports" / "season_sim.md").read_text(encoding="utf-8")
    assert text == report_markdown(result)


def test_report_is_ascii_and_names_the_key_odds():
    games, upcoming = synthetic_league()
    text = report_markdown(run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=200))
    text.encode("ascii")
    for label in ("Mean wins", "P(10+ wins)", "P(title game)", "P(Big Ten champ)", "P(CFP)", "2026-09-01"):
        assert label in text


def test_report_before_any_game_is_played():
    games, upcoming = synthetic_league()
    games = games[~games["completed"]]
    text = report_markdown(run_simulation(games, upcoming, season=2026, sigma=16.0, team="A", n_sims=50))
    assert "no games played yet" in text
