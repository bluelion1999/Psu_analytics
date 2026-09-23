import pytest

from psu import cli, config


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = config.Settings(
        api_key=None,
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "psu.duckdb",
        current_season=2026,
        min_interval_s=0,
    )
    monkeypatch.setattr(config, "load_settings", lambda: s)
    return s


def test_ingest_without_key_or_cache_explains_how_to_fix(settings, capsys):
    assert cli.main(["ingest", "--seasons", "2024"]) == 2
    assert "CFBD_API_KEY" in capsys.readouterr().err


def test_bad_season_spec_is_a_usage_error(settings, capsys):
    assert cli.main(["ingest", "--seasons", "2031"]) == 2
    assert "2022-2026" in capsys.readouterr().err


def test_status_lists_all_tables(settings, capsys):
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "plays" in out and "ratings_sp" in out


def test_status_does_not_create_the_database(settings, capsys):
    assert cli.main(["status"]) == 0
    capsys.readouterr()
    assert not settings.db_path.exists()


def test_budget_exhaustion_exits_cleanly(settings, capsys, monkeypatch, fake_cfbd):
    monkeypatch.setattr(cli, "make_fetch", lambda s: fake_cfbd)
    assert cli.main(["ingest", "--seasons", "2024", "--max-calls", "3"]) == 3
    captured = capsys.readouterr()
    out, err = captured.out, captured.err
    assert "budget" in err.lower()
    assert "plays" in out
    assert len(fake_cfbd.calls) == 3


def test_ingest_success_reports_calls_and_counts(settings, capsys, monkeypatch, fake_cfbd):
    monkeypatch.setattr(cli, "make_fetch", lambda s: fake_cfbd)
    assert cli.main(["ingest", "--seasons", "2024"]) == 0
    out = capsys.readouterr().out
    assert "API calls this run: 17" in out
    assert "plays" in out


def test_build_requires_ingested_data(settings, capsys):
    assert cli.main(["build"]) == 2
    assert "psu ingest" in capsys.readouterr().err


def test_build_rejects_bad_garbage_spec(settings, capsys):
    assert cli.main(["build", "--garbage", "30,20"]) == 2
    assert "38,28,22" in capsys.readouterr().err


def test_build_writes_tables(settings, capsys):
    from conftest import seed_raw_tables
    from psu import db

    con = db.connect(settings.db_path)
    seed_raw_tables(con)
    con.close()
    assert cli.main(["build", "--alpha", "1"]) == 0
    out = capsys.readouterr().out
    assert "team_offense" in out and "team_adjusted" in out


def test_train_requires_ingested_data(settings, capsys):
    assert cli.main(["train"]) == 2
    assert "psu ingest" in capsys.readouterr().err


def test_train_reports_and_writes_outputs(settings, capsys, monkeypatch):
    from conftest import seed_raw_tables, synthetic_features
    from psu import db

    con = db.connect(settings.db_path)
    seed_raw_tables(con)
    con.close()
    monkeypatch.setattr(cli, "load_features", lambda con, **kw: synthetic_features())
    assert cli.main(["train"]) == 0
    assert "Vegas MAE" in capsys.readouterr().out
    assert (settings.db_path.parent / "models" / "game_model.joblib").exists()


def test_train_with_too_few_seasons_is_a_clean_error(settings, capsys, monkeypatch):
    from conftest import seed_raw_tables, synthetic_features
    from psu import db

    con = db.connect(settings.db_path)
    seed_raw_tables(con)
    con.close()
    few = synthetic_features()
    monkeypatch.setattr(cli, "load_features", lambda con, **kw: few[few["season"] >= 2024])
    assert cli.main(["train"]) == 2
    assert "3 complete seasons" in capsys.readouterr().err


def _trained(settings):
    from conftest import seed_league_db
    from psu import db

    reports = settings.db_path.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "game_model.json").write_text('{"sigma": 16.0}', encoding="utf-8")
    con = db.connect(settings.db_path)
    try:
        seed_league_db(con)
    finally:
        con.close()


def test_simulate_requires_trained_model(settings, capsys):
    assert cli.main(["simulate"]) == 2
    assert "psu train" in capsys.readouterr().err
    assert not settings.db_path.exists()


def test_simulate_writes_tables_and_prints_summary(settings, capsys):
    import duckdb

    _trained(settings)
    assert cli.main(["simulate", "--sims", "200", "--team", "A"]) == 0
    out = capsys.readouterr().out
    assert "P(10+ wins)" in out and "Season simulation: A 2026" in out
    con = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        assert con.execute("SELECT n_sims FROM sim_team_summary").fetchone()[0] == 200
        assert con.execute("SELECT count(*) FROM sim_conference").fetchone()[0] == 4
    finally:
        con.close()
    assert (settings.db_path.parent / "reports" / "season_sim.md").exists()


def test_simulate_rejects_too_large_tau(settings, capsys):
    _trained(settings)
    assert cli.main(["simulate", "--tau", "20", "--team", "A"]) == 2
    assert "tau" in capsys.readouterr().err


def test_simulate_unknown_team_is_a_usage_error(settings, capsys):
    import duckdb

    _trained(settings)
    assert cli.main(["simulate", "--team", "Nobody"]) == 2
    assert "Nobody" in capsys.readouterr().err
    con = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    finally:
        con.close()
    assert "sim_team_summary" not in tables
