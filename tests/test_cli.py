import pandas as pd
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
    assert f"{config.FIRST_SEASON}-{settings.current_season}" in capsys.readouterr().err


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
    assert "API calls this run: 18" in out
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


def test_simulate_backfill_requires_trained_model(settings, capsys):
    assert cli.main(["simulate", "--backfill"]) == 2
    assert "psu train" in capsys.readouterr().err


def test_trained_alpha_and_shrink_plays_reads_the_saved_report(settings):
    reports = settings.db_path.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "game_model.json").write_text('{"alpha": 5.0, "shrink_plays": 10}', encoding="utf-8")
    assert cli._trained_alpha_and_shrink_plays(settings.db_path.parent) == (5.0, 10)


def test_trained_alpha_and_shrink_plays_falls_back_to_defaults_when_absent(settings):
    reports = settings.db_path.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "game_model.json").write_text('{"sigma": 16.0}', encoding="utf-8")  # older report: no alpha recorded
    assert cli._trained_alpha_and_shrink_plays(settings.db_path.parent) == (config.TRAIN_ALPHA, config.SHRINK_PLAYS)
    assert cli._trained_alpha_and_shrink_plays(settings.db_path.parent / "missing") == (
        config.TRAIN_ALPHA,
        config.SHRINK_PLAYS,
    )


def test_backfill_uses_the_trained_alpha_and_shrink_plays(settings, monkeypatch):
    _trained(settings)
    reports = settings.db_path.parent / "reports"
    (reports / "game_model.json").write_text('{"sigma": 16.0, "alpha": 3.0, "shrink_plays": 40}', encoding="utf-8")
    seen = {}

    def fake_backfill_history(con, **kw):
        seen.update(kw)
        return pd.DataFrame()

    monkeypatch.setattr(cli, "backfill_history", fake_backfill_history)
    assert cli._backfill(settings, team="A", n_sims=50, seed=0, tau=5.0) == 0
    assert seen["alpha"] == 3.0 and seen["shrink_plays"] == 40


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


def test_parser_defaults_come_from_config():
    parser = cli.build_parser()
    b = parser.parse_args(["build"])
    assert (b.garbage, b.alpha) == (config.GARBAGE, config.BUILD_ALPHA)
    t = parser.parse_args(["train"])
    assert (t.alpha, t.shrink_plays) == (config.TRAIN_ALPHA, config.SHRINK_PLAYS)
    s = parser.parse_args(["simulate"])
    assert (s.sims, s.seed, s.tau, s.team) == (config.SIM_N, config.SIM_SEED, config.SIM_TAU, config.TEAM)
    assert s.backfill is False
    assert parser.parse_args(["simulate", "--backfill"]).backfill is True


def test_database_in_use_is_a_clean_error(settings, capsys, monkeypatch):
    import duckdb

    from psu import db

    def locked(*args, **kwargs):
        raise duckdb.IOException("Could not set lock on file: held by PID 1234")

    monkeypatch.setattr(db, "connect", locked)
    assert cli.main(["build"]) == 2
    err = capsys.readouterr().err
    assert "in use by another process" in err and "PID 1234" in err


def _stub_steps(monkeypatch, codes=None, api_calls=4):
    """Replace refresh's four steps with recorders; codes maps step -> exit code (default 0)."""
    from types import SimpleNamespace

    from psu.ingest import IngestResult

    codes = codes or {}
    calls = []

    def ingest_step(settings, seasons, max_calls):
        calls.append(("ingest", seasons, max_calls))
        code = codes.get("ingest", 0)
        return code, (IngestResult(api_calls=api_calls, row_counts={}) if code == 0 else None)

    def build_step(args, settings):
        calls.append(("build", args.garbage, args.alpha))
        return codes.get("build", 0)

    def train_step(args, settings):
        calls.append(("train", args.alpha, args.shrink_plays))
        return codes.get("train", 0)

    def simulate_step(settings, *, team, n_sims, seed, tau):
        calls.append(("simulate", team, n_sims, seed, tau))
        code = codes.get("simulate", 0)
        return code, (SimpleNamespace(team=team, mean_wins=9.47) if code == 0 else None)

    monkeypatch.setattr(cli, "_ingest", ingest_step)
    monkeypatch.setattr(cli, "cmd_build", build_step)
    monkeypatch.setattr(cli, "cmd_train", train_step)
    monkeypatch.setattr(cli, "_simulate", simulate_step)
    return calls


def test_refresh_runs_every_step_in_order_with_shared_defaults(settings, capsys, monkeypatch):
    calls = _stub_steps(monkeypatch)
    assert cli.main(["refresh"]) == 0
    assert calls == [
        ("ingest", [2026], None),
        ("build", config.GARBAGE, config.BUILD_ALPHA),
        ("train", config.TRAIN_ALPHA, config.SHRINK_PLAYS),
        ("simulate", config.TEAM, config.SIM_N, config.SIM_SEED, config.SIM_TAU),
    ]
    out = capsys.readouterr().out
    assert out.index("== ingest ==") < out.index("== build ==") < out.index("== train ==") < out.index("== simulate ==")
    assert "refresh ok in" in out and "4 API calls" in out and "Penn State mean wins 9.47" in out


def test_refresh_passes_flags_through(settings, monkeypatch):
    calls = _stub_steps(monkeypatch)
    assert cli.main(["refresh", "--max-calls", "7", "--sims", "500", "--seed", "3"]) == 0
    assert calls[0] == ("ingest", [2026], 7)
    assert calls[-1] == ("simulate", config.TEAM, 500, 3, config.SIM_TAU)


def test_refresh_stops_at_the_first_failing_step(settings, capsys, monkeypatch):
    calls = _stub_steps(monkeypatch, codes={"build": 2})
    assert cli.main(["refresh"]) == 2
    assert [c[0] for c in calls] == ["ingest", "build"]
    captured = capsys.readouterr()
    assert "refresh stopped at build (exit 2)" in captured.err
    assert "refresh ok" not in captured.out


def test_refresh_budget_stop_returns_3(settings, capsys, monkeypatch):
    calls = _stub_steps(monkeypatch, codes={"ingest": 3})
    assert cli.main(["refresh"]) == 3
    assert [c[0] for c in calls] == ["ingest"]
    assert "refresh stopped at ingest (exit 3)" in capsys.readouterr().err


def test_refresh_skip_ingest_never_ingests(settings, capsys, monkeypatch):
    calls = _stub_steps(monkeypatch)
    assert cli.main(["refresh", "--skip-ingest"]) == 0
    assert [c[0] for c in calls] == ["build", "train", "simulate"]
    out = capsys.readouterr().out
    assert "== ingest ==" not in out and "0 API calls" in out


def test_refresh_without_key_stops_at_ingest(settings, capsys):
    assert cli.main(["refresh"]) == 2
    err = capsys.readouterr().err
    assert "CFBD_API_KEY" in err and "refresh stopped at ingest (exit 2)" in err


def test_refresh_skip_ingest_on_empty_db_stops_at_build(settings, capsys):
    assert cli.main(["refresh", "--skip-ingest"]) == 2
    err = capsys.readouterr().err
    assert "psu ingest" in err and "refresh stopped at build (exit 2)" in err


def test_database_error_that_is_not_a_lock_is_not_blamed_on_another_process(settings, capsys, monkeypatch):
    import duckdb

    from psu import db

    def unreadable(*args, **kwargs):
        raise duckdb.IOException("Cannot open file: Permission denied")

    monkeypatch.setattr(db, "connect", unreadable)
    assert cli.main(["build"]) == 2
    err = capsys.readouterr().err
    assert "cannot open or write" in err and "Permission denied" in err
    assert "in use by another process" not in err


def test_refresh_database_locked_mid_run_reports_the_step(settings, capsys, monkeypatch):
    import duckdb

    calls = _stub_steps(monkeypatch)

    def locked_train(args, settings):
        calls.append(("train",))
        raise duckdb.IOException("Could not set lock on file: held by PID 1234")

    monkeypatch.setattr(cli, "cmd_train", locked_train)
    assert cli.main(["refresh"]) == 2
    assert [c[0] for c in calls] == ["ingest", "build", "train"]
    err = capsys.readouterr().err
    assert "in use by another process" in err and "refresh stopped at train (exit 2)" in err
