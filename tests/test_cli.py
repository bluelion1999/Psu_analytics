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
