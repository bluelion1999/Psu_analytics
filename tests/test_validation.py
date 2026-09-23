"""Checks our data and metrics against CFBD's own advanced season stats for Penn State.

Needs the real data/psu.duckdb from `psu ingest`; skipped when it is absent.

CFBD's advanced stats count every play with a PPA value (including End Period rows and return
touchdowns charged to the defense) and include garbage time. Under that convention our data matches
CFBD exactly. Our metrics use scrimmage plays only, so they are compared with looser tolerances.
"""

from pathlib import Path

import duckdb
import pytest

from psu.build import _PLAY_COLUMNS
from psu.metrics import efficiency, havoc_rate
from psu.transform import enrich_plays

DB = Path(__file__).resolve().parents[1] / "data" / "psu.duckdb"
SEASONS = (2022, 2023, 2024, 2025)
TEAM = "Penn State"

pytestmark = pytest.mark.skipif(not DB.exists(), reason="needs data/psu.duckdb from `psu ingest`")


@pytest.fixture(scope="module")
def ours_and_cfbd():
    con = duckdb.connect(str(DB), read_only=True)
    try:
        season_list = ", ".join(map(str, SEASONS))
        plays = con.execute(f"SELECT {_PLAY_COLUMNS} FROM plays WHERE season IN ({season_list})").df()
        games = con.execute("SELECT id, neutral_site FROM games").df()
        drives = con.execute(
            f"SELECT id, offense, start_offense_score, start_defense_score FROM drives WHERE season IN ({season_list})"
        ).df()
        box = con.execute(
            f"SELECT game_id, season, team, category, stat FROM team_game_stats WHERE season IN ({season_list})"
        ).df()
        cfbd = con.execute(f"SELECT * FROM advanced_season WHERE team = '{TEAM}'").df().set_index("season")
    finally:
        con.close()
    enriched = enrich_plays(plays, games, drives)
    offense = efficiency(enriched, "offense", exclude_garbage=False).set_index(["team", "season"]).loc[TEAM]
    defense = efficiency(enriched, "defense", exclude_garbage=False).set_index(["team", "season"]).loc[TEAM]
    havoc = havoc_rate(box, enriched).set_index(["team", "season"]).loc[TEAM]
    all_ppa = plays[plays["ppa"].notna()]
    return offense, defense, havoc, cfbd, all_ppa


@pytest.mark.parametrize("season", SEASONS)
@pytest.mark.parametrize("side", ["offense", "defense"])
def test_data_matches_cfbd_exactly_under_its_convention(ours_and_cfbd, season, side):
    *_, cfbd, all_ppa = ours_and_cfbd
    mine = all_ppa[(all_ppa["season"] == season) & (all_ppa[side] == TEAM)]
    assert len(mine) == cfbd.loc[season, f"{side}_plays"]
    assert mine["ppa"].mean() == pytest.approx(cfbd.loc[season, f"{side}_ppa"], abs=1e-4)


@pytest.mark.parametrize("season", SEASONS)
@pytest.mark.parametrize("side", ["offense", "defense"])
def test_scrimmage_metrics_roughly_match_cfbd(ours_and_cfbd, season, side):
    offense, defense, _, cfbd, _ = ours_and_cfbd
    ours = (offense if side == "offense" else defense).loc[season]
    theirs = cfbd.loc[season]
    assert ours["plays"] == pytest.approx(theirs[f"{side}_plays"], rel=0.025)
    assert ours["epa_per_play"] == pytest.approx(theirs[f"{side}_ppa"], abs=0.03)
    assert ours["success_rate"] == pytest.approx(theirs[f"{side}_success_rate"], abs=0.015)
    # CFBD doesn't publish its explosiveness formula; the observed gap is up to 0.11 on defense.
    assert ours["explosiveness"] == pytest.approx(theirs[f"{side}_explosiveness"], abs=0.12)


@pytest.mark.parametrize("season", SEASONS)
def test_havoc_matches_cfbd(ours_and_cfbd, season):
    _, _, havoc, cfbd, _ = ours_and_cfbd
    assert havoc.loc[season, "havoc_rate"] == pytest.approx(cfbd.loc[season, "defense_havoc_total"], abs=0.02)
