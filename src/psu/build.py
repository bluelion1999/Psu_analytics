"""Materialize the Phase 2 metric tables in DuckDB from the raw Phase 1 tables (no API calls)."""
from __future__ import annotations

import duckdb
import pandas as pd

from psu.adjust import opponent_adjust
from psu.metrics import efficiency, havoc_rate, red_zone, turnover_margin
from psu.transform import Explosive, GarbageTime, enrich_plays

SPLITS = ("down", "quarter", "score_state", "venue", "opp_conference")
DERIVED_TABLES = (
    "plays_enriched", "team_offense", "team_defense", "team_splits",
    "team_adjusted", "team_havoc", "team_turnovers", "team_red_zone",
)
_PLAY_COLUMNS = (
    "id, game_id, drive_id, season, week, season_type, offense, offense_conference, defense, "
    "defense_conference, home, away, period, down, distance, yards_to_goal, yards_gained, play_type, "
    "play_text, ppa, offense_score, defense_score"
)


def fbs_teams(games: pd.DataFrame) -> pd.DataFrame:
    sides = [
        games.loc[games[f"{side}_classification"] == "fbs", ["season", f"{side}_team"]].rename(
            columns={f"{side}_team": "team"}
        )
        for side in ("home", "away")
    ]
    return pd.concat(sides).drop_duplicates().reset_index(drop=True)


def _splits(enriched: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side, opp_conf in (("offense", "defense_conference"), ("defense", "offense_conference")):
        with_opp = enriched.assign(opp_conference=enriched[opp_conf].fillna("FCS/other"))
        for split in SPLITS:
            frame = efficiency(with_opp, side, by=("season", split)).rename(columns={split: "split_value"})
            frame["split_value"] = frame["split_value"].astype(str)
            frame.insert(0, "split", split)
            frame.insert(0, "side", side)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _adjusted(enriched: pd.DataFrame, alpha: float) -> pd.DataFrame:
    def renamed(value: str, tag: str) -> pd.DataFrame:
        return opponent_adjust(enriched, value, alpha=alpha).rename(columns={
            "off_raw": f"off_{tag}_raw", "off_adj": f"off_{tag}_adj",
            "def_raw": f"def_{tag}_raw", "def_adj": f"def_{tag}_adj",
        })

    epa = renamed("ppa", "epa")
    sr = renamed("success", "sr").drop(columns=["off_plays", "def_plays"])
    merged = epa.merge(sr, on=["season", "team"])
    return merged[[
        "season", "team", "off_epa_raw", "off_epa_adj", "def_epa_raw", "def_epa_adj",
        "off_sr_raw", "off_sr_adj", "def_sr_raw", "def_sr_adj", "off_plays", "def_plays",
    ]]


def _red_zone(enriched: pd.DataFrame, drives: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side in ("offense", "defense"):
        frame = red_zone(enriched, drives, side)
        frame.insert(0, "side", side)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    con.register("_derived", df)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _derived')
    finally:
        con.unregister("_derived")


def build(
    con: duckdb.DuckDBPyConnection,
    *,
    garbage: GarbageTime = GarbageTime(),
    explosive: Explosive = Explosive(),
    alpha: float = 50.0,
) -> dict[str, int]:
    plays = con.execute(f"SELECT {_PLAY_COLUMNS} FROM plays").df()
    games = con.execute(
        "SELECT id, season, neutral_site, home_team, away_team, home_classification, away_classification FROM games"
    ).df()
    drives = con.execute(
        "SELECT id, season, offense, defense, start_offense_score, end_offense_score FROM drives"
    ).df()
    box = con.execute("SELECT game_id, season, team, category, stat FROM team_game_stats").df()

    enriched = enrich_plays(plays, games, garbage=garbage, explosive=explosive)
    fbs = fbs_teams(games)

    def fbs_only(df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(fbs, on=["season", "team"])

    tables = {
        "plays_enriched": enriched,
        "team_offense": fbs_only(efficiency(enriched, "offense")),
        "team_defense": fbs_only(efficiency(enriched, "defense")),
        "team_splits": fbs_only(_splits(enriched)),
        "team_adjusted": fbs_only(_adjusted(enriched, alpha)),
        "team_havoc": fbs_only(havoc_rate(box, enriched)),
        "team_turnovers": fbs_only(turnover_margin(box)),
        "team_red_zone": fbs_only(_red_zone(enriched, drives)),
    }
    for name, df in tables.items():
        _write(con, name, df)
    return {name: len(df) for name, df in tables.items()}
