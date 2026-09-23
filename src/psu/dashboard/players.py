"""Season player tables (passing, rushing, receiving) from CFBD box scores."""

from __future__ import annotations

import duckdb
import pandas as pd

from psu.dashboard.common import require

COLUMNS = {
    "passing": ["player", "games", "comp", "att", "comp_pct", "yards", "yds_per_att", "td", "int"],
    "rushing": ["player", "games", "carries", "yards", "yds_per_carry", "td", "long"],
    "receiving": ["player", "games", "receptions", "yards", "yds_per_rec", "td", "long"],
}
VOLUME = {"passing": "att", "rushing": "carries", "receiving": "receptions"}
_VOLUME_STAT = {"rushing": "CAR", "receiving": "REC"}
_RATE = {"passing": "yds_per_att", "rushing": "yds_per_carry", "receiving": "yds_per_rec"}


def teams(con: duckdb.DuckDBPyConnection, season: int) -> list[str]:
    require(con, "player_game_stats")
    return (
        con.execute("SELECT DISTINCT team FROM player_game_stats WHERE season = ? ORDER BY team", [season])
        .df()["team"]
        .tolist()
    )


def _num(wide: pd.DataFrame, stat_type: str) -> pd.Series:
    if stat_type not in wide:
        return pd.Series(0.0, index=wide.index)
    return pd.to_numeric(wide[stat_type], errors="coerce").fillna(0.0)


def _per_game(kind: str, wide: pd.DataFrame) -> pd.DataFrame:
    if kind == "passing":
        if "C/ATT" in wide:
            parts = wide["C/ATT"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)").astype(float).fillna(0.0)
        else:
            parts = pd.DataFrame({0: 0.0, 1: 0.0}, index=wide.index)
        return pd.DataFrame(
            {
                "comp": parts[0],
                "att": parts[1],
                "yards": _num(wide, "YDS"),
                "td": _num(wide, "TD"),
                "int": _num(wide, "INT"),
            }
        )
    return pd.DataFrame(
        {
            VOLUME[kind]: _num(wide, _VOLUME_STAT[kind]),
            "yards": _num(wide, "YDS"),
            "td": _num(wide, "TD"),
            "long": _num(wide, "LONG"),
        }
    )


def player_table(con: duckdb.DuckDBPyConnection, season: int, team: str, kind: str, minimum: int = 0) -> pd.DataFrame:
    if kind not in COLUMNS:
        raise ValueError(f"kind must be one of {sorted(COLUMNS)}, not {kind!r}")
    require(con, "player_game_stats")
    raw = con.execute(
        "SELECT game_id, athlete_id, athlete_name, stat_type, stat FROM player_game_stats "
        "WHERE season = ? AND team = ? AND category = ?",
        [season, team, kind],
    ).df()
    raw = raw[(raw["athlete_name"].fillna("").str.strip().str.lower() != "team") & raw["athlete_id"].notna()]
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS[kind])
    wide = raw.pivot_table(
        index=["game_id", "athlete_id"], columns="stat_type", values="stat", aggfunc="first"
    ).reset_index()
    names = raw.groupby("athlete_id")["athlete_name"].first()
    per_game = _per_game(kind, wide)
    named = {c: (c, "max" if c == "long" else "sum") for c in per_game.columns}
    named["games"] = ("yards", "size")
    out = (
        per_game.assign(player=wide["athlete_id"].map(names).to_numpy(), athlete_id=wide["athlete_id"].to_numpy())
        .groupby(["athlete_id", "player"], as_index=False)
        .agg(**named)
    )
    volume = out[VOLUME[kind]]
    out[_RATE[kind]] = out["yards"] / volume.where(volume > 0)
    if kind == "passing":
        out["comp_pct"] = out["comp"] / volume.where(volume > 0)
    out = out[volume >= minimum]
    for column in COLUMNS[kind]:
        if column not in ("player", "comp_pct", _RATE[kind]):
            out[column] = out[column].astype(int)
    return out.sort_values(["yards", "player"], ascending=[False, True])[COLUMNS[kind]].reset_index(drop=True)
