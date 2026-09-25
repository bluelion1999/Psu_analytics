"""Players: season passing, rushing and receiving tables from box scores."""

import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load_or_stop, page_header, season_picker

KINDS = {"passing": ("Minimum attempts", 20), "rushing": ("Minimum carries", 10), "receiving": ("Minimum catches", 5)}

season = season_picker()
page_header("Players", "Season passing, rushing and receiving from box scores, for any FBS team.")

teams = load_or_stop("players.teams", season)
if not teams:
    st.info(f"No player box scores for {season} yet.")
    st.stop()
team = st.sidebar.selectbox("Team", teams, index=teams.index(TEAM) if TEAM in teams else 0)
kind = st.radio("Stat", list(KINDS), horizontal=True, format_func=str.title)
label, default = KINDS[kind]
minimum = st.slider(label, 0, 200, default)

table = load_or_stop("players.player_table", season, team, kind, minimum)
if table.empty:
    st.info("No players meet the minimum.")
else:
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "player": st.column_config.TextColumn("Player", width="medium"),
            "games": "G",
            "comp": "Comp",
            "att": "Att",
            "carries": "Car",
            "receptions": "Rec",
            "yards": "Yds",
            "td": "TD",
            "int": "Int",
            "long": "Long",
            "comp_pct": st.column_config.NumberColumn("Comp %", format="percent"),
            "yds_per_att": st.column_config.NumberColumn("Yds/att", format="%.1f"),
            "yds_per_carry": st.column_config.NumberColumn("Yds/car", format="%.1f"),
            "yds_per_rec": st.column_config.NumberColumn("Yds/rec", format="%.1f"),
        },
    )
st.caption("From CFBD box scores. Plays don't name players, so per-player EPA isn't available.")
