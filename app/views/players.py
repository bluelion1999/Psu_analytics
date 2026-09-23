"""Players: season passing, rushing and receiving tables from box scores."""
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load_or_stop, season_picker

KINDS = {"passing": ("Minimum attempts", 20), "rushing": ("Minimum carries", 10), "receiving": ("Minimum catches", 5)}

season = season_picker()
st.title("Players")

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
    rate = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(table, hide_index=True, column_config={
        "comp_pct": st.column_config.NumberColumn("Comp %", format="percent"),
        "yds_per_att": rate, "yds_per_carry": rate, "yds_per_rec": rate,
    })
st.caption("From CFBD box scores. Plays don't name players, so per-player EPA isn't available.")
