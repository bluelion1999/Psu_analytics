"""Efficiency trends: EPA/play and success rate by season, and game by game within a season."""

import altair as alt
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load_or_note, season_picker

season = season_picker()
st.title("Efficiency trends")

st.subheader("By season")
trends = load_or_note("trends.season_trends", TEAM)
if trends is not None:
    if trends.empty:
        st.info("No season metrics yet; run `psu build` first.")
    else:
        chart = (
            alt.Chart(trends)
            .mark_line(point=True)
            .encode(
                x=alt.X("season:O", title="Season"),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("group:N", title=None),
                tooltip=["season:O", "group:N", alt.Tooltip("value:Q", format=".3f")],
            )
            .properties(width=320, height=220)
            .facet(facet=alt.Facet("metric:N", title=None), columns=2)
            .resolve_scale(y="independent")
        )
        st.altair_chart(chart)

st.subheader(f"Game by game, {season}")
weekly = load_or_note("trends.weekly_trends", season, TEAM)
if weekly is not None:
    if weekly.empty:
        st.info(f"No {TEAM} plays for {season} yet.")
    else:
        order = list(dict.fromkeys(weekly["game"]))
        for column, title in (("epa_per_play", "EPA/play"), ("success_rate", "Success rate")):
            chart = (
                alt.Chart(weekly)
                .mark_line(point=True)
                .encode(
                    x=alt.X("game:N", sort=order, title=None),
                    y=alt.Y(f"{column}:Q", title=title),
                    color=alt.Color("side:N", title=None),
                    tooltip=["game:N", "side:N", alt.Tooltip(f"{column}:Q", format=".3f"), "plays:Q"],
                )
            )
            st.altair_chart(chart)
        st.caption("Garbage time is excluded. For the defense, lower is better.")
