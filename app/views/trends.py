"""Efficiency trends: EPA/play and success rate by season, and game by game within a season."""

import altair as alt
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load_or_note, page_header, season_picker, show_chart

season = season_picker()
page_header("Efficiency trends", f"{TEAM}'s EPA per play and success rate, season by season and game by game.")

st.subheader("By season")
trends = load_or_note("trends.season_trends", TEAM)
if trends is not None:
    if trends.empty:
        st.info("No season metrics yet; run `psu build` first.")
    else:
        show_chart(
            alt.Chart(trends)
            .mark_line(point=True)
            .encode(
                x=alt.X("season:O", title="Season"),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("group:N", title=None, sort=[TEAM]),  # the team gets the primary blue
                tooltip=[
                    alt.Tooltip("season:O", title="Season"),
                    alt.Tooltip("group:N", title="Group"),
                    alt.Tooltip("value:Q", title="Value", format=".3f"),
                ],
            )
            .properties(width=340, height=220)
            .facet(facet=alt.Facet("metric:N", title=None), columns=2)
            .resolve_scale(y="independent")
        )

st.subheader(f"Game by game, {season}")
weekly = load_or_note("trends.weekly_trends", season, TEAM)
if weekly is not None:
    if weekly.empty:
        st.info(f"No {TEAM} plays for {season} yet.")
    else:
        order = list(dict.fromkeys(weekly["game"]))
        columns = st.columns(2)
        for column, (field, title) in zip(
            columns, (("epa_per_play", "EPA/play"), ("success_rate", "Success rate")), strict=True
        ):
            with column:
                st.markdown(f"**{title}**")
                show_chart(
                    alt.Chart(weekly)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("game:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=90)),
                        y=alt.Y(f"{field}:Q", title=None),
                        color=alt.Color("side:N", title=None, sort=["offense", "defense"]),
                        tooltip=[
                            alt.Tooltip("game:N", title="Game"),
                            alt.Tooltip("side:N", title="Side"),
                            alt.Tooltip(f"{field}:Q", title=title, format=".3f"),
                            alt.Tooltip("plays:Q", title="Plays"),
                        ],
                    )
                    .properties(height=260)
                )
        st.caption("Garbage time is excluded. For the defense, lower is better.")
