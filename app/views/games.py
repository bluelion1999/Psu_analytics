"""Game explorer: line score, box score, drive chart, win probability and biggest plays for one game."""

import altair as alt
import pandas as pd
import streamlit as st

from psu.config import TEAM
from psu.dashboard.common import db_path
from psu.dashboard.ui import load, load_or_note, load_or_stop, page_header, season_picker, show_chart
from psu.simulate import MissingModel, load_sigmas

season = season_picker()
page_header("Game explorer", f"Box score, drives, win probability and the biggest plays from any {TEAM} game.")

options = load_or_stop("games.game_options", season, TEAM)
if options.empty:
    st.info(f"No completed {TEAM} games in {season} yet.")
    st.stop()
labels = dict(zip(options["game_id"].tolist(), options["label"].tolist(), strict=True))
game_id = st.selectbox("Game", list(labels), format_func=labels.get)

scores = load_or_stop("games.line_scores", game_id)
st.dataframe(scores, hide_index=True, width="stretch", column_config={"team": "Team"})
home = scores["team"].iloc[-1]

box_col, drive_col = st.columns(2)
with box_col:
    st.subheader("Box score")
    box = load_or_note("games.box_score", game_id)
    if box is not None:
        st.dataframe(box, hide_index=True, width="stretch", column_config={"stat": "Stat"})
with drive_col:
    st.subheader("Drives")
    drives = load_or_note("games.drive_chart", game_id)
    if drives is not None and not drives.empty:
        show_chart(
            alt.Chart(drives)
            .mark_bar(cornerRadius=3, height={"band": 0.7})
            .encode(
                x=alt.X("start_pos:Q", title="Yards from own goal", scale=alt.Scale(domain=[0, 100])),
                x2="end_pos:Q",
                y=alt.Y("drive_number:O", title="Drive"),
                color=alt.Color("offense:N", title=None, sort=[TEAM]),  # the team gets the primary blue
                tooltip=[
                    alt.Tooltip("drive_number:O", title="Drive"),
                    alt.Tooltip("offense:N", title="Offense"),
                    alt.Tooltip("quarter:Q", title="Quarter"),
                    alt.Tooltip("result:N", title="Result"),
                    alt.Tooltip("plays:Q", title="Plays"),
                    alt.Tooltip("yards:Q", title="Yards"),
                    alt.Tooltip("time:N", title="Time"),
                ],
            )
        )

st.subheader("Win probability")
try:
    sigma = load_sigmas(db_path().parent)[load("games.game_phase", game_id)]
except MissingModel as e:
    st.info(str(e))
else:
    wp = load_or_note("games.win_probability", game_id, sigma)
    if wp is not None:
        line = (
            alt.Chart(wp)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X("minute:Q", title="Minute", scale=alt.Scale(domain=[0, 60])),
                y=alt.Y(
                    "home_wp:Q",
                    title=f"{home} win probability",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                tooltip=[
                    alt.Tooltip("minute:Q", title="Minute", format=".1f"),
                    alt.Tooltip("home_margin:Q", title=f"{home} margin"),
                    alt.Tooltip("home_wp:Q", title="Win prob", format=".0%"),
                ],
            )
        )
        even = (
            alt.Chart(pd.DataFrame({"y": [0.5]}))
            .mark_rule(strokeDash=[4, 4], opacity=0.5, color="#8A94A6")
            .encode(y="y:Q")
        )
        show_chart((line + even).properties(height=300))
        st.caption("Estimated from the score, time left and the model's pregame line (not a play-level model).")

st.subheader("Biggest plays")
plays = load_or_note("games.top_plays", game_id)
if plays is not None:
    st.dataframe(
        plays,
        hide_index=True,
        width="stretch",
        column_config={
            "quarter": "Qtr",
            "clock": "Clock",
            "offense": "Offense",
            "down": "Down",
            "distance": "To go",
            "epa": st.column_config.NumberColumn("EPA", format="%+.2f"),
            "text": st.column_config.TextColumn("Play", width="large"),
        },
    )
