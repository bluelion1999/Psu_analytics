"""Game explorer: line score, box score, drive chart, win probability and biggest plays for one game."""

import altair as alt
import streamlit as st

from psu.config import TEAM
from psu.dashboard.common import db_path
from psu.dashboard.ui import load_or_note, load_or_stop, season_picker
from psu.simulate import MissingModel, load_sigma

season = season_picker()
st.title("Game explorer")

options = load_or_stop("games.game_options", season, TEAM)
if options.empty:
    st.info(f"No completed {TEAM} games in {season} yet.")
    st.stop()
labels = dict(zip(options["game_id"].tolist(), options["label"].tolist(), strict=True))
game_id = st.selectbox("Game", list(labels), format_func=labels.get)

scores = load_or_stop("games.line_scores", game_id)
st.dataframe(scores, hide_index=True)
home = scores["team"].iloc[-1]

box_col, drive_col = st.columns(2)
with box_col:
    st.subheader("Box score")
    box = load_or_note("games.box_score", game_id)
    if box is not None:
        st.dataframe(box, hide_index=True)
with drive_col:
    st.subheader("Drives")
    drives = load_or_note("games.drive_chart", game_id)
    if drives is not None and not drives.empty:
        chart = (
            alt.Chart(drives)
            .mark_bar()
            .encode(
                x=alt.X("start_pos:Q", title="Yards from own goal", scale=alt.Scale(domain=[0, 100])),
                x2="end_pos:Q",
                y=alt.Y("drive_number:O", title="Drive"),
                color=alt.Color("offense:N", title=None),
                tooltip=["drive_number:O", "offense:N", "quarter:Q", "result:N", "plays:Q", "yards:Q", "time:N"],
            )
        )
        st.altair_chart(chart)

st.subheader("Win probability")
try:
    sigma = load_sigma(db_path().parent)
except MissingModel as e:
    st.info(str(e))
else:
    wp = load_or_note("games.win_probability", game_id, sigma)
    if wp is not None:
        chart = (
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
                    alt.Tooltip("minute:Q", format=".1f"),
                    "home_margin:Q",
                    alt.Tooltip("home_wp:Q", format=".0%"),
                ],
            )
        )
        st.altair_chart(chart)
        st.caption("Estimated from the score, time left and the model's pregame line (not a play-level model).")

st.subheader("Biggest plays")
plays = load_or_note("games.top_plays", game_id)
if plays is not None:
    st.dataframe(plays, hide_index=True, column_config={"epa": st.column_config.NumberColumn("EPA", format="%+.2f")})
