"""Season overview: record, schedule with results and pregame lines, and key metrics against conference and FBS."""
import pandas as pd
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import load, load_or_note, load_or_stop, season_picker

season = season_picker()
st.title(f"{TEAM} {season}")

schedule = load_or_stop("overview.schedule", season, TEAM)
if schedule.empty:
    st.info(f"No {TEAM} games found for {season}.")
    st.stop()

rec = load_or_stop("overview.record", season, TEAM)
record_col, conf_col, left_col = st.columns(3)
record_col.metric("Record", f"{rec['wins']}-{rec['losses']}")
conf_col.metric("Conference", f"{rec['conf_wins']}-{rec['conf_losses']}")
left_col.metric("Games left", rec["remaining"])

st.subheader("Schedule")
score = schedule["team_points"].astype("Int64").astype(str) + "-" + schedule["opp_points"].astype("Int64").astype(str)
table = pd.DataFrame({
    "Week": schedule["week"],
    "Date": pd.to_datetime(schedule["start_date"]).dt.date,
    "Opponent": schedule["opponent"],
    "Venue": schedule["venue"].str.title(),
    "Result": (schedule["result"].fillna("") + " " + score).where(schedule["completed"], ""),
    "Vegas": schedule["vegas_margin"],
    "Model": schedule["model_margin"],
    "Win prob": schedule["win_prob"] * 100,
})
st.dataframe(table, hide_index=True, column_config={
    "Vegas": st.column_config.NumberColumn(format="%+.1f"),
    "Model": st.column_config.NumberColumn(format="%+.1f"),
    "Win prob": st.column_config.NumberColumn(format="%.0f%%"),
})
st.caption(
    f"Lines are from {TEAM}'s side: +7 means {TEAM} is favoured by 7. "
    "For games already played the model line is in-sample, so it flatters the model."
)

st.subheader("Key metrics")
metrics = load_or_note("overview.metric_comparison", season, TEAM)
if metrics is not None:
    conference = load("common.team_conference", season, TEAM) or "Conference"
    shown = metrics.drop(columns="higher_is_better").rename(columns={
        "metric": "Metric", "value": TEAM, "conference_avg": f"{conference} avg", "national_avg": "FBS avg",
    })
    number = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(shown, hide_index=True, column_config={c: number for c in shown.columns if c != "Metric"})
    st.caption("For defense EPA/play and success rate, lower is better. Blank means no plays yet this season.")
