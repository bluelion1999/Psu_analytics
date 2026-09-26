"""Season overview: record, schedule with results and pregame lines, and key metrics against conference and FBS."""

import pandas as pd
import streamlit as st

from psu.config import TEAM
from psu.dashboard.common import MissingData
from psu.dashboard.ui import (
    chance_column,
    fmt,
    load,
    load_or_note,
    load_or_stop,
    page_header,
    result_color,
    season_picker,
    stat_tiles,
)

season = season_picker()
page_header(f"{TEAM} {season}", "Results, pregame lines and how the team stacks up against its conference and FBS.")

schedule = load_or_stop("overview.schedule", season, TEAM)
if schedule.empty:
    st.info(f"No {TEAM} games found for {season}.")
    st.stop()

rec = load_or_stop("overview.record", season, TEAM)
stat_tiles(
    [
        ("Record", f"{rec['wins']}-{rec['losses']}"),
        ("Conference", f"{rec['conf_wins']}-{rec['conf_losses']}"),
        ("Games left", str(rec["remaining"])),
    ]
)

st.subheader("Schedule")
score = schedule["team_points"].astype("Int64").astype(str) + "-" + schedule["opp_points"].astype("Int64").astype(str)
table = pd.DataFrame(
    {
        "Week": schedule["week"],
        "Date": pd.to_datetime(schedule["start_date"]).dt.date,
        "Opponent": schedule["opponent"],
        "Venue": schedule["venue"].str.title(),
        "Result": (schedule["result"].fillna("") + " " + score).where(schedule["completed"], ""),
        "Vegas": fmt(schedule["vegas_margin"], "+.1f"),
        "Model": fmt(schedule["model_margin"], "+.1f"),
        "Win prob": schedule["win_prob"] * 100,
    }
)
st.dataframe(
    table.style.map(result_color, subset=["Result"]),
    hide_index=True,
    width="stretch",
    column_config={"Win prob": chance_column("Win prob")},
)
st.caption(
    f"Lines are from {TEAM}'s side: +7 means {TEAM} is favoured by 7. "
    "For games already played the model line is in-sample, so it flatters the model."
)

st.subheader("Key metrics")
metrics = load_or_note("overview.metric_comparison", season, TEAM)
if metrics is not None:
    try:
        conference = load("common.team_conference", season, TEAM) or "Conference"
    except MissingData:
        conference = "Conference"
    shown = metrics.drop(columns="higher_is_better").rename(
        columns={
            "metric": "Metric",
            "value": TEAM,
            "conference_avg": f"{conference} avg",
            "national_avg": "FBS avg",
        }
    )
    for column in shown.columns:
        if column != "Metric":
            shown[column] = fmt(shown[column], ".3f")
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("For defense EPA/play and success rate, lower is better. Blank means no plays yet this season.")
