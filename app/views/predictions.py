"""Predictions: upcoming games, the season simulation and the next FBS slate."""

import altair as alt
import pandas as pd
import streamlit as st

from psu.config import TEAM
from psu.dashboard.ui import fmt, load_or_note, season_picker

season = season_picker()
st.title("Predictions")

st.subheader(f"{TEAM} upcoming games")
games = load_or_note("predictions.upcoming", season, TEAM)
if games is not None:
    if games.empty:
        st.info(f"No upcoming {TEAM} games in {season}.")
    else:
        st.dataframe(
            pd.DataFrame(
                {
                    "Week": games["week"],
                    "Date": pd.to_datetime(games["start_date"]).dt.date,
                    "Opponent": games["opponent"],
                    "Venue": games["venue"].str.title(),
                    "Model": fmt(games["model_margin"], "+.1f"),
                    "Win prob": fmt(games["win_prob"], ".0%"),
                    "Vegas": fmt(games["vegas_margin"], "+.1f"),
                }
            ),
            hide_index=True,
        )

st.subheader("Season simulation")
summary = load_or_note("predictions.sim_summary", TEAM)
if summary is not None and int(summary["season"]) != season:
    st.info(f"The season simulation covers {int(summary['season'])} only; pick it in the sidebar.")
elif summary is not None:
    tiles = st.columns(5)
    tiles[0].metric("Mean wins", f"{summary['mean_wins']:.1f}")
    for tile, (label, key) in zip(
        tiles[1:],
        (
            ("P(10+ wins)", "p_10_plus"),
            ("P(title game)", "p_title_game"),
            ("P(Big Ten champ)", "p_conf_champ"),
            ("P(CFP)", "p_cfp"),
        ),
        strict=True,
    ):
        tile.metric(label, f"{summary[key]:.0%}")
    as_of = pd.Timestamp(summary["as_of"]).date() if pd.notna(summary["as_of"]) else "preseason"
    st.caption(
        f"{int(summary['n_sims']):,} simulated {int(summary['season'])} seasons (tau {summary['tau']:g}), "
        f"results through {as_of}. Re-run `psu simulate` after new games."
    )
    totals = load_or_note("predictions.sim_win_totals", TEAM)
    if totals is not None and not totals.empty:
        st.altair_chart(
            alt.Chart(totals)
            .mark_bar()
            .encode(
                x=alt.X("wins:O", title="Regular-season wins"),
                y=alt.Y("prob:Q", title="Probability", axis=alt.Axis(format="%")),
                tooltip=["wins:O", alt.Tooltip("prob:Q", format=".1%")],
            )
        )
    race = load_or_note("predictions.sim_conference")
    if race is not None:
        st.markdown("**Big Ten title race**")
        percent = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(
            race.assign(p_title_game=race["p_title_game"] * 100, p_conf_champ=race["p_conf_champ"] * 100),
            hide_index=True,
            column_config={
                "mean_conf_wins": st.column_config.NumberColumn("Mean conf wins", format="%.2f"),
                "p_title_game": percent,
                "p_conf_champ": percent,
            },
        )

st.subheader("Odds over time")
history = load_or_note("predictions.sim_history", season, TEAM)
if history is not None:
    if history.empty:
        if summary is not None and int(summary["season"]) == season:
            st.info("No simulation history yet. Run `psu simulate --backfill` to replay earlier weeks.")
        else:
            st.info(f"No simulation history for {season}.")
    else:
        labels = {"p_cfp": "CFP", "p_conf_champ": "Big Ten champ", "p_title_game": "Title game"}
        long = history.melt(id_vars=["as_of_slate"], value_vars=list(labels), var_name="odds", value_name="prob")
        long["odds"] = long["odds"].map(labels)
        st.altair_chart(
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("as_of_slate:O", title="Before week"),
                y=alt.Y("prob:Q", title="Probability", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("odds:N", title=None),
                tooltip=["as_of_slate:O", "odds:N", alt.Tooltip("prob:Q", format=".1%")],
            ),
            use_container_width=True,
        )
        if history["backfilled"].any():
            st.caption(
                "Earlier weeks are replays: team ratings as of each week, but model coefficients fitted with "
                "this season's results in view."
            )

with st.expander("Next week's FBS games"):
    slate = load_or_note("predictions.next_slate", season)
    if slate is not None:
        if slate.empty:
            st.info(f"No upcoming games in {season}.")
        else:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Week": slate["week"],
                        "Date": pd.to_datetime(slate["start_date"]).dt.date,
                        "Away": slate["away_team"],
                        "Home": slate["home_team"],
                        "Neutral": slate["neutral_site"].map({True: "Neutral", False: ""}),
                        "Model (home)": fmt(slate["pred_margin"], "+.1f"),
                        "Win prob (home)": fmt(slate["home_win_prob"], ".0%"),
                        "Vegas (home)": fmt(slate["vegas_margin"], "+.1f"),
                    }
                ),
                hide_index=True,
            )
