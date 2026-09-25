"""Streamlit helpers shared by the pages: cached reads, the season picker, and friendly missing-data messages."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from psu.dashboard.common import MissingData, db_path, read


@st.cache_data(ttl=600, show_spinner=False)
def _load(db: str, name: str, *args: Any) -> Any:
    module, func = name.split(".")
    return read(getattr(importlib.import_module(f"psu.dashboard.{module}"), func), *args, path=Path(db))


def load(name: str, *args: Any) -> Any:
    """Run psu.dashboard.<module>.<function>(con, *args) on a fresh read-only connection, cached for 10 minutes."""
    return _load(str(db_path()), name, *args)


def load_or_stop(name: str, *args: Any) -> Any:
    try:
        return load(name, *args)
    except MissingData as e:
        st.info(str(e))
        st.stop()


def load_or_note(name: str, *args: Any) -> Any:
    try:
        return load(name, *args)
    except MissingData as e:
        st.info(str(e))
        return None


def fmt(series: pd.Series, spec: str) -> pd.Series:
    """Format a numeric series, blanking missing values instead of showing "None"."""
    return series.map(lambda v: "" if pd.isna(v) else format(v, spec))


def season_picker() -> int:
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
    options = load_or_stop("common.seasons")
    if not options:
        st.info("No games loaded yet; run `psu ingest` first")
        st.stop()
    saved = st.session_state.get("_season")
    if saved not in options:
        saved = options[0]
    season = st.sidebar.selectbox("Season", options, index=options.index(saved))
    st.session_state["_season"] = season
    return season
