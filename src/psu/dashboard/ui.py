"""Streamlit helpers shared by the pages: cached reads, the season picker, friendly missing-data messages, and the
Penn State look (chart theme, page headers, stat tiles)."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from psu.dashboard.common import MissingData, db_path, read

PSU_NAVY = "#041E42"
CHART_BLUE = "#4A7DD4"  # brighter Penn State blue: navy #1E407C disappears on the dark background
PALETTE = [CHART_BLUE, "#96BEE6", "#E0A526", "#8A94A6", "#C8553D", "#4C9F70"]
WIN_COLOR = "#2E8B57"
LOSS_COLOR = "#C8553D"
_MUTED = "#8A94A6"  # readable on both the light and the dark background
_GRID = "rgba(138, 148, 166, 0.2)"


@alt.theme.register("psu", enable=True)
def psu_chart_theme() -> alt.theme.ThemeConfig:
    """One chart style for every page: Penn State palette, quiet axes, horizontal labels, transparent background."""
    axis = {
        "labelColor": _MUTED,
        "titleColor": _MUTED,
        "gridColor": _GRID,
        "tickColor": _GRID,
        "domain": False,
        "labelFontSize": 11,
        "titleFontSize": 12,
        "titleFontWeight": 500,
    }
    return {
        "config": {
            "background": "transparent",
            "font": "'Source Sans Pro', 'Segoe UI', sans-serif",
            "view": {"stroke": None},
            "axis": axis,
            "axisX": {"labelAngle": 0, "grid": False},
            "legend": {"labelColor": _MUTED, "titleColor": _MUTED, "orient": "top", "labelFontSize": 12},
            "header": {"labelColor": _MUTED, "titleColor": _MUTED, "labelFontSize": 13, "labelFontWeight": 600},
            "range": {"category": PALETTE},
            "mark": {"color": CHART_BLUE},
            "line": {"strokeWidth": 2.5},
            "point": {"filled": True, "size": 60},
            "bar": {"color": CHART_BLUE, "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
        }
    }


def show_chart(chart: alt.TopLevelMixin) -> None:
    """Render with the Penn State theme (Streamlit's own chart theme would override it)."""
    faceted = isinstance(chart, alt.FacetChart)
    st.altair_chart(chart, theme=None, width="content" if faceted else "stretch")


def page_header(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)


def stat_tiles(items: Sequence[tuple[str, str]]) -> None:
    """A row of bordered KPI cards, one per (label, value)."""
    for column, (label, value) in zip(st.columns(len(items)), items, strict=True):
        column.container(border=True).metric(label, value)


def chance_column(label: str) -> st.column_config.ProgressColumn:
    """A 0-100 bar with a whole-number percent; feed it probabilities multiplied by 100."""
    return st.column_config.ProgressColumn(label, format="%.0f%%", min_value=0, max_value=100)


def result_color(value: str) -> str:
    """Cell style for a schedule result such as "W 34-10" or "L 20-24"."""
    if value.startswith("W"):
        return f"color: {WIN_COLOR}; font-weight: 600"
    if value.startswith("L"):
        return f"color: {LOSS_COLOR}; font-weight: 600"
    return ""


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
