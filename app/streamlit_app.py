"""Penn State football dashboard. Run: streamlit run app/streamlit_app.py (reads data/psu.duckdb; no API calls)."""

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="Penn State analytics", page_icon=str(ASSETS / "icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))
st.navigation(
    [
        st.Page("views/overview.py", title="Season overview", icon=":material/dashboard:", default=True),
        st.Page("views/trends.py", title="Efficiency trends", icon=":material/show_chart:"),
        st.Page("views/games.py", title="Game explorer", icon=":material/sports_score:"),
        st.Page("views/players.py", title="Players", icon=":material/groups:"),
        st.Page("views/predictions.py", title="Predictions", icon=":material/insights:"),
    ]
).run()
