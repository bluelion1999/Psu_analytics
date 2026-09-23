"""Penn State football dashboard. Run: streamlit run app/streamlit_app.py (reads data/psu.duckdb; no API calls)."""
import streamlit as st

st.set_page_config(page_title="Penn State analytics", layout="wide")
st.navigation([
    st.Page("views/overview.py", title="Season overview", default=True),
    st.Page("views/trends.py", title="Efficiency trends"),
    st.Page("views/games.py", title="Game explorer"),
    st.Page("views/players.py", title="Players"),
    st.Page("views/predictions.py", title="Predictions"),
]).run()
