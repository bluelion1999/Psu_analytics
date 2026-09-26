import altair as alt
import pandas as pd

from psu.dashboard.ui import CHART_BLUE, LOSS_COLOR, PALETTE, WIN_COLOR, psu_chart_theme, result_color


def test_psu_chart_theme_is_enabled_with_the_penn_state_palette():
    assert alt.theme.active == "psu"
    config = psu_chart_theme()["config"]
    assert config["range"]["category"] == PALETTE and PALETTE[0] == CHART_BLUE
    assert config["axisX"]["labelAngle"] == 0  # no sideways week numbers
    assert config["background"] == "transparent"  # works on the light and the dark theme


def test_charts_pick_up_the_theme():
    spec = alt.Chart(pd.DataFrame({"x": [1], "y": [2]})).mark_line().encode(x="x", y="y").to_dict()
    assert spec["config"]["range"]["category"] == PALETTE


def test_result_color_marks_wins_and_losses():
    assert WIN_COLOR in result_color("W 34-10")
    assert LOSS_COLOR in result_color("L 20-24")
    assert result_color("") == ""
