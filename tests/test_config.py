from datetime import date

import pytest

from psu.config import current_season, parse_seasons


def test_current_season_rolls_over_in_march():
    assert current_season(date(2026, 9, 22)) == 2026
    assert current_season(date(2027, 1, 10)) == 2026  # bowl season belongs to the prior fall
    assert current_season(date(2027, 3, 1)) == 2027


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("2024", [2024]),
        ("2022-2024", [2022, 2023, 2024]),
        ("2022, 2024-2025", [2022, 2024, 2025]),
        ("2025,2024,2025", [2024, 2025]),
    ],
)
def test_parse_seasons(spec, expected):
    assert parse_seasons(spec, first=2022, last=2026) == expected


@pytest.mark.parametrize("spec", ["", "2021", "2027", "2025-2023", "twenty", "2022-"])
def test_parse_seasons_rejects(spec):
    with pytest.raises(ValueError):
        parse_seasons(spec, first=2022, last=2026)
