from types import SimpleNamespace
from unittest.mock import patch

import cfbd

from psu.cfbd_api import ENDPOINTS, make_cfbd_fetch


def test_every_endpoint_maps_to_a_real_cfbd_method():
    for endpoint, (cls_name, method) in ENDPOINTS.items():
        assert hasattr(getattr(cfbd, cls_name), f"{method}_with_http_info"), endpoint


def test_fetch_returns_raw_json_and_converts_enums():
    fake = SimpleNamespace(raw_data=b'[{"id": "1", "playType": "Rush"}]')
    with patch.object(cfbd.PlaysApi, "get_plays_with_http_info", return_value=fake) as method:
        fetch = make_cfbd_fetch("test-key")
        data = fetch(
            "plays",
            {"year": 2024, "week": 1, "season_type": "regular", "classification": "fbs", "team": None},
        )
    assert data == [{"id": "1", "playType": "Rush"}]
    kwargs = method.call_args.kwargs
    assert kwargs["year"] == 2024 and kwargs["week"] == 1
    assert kwargs["season_type"] == cfbd.SeasonType("regular")
    assert kwargs["classification"] == cfbd.DivisionClassification("fbs")
    assert kwargs["_preload_content"] is False
    assert "team" not in kwargs
