"""Offline unit tests for marine_engine.providers.sediment.bgs.

Uses a small synthetic square polygon for AOI geometry encoding and canned
ArcGIS REST-shaped JSON payloads (GeoJSON query responses and `identify`
responses) -- never the real PL854 AOI -- and never touches the network.
"""

from urllib.parse import parse_qs

import pytest
import requests
from shapely.geometry import Polygon

from marine_engine.providers.sediment import bgs as sediment_bgs


class FakeJsonResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _square_polygon() -> Polygon:
    # Standard shapely/GeoJSON counter-clockwise winding.
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])


def _shoelace_signed_area(coords: list[list[float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):  # noqa: B905 -- deliberately offset
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _feature(feature_id: int) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [1.0, 53.0]},
        "properties": {"id": feature_id},
    }


# --- _esri_polygon_geometry_json ---------------------------------------------


def test_esri_polygon_geometry_json_orients_exterior_clockwise():
    geometry_json = sediment_bgs._esri_polygon_geometry_json(_square_polygon())

    assert geometry_json["spatialReference"] == {"wkid": 4326}
    assert len(geometry_json["rings"]) == 1

    ring = geometry_json["rings"][0]
    assert _shoelace_signed_area(ring) < 0  # Esri convention: exterior clockwise


# --- _query_layer_paginated (via the public fetch_* wrappers) ----------------
#
# Uses POST (form-encoded body), never GET: the real PL854 AOI polygon has
# over 1500 vertices, and a GET query string that large gets the connection
# reset by the live service (confirmed against the real AOI) -- so these
# fakes patch `post_with_retries` and parse `resultOffset` back out of the
# form-encoded body, not out of a `params` dict.


def _offset_from_form_body(data: bytes) -> int:
    parsed = parse_qs(data.decode("utf-8"))
    return int(parsed["resultOffset"][0])


def test_query_layer_paginated_handles_more_than_page_size_records(monkeypatch):
    call_count = {"n": 0}

    def fake_post(url, data, *, headers=None):
        call_count["n"] += 1
        offset = _offset_from_form_body(data)
        if offset == 0:
            page = [_feature(1), _feature(2)]
        elif offset == 2:
            page = [_feature(3), _feature(4)]
        elif offset == 4:
            page = [_feature(5)]
        else:  # pragma: no cover -- would indicate a pagination bug
            raise AssertionError(f"unexpected resultOffset {offset}")
        return FakeJsonResponse({"type": "FeatureCollection", "features": page})

    monkeypatch.setattr(sediment_bgs, "post_with_retries", fake_post)

    features = sediment_bgs.fetch_psa_observations(_square_polygon(), page_size=2)

    assert len(features) == 5
    assert call_count["n"] == 3


def test_query_layer_paginated_stops_on_short_page_even_if_exceeded_transfer_limit_true(
    monkeypatch,
):
    call_count = {"n": 0}

    def fake_post(url, data, *, headers=None):
        call_count["n"] += 1
        page = [_feature(1), _feature(2), _feature(3)]
        return FakeJsonResponse(
            {
                "type": "FeatureCollection",
                "exceededTransferLimit": True,  # observed live even on a short page
                "features": page,
            }
        )

    monkeypatch.setattr(sediment_bgs, "post_with_retries", fake_post)

    features = sediment_bgs.fetch_seabed_sediments_250k(_square_polygon(), page_size=10)

    assert len(features) == 3
    assert call_count["n"] == 1


def test_query_layer_paginated_raises_on_in_body_error(monkeypatch):
    def fake_post(url, data, *, headers=None):
        return FakeJsonResponse({"error": {"code": 400, "message": "boom"}})

    monkeypatch.setattr(sediment_bgs, "post_with_retries", fake_post)

    with pytest.raises(sediment_bgs.BgsSedimentUnavailableError):
        sediment_bgs.fetch_seabed_sediments_250k(_square_polygon())


def test_query_layer_paginated_raises_on_network_failure(monkeypatch):
    def fake_post(url, data, *, headers=None):
        raise requests.ConnectionError("simulated")

    monkeypatch.setattr(sediment_bgs, "post_with_retries", fake_post)

    with pytest.raises(sediment_bgs.BgsSedimentUnavailableError) as exc_info:
        sediment_bgs.fetch_psa_observations(_square_polygon())

    assert isinstance(exc_info.value.__cause__, requests.ConnectionError)


# --- fetch_predictive_percentage_at_point ------------------------------------


def test_fetch_predictive_percentage_at_point_parses_pixel_value(monkeypatch):
    def fake_get(url, params=None):
        return FakeJsonResponse(
            {"results": [{"layerId": 9, "attributes": {"Stretch.Pixel Value": "2"}}]}
        )

    monkeypatch.setattr(sediment_bgs, "get_with_retries", fake_get)

    value = sediment_bgs.fetch_predictive_percentage_at_point(1.8, 53.4, 9)

    assert value == 2.0
    assert isinstance(value, float)


def test_fetch_predictive_percentage_at_point_returns_none_when_no_results(monkeypatch):
    def fake_get(url, params=None):
        return FakeJsonResponse({"results": []})

    monkeypatch.setattr(sediment_bgs, "get_with_retries", fake_get)

    value = sediment_bgs.fetch_predictive_percentage_at_point(1.8, 53.4, 9)

    assert value is None


def test_fetch_predictive_percentage_at_point_raises_on_error(monkeypatch):
    def fake_get(url, params=None):
        return FakeJsonResponse({"error": {"code": 400, "message": "boom"}})

    monkeypatch.setattr(sediment_bgs, "get_with_retries", fake_get)

    with pytest.raises(sediment_bgs.BgsSedimentUnavailableError):
        sediment_bgs.fetch_predictive_percentage_at_point(1.8, 53.4, 9)
