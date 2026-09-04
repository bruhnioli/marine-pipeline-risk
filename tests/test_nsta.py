"""Offline unit tests for marine_engine.providers.nsta.

These use small synthetic GeoJSON fixtures that mirror the real NSTA schema
(field names confirmed by inspecting the live ArcGIS layer during MAR-002)
rather than the real ~1300-vertex PL854 geometry. No network access.
"""

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from marine_engine.providers import nsta


def _feature(
    nstapipno: str | None, coordinates: list[tuple[float, float]] | None = None, **overrides
) -> dict:
    coordinates = coordinates or [(1.7, 53.37), (1.85, 53.38), (2.0, 53.39)]
    properties = {
        "FEATURE_ID": "test-feature-id",
        "NSTAPIPNO": nstapipno,
        "PIPE_NAME": "LOGGS PP TO ANGLIA YD GAS LINE",
        "INF_TYPE": "PIPELINE",
        "REP_GROUP": "ITHACA ENERGY",
        "FLUID": "GAS",
        "PIPE_SYS": None,
        "DESCRIPTIO": "LOGGS PP TO ANGLIA YD 12IN GAS LINE",
        "STATUS": "NOT IN USE",
        "DIAMETERMM": 304.8,
        "LENGTH_M": 23489.02,
    }
    properties.update(overrides)
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "LineString", "coordinates": [list(c) for c in coordinates]},
    }


def _feature_collection(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


# --- PL854 selection from a representative schema --------------------------


def test_select_pipeline_records_finds_pl854():
    fc = _feature_collection(_feature("PL854"), _feature("PL999"))

    feature = nsta.select_pipeline_records(fc, "PL854")

    assert feature["properties"]["NSTAPIPNO"] == "PL854"


def test_select_pipeline_records_zero_matches():
    fc = _feature_collection(_feature("PL999"))

    with pytest.raises(nsta.PipelineNotFoundError):
        nsta.select_pipeline_records(fc, "PL854")


def test_select_pipeline_records_empty_collection():
    fc = _feature_collection()

    with pytest.raises(nsta.PipelineNotFoundError):
        nsta.select_pipeline_records(fc, "PL854")


def test_select_pipeline_records_duplicate_matches():
    fc = _feature_collection(_feature("PL854"), _feature("PL854"))

    with pytest.raises(nsta.AmbiguousPipelineError):
        nsta.select_pipeline_records(fc, "PL854")


# --- fetch_pipeline fallback behaviour (network calls monkeypatched) -------


def test_fetch_pipeline_falls_back_to_removed_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    responses = {
        nsta.ACTIVE_PIPELINE_SERVICE_URL: _feature_collection(),  # zero matches
        nsta.REMOVED_PIPELINE_SERVICE_URL: _feature_collection(_feature("PL854")),
    }

    def fake_query_service(service_url: str, pipeline_number: str, timeout: float = 30.0) -> dict:
        return responses[service_url]

    monkeypatch.setattr(nsta, "_query_service", fake_query_service)

    result = nsta.fetch_pipeline("PL854", cache_dir=tmp_path)

    assert result.source_label == "removed"
    assert result.feature["properties"]["NSTAPIPNO"] == "PL854"
    assert (tmp_path / "PL854_active.geojson").exists()
    assert (tmp_path / "PL854_removed.geojson").exists()


def test_fetch_pipeline_raises_when_absent_from_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(nsta, "_query_service", lambda *a, **k: _feature_collection())

    with pytest.raises(nsta.PipelineNotFoundError):
        nsta.fetch_pipeline("PL854", cache_dir=tmp_path)


def test_fetch_pipeline_raises_on_ambiguous_active_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        nsta,
        "_query_service",
        lambda *a, **k: _feature_collection(_feature("PL854"), _feature("PL854")),
    )

    with pytest.raises(nsta.AmbiguousPipelineError):
        nsta.fetch_pipeline("PL854", cache_dir=tmp_path)


# --- geometry validation -----------------------------------------------------


def test_geometry_from_feature_valid_linestring():
    geometry = nsta.geometry_from_feature(_feature("PL854"))

    assert geometry.geom_type == "LineString"


def test_geometry_from_feature_missing_geometry_raises():
    feature = _feature("PL854")
    feature["geometry"] = None

    with pytest.raises(nsta.InvalidGeometryError):
        nsta.geometry_from_feature(feature)


def test_geometry_from_feature_wrong_type_raises():
    feature = _feature("PL854")
    feature["geometry"] = {"type": "Point", "coordinates": [1.7, 53.37]}

    with pytest.raises(nsta.InvalidGeometryError):
        nsta.geometry_from_feature(feature)


def test_geometry_from_feature_merges_contiguous_multilinestring():
    feature = _feature("PL854")
    feature["geometry"] = {
        "type": "MultiLineString",
        "coordinates": [[[1.7, 53.37], [1.85, 53.38]], [[1.85, 53.38], [2.0, 53.39]]],
    }

    geometry = nsta.geometry_from_feature(feature)

    assert geometry.geom_type == "LineString"


# --- CRS transformation and geodetic length calculation ---------------------


def test_to_working_crs_reprojects_to_metric_crs():
    geometry = nsta.geometry_from_feature(_feature("PL854"))

    projected = nsta.to_working_crs(geometry, source_crs="EPSG:4326", working_crs="EPSG:32631")

    # UTM31N easting/northing are on the order of 1e5-1e6 m, not degrees.
    assert projected.bounds[0] > 1000
    assert projected.geom_type == "LineString"


def test_geometry_length_matches_independent_haversine_estimate():
    # Two points ~50 km apart in the PL854 area; independently computed via
    # the haversine formula so this doesn't just check the library against
    # itself.
    lon1, lat1 = 1.65160614, 53.36780220
    lon2, lat2 = 2.00197393, 53.38922923

    feature = _feature("PL854", coordinates=[(lon1, lat1), (lon2, lat2)])
    geometry = nsta.geometry_from_feature(feature)
    projected = nsta.to_working_crs(geometry, working_crs="EPSG:32631")

    geometry_length_m = nsta.compute_geometry_length_m(projected)

    r = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    haversine_m = 2 * r * math.asin(math.sqrt(a))

    assert geometry_length_m == pytest.approx(haversine_m, rel=0.01)


# --- reference validation (informational, never raises) ---------------------


def test_validate_against_reference_matches_expected_pl854():
    checks = nsta.validate_against_reference(
        _feature("PL854")["properties"], geometry_length_m=23_489.0
    )
    by_name = {c.name: c for c in checks}

    assert by_name["pipeline_number"].note == "match"
    assert by_name["diameter_mm"].difference_pct == pytest.approx(0.0, abs=1e-6)
    assert by_name["anglia_loggs_relationship"].note == "both keywords present"


def test_validate_against_reference_surfaces_length_discrepancy_without_raising():
    properties = _feature("PL854")["properties"]
    properties["LENGTH_M"] = 20_000.0  # deliberately off from the ~23.7 km reference

    checks = nsta.validate_against_reference(properties, geometry_length_m=20_000.0)
    by_name = {c.name: c for c in checks}

    assert by_name["source_length_m (LENGTH_M)"].difference_pct < -10


def test_validate_against_reference_handles_missing_source_length():
    properties = _feature("PL854")["properties"]
    properties["LENGTH_M"] = None

    checks = nsta.validate_against_reference(properties, geometry_length_m=23_489.0)
    by_name = {c.name: c for c in checks}

    assert by_name["source_length_m (LENGTH_M)"].note == "not reported by source"


# --- canonical schema creation ----------------------------------------------


def test_build_canonical_gdf_has_required_and_extra_columns():
    feature = _feature("PL854")
    geometry = nsta.to_working_crs(nsta.geometry_from_feature(feature))

    gdf = nsta.build_canonical_gdf(
        feature=feature,
        geometry_working_crs=geometry,
        working_crs="EPSG:32631",
        source_crs="EPSG:4326",
        source_title=nsta.ACTIVE_SOURCE_TITLE,
        geometry_length_m=23_489.0,
        retrieved_at=datetime.now(UTC),
    )

    required = {
        "pipeline_id",
        "source",
        "source_feature_id",
        "source_crs",
        "working_crs",
        "source_length_m",
        "geometry_length_m",
        "diameter_mm",
        "status",
        "geometry",
    }
    assert required.issubset(set(gdf.columns))
    assert len(gdf) == 1
    assert gdf.crs.to_string() == "EPSG:32631"
    assert gdf.iloc[0]["pipeline_id"] == "PL854"


def test_write_canonical_gpkg_roundtrip(tmp_path: Path):
    feature = _feature("PL854")
    geometry = nsta.to_working_crs(nsta.geometry_from_feature(feature))
    gdf = nsta.build_canonical_gdf(
        feature=feature,
        geometry_working_crs=geometry,
        working_crs="EPSG:32631",
        source_crs="EPSG:4326",
        source_title=nsta.ACTIVE_SOURCE_TITLE,
        geometry_length_m=23_489.0,
        retrieved_at=datetime.now(UTC),
    )

    output_path = tmp_path / "pl854" / "pipeline.gpkg"
    nsta.write_canonical_gpkg(gdf, output_path)

    assert output_path.exists()

    import geopandas as gpd

    roundtrip = gpd.read_file(output_path, layer="pipeline")
    assert len(roundtrip) == 1
    assert roundtrip.iloc[0]["pipeline_id"] == "PL854"
    assert roundtrip.crs.to_string() == "EPSG:32631"
