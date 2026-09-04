"""Offline unit tests for marine_engine.preprocessing.chainage.

Uses small synthetic line geometries with simple round-number coordinates
(chainage math doesn't care about real-world location) -- never the real
~23.5 km PL854 route -- and never touches the network.
"""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from marine_engine.preprocessing import chainage


def _write_pipeline_gpkg(
    path: Path,
    pipeline_id: str = "PL854",
    crs: str = "EPSG:32631",
    geometry=None,
    extra_attrs: dict | None = None,
) -> LineString:
    geometry = geometry or LineString([(500000.0, 5900000.0), (500108.0, 5900000.0)])
    record = {"pipeline_id": pipeline_id, "source": "test", "status": "ACTIVE"}
    if extra_attrs:
        record.update(extra_attrs)
    gdf = gpd.GeoDataFrame([record], geometry=[geometry], crs=crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG", layer="pipeline")
    return geometry


def _write_aoi_gpkg(path: Path, geometry, crs: str = "EPSG:32631") -> None:
    gdf = gpd.GeoDataFrame([{"study_id": "PL854"}], geometry=[geometry], crs=crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG", layer="study_aoi")


# --- compute_chainage_stations (pure) ---------------------------------------


def test_exact_interval_length_no_duplicate_terminal():
    route = LineString([(0, 0), (100, 0)])  # exactly 100 m, exactly 4*25
    result = chainage.compute_chainage_stations(route, 25)

    chainages = [s.chainage_m for s in result.stations]
    assert chainages == [0, 25, 50, 75, 100]
    assert result.regular_station_count == 5
    assert result.terminal_residual_m == 0.0
    assert sum(s.is_terminal for s in result.stations) == 1
    assert result.stations[-1].is_terminal is True


def test_non_exact_interval_length_appends_terminal():
    route = LineString([(0, 0), (108, 0)])
    result = chainage.compute_chainage_stations(route, 25)

    chainages = [s.chainage_m for s in result.stations]
    assert chainages == [0, 25, 50, 75, 100, 108]
    assert result.regular_station_count == 5
    assert result.terminal_residual_m == pytest.approx(8.0)


def test_configurable_interval_changes_station_count():
    route = LineString([(0, 0), (100, 0)])
    result_10 = chainage.compute_chainage_stations(route, 10)
    result_20 = chainage.compute_chainage_stations(route, 20)

    assert len(result_10.stations) == 11  # 0,10,...,100
    assert len(result_20.stations) == 6  # 0,20,...,100


def test_first_station_is_zero_and_last_equals_route_length():
    route = LineString([(0, 0), (108, 0)])
    result = chainage.compute_chainage_stations(route, 25)

    assert result.stations[0].chainage_m == 0
    assert result.stations[-1].chainage_m == pytest.approx(route.length)


def test_stations_are_monotonic_with_unique_index_and_chainage():
    route = LineString([(0, 0), (108, 0)])
    result = chainage.compute_chainage_stations(route, 25)

    chainages = [s.chainage_m for s in result.stations]
    assert chainages == sorted(chainages)
    assert len(set(chainages)) == len(chainages)

    indices = [s.station_index for s in result.stations]
    assert indices == list(range(len(indices)))


def test_rejects_nonpositive_interval():
    route = LineString([(0, 0), (100, 0)])
    with pytest.raises(ValueError):
        chainage.compute_chainage_stations(route, 0)
    with pytest.raises(ValueError):
        chainage.compute_chainage_stations(route, -5)


def test_uses_geometry_interpolation_not_vertex_spacing():
    # An intermediate vertex NOT on a 25 m boundary: if station generation
    # were vertex-based rather than true linear referencing, chainage 25
    # would not land exactly at (25, 0).
    route = LineString([(0, 0), (13, 0), (100, 0)])
    result = chainage.compute_chainage_stations(route, 25)

    station_25 = next(s for s in result.stations if s.chainage_m == 25)
    assert station_25.point.x == pytest.approx(25.0)
    assert station_25.point.y == pytest.approx(0.0)


def test_deterministic_repeated_computation():
    route = LineString([(0, 0), (108, 0)])
    first = chainage.compute_chainage_stations(route, 25)
    second = chainage.compute_chainage_stations(route, 25)

    assert [s.chainage_m for s in first.stations] == [s.chainage_m for s in second.stations]
    assert [(s.point.x, s.point.y) for s in first.stations] == [
        (s.point.x, s.point.y) for s in second.stations
    ]


# --- project_point_to_route (pure) -------------------------------------------


def test_project_point_to_route_point_on_route():
    route = LineString([(0, 0), (100, 0)])
    projection = chainage.project_point_to_route(route, Point(40, 0))

    assert projection.chainage_m == pytest.approx(40.0)
    assert projection.distance_m == pytest.approx(0.0, abs=1e-9)
    assert projection.nearest_point.x == pytest.approx(40.0)
    assert projection.nearest_point.y == pytest.approx(0.0)
    assert projection.fraction_along_route == pytest.approx(0.4)


def test_project_point_to_route_point_off_route():
    route = LineString([(0, 0), (100, 0)])
    projection = chainage.project_point_to_route(route, Point(30, 40))  # 40 m abeam chainage 30

    assert projection.chainage_m == pytest.approx(30.0)
    assert projection.distance_m == pytest.approx(40.0)
    assert projection.nearest_point.x == pytest.approx(30.0)
    assert projection.nearest_point.y == pytest.approx(0.0)


def test_project_point_to_route_beyond_route_end_clamps_to_terminus():
    route = LineString([(0, 0), (100, 0)])
    projection = chainage.project_point_to_route(route, Point(150, 0))

    assert projection.chainage_m == pytest.approx(100.0)
    assert projection.distance_m == pytest.approx(50.0)


def test_project_point_to_route_never_reports_distance_zero_from_offset():
    """`distance_m` must always stay visible -- never collapsed away by a caller."""

    route = LineString([(0, 0), (200, 0)])
    projection = chainage.project_point_to_route(route, Point(100, 5))

    assert projection.distance_m == pytest.approx(5.0)
    assert projection.chainage_m == pytest.approx(100.0)


# --- KP label formatting ------------------------------------------------------


@pytest.mark.parametrize(
    ("chainage_m", "expected"),
    [
        (0, "KP 0+000"),
        (25, "KP 0+025"),
        (1250, "KP 1+250"),
        (23475, "KP 23+475"),
        (23480.669373619585, "KP 23+480.67"),
    ],
)
def test_format_kp_label(chainage_m, expected):
    assert chainage.format_kp_label(chainage_m) == expected


# --- route resolution (pure) --------------------------------------------------


def test_resolve_continuous_route_single_linestring():
    line = LineString([(0, 0), (100, 0)])
    result = chainage._resolve_continuous_route([line], "PL854")
    assert result.geom_type == "LineString"


def test_resolve_continuous_route_merges_contiguous_parts():
    parts = [LineString([(0, 0), (50, 0)]), LineString([(50, 0), (100, 0)])]
    result = chainage._resolve_continuous_route(parts, "PL854")
    assert result.geom_type == "LineString"
    assert result.length == pytest.approx(100.0)


def test_resolve_continuous_route_rejects_disconnected_parts():
    parts = [LineString([(0, 0), (50, 0)]), LineString([(60, 0), (100, 0)])]  # 10 m gap
    with pytest.raises(chainage.InvalidPipelineRouteError):
        chainage._resolve_continuous_route(parts, "PL854")


def test_resolve_continuous_route_rejects_missing_geometry():
    with pytest.raises(chainage.InvalidPipelineRouteError):
        chainage._resolve_continuous_route([None], "PL854")


def test_resolve_continuous_route_rejects_non_linear():
    with pytest.raises(chainage.InvalidPipelineRouteError):
        chainage._resolve_continuous_route([Point(0, 0)], "PL854")


# --- chainage direction / origin provenance -----------------------------------


def test_determine_chainage_origin_falls_back_without_from_to_metadata():
    decision = chainage.determine_chainage_origin({"pipe_name": "LOGGS PP TO ANGLIA YD GAS LINE"})

    assert decision.basis == "source_geometry_start"
    assert "pipe_name" in decision.note


def test_determine_chainage_origin_uses_authoritative_from_to_when_present():
    decision = chainage.determine_chainage_origin(
        {"from_installation": "Anglia A", "to_installation": "LOGGS PP"}
    )

    assert decision.basis == "source_from_to_metadata"


def test_compute_route_endpoints_reports_both_crs():
    route = LineString([(413365.0, 5906432.0), (426693.0, 5906208.0)])
    endpoints = chainage.compute_route_endpoints(route, "EPSG:32631")

    assert endpoints.start_working_crs == (413365.0, 5906432.0)
    assert 1.0 < endpoints.start_wgs84[0] < 2.5
    assert 52.5 < endpoints.start_wgs84[1] < 54.0


# --- canonical schema ----------------------------------------------------------


def test_build_canonical_chainage_gdf_schema():
    route = LineString([(0, 0), (108, 0)])
    result = chainage.compute_chainage_stations(route, 25)

    gdf = chainage.build_canonical_chainage_gdf(
        pipeline_id="PL854",
        stations=result.stations,
        interval_m=25,
        chainage_origin_basis="source_geometry_start",
        working_crs="EPSG:32631",
    )

    required = {
        "pipeline_id",
        "station_index",
        "chainage_m",
        "kp_label",
        "chainage_interval_m",
        "is_terminal",
        "chainage_origin_basis",
        "easting",
        "northing",
        "geometry",
    }
    assert required.issubset(set(gdf.columns))
    assert len(gdf) == len(result.stations)
    assert gdf.crs.to_string() == "EPSG:32631"


# --- validate_chainage_gdf ------------------------------------------------------


def test_validate_chainage_gdf_point_on_line_and_counts():
    route = LineString([(0, 0), (100, 0)])
    result = chainage.compute_chainage_stations(route, 25)
    gdf = chainage.build_canonical_chainage_gdf(
        pipeline_id="PL854",
        stations=result.stations,
        interval_m=25,
        chainage_origin_basis="source_geometry_start",
        working_crs="EPSG:32631",
    )

    report = chainage.validate_chainage_gdf(gdf, route, None, "EPSG:32631")

    assert report.max_point_to_route_distance_m < 1e-6
    assert report.station_count == 5


def test_validate_chainage_gdf_rejects_crs_mismatch():
    route = LineString([(0, 0), (100, 0)])
    result = chainage.compute_chainage_stations(route, 25)
    gdf = chainage.build_canonical_chainage_gdf(
        pipeline_id="PL854",
        stations=result.stations,
        interval_m=25,
        chainage_origin_basis="source_geometry_start",
        working_crs="EPSG:4326",  # deliberately wrong
    )

    with pytest.raises(chainage.ChainageValidationError):
        chainage.validate_chainage_gdf(gdf, route, None, "EPSG:32631")


def test_validate_chainage_gdf_rejects_out_of_aoi_station():
    route = LineString([(0, 0), (100, 0)])
    result = chainage.compute_chainage_stations(route, 25)
    gdf = chainage.build_canonical_chainage_gdf(
        pipeline_id="PL854",
        stations=result.stations,
        interval_m=25,
        chainage_origin_basis="source_geometry_start",
        working_crs="EPSG:32631",
    )

    tiny_aoi = Point(0, 0).buffer(10)  # far too small to contain the whole route

    with pytest.raises(chainage.ChainageValidationError):
        chainage.validate_chainage_gdf(gdf, route, tiny_aoi, "EPSG:32631")


# --- end-to-end build_chainage ---------------------------------------------------


def test_build_chainage_end_to_end_with_aoi(tmp_path: Path):
    pipeline_path = tmp_path / "pl854" / "pipeline.gpkg"
    line = _write_pipeline_gpkg(pipeline_path)
    aoi_path = tmp_path / "pl854" / "aoi.gpkg"
    _write_aoi_gpkg(aoi_path, line.buffer(500))
    output_path = tmp_path / "pl854" / "chainage_25m.gpkg"

    report = chainage.build_chainage(
        pipeline_gpkg_path=pipeline_path,
        aoi_gpkg_path=aoi_path,
        pipeline_id="PL854",
        study_id="PL854",
        interval_m=25,
        working_crs="EPSG:32631",
        output_path=output_path,
    )

    assert output_path.exists()
    assert report.total_station_count == 6  # 0,25,50,75,100,108
    assert report.regular_station_count == 5
    assert report.terminal_residual_m == pytest.approx(8.0)
    assert report.all_within_aoi is True
    assert report.max_point_to_route_distance_m < 1e-6
    assert report.chainage_origin_basis == "source_geometry_start"

    roundtrip = gpd.read_file(output_path, layer="chainage_points")
    assert len(roundtrip) == 6
    assert roundtrip.crs.to_string() == "EPSG:32631"
    assert roundtrip["chainage_m"].is_monotonic_increasing
    assert roundtrip["station_index"].is_unique
    assert roundtrip["chainage_m"].is_unique


def test_build_chainage_missing_pipeline_file_raises(tmp_path: Path):
    with pytest.raises(chainage.InvalidPipelineRouteError):
        chainage.build_chainage(
            pipeline_gpkg_path=tmp_path / "missing.gpkg",
            aoi_gpkg_path=None,
            pipeline_id="PL854",
            study_id="PL854",
            interval_m=25,
            working_crs="EPSG:32631",
            output_path=tmp_path / "chainage.gpkg",
        )


def test_build_chainage_disconnected_multipart_raises(tmp_path: Path):
    pipeline_path = tmp_path / "pl854" / "pipeline.gpkg"
    gap_geometry = MultiLineString([[(0, 0), (50, 0)], [(60, 0), (100, 0)]])
    _write_pipeline_gpkg(pipeline_path, geometry=gap_geometry)

    with pytest.raises(chainage.InvalidPipelineRouteError):
        chainage.build_chainage(
            pipeline_gpkg_path=pipeline_path,
            aoi_gpkg_path=None,
            pipeline_id="PL854",
            study_id="PL854",
            interval_m=25,
            working_crs="EPSG:32631",
            output_path=tmp_path / "chainage.gpkg",
        )


def test_build_chainage_rejects_crs_mismatch(tmp_path: Path):
    pipeline_path = tmp_path / "pl854" / "pipeline.gpkg"
    _write_pipeline_gpkg(pipeline_path, crs="EPSG:32630")  # a different UTM zone

    with pytest.raises(chainage.InvalidPipelineRouteError):
        chainage.build_chainage(
            pipeline_gpkg_path=pipeline_path,
            aoi_gpkg_path=None,
            pipeline_id="PL854",
            study_id="PL854",
            interval_m=25,
            working_crs="EPSG:32631",
            output_path=tmp_path / "chainage.gpkg",
        )


def test_build_chainage_deterministic_repeated_output(tmp_path: Path):
    pipeline_path = tmp_path / "pl854" / "pipeline.gpkg"
    _write_pipeline_gpkg(pipeline_path)

    output_path1 = tmp_path / "run1.gpkg"
    output_path2 = tmp_path / "run2.gpkg"
    kwargs = {
        "pipeline_gpkg_path": pipeline_path,
        "aoi_gpkg_path": None,
        "pipeline_id": "PL854",
        "study_id": "PL854",
        "interval_m": 25,
        "working_crs": "EPSG:32631",
    }
    chainage.build_chainage(output_path=output_path1, **kwargs)
    chainage.build_chainage(output_path=output_path2, **kwargs)

    gdf1 = gpd.read_file(output_path1, layer="chainage_points")
    gdf2 = gpd.read_file(output_path2, layer="chainage_points")

    assert gdf1["chainage_m"].tolist() == gdf2["chainage_m"].tolist()
    assert list(gdf1.geometry.apply(lambda p: (p.x, p.y))) == list(
        gdf2.geometry.apply(lambda p: (p.x, p.y))
    )
