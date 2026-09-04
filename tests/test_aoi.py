"""Offline unit tests for marine_engine.preprocessing.aoi.

Uses small synthetic line geometries -- never the real ~1300-vertex PL854
route -- and never touches the network. The synthetic EPSG:32631 coordinates
below were computed once via pyproj from (1.70 E, 53.30 N) / (1.90 E, 53.30 N)
so that reprojecting them back to WGS84 lands near the real PL854 area and
exercises the North-Sea plausibility check meaningfully.
"""

import math
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from marine_engine.preprocessing import aoi

# A ~13,330 m straight line in EPSG:32631, verified to reproject to
# approximately (1.70-1.90 E, 53.30 N) -- inside the North Sea plausibility band.
LINE_START = (413365.0, 5906432.0)
LINE_END = (426693.0, 5906208.0)
STRAIGHT_LINE_32631 = LineString([LINE_START, LINE_END])
STRAIGHT_LINE_LENGTH_M = math.dist(LINE_START, LINE_END)


def _write_pipeline_gpkg(
    path: Path,
    pipeline_id: str = "PL854",
    crs: str = "EPSG:32631",
    geometry=None,
    rows: int = 1,
) -> None:
    geometry = geometry or STRAIGHT_LINE_32631
    records = [
        {"pipeline_id": pipeline_id, "source": "test", "status": "ACTIVE"} for _ in range(rows)
    ]
    geometries = [geometry] * rows
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG", layer="pipeline")


# --- load_canonical_pipeline / missing / invalid input ----------------------


def test_load_canonical_pipeline_reads_matching_geometry(tmp_path: Path):
    gpkg_path = tmp_path / "pipeline.gpkg"
    _write_pipeline_gpkg(gpkg_path)

    geometry, source_crs = aoi.load_canonical_pipeline(gpkg_path, "PL854")

    assert geometry.geom_type == "LineString"
    assert source_crs == "EPSG:32631"


def test_load_canonical_pipeline_missing_file_raises(tmp_path: Path):
    with pytest.raises(aoi.InvalidPipelineInputError):
        aoi.load_canonical_pipeline(tmp_path / "does_not_exist.gpkg", "PL854")


def test_load_canonical_pipeline_wrong_pipeline_id_raises(tmp_path: Path):
    gpkg_path = tmp_path / "pipeline.gpkg"
    _write_pipeline_gpkg(gpkg_path, pipeline_id="PL999")

    with pytest.raises(aoi.InvalidPipelineInputError):
        aoi.load_canonical_pipeline(gpkg_path, "PL854")


def test_load_canonical_pipeline_merges_multiple_connected_parts(tmp_path: Path):
    gpkg_path = tmp_path / "pipeline.gpkg"
    _write_pipeline_gpkg(gpkg_path, rows=2)  # two rows, same pipeline_id -> unioned

    geometry, _ = aoi.load_canonical_pipeline(gpkg_path, "PL854")

    assert geometry.geom_type in ("LineString", "MultiLineString")


# --- _validate_and_merge_geometries (pure, no file I/O) ---------------------


def test_validate_and_merge_geometries_rejects_missing():
    with pytest.raises(aoi.InvalidPipelineInputError):
        aoi._validate_and_merge_geometries([None], "PL854")


def test_validate_and_merge_geometries_rejects_invalid_geometry():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])  # self-intersecting
    with pytest.raises(aoi.InvalidPipelineInputError):
        aoi._validate_and_merge_geometries([bowtie], "PL854")


def test_validate_and_merge_geometries_rejects_non_linear():
    with pytest.raises(aoi.InvalidPipelineInputError):
        aoi._validate_and_merge_geometries([Point(0, 0)], "PL854")


def test_validate_and_merge_geometries_unions_multilinestring_parts():
    multi = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])
    result = aoi._validate_and_merge_geometries([multi], "PL854")
    assert result.is_valid


# --- CRS validation / transformation ----------------------------------------


def test_ensure_projected_working_crs_transforms_wgs84_input():
    line_wgs84 = LineString([(1.7, 53.30), (1.9, 53.30)])

    result = aoi.ensure_projected_working_crs(line_wgs84, "EPSG:4326", "EPSG:32631")

    # UTM31N easting/northing are on the order of 1e5-1e6 m, not degrees.
    assert result.bounds[0] > 1000


def test_ensure_projected_working_crs_noop_when_already_matching():
    result = aoi.ensure_projected_working_crs(STRAIGHT_LINE_32631, "EPSG:32631", "EPSG:32631")
    assert result.equals(STRAIGHT_LINE_32631)


def test_ensure_projected_working_crs_rejects_geographic_working_crs():
    with pytest.raises(aoi.InvalidPipelineInputError):
        aoi.ensure_projected_working_crs(STRAIGHT_LINE_32631, "EPSG:32631", "EPSG:4326")


# --- corridor buffer generation ---------------------------------------------


def test_build_corridor_buffer_produces_valid_polygon():
    result = aoi.build_corridor_buffer(STRAIGHT_LINE_32631, 5000)

    assert result.is_valid
    assert result.geom_type in ("Polygon", "MultiPolygon")


def test_build_corridor_buffer_rejects_nonpositive_distance():
    with pytest.raises(ValueError):
        aoi.build_corridor_buffer(STRAIGHT_LINE_32631, 0)
    with pytest.raises(ValueError):
        aoi.build_corridor_buffer(STRAIGHT_LINE_32631, -100)


def test_build_corridor_buffer_distance_is_configurable():
    small = aoi.build_corridor_buffer(STRAIGHT_LINE_32631, 1000)
    large = aoi.build_corridor_buffer(STRAIGHT_LINE_32631, 5000)

    assert large.area > small.area


def test_build_corridor_buffer_contains_the_pipeline():
    result = aoi.build_corridor_buffer(STRAIGHT_LINE_32631, 5000)
    assert result.contains(STRAIGHT_LINE_32631)


def test_build_corridor_buffer_matches_analytical_stadium_area():
    buffer_m = 5000.0
    result = aoi.build_corridor_buffer(STRAIGHT_LINE_32631, buffer_m)

    # A straight line's buffer is a "stadium": a rectangle (length x 2*buffer)
    # plus two semicircular end caps (together one full circle of the buffer
    # radius). Independent of the buffer implementation under test.
    expected_area = STRAIGHT_LINE_LENGTH_M * 2 * buffer_m + math.pi * buffer_m**2

    assert result.area == pytest.approx(expected_area, rel=0.01)


# --- sanity checks -----------------------------------------------------------


def test_run_sanity_checks_reports_expected_metrics():
    buffer_m = 5000.0
    polygon = aoi.build_corridor_buffer(STRAIGHT_LINE_32631, buffer_m)

    metrics = aoi.run_sanity_checks(STRAIGHT_LINE_32631, polygon, buffer_m, "EPSG:32631")

    assert metrics.min_boundary_distance_m == pytest.approx(buffer_m, rel=0.05)
    assert metrics.area_km2 == pytest.approx(polygon.area / 1_000_000.0)
    lon, lat = metrics.centroid_wgs84
    assert 1.0 < lon < 2.5
    assert 52.5 < lat < 54.0


def test_run_sanity_checks_raises_if_aoi_does_not_contain_pipeline():
    unrelated_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    with pytest.raises(aoi.InvalidAoiGeometryError):
        aoi.run_sanity_checks(STRAIGHT_LINE_32631, unrelated_polygon, 5000, "EPSG:32631")


def test_run_sanity_checks_raises_on_implausible_buffer_distance():
    # A polygon that only loosely surrounds the line -- far from a uniform
    # 5000 m offset -- should be caught by the distance-tolerance check.
    minx, miny, maxx, maxy = STRAIGHT_LINE_32631.buffer(50).bounds
    tiny_buffer_polygon = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])

    with pytest.raises(aoi.InvalidAoiGeometryError):
        aoi.run_sanity_checks(STRAIGHT_LINE_32631, tiny_buffer_polygon, 5000, "EPSG:32631")


def test_is_plausible_north_sea_location():
    assert aoi._is_plausible_north_sea_location(1.8, 53.3) is True
    assert aoi._is_plausible_north_sea_location(-70.0, 40.0) is False  # e.g. US east coast


def test_run_sanity_checks_raises_if_centroid_outside_north_sea():
    # A valid, well-formed buffer polygon that simply isn't in the North Sea.
    far_away_line = LineString([(0, 0), (10_000, 0)])
    far_away_polygon = far_away_line.buffer(5000)

    with pytest.raises(aoi.InvalidAoiGeometryError):
        aoi.run_sanity_checks(far_away_line, far_away_polygon, 5000, "EPSG:32631")


# --- canonical schema / GeoPackage output -----------------------------------


def test_build_canonical_aoi_gdf_has_required_columns():
    from datetime import UTC, datetime

    polygon = aoi.build_corridor_buffer(STRAIGHT_LINE_32631, 5000)
    gdf = aoi.build_canonical_aoi_gdf(
        study_id="PL854",
        pipeline_id="PL854",
        buffer_m=5000,
        working_crs="EPSG:32631",
        aoi_geometry=polygon,
        area_km2=polygon.area / 1_000_000.0,
        generated_at=datetime.now(UTC),
    )

    required = {
        "study_id",
        "pipeline_id",
        "corridor_buffer_m",
        "working_crs",
        "area_km2",
        "geometry",
    }
    assert required.issubset(set(gdf.columns))
    assert len(gdf) == 1
    assert gdf.crs.to_string() == "EPSG:32631"


def test_write_and_read_back_aoi_gpkg(tmp_path: Path):
    from datetime import UTC, datetime

    polygon = aoi.build_corridor_buffer(STRAIGHT_LINE_32631, 5000)
    gdf = aoi.build_canonical_aoi_gdf(
        study_id="PL854",
        pipeline_id="PL854",
        buffer_m=5000,
        working_crs="EPSG:32631",
        aoi_geometry=polygon,
        area_km2=polygon.area / 1_000_000.0,
        generated_at=datetime.now(UTC),
    )

    output_path = tmp_path / "pl854" / "aoi.gpkg"
    aoi.write_aoi_gpkg(gdf, output_path)

    roundtrip = gpd.read_file(output_path, layer="study_aoi")
    assert len(roundtrip) == 1
    assert roundtrip.iloc[0]["study_id"] == "PL854"
    assert roundtrip.crs.to_string() == "EPSG:32631"
    assert roundtrip.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")
    assert roundtrip.geometry.iloc[0].is_valid


# --- end-to-end build_aoi ----------------------------------------------------


def test_build_aoi_end_to_end_with_synthetic_pipeline(tmp_path: Path):
    pipeline_gpkg_path = tmp_path / "pl854" / "pipeline.gpkg"
    _write_pipeline_gpkg(pipeline_gpkg_path)
    output_path = tmp_path / "pl854" / "aoi.gpkg"

    report = aoi.build_aoi(
        pipeline_gpkg_path=pipeline_gpkg_path,
        pipeline_id="PL854",
        study_id="PL854",
        buffer_m=5000,
        working_crs="EPSG:32631",
        output_path=output_path,
    )

    assert output_path.exists()
    assert report.corridor_buffer_m == 5000
    assert report.working_crs == "EPSG:32631"
    assert report.area_km2 > 0

    roundtrip = gpd.read_file(output_path, layer="study_aoi")
    aoi_polygon = roundtrip.geometry.iloc[0]
    assert aoi_polygon.contains(STRAIGHT_LINE_32631)

    west, south, east, north = report.bounds_wgs84
    assert -4.0 <= west < east <= 9.0
    assert 51.0 <= south < north <= 62.0


def test_build_aoi_missing_pipeline_file_raises(tmp_path: Path):
    with pytest.raises(aoi.InvalidPipelineInputError):
        aoi.build_aoi(
            pipeline_gpkg_path=tmp_path / "missing.gpkg",
            pipeline_id="PL854",
            study_id="PL854",
            buffer_m=5000,
            working_crs="EPSG:32631",
            output_path=tmp_path / "aoi.gpkg",
        )


def test_build_aoi_wgs84_bounds_match_independent_reprojection(tmp_path: Path):
    import pyproj

    pipeline_gpkg_path = tmp_path / "pl854" / "pipeline.gpkg"
    _write_pipeline_gpkg(pipeline_gpkg_path)
    output_path = tmp_path / "pl854" / "aoi.gpkg"

    report = aoi.build_aoi(
        pipeline_gpkg_path=pipeline_gpkg_path,
        pipeline_id="PL854",
        study_id="PL854",
        buffer_m=5000,
        working_crs="EPSG:32631",
        output_path=output_path,
    )

    # Independently reproject every boundary vertex of the actual AOI polygon
    # (not the working-CRS bounding-box corners, which lie outside a rounded
    # "stadium" shape and would test the wrong thing) via a raw pyproj
    # Transformer, bypassing geopandas' `.to_crs()` entirely.
    roundtrip = gpd.read_file(output_path, layer="study_aoi")
    aoi_polygon = roundtrip.geometry.iloc[0]
    transformer = pyproj.Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    xs, ys = aoi_polygon.exterior.coords.xy
    lons, lats = transformer.transform(list(xs), list(ys))

    west, south, east, north = report.bounds_wgs84
    assert west == pytest.approx(min(lons), abs=1e-6)
    assert east == pytest.approx(max(lons), abs=1e-6)
    assert south == pytest.approx(min(lats), abs=1e-6)
    assert north == pytest.approx(max(lats), abs=1e-6)
