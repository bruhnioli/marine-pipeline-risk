"""Offline unit tests for marine_engine.preprocessing.bathymetry.

Uses small synthetic rasters and geometries with simple round-number
coordinates -- never the real ~481x107 EMODnet raster or the real PL854
chainage -- and never touches the network. Live WFS/WCS behaviour is
covered separately in tests/test_bathymetry_sources.py and
tests/test_bathymetry_live.py.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import Point, Polygon, box

from marine_engine.preprocessing import bathymetry
from marine_engine.providers.bathymetry import emodnet


def _write_raster(
    path: Path,
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: str = "EPSG:4326",
    nodata: float | None = None,
    dtype: str = "float64",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array.astype(dtype), 1)
    return path


def _write_uniform_raster_covering(
    tmp_path: Path,
    points: list[tuple[float, float]],
    crs: str,
    value: float = 25.0,
    pad: float = 1000.0,
    size: int = 10,
    name: str = "canonical.tif",
) -> Path:
    """A small raster of one constant value, generously covering `points`."""

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    transform = rasterio.transform.from_bounds(
        min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad, size, size
    )
    array = np.full((size, size), value, dtype="float32")
    return _write_raster(tmp_path / name, array, transform, crs=crs, nodata=np.nan, dtype="float32")


def _make_chainage_gdf(
    points: list[tuple[float, float]], crs: str = "EPSG:32631", pipeline_id: str = "PL854"
) -> gpd.GeoDataFrame:
    records = [
        {
            "pipeline_id": pipeline_id,
            "station_index": i,
            "chainage_m": i * 25.0,
            "kp_label": f"KP {i * 25.0 / 1000.0:.3f}",
        }
        for i in range(len(points))
    ]
    return gpd.GeoDataFrame(records, geometry=[Point(p) for p in points], crs=crs)


def _source_ref(identifier: str, geometry_wgs84, **overrides) -> emodnet.SourceReferenceFeature:
    fields = {
        "identifier": identifier,
        "source_type": "CDI",
        "edmo_id": 2607,
        "release": "2024",
        "date_start": None,
        "date_end": None,
        "metadata_url": None,
        "geometry_wgs84": geometry_wgs84,
    }
    fields.update(overrides)
    return emodnet.SourceReferenceFeature(**fields)


def _qi_feature(identifier: str, geometry_wgs84, **overrides) -> emodnet.QualityIndexFeature:
    fields = {
        "identifier": identifier,
        "source_type": "CDI",
        "combined": 76.9,
        "horizontal": 2,
        "vertical": 3,
        "age": 3,
        "purpose": 3,
        "release": "2024",
        "geometry_wgs84": geometry_wgs84,
    }
    fields.update(overrides)
    return emodnet.QualityIndexFeature(**fields)


# --- inspect_raw_raster / malformed raster failure ---------------------------


def test_inspect_raw_raster_missing_file_raises(tmp_path):
    with pytest.raises(bathymetry.InvalidRawRasterError):
        bathymetry.inspect_raw_raster(tmp_path / "does_not_exist.tif")


def test_inspect_raw_raster_all_nodata_raises(tmp_path):
    array = np.full((5, 5), -9999.0)
    transform = rasterio.transform.from_origin(0.0, 1.0, 0.1, 0.1)
    path = _write_raster(tmp_path / "all_nodata.tif", array, transform, nodata=-9999.0)

    with pytest.raises(bathymetry.InvalidRawRasterError):
        bathymetry.inspect_raw_raster(path)


def test_inspect_raw_raster_excludes_nodata_from_stats(tmp_path):
    array = np.array([[-9999.0, -20.0], [-30.0, -9999.0]])
    transform = rasterio.transform.from_origin(0.0, 1.0, 0.1, 0.1)
    path = _write_raster(tmp_path / "with_nodata.tif", array, transform, nodata=-9999.0)

    stats = bathymetry.inspect_raw_raster(path)

    assert stats.finite_count == 2
    assert stats.nan_count == 0
    assert stats.mean == pytest.approx(-25.0)
    assert stats.nodata_value == pytest.approx(-9999.0)


def test_inspect_raw_raster_excludes_nan_pixels_with_no_declared_nodata(tmp_path):
    array = np.array([[-20.0, np.nan], [-30.0, -25.0]])
    transform = rasterio.transform.from_origin(0.0, 1.0, 0.1, 0.1)
    path = _write_raster(tmp_path / "with_nan.tif", array, transform)

    stats = bathymetry.inspect_raw_raster(path)

    assert stats.finite_count == 3
    assert stats.nan_count == 1
    assert stats.nodata_value is None


# --- determine_sign_convention (empirical, never assumed) --------------------


def _stats(mean: float, low: float = None, high: float = None) -> bathymetry.RawRasterStats:
    return bathymetry.RawRasterStats(
        width=10,
        height=10,
        source_crs="EPSG:4326",
        nodata_value=None,
        total_pixels=100,
        nan_count=0,
        finite_count=100,
        min=low if low is not None else mean - 5,
        max=high if high is not None else mean + 5,
        mean=mean,
        median=mean,
    )


def test_determine_sign_convention_detects_negative_elevation():
    assert (
        bathymetry.determine_sign_convention(_stats(mean=-26.2))
        == bathymetry.SIGN_NEGATIVE_ELEVATION
    )


def test_determine_sign_convention_detects_positive_down_depth():
    assert (
        bathymetry.determine_sign_convention(_stats(mean=24.5))
        == bathymetry.SIGN_POSITIVE_DOWN_DEPTH
    )


def test_determine_sign_convention_raises_when_magnitude_implausible_in_either_sign():
    with pytest.raises(bathymetry.AmbiguousSignConventionError):
        bathymetry.determine_sign_convention(_stats(mean=500.0))


def test_determine_sign_convention_raises_near_zero_mean():
    with pytest.raises(bathymetry.AmbiguousSignConventionError):
        bathymetry.determine_sign_convention(_stats(mean=0.01))


def test_determine_sign_convention_honours_custom_reference_range():
    result = bathymetry.determine_sign_convention(
        _stats(mean=-120.0), reference_depth_range_m=(100.0, 140.0)
    )
    assert result == bathymetry.SIGN_NEGATIVE_ELEVATION


# --- to_canonical_depth (positive-down conversion, no double flip) -----------


def test_to_canonical_depth_flips_negative_elevation():
    raw = np.array([[-10.0, -20.0], [-30.0, np.nan]])
    result = bathymetry.to_canonical_depth(raw, bathymetry.SIGN_NEGATIVE_ELEVATION)

    assert result[0, 0] == pytest.approx(10.0)
    assert result[1, 0] == pytest.approx(30.0)
    assert np.isnan(result[1, 1])


def test_to_canonical_depth_passes_through_positive_down_depth():
    raw = np.array([[10.0, 20.0]])
    result = bathymetry.to_canonical_depth(raw, bathymetry.SIGN_POSITIVE_DOWN_DEPTH)

    assert result[0, 0] == pytest.approx(10.0)
    assert result[0, 1] == pytest.approx(20.0)


def test_to_canonical_depth_unknown_convention_raises():
    with pytest.raises(ValueError):
        bathymetry.to_canonical_depth(np.array([[1.0]]), "not_a_real_convention")


# --- reproject_and_resample (CRS reprojection, 100 m grid, bilinear) ---------


def test_reproject_and_resample_produces_target_resolution():
    array = np.full((20, 20), 25.0)
    transform = rasterio.transform.from_origin(0.0, 54.0, 0.01, 0.01)
    bounds = rasterio.transform.array_bounds(20, 20, transform)

    _, dst_transform = bathymetry.reproject_and_resample(
        array, transform, "EPSG:4326", "EPSG:32631", bounds, target_resolution_m=100.0
    )

    assert dst_transform.a == pytest.approx(100.0)
    assert dst_transform.e == pytest.approx(-100.0)


def test_reproject_and_resample_actually_reprojects_not_a_noop():
    array = np.full((20, 20), 25.0)
    transform = rasterio.transform.from_origin(0.0, 54.0, 0.01, 0.01)
    bounds = rasterio.transform.array_bounds(20, 20, transform)

    result, dst_transform = bathymetry.reproject_and_resample(
        array, transform, "EPSG:4326", "EPSG:32631", bounds, target_resolution_m=100.0
    )

    # A metre-based UTM pixel size is nothing like the original 0.01 deg spacing.
    assert abs(dst_transform.a - 0.01) > 1.0
    assert result.shape != array.shape


def test_reproject_and_resample_uses_bilinear_resampling(monkeypatch):
    captured = {}
    original_reproject = bathymetry.reproject

    def spy_reproject(**kwargs):
        captured["resampling"] = kwargs["resampling"]
        return original_reproject(**kwargs)

    monkeypatch.setattr(bathymetry, "reproject", spy_reproject)

    array = np.full((10, 10), 25.0)
    transform = rasterio.transform.from_origin(0.0, 54.0, 0.01, 0.01)
    bounds = rasterio.transform.array_bounds(10, 10, transform)

    bathymetry.reproject_and_resample(array, transform, "EPSG:4326", "EPSG:32631", bounds)

    assert captured["resampling"] == bathymetry.Resampling.bilinear


# --- mask_to_aoi_polygon (real polygon, not bounding box) --------------------


def test_mask_to_aoi_polygon_masks_pixels_outside_polygon_within_bbox():
    array = np.full((10, 10), 25.0)
    transform = rasterio.transform.from_origin(500000.0, 5900100.0, 10.0, 10.0)
    # A triangle inscribed in the array's own bounding box: if masking only
    # used the bbox, nothing would be clipped. It should clip roughly half.
    triangle = Polygon([(500000.0, 5900000.0), (500100.0, 5900000.0), (500000.0, 5900100.0)])

    masked, _ = bathymetry.mask_to_aoi_polygon(array, transform, "EPSG:32631", triangle)

    valid = ~np.isnan(masked)
    assert 0 < valid.sum() < array.size


def test_mask_to_aoi_polygon_full_bbox_rectangle_keeps_all_pixels():
    array = np.full((10, 10), 25.0)
    transform = rasterio.transform.from_origin(500000.0, 5900100.0, 10.0, 10.0)
    rect = box(500000.0, 5900000.0, 500100.0, 5900100.0)

    masked, _ = bathymetry.mask_to_aoi_polygon(array, transform, "EPSG:32631", rect)

    assert np.isnan(masked).sum() == 0


# --- _canonical_stats / write_canonical_raster (NoData, tags) ----------------


def test_canonical_stats_all_nodata_reports_zero_valid():
    stats = bathymetry._canonical_stats(np.full((5, 5), np.nan))

    assert stats.valid_pixel_count == 0
    assert stats.valid_percent == 0.0
    assert np.isnan(stats.depth_min)


def test_canonical_stats_mixed_valid_and_nodata():
    stats = bathymetry._canonical_stats(np.array([[10.0, np.nan], [30.0, 20.0]]))

    assert stats.valid_pixel_count == 3
    assert stats.nodata_pixel_count == 1
    assert stats.depth_min == pytest.approx(10.0)
    assert stats.depth_max == pytest.approx(30.0)
    assert stats.depth_mean == pytest.approx(20.0)


def test_write_canonical_raster_sets_nan_nodata_and_tags(tmp_path):
    array = np.array([[10.0, np.nan], [30.0, 20.0]])
    transform = rasterio.transform.from_origin(500000.0, 5900100.0, 100.0, 100.0)
    out_path = tmp_path / "canonical.tif"

    bathymetry.write_canonical_raster(
        array, transform, "EPSG:32631", out_path, tags={"vertical_datum": "LAT", "product": "Test"}
    )

    with rasterio.open(out_path) as dataset:
        assert dataset.crs.to_string() == "EPSG:32631"
        assert np.isnan(dataset.nodata)
        band = dataset.read(1)
        assert np.isnan(band[0, 1])
        tags = dataset.tags()
        assert tags["vertical_datum"] == "LAT"
        assert tags["product"] == "Test"


# --- build_canonical_dtm: full pipeline, provenance JSON, determinism --------


def test_build_canonical_dtm_full_pipeline_negative_elevation_source(tmp_path):
    raw_array = np.full((40, 40), -25.0)
    raw_transform = rasterio.transform.from_origin(500000.0, 5901000.0, 25.0, 25.0)
    raw_path = _write_raster(tmp_path / "raw.tif", raw_array, raw_transform, crs="EPSG:32631")
    aoi_geometry = box(500100.0, 5900100.0, 500900.0, 5900900.0)

    report = bathymetry.build_canonical_dtm(
        raw_path=raw_path,
        raw_manifest_entry={"sha256": "deadbeef"},
        aoi_geometry_working=aoi_geometry,
        working_crs="EPSG:32631",
        aoi_identifier="TEST_AOI",
        output_raster_path=tmp_path / "canonical.tif",
        output_metadata_path=tmp_path / "canonical.json",
        target_resolution_m=100.0,
        reference_depth_range_m=(20.0, 28.0),
    )

    assert report.sign_convention_observed == bathymetry.SIGN_NEGATIVE_ELEVATION
    # A single correct sign flip of -25 -> +25; neither zero flips (-25) nor
    # a double flip (also -25 for pure negation) would pass this assertion.
    assert report.canonical_stats.depth_mean == pytest.approx(25.0, abs=0.5)
    assert report.canonical_stats.depth_mean > 0
    assert report.output_resolution_m == 100.0
    assert report.output_raster_path.exists()
    assert report.output_metadata_path.exists()

    with rasterio.open(report.output_raster_path) as dataset:
        assert dataset.crs.to_string() == "EPSG:32631"
        assert abs(dataset.transform.a - 100.0) < 1e-6
        assert np.isnan(dataset.nodata)

    metadata = json.loads(report.output_metadata_path.read_text())
    for key in (
        "product",
        "source_version",
        "source_path",
        "source_sha256",
        "source_crs",
        "output_crs",
        "source_nominal_resolution_m",
        "output_analysis_grid_spacing_m",
        "resampling_method",
        "vertical_datum",
        "source_sign_convention",
        "canonical_sign_convention",
        "aoi_identifier",
        "processing_timestamp",
        "software_version",
        "scientific_limitations",
    ):
        assert key in metadata, f"missing provenance key: {key}"
    assert metadata["vertical_datum"] == "LAT"
    assert metadata["source_sign_convention"] == "negative_elevation"
    assert metadata["canonical_sign_convention"] == "positive_down_depth_lat_m"
    assert metadata["source_sha256"] == "deadbeef"
    assert len(metadata["scientific_limitations"]) == 5
    # Explicit LAT/MSL separation: the canonical LAT product's own metadata
    # never references MSL -- the two are stored completely separately.
    metadata_text = json.dumps(metadata).lower()
    assert "msl" not in metadata_text
    assert "mean sea level" not in metadata_text


def test_build_canonical_dtm_reprojects_positive_down_source_from_wgs84(tmp_path):
    raw_array = np.full((20, 20), 24.0)
    raw_transform = rasterio.transform.from_origin(1.60, 53.40, 0.01, 0.01)
    raw_path = _write_raster(tmp_path / "raw_wgs84.tif", raw_array, raw_transform, crs="EPSG:4326")

    aoi_wgs84 = box(1.62, 53.38, 1.70, 53.39)
    aoi_working = gpd.GeoSeries([aoi_wgs84], crs="EPSG:4326").to_crs("EPSG:32631").iloc[0]

    report = bathymetry.build_canonical_dtm(
        raw_path=raw_path,
        raw_manifest_entry=None,
        aoi_geometry_working=aoi_working,
        working_crs="EPSG:32631",
        aoi_identifier="TEST_AOI_WGS84",
        output_raster_path=tmp_path / "canonical_wgs84.tif",
        output_metadata_path=tmp_path / "canonical_wgs84.json",
        target_resolution_m=100.0,
    )

    assert report.sign_convention_observed == bathymetry.SIGN_POSITIVE_DOWN_DEPTH
    assert report.canonical_stats.valid_pixel_count > 0
    assert report.canonical_stats.depth_mean == pytest.approx(24.0, abs=1.0)
    assert report.source_sha256 is None  # no manifest entry supplied -- null, not fabricated
    with rasterio.open(report.output_raster_path) as dataset:
        assert dataset.crs.to_string() == "EPSG:32631"


def test_build_canonical_dtm_is_deterministic_across_runs(tmp_path):
    raw_array = np.full((20, 20), -25.0)
    raw_transform = rasterio.transform.from_origin(500000.0, 5900500.0, 25.0, 25.0)
    raw_path = _write_raster(tmp_path / "raw.tif", raw_array, raw_transform, crs="EPSG:32631")
    aoi_geometry = box(500050.0, 5900050.0, 500450.0, 5900450.0)

    def run(tag: str) -> bathymetry.CanonicalDtmReport:
        return bathymetry.build_canonical_dtm(
            raw_path=raw_path,
            raw_manifest_entry=None,
            aoi_geometry_working=aoi_geometry,
            working_crs="EPSG:32631",
            aoi_identifier="TEST",
            output_raster_path=tmp_path / f"out_{tag}.tif",
            output_metadata_path=tmp_path / f"out_{tag}.json",
        )

    report_a = run("a")
    report_b = run("b")

    with (
        rasterio.open(report_a.output_raster_path) as ds_a,
        rasterio.open(report_b.output_raster_path) as ds_b,
    ):
        np.testing.assert_array_equal(ds_a.read(1), ds_b.read(1))
    assert report_a.canonical_stats.depth_mean == report_b.canonical_stats.depth_mean


# --- sample_chainage_bathymetry: depth, source-ref/QI attribution -----------


def test_sample_chainage_bathymetry_reads_expected_depth_values(tmp_path):
    working_crs = "EPSG:32631"
    points = [(500000.0, 5900000.0), (500020.0, 5900020.0)]
    raster_path = _write_uniform_raster_covering(tmp_path, points, working_crs, value=25.0)
    chainage_gdf = _make_chainage_gdf(points, crs=working_crs)

    df = bathymetry.sample_chainage_bathymetry(
        chainage_gdf=chainage_gdf,
        canonical_raster_path=raster_path,
        source_reference_features=[],
        quality_index_features=[],
        working_crs=working_crs,
    )

    assert len(df) == 2
    assert df["depth_lat_m"].iloc[0] == pytest.approx(25.0)
    assert list(df.columns) == list(bathymetry.CHAINAGE_OUTPUT_COLUMNS)


def test_sample_chainage_bathymetry_retains_out_of_extent_stations_as_null_depth(tmp_path):
    working_crs = "EPSG:32631"
    in_bounds_point = (500000.0, 5900000.0)
    raster_path = _write_uniform_raster_covering(
        tmp_path, [in_bounds_point], working_crs, value=25.0, pad=50.0, size=5
    )
    far_outside_point = (9_000_000.0, 9_000_000.0)
    chainage_gdf = _make_chainage_gdf([in_bounds_point, far_outside_point], crs=working_crs)

    df = bathymetry.sample_chainage_bathymetry(
        chainage_gdf=chainage_gdf,
        canonical_raster_path=raster_path,
        source_reference_features=[],
        quality_index_features=[],
        working_crs=working_crs,
    )

    assert len(df) == 2  # both stations retained, never dropped
    assert df["depth_lat_m"].iloc[0] == pytest.approx(25.0)
    # DataFrame construction from records normalizes a missing value to NaN
    # even in an otherwise-object/string column -- pd.isna is the correct
    # check here, not `is None` (the row dict itself does hold None).
    assert pd.isna(df["depth_lat_m"].iloc[1])


def test_sample_chainage_bathymetry_depth_available_even_when_attribution_empty(tmp_path):
    working_crs = "EPSG:32631"
    point = (500000.0, 5900000.0)
    raster_path = _write_uniform_raster_covering(tmp_path, [point], working_crs, value=27.5)
    chainage_gdf = _make_chainage_gdf([point], crs=working_crs)

    df = bathymetry.sample_chainage_bathymetry(
        chainage_gdf=chainage_gdf,
        canonical_raster_path=raster_path,
        source_reference_features=[],  # simulates source_attribution_status == unavailable
        quality_index_features=[],
        working_crs=working_crs,
    )

    assert df["depth_lat_m"].iloc[0] == pytest.approx(27.5)
    assert pd.isna(df["source_reference_id"].iloc[0])
    assert pd.isna(df["source_reference_type"].iloc[0])
    assert pd.isna(df["qi_combined"].iloc[0])


def test_sample_chainage_bathymetry_matches_containing_source_reference_polygon(tmp_path):
    working_crs = "EPSG:32631"
    poly_wgs84 = box(1.60, 53.35, 1.90, 53.40)
    poly_working = gpd.GeoSeries([poly_wgs84], crs="EPSG:4326").to_crs(working_crs).iloc[0]
    inside_point = (poly_working.centroid.x, poly_working.centroid.y)
    minx, miny, maxx, maxy = poly_working.bounds
    outside_point = (maxx + 50_000.0, maxy + 50_000.0)

    raster_path = _write_uniform_raster_covering(
        tmp_path, [inside_point, outside_point], working_crs
    )
    chainage_gdf = _make_chainage_gdf([inside_point, outside_point], crs=working_crs)

    source_ref = _source_ref("121954", poly_wgs84)
    qi = _qi_feature("121954", poly_wgs84)

    df = bathymetry.sample_chainage_bathymetry(
        chainage_gdf=chainage_gdf,
        canonical_raster_path=raster_path,
        source_reference_features=[source_ref],
        quality_index_features=[qi],
        working_crs=working_crs,
    )

    assert df["source_reference_id"].iloc[0] == "121954"
    assert df["source_reference_type"].iloc[0] == "CDI"
    assert df["qi_combined"].iloc[0] == pytest.approx(76.9)
    # The far-outside point matches no polygon -- null, never fabricated.
    assert pd.isna(df["source_reference_id"].iloc[1])
    assert pd.isna(df["qi_combined"].iloc[1])


def test_sample_chainage_bathymetry_preserves_raw_qi_classes_unmodified(tmp_path):
    working_crs = "EPSG:32631"
    point = (500000.0, 5900000.0)
    point_wgs84 = gpd.GeoSeries([Point(*point)], crs=working_crs).to_crs("EPSG:4326").iloc[0]
    poly_wgs84 = point_wgs84.buffer(0.5)

    raster_path = _write_uniform_raster_covering(tmp_path, [point], working_crs)
    chainage_gdf = _make_chainage_gdf([point], crs=working_crs)
    qi = _qi_feature("999", poly_wgs84, horizontal=1, vertical=4, age=2, purpose=3, combined=42.42)

    df = bathymetry.sample_chainage_bathymetry(
        chainage_gdf=chainage_gdf,
        canonical_raster_path=raster_path,
        source_reference_features=[],
        quality_index_features=[qi],
        working_crs=working_crs,
    )

    assert df["qi_horizontal"].iloc[0] == 1
    assert df["qi_vertical"].iloc[0] == 4
    assert df["qi_age"].iloc[0] == 2
    assert df["qi_purpose"].iloc[0] == 3
    assert df["qi_combined"].iloc[0] == pytest.approx(42.42)


def test_write_chainage_bathymetry_round_trips_via_parquet(tmp_path):
    df = pd.DataFrame(
        [
            {
                "pipeline_id": "PL854",
                "station_index": 0,
                "chainage_m": 0.0,
                "kp_label": "KP 0.000",
                "depth_lat_m": 25.0,
                "bathymetry_source_product": "Test",
                "source_reference_id": None,
                "source_reference_type": None,
                "qi_age": None,
                "qi_horizontal": None,
                "qi_vertical": None,
                "qi_purpose": None,
                "qi_combined": None,
            }
        ],
        columns=list(bathymetry.CHAINAGE_OUTPUT_COLUMNS),
    )
    out_path = tmp_path / "chainage_bathymetry.parquet"

    result_path = bathymetry.write_chainage_bathymetry(df, out_path)

    assert result_path == out_path
    reloaded = pd.read_parquet(out_path)
    assert len(reloaded) == 1
    assert reloaded["depth_lat_m"].iloc[0] == pytest.approx(25.0)


# --- print_bathymetry_report --------------------------------------------------


def test_print_bathymetry_report_includes_expected_sections(tmp_path):
    import io

    raw_array = np.full((10, 10), -25.0)
    raw_transform = rasterio.transform.from_origin(500000.0, 5900250.0, 25.0, 25.0)
    raw_path = _write_raster(tmp_path / "raw.tif", raw_array, raw_transform, crs="EPSG:32631")
    aoi_geometry = box(500050.0, 5900050.0, 500200.0, 5900200.0)

    report = bathymetry.build_canonical_dtm(
        raw_path=raw_path,
        raw_manifest_entry=None,
        aoi_geometry_working=aoi_geometry,
        working_crs="EPSG:32631",
        aoi_identifier="TEST",
        output_raster_path=tmp_path / "canonical.tif",
        output_metadata_path=tmp_path / "canonical.json",
        target_resolution_m=25.0,
    )
    chainage_df = pd.DataFrame(
        [
            {
                "pipeline_id": "PL854",
                "station_index": 0,
                "chainage_m": 0.0,
                "kp_label": "KP 0.000",
                "depth_lat_m": 25.0,
                "bathymetry_source_product": "Test",
                "source_reference_id": "121954",
                "source_reference_type": "CDI",
                "qi_age": 3,
                "qi_horizontal": 2,
                "qi_vertical": 3,
                "qi_purpose": 3,
                "qi_combined": 76.9,
            }
        ],
        columns=list(bathymetry.CHAINAGE_OUTPUT_COLUMNS),
    )

    buffer = io.StringIO()
    bathymetry.print_bathymetry_report(
        report,
        chainage_df,
        attribution_status="available",
        attribution_notes="",
        msl_notes="not available; no tile found",
        file=buffer,
    )
    output = buffer.getvalue()

    assert "Raw EMODnet" in output
    assert "Canonical DTM" in output
    assert "PL854 chainage" in output
    assert "source_attribution_status: available" in output
    assert "LIMITATION" in output
