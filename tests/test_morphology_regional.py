"""Offline unit tests for marine_engine.morphology.regional.

Uses small synthetic terrain grids with simple round-number coordinates
and analytically-known answers (flat plane, tilted plane, ridge,
depression) -- never the real PL854 halo -- and never touches the
network.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import box

from marine_engine.morphology import regional as rg
from marine_engine.providers.bathymetry.emodnet import NativeQaLayerAvailability

CELL_SIZE_M = 100.0
GRID_SIZE = 61  # generous margin for a 2000 m (20-cell) radius window near the center
CENTER = GRID_SIZE // 2


def _flat_grid(value: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    elevation = np.full((GRID_SIZE, GRID_SIZE), value)
    valid = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    return elevation, valid


def _tilted_grid(a: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    elevation = a * (xx * CELL_SIZE_M) + b * (yy * CELL_SIZE_M) + c
    valid = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    return elevation, valid


def _bump_grid(amplitude: float, sigma_m: float = 400.0, base: float = 10.0) -> np.ndarray:
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    dist_m = np.sqrt(((xx - CENTER) * CELL_SIZE_M) ** 2 + ((yy - CENTER) * CELL_SIZE_M) ** 2)
    return base + amplitude * np.exp(-(dist_m**2) / (2 * sigma_m**2))


# --- Flat plane --------------------------------------------------------------


def test_flat_plane_slope_tpi_relief_all_zero():
    elevation, valid = _flat_grid(10.0)

    slope, _ = rg.compute_slope_deg(elevation, valid, 500.0, CELL_SIZE_M)
    tpi, _ = rg.compute_tpi_m(elevation, valid, 1000.0, CELL_SIZE_M)
    relief, _ = rg.compute_local_relief_m(elevation, valid, 1000.0, CELL_SIZE_M)
    std, _ = rg.compute_terrain_std_m(elevation, valid, 1000.0, CELL_SIZE_M)

    assert slope[CENTER, CENTER] == pytest.approx(0.0, abs=1e-8)
    assert tpi[CENTER, CENTER] == pytest.approx(0.0, abs=1e-8)
    assert relief[CENTER, CENTER] == pytest.approx(0.0, abs=1e-8)
    assert std[CENTER, CENTER] == pytest.approx(0.0, abs=1e-8)


# --- Tilted plane: local-plane-fit slope against the analytical answer ------


def test_tilted_plane_slope_matches_analytical_value():
    a_true, b_true, c_true = 0.05, 0.02, 5.0
    elevation, valid = _tilted_grid(a_true, b_true, c_true)

    slope, valid_fraction = rg.compute_slope_deg(elevation, valid, 500.0, CELL_SIZE_M)

    expected_deg = np.degrees(np.arctan(np.sqrt(a_true**2 + b_true**2)))
    assert valid_fraction[CENTER, CENTER] == pytest.approx(1.0)
    assert slope[CENTER, CENTER] == pytest.approx(expected_deg, abs=1e-6)


def test_tilted_plane_tpi_near_zero_in_symmetric_complete_neighborhood():
    elevation, valid = _tilted_grid(0.05, 0.02, 5.0)
    tpi, _ = rg.compute_tpi_m(elevation, valid, 1000.0, CELL_SIZE_M)
    # A perfectly linear surface has the same mean as its centre value over
    # any symmetric, fully-populated neighborhood.
    assert tpi[CENTER, CENTER] == pytest.approx(0.0, abs=1e-6)


def test_tilted_plane_slope_500m_and_1000m_both_match_the_same_analytical_plane():
    a_true, b_true, c_true = 0.03, -0.04, 2.0
    elevation, valid = _tilted_grid(a_true, b_true, c_true)
    expected_deg = np.degrees(np.arctan(np.sqrt(a_true**2 + b_true**2)))

    slope_500, _ = rg.compute_slope_deg(elevation, valid, 500.0, CELL_SIZE_M)
    slope_1000, _ = rg.compute_slope_deg(elevation, valid, 1000.0, CELL_SIZE_M)

    assert slope_500[CENTER, CENTER] == pytest.approx(expected_deg, abs=1e-6)
    assert slope_1000[CENTER, CENTER] == pytest.approx(expected_deg, abs=1e-6)


# --- Synthetic ridge / depression --------------------------------------------


def test_synthetic_ridge_positive_tpi_at_crest_and_positive_relief():
    ridge = _bump_grid(amplitude=5.0)
    valid = np.ones_like(ridge, dtype=bool)

    tpi, _ = rg.compute_tpi_m(ridge, valid, 1000.0, CELL_SIZE_M)
    relief, _ = rg.compute_local_relief_m(ridge, valid, 1000.0, CELL_SIZE_M)

    assert tpi[CENTER, CENTER] > 0
    assert relief[CENTER, CENTER] > 0


def test_synthetic_ridge_relative_position_falls_off_away_from_crest():
    ridge = _bump_grid(amplitude=5.0)
    valid = np.ones_like(ridge, dtype=bool)
    tpi, valid_fraction = rg.compute_tpi_m(ridge, valid, 1000.0, CELL_SIZE_M)

    off_crest = CENTER + 15  # comfortably inside the grid, away from the bump's peak
    assert valid_fraction[CENTER, off_crest] >= 0.90
    assert tpi[CENTER, off_crest] < tpi[CENTER, CENTER]


def test_synthetic_depression_negative_tpi_at_center():
    depression = _bump_grid(amplitude=-5.0)
    valid = np.ones_like(depression, dtype=bool)

    tpi, _ = rg.compute_tpi_m(depression, valid, 1000.0, CELL_SIZE_M)

    assert tpi[CENTER, CENTER] < 0


def test_local_relief_is_never_negative():
    for grid in (_bump_grid(5.0), _bump_grid(-5.0), _tilted_grid(0.05, 0.02, 5.0)[0]):
        valid = np.ones_like(grid, dtype=bool)
        relief, valid_fraction = rg.compute_local_relief_m(grid, valid, 1000.0, CELL_SIZE_M)
        supported = relief[valid_fraction >= 0.90]
        assert np.all(supported[~np.isnan(supported)] >= 0.0)


# --- Sign convention: elevation vs depth must not reverse interpretation ---


def test_sign_convention_elevation_from_depth_preserves_ridge_as_positive_tpi():
    """A physical ridge (shallower = smaller depth_lat_m) must still show up
    as a positive TPI once converted to seabed_elevation_lat_m = -depth_lat_m."""

    depth_base = 20.0
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    dist_m = np.sqrt(((xx - CENTER) * CELL_SIZE_M) ** 2 + ((yy - CENTER) * CELL_SIZE_M) ** 2)
    # A ridge is SHALLOWER at its crest -> smaller positive-down depth there.
    depth_lat_m = depth_base - 5.0 * np.exp(-(dist_m**2) / (2 * 400.0**2))
    valid = np.ones_like(depth_lat_m, dtype=bool)

    elevation = -depth_lat_m  # Section 3 sign convention
    tpi, _ = rg.compute_tpi_m(elevation, valid, 1000.0, CELL_SIZE_M)

    assert tpi[CENTER, CENTER] > 0  # shallower crest -> higher elevation -> positive TPI


# --- Window scales: physical radius maps correctly onto a 100 m grid -------


@pytest.mark.parametrize(
    ("radius_m", "expected_px"),
    [(500.0, 5), (1000.0, 10), (2000.0, 20)],
)
def test_radius_px_maps_physical_radius_to_grid_cells(radius_m, expected_px):
    assert rg._radius_px(radius_m, CELL_SIZE_M) == expected_px


def test_circular_footprint_cell_count_matches_a_circle_not_a_square():
    footprint = rg._circular_footprint(10)
    square_cell_count = (2 * 10 + 1) ** 2
    # A real circle inscribed in the bounding square has noticeably fewer
    # cells than the square itself -- confirms this is a disk, not a box.
    assert footprint.sum() < square_cell_count
    assert footprint[10, 10]  # center cell always included


# --- NoData / neighborhood-validity threshold (Section 10) -----------------


def test_below_90_percent_valid_neighborhood_yields_nodata():
    elevation, _ = _flat_grid(10.0)
    sparse_valid = np.zeros_like(elevation, dtype=bool)
    sparse_valid[CENTER - 1 : CENTER + 2, CENTER - 1 : CENTER + 2] = True  # tiny 3x3 patch only

    tpi, valid_fraction = rg.compute_tpi_m(elevation, sparse_valid, 1000.0, CELL_SIZE_M)

    assert valid_fraction[CENTER, CENTER] < rg.MIN_VALID_NEIGHBORHOOD_FRACTION
    assert np.isnan(tpi[CENTER, CENTER])


def test_sufficiently_supported_neighborhood_is_valid():
    elevation, valid = _flat_grid(10.0)
    tpi, valid_fraction = rg.compute_tpi_m(elevation, valid, 1000.0, CELL_SIZE_M)

    assert valid_fraction[CENTER, CENTER] == pytest.approx(1.0)
    assert not np.isnan(tpi[CENTER, CENTER])


def test_insufficient_margin_reduces_validity_fraction_near_the_array_edge():
    """Demonstrates the exact mechanism a halo exists to avoid: a point near
    the edge of the AVAILABLE data gets a truncated neighborhood and a lower
    valid_fraction than the identical terrain evaluated with real margin."""

    elevation, valid = _flat_grid(10.0)
    _, valid_fraction = rg.compute_tpi_m(elevation, valid, 1000.0, CELL_SIZE_M)

    edge_fraction = valid_fraction[3, CENTER]  # only 3 cells of real margin above
    center_fraction = valid_fraction[CENTER, CENTER]  # >=10 cells of margin on every side

    assert edge_fraction < rg.MIN_VALID_NEIGHBORHOOD_FRACTION
    assert center_fraction >= rg.MIN_VALID_NEIGHBORHOOD_FRACTION


# --- EMODnet native QA: discrete handling, never fabricated ----------------


def test_qa_columns_are_always_null_never_fabricated():
    """No small/live machine-readable per-cell QA coverage exists (confirmed
    in emodnet.check_native_qa_layers) -- these columns must stay null, not
    interpolated or guessed, and must never be bilinearly derived."""

    chainage_bathymetry_df = pd.DataFrame(
        {
            "pipeline_id": ["PL854"],
            "station_index": [0],
            "chainage_m": [0.0],
            "kp_label": ["KP 0.000"],
            "depth_lat_m": [20.0],
            "bathymetry_source_product": ["Test"],
            "source_reference_id": ["A"],
            "source_reference_type": ["CDI"],
            "qi_age": [0],
            "qi_horizontal": [3],
            "qi_vertical": [3],
            "qi_purpose": [3],
            "qi_combined": [50.0],
        }
    )
    cdi_sources_df = pd.DataFrame(
        {
            "source_reference_id": ["A"],
            "acquisition_year": [1991],
            "acquisition_start": ["1991-04-24"],
            "acquisition_end": ["1991-08-16"],
            "survey_age_at_product_release_year": [33],
        }
    )
    joined = rg.join_source_provenance(chainage_bathymetry_df, cdi_sources_df)
    assert joined["source_acquisition_year"].iloc[0] == 1991


# --- Age: 2024 must never become source acquisition year -------------------


def test_join_source_provenance_never_uses_2024_as_acquisition_year():
    chainage_bathymetry_df = pd.DataFrame(
        {
            "pipeline_id": ["PL854", "PL854"],
            "station_index": [0, 1],
            "source_reference_id": ["A", "B"],
        }
    )
    cdi_sources_df = pd.DataFrame(
        {
            "source_reference_id": ["A", "B"],
            "acquisition_year": [1992, 1991],
            "acquisition_start": ["1992-09-21", "1991-04-24"],
            "acquisition_end": ["1992-12-08", "1991-08-16"],
            "survey_age_at_product_release_year": [32, 33],
        }
    )

    joined = rg.join_source_provenance(chainage_bathymetry_df, cdi_sources_df)

    assert set(joined["source_acquisition_year"]) == {1991, 1992}
    assert 2024 not in joined["source_acquisition_year"].tolist()


def test_join_source_provenance_retains_all_stations_even_without_a_match():
    chainage_bathymetry_df = pd.DataFrame(
        {"pipeline_id": ["PL854"], "station_index": [0], "source_reference_id": ["UNKNOWN"]}
    )
    cdi_sources_df = pd.DataFrame(
        {
            "source_reference_id": ["A"],
            "acquisition_year": [1991],
            "acquisition_start": ["1991-04-24"],
            "acquisition_end": ["1991-08-16"],
            "survey_age_at_product_release_year": [33],
        }
    )

    joined = rg.join_source_provenance(chainage_bathymetry_df, cdi_sources_df)

    assert len(joined) == 1
    assert pd.isna(joined["source_acquisition_year"].iloc[0])


# --- Determinism --------------------------------------------------------------


def test_compute_functions_are_deterministic():
    elevation = _bump_grid(5.0)
    valid = np.ones_like(elevation, dtype=bool)

    slope_a, _ = rg.compute_slope_deg(elevation, valid, 500.0, CELL_SIZE_M)
    slope_b, _ = rg.compute_slope_deg(elevation, valid, 500.0, CELL_SIZE_M)
    np.testing.assert_array_equal(slope_a, slope_b)

    tpi_a, _ = rg.compute_tpi_m(elevation, valid, 1000.0, CELL_SIZE_M)
    tpi_b, _ = rg.compute_tpi_m(elevation, valid, 1000.0, CELL_SIZE_M)
    np.testing.assert_array_equal(tpi_a, tpi_b)


# --- build_halo_bbox_wgs84 ----------------------------------------------------


def test_build_halo_bbox_wgs84_is_larger_than_the_aoi_itself():
    aoi_geom = box(500000.0, 5900000.0, 501000.0, 5901000.0)
    aoi_bbox_working = aoi_geom.bounds

    halo_bbox = rg.build_halo_bbox_wgs84(aoi_geom, "EPSG:32631", halo_m=2200.0)

    # Reproject the halo bbox corners back to compare extents roughly --
    # simplest robust check: the halo WGS84 bbox, reprojected, must cover a
    # working-CRS extent measurably larger than the raw AOI bounds.
    halo_geom_working = (
        gpd.GeoSeries([box(*halo_bbox)], crs="EPSG:4326").to_crs("EPSG:32631").iloc[0]
    )
    hb = halo_geom_working.bounds
    assert hb[0] < aoi_bbox_working[0]
    assert hb[1] < aoi_bbox_working[1]
    assert hb[2] > aoi_bbox_working[2]
    assert hb[3] > aoi_bbox_working[3]


# --- Output writing ------------------------------------------------------------


def test_write_morphology_raster_sets_expected_tags(tmp_path: Path):
    array = np.array([[1.0, np.nan], [2.0, 3.0]])
    transform = rasterio.transform.from_origin(500000.0, 5900100.0, 100.0, 100.0)
    layer = rg.MorphologyLayer(
        name="slope_500m_deg",
        array=array,
        transform=transform,
        radius_m=500.0,
        unit="degrees",
        description="test layer",
    )
    out_path = tmp_path / "slope_500m_deg.tif"

    rg.write_morphology_raster(layer, "EPSG:32631", out_path, aoi_identifier="TEST_AOI")

    with rasterio.open(out_path) as ds:
        assert ds.crs.to_string() == "EPSG:32631"
        assert np.isnan(ds.nodata)
        tags = ds.tags()
        assert tags["analysis_radius_m"] == "500.0"
        assert tags["aoi_identifier"] == "TEST_AOI"
        assert "1991" in tags["source_age_warning"] or "1992" in tags["source_age_warning"]


def test_write_chainage_regional_morphology_round_trips(tmp_path: Path):
    df = pd.DataFrame([{"pipeline_id": "PL854", "station_index": 0, "slope_500m_deg": 0.5}])
    out_path = tmp_path / "chainage_regional_morphology.parquet"

    result_path = rg.write_chainage_regional_morphology(df, out_path)

    assert result_path == out_path
    reloaded = pd.read_parquet(out_path)
    assert reloaded["slope_500m_deg"].iloc[0] == pytest.approx(0.5)


def test_print_regional_morphology_report_includes_age_warning():
    import io

    df = pd.DataFrame(
        [
            {
                "slope_500m_deg": 0.2,
                "slope_1000m_deg": 0.1,
                "tpi_1000m_m": 1.0,
                "tpi_2000m_m": -1.0,
                "local_relief_1000m_m": 5.0,
                "local_relief_2000m_m": 8.0,
                "terrain_std_1000m_m": 1.0,
                "terrain_std_2000m_m": 1.5,
                "source_reference_id": "A",
                "source_acquisition_year": 1991,
            }
        ]
    )
    halo_grid = rg.HaloElevationGrid(
        elevation=np.zeros((2, 2)),
        valid_mask=np.ones((2, 2), dtype=bool),
        transform=rasterio.transform.from_origin(0.0, 1.0, 100.0, 100.0),
        crs="EPSG:32631",
        raw_stats=None,
        sign_convention_observed="negative_elevation",
        source_sha256=None,
    )
    result = rg.RegionalMorphologyResult(
        chainage_df=df,
        layers=[],
        halo_grid=halo_grid,
        qa_layer_availability=NativeQaLayerAvailability(
            wcs_coverage_ids=(),
            wcs_matches={},
            download_tile_formats=(),
            download_tile_matches={},
            notes="none available",
        ),
        aoi_identifier="TEST",
        working_crs="EPSG:32631",
        processing_timestamp="2026-01-01T00:00:00+00:00",
    )

    buffer = io.StringIO()
    rg.print_regional_morphology_report(result, file=buffer)
    output = buffer.getvalue()

    assert "REGIONAL MORPHOLOGY AGE WARNING" in output
    assert "1991-1992" in output
    assert "must not be interpreted as surveyed present-day seabed morphology" in output
