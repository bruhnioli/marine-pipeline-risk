"""Offline unit tests for marine_engine.metocean.evidence.

Uses small synthetic grids, hand-built xarray current/wave datasets, and
synthetic chainage geometries -- a handful of hand-placed support nodes and
tiny datasets with simple round-number values -- never the real PL854
route, real Copernicus Marine current/wave products, or any other real
project data -- and never touches the network.

The single most important behaviour under test throughout this file: PL854's
941 densely-spaced (25 m) chainage stations must collapse onto a much
smaller number of real Copernicus Marine model grid cells ("support nodes")
-- never one fabricated per-station time series (see `evidence.py`'s own
module docstring, Sections 4-5, 19-23).
"""

import io

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point

from marine_engine.metocean import current as current_module
from marine_engine.metocean import evidence
from marine_engine.preprocessing import chainage

WORKING_CRS = "EPSG:32631"


# --- shared helpers -----------------------------------------------------------


def _support_node_from_working_point(
    node_id: str, grid_i: int, grid_j: int, point_working: Point, crs: str
) -> evidence.SupportNode:
    """A SupportNode whose lon/lat are back-projected from a working-CRS point.

    Mirrors this repo's established `_feature_from_point_working` pattern
    (tests/test_sediment_evidence.py) -- build geometry with simple
    round-number coordinates in the working CRS, then back-project to
    WGS84 only because `SupportNode.longitude/latitude` requires it.
    """

    point_wgs84 = gpd.GeoSeries([point_working], crs=crs).to_crs("EPSG:4326").iloc[0]
    return evidence.SupportNode(
        node_id=node_id,
        grid_i=grid_i,
        grid_j=grid_j,
        longitude=float(point_wgs84.x),
        latitude=float(point_wgs84.y),
    )


def _make_route(length_m: float = 500.0) -> LineString:
    """A due-east route of a given length, entirely synthetic."""

    return LineString([(500000.0, 5900000.0), (500000.0 + length_m, 5900000.0)])


def _make_chainage_gdf(route: LineString, crs: str, interval_m: float = 100.0) -> gpd.GeoDataFrame:
    stations_result = chainage.compute_chainage_stations(route, interval_m)
    return chainage.build_canonical_chainage_gdf(
        pipeline_id="PL854-TEST",
        stations=stations_result.stations,
        interval_m=interval_m,
        chainage_origin_basis="source_geometry_start",
        working_crs=crs,
    )


def _make_node_mapping(n: int, node_id: str, distances_m: list[float]) -> pd.DataFrame:
    """A hand-built mapping frame shaped like `map_points_to_nearest_node`'s output."""

    return pd.DataFrame({"node_id": [node_id] * n, "distance_m": distances_m})


def _make_primary_current_dataset(
    *,
    times: pd.DatetimeIndex,
    depths: list[float],
    latitudes: list[float],
    longitudes: list[float],
    uo_values: np.ndarray,
    vo_values: np.ndarray,
) -> xr.Dataset:
    """A tiny synthetic 4D (time, depth, latitude, longitude) current dataset."""

    return xr.Dataset(
        {
            "uo": (("time", "depth", "latitude", "longitude"), uo_values),
            "vo": (("time", "depth", "latitude", "longitude"), vo_values),
        },
        coords={
            "time": times,
            "depth": depths,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )


def _make_long_term_current_dataset(
    *,
    times: pd.DatetimeIndex,
    latitudes: list[float],
    longitudes: list[float],
    uo_values: np.ndarray,
    vo_values: np.ndarray,
) -> xr.Dataset:
    """A tiny synthetic 2D (time, latitude, longitude) surface current dataset."""

    return xr.Dataset(
        {
            "uo": (("time", "latitude", "longitude"), uo_values),
            "vo": (("time", "latitude", "longitude"), vo_values),
        },
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )


def _make_wave_dataset(
    *,
    times: pd.DatetimeIndex,
    latitudes: list[float],
    longitudes: list[float],
    variables: dict[str, np.ndarray],
) -> xr.Dataset:
    """A tiny synthetic 2D (time, latitude, longitude) wave dataset.

    `variables` maps a wave variable name (e.g. "VHM0") to an array shaped
    (time, latitude, longitude); any optional variable `normalize_wave`
    knows about may simply be omitted to simulate an absent product field.
    """

    data_vars = {
        name: (("time", "latitude", "longitude"), values) for name, values in variables.items()
    }
    return xr.Dataset(
        data_vars,
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )


def _make_nontrivial_metocean_kwargs(chainage_gdf: gpd.GeoDataFrame) -> dict:
    """One real support node per product, every station mapped onto it.

    A hand-built (not pipeline-derived) set of mapping/stats inputs for
    `build_chainage_metocean_evidence`, exercising its merges with actual
    (non-null) data rather than the all-empty acquisition-failure case.
    """

    n = len(chainage_gdf)
    current_mapping = _make_node_mapping(n, "CUR_A", [10.0 * (i + 1) for i in range(n)])
    current_stats = pd.DataFrame(
        {
            "current_node_id": ["CUR_A"],
            "representative_sample_depth_m": [12.0],
            "current_speed_mean_m_s": [0.30],
            "current_speed_median_m_s": [0.28],
            "current_speed_p90_m_s": [0.55],
            "current_speed_p95_m_s": [0.60],
            "current_speed_p99_m_s": [0.70],
            "current_speed_max_m_s": [0.75],
            "valid_hour_count": [720],
        }
    )
    long_term_mapping = _make_node_mapping(n, "LT_A", [50.0 * (i + 1) for i in range(n)])
    long_term_stats = pd.DataFrame(
        {
            "current_lt_node_id": ["LT_A"],
            "surface_current_speed_mean_m_s": [0.20],
            "surface_current_speed_p95_m_s": [0.45],
            "surface_current_speed_p99_m_s": [0.55],
            "surface_current_speed_max_m_s": [0.60],
        }
    )
    wave_mapping = _make_node_mapping(n, "WAVE_A", [100.0 * (i + 1) for i in range(n)])
    wave_stats = pd.DataFrame(
        {
            "wave_node_id": ["WAVE_A"],
            "hs_mean_m": [1.1],
            "hs_median_m": [1.0],
            "hs_p90_m": [1.8],
            "hs_p95_m": [2.0],
            "hs_p99_m": [2.3],
            "hs_max_m": [2.5],
            "tp_median_s": [8.0],
            "tp_p95_s": [9.5],
            "valid_3hour_count": [2000],
        }
    )
    return {
        "chainage_gdf": chainage_gdf,
        "canonical_depth_df": None,
        "current_mapping": current_mapping,
        "current_stats": current_stats,
        "current_node_bathymetry": {"CUR_A": 30.0},
        "long_term_mapping": long_term_mapping,
        "long_term_stats": long_term_stats,
        "wave_mapping": wave_mapping,
        "wave_stats": wave_stats,
        "wave_node_bathymetry": {"WAVE_A": 28.0},
    }


# --- identify_wet_grid_cells ---------------------------------------------------


def test_identify_wet_grid_cells_only_returns_true_cells():
    longitude = np.array([1.0, 1.1, 1.2])
    latitude = np.array([53.0, 53.1])
    wet_mask_2d = np.array(
        [
            [True, False, True],
            [False, True, False],
        ]
    )

    nodes = evidence.identify_wet_grid_cells(longitude, latitude, wet_mask_2d, "CUR")

    assert len(nodes) == int(np.sum(wet_mask_2d))
    assert len(nodes) == 3
    for node in nodes:
        assert node.longitude == pytest.approx(longitude[node.grid_i])
        assert node.latitude == pytest.approx(latitude[node.grid_j])
        assert node.node_id == f"CUR_{node.grid_j:04d}_{node.grid_i:04d}"


# --- map_points_to_nearest_node -------------------------------------------------


def test_map_points_to_nearest_node_collapses_many_points_onto_few_nodes():
    """The core dense-chainage -> sparse-real-node proof (Sections 4-5, 19)."""

    route_x_start = 500000.0
    route_length = 500.0
    n_points = 50
    xs = np.linspace(route_x_start, route_x_start + route_length, n_points)
    points_working = gpd.GeoSeries([Point(x, 5900000.0) for x in xs], crs=WORKING_CRS)

    # Only 4 coarse support nodes spread along the same line (simulating a
    # ~1.5 km real model grid against 941 dense 25 m chainage stations).
    node_xs = [
        route_x_start,
        route_x_start + route_length / 3.0,
        route_x_start + 2.0 * route_length / 3.0,
        route_x_start + route_length,
    ]
    nodes = [
        _support_node_from_working_point(f"NODE_{i}", i, 0, Point(x, 5900000.0), WORKING_CRS)
        for i, x in enumerate(node_xs)
    ]

    result = evidence.map_points_to_nearest_node(points_working, nodes, WORKING_CRS)

    assert len(result) == n_points
    unique_nodes_used = set(result["node_id"])
    # <=4 is trivially guaranteed by only supplying 4 nodes; the meaningful
    # claims are that far fewer than 50 labels are used (real collapsing,
    # not one row silently dropped per node) and more than 1 (proving the
    # nearest-neighbour search actually discriminates between nodes rather
    # than e.g. always returning the same index regardless of distance).
    assert len(unique_nodes_used) <= 4
    assert 1 < len(unique_nodes_used) < n_points
    # The leftmost/rightmost points are unambiguously nearest to the
    # leftmost/rightmost nodes -- a direct, hand-verifiable spatial check.
    assert result.iloc[0]["node_id"] == "NODE_0"
    assert result.iloc[-1]["node_id"] == "NODE_3"


def test_map_points_to_nearest_node_distance_is_correct():
    node_a = _support_node_from_working_point(
        "NODE_A", 0, 0, Point(500000.0, 5900000.0), WORKING_CRS
    )
    node_b = _support_node_from_working_point(
        "NODE_B", 1, 0, Point(500300.0, 5900000.0), WORKING_CRS
    )
    # 25 m due north of NODE_A, far closer to A than to B (300 m away).
    query_point = gpd.GeoSeries([Point(500000.0, 5900025.0)], crs=WORKING_CRS)

    result = evidence.map_points_to_nearest_node(query_point, [node_a, node_b], WORKING_CRS)

    assert result.iloc[0]["node_id"] == "NODE_A"
    assert result.iloc[0]["distance_m"] == pytest.approx(25.0, abs=1e-2)


# --- build_support_node_table ---------------------------------------------------


def test_build_support_node_table_only_includes_actually_used_nodes():
    nodes = [
        evidence.SupportNode(
            node_id=f"N{i}", grid_i=i, grid_j=0, longitude=1.0 + i * 0.01, latitude=53.0
        )
        for i in range(5)
    ]
    assigned_node_ids = pd.Series(["N0", "N2", "N0", "N2"])
    assigned_distances_m = pd.Series([10.0, 20.0, 30.0, 40.0])

    result = evidence.build_support_node_table(
        nodes,
        assigned_node_ids,
        assigned_distances_m,
        source_product="TEST_PRODUCT",
        source_dataset="TEST_DATASET",
        evidence_role="TEST_ROLE",
    )

    assert len(result) == 2
    assert set(result["node_id"]) == {"N0", "N2"}


def test_build_support_node_table_station_counts_and_distance_range_correct():
    node = evidence.SupportNode(node_id="NODE_A", grid_i=0, grid_j=0, longitude=1.0, latitude=53.0)
    assigned_node_ids = pd.Series(["NODE_A", "NODE_A", "NODE_A"])
    assigned_distances_m = pd.Series([100.0, 250.0, 400.0])

    result = evidence.build_support_node_table(
        [node],
        assigned_node_ids,
        assigned_distances_m,
        source_product="TEST_PRODUCT",
        source_dataset="TEST_DATASET",
        evidence_role="TEST_ROLE",
    )

    row = result.iloc[0]
    assert row["station_count_assigned"] == 3
    assert row["min_chainage_distance_to_node_m"] == pytest.approx(100.0)
    assert row["max_chainage_distance_to_node_m"] == pytest.approx(400.0)


# --- normalize_primary_current ---------------------------------------------------


def test_normalize_primary_current_one_row_per_node_per_timestamp():
    times = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    depths = [0.0, 5.0, 10.0]
    latitudes = [53.30, 53.31]
    longitudes = [1.70, 1.71]
    uo_values = np.full((3, 3, 2, 2), 0.1)
    vo_values = np.full((3, 3, 2, 2), 0.2)
    ds = _make_primary_current_dataset(
        times=times,
        depths=depths,
        latitudes=latitudes,
        longitudes=longitudes,
        uo_values=uo_values,
        vo_values=vo_values,
    )
    node0 = evidence.SupportNode(
        node_id="CUR_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )
    node1 = evidence.SupportNode(
        node_id="CUR_0001_0001", grid_i=1, grid_j=1, longitude=1.71, latitude=53.31
    )
    nodes = [node0, node1]

    result = evidence.normalize_primary_current(
        ds,
        nodes=nodes,
        model_bathymetry_by_node_id={node0.node_id: 25.0, node1.node_id: 30.0},
        source_dataset="TEST_DATASET",
        evidence_role="PRIMARY_CURRENT",
    )

    # Exactly nodes * timestamps rows -- never one row per chainage station.
    assert len(result) == len(nodes) * len(times)
    assert set(result["current_node_id"]) == {node0.node_id, node1.node_id}


def test_normalize_primary_current_selects_deepest_valid_level_per_row():
    times = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
    depths = [0.0, 5.0, 10.0]
    latitudes = [53.30]
    longitudes = [1.70]
    uo_values = np.zeros((1, 3, 1, 1))
    vo_values = np.zeros((1, 3, 1, 1))
    uo_values[0, 0, 0, 0] = 1.0  # depth 0.0 -- valid, but shallower
    vo_values[0, 0, 0, 0] = 0.0
    uo_values[0, 1, 0, 0] = 3.0  # depth 5.0 -- valid, and the deepest valid one
    vo_values[0, 1, 0, 0] = 4.0
    uo_values[0, 2, 0, 0] = np.nan  # depth 10.0 -- invalid (below seafloor)
    vo_values[0, 2, 0, 0] = np.nan
    ds = _make_primary_current_dataset(
        times=times,
        depths=depths,
        latitudes=latitudes,
        longitudes=longitudes,
        uo_values=uo_values,
        vo_values=vo_values,
    )
    node = evidence.SupportNode(
        node_id="CUR_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )

    result = evidence.normalize_primary_current(
        ds,
        nodes=[node],
        model_bathymetry_by_node_id={},
        source_dataset="TEST_DATASET",
        evidence_role="PRIMARY_CURRENT",
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["current_sample_depth_m"] == pytest.approx(5.0)
    assert row["depth_level_index"] == 1
    expected_speed = float(
        current_module.compute_current_speed_m_s(np.array([3.0]), np.array([4.0]))[0]
    )
    expected_direction = float(
        current_module.compute_current_direction_to_deg(np.array([3.0]), np.array([4.0]))[0]
    )
    assert row["current_speed_m_s"] == pytest.approx(expected_speed)
    assert row["current_direction_to_deg"] == pytest.approx(expected_direction)


def test_normalize_primary_current_height_above_model_bed():
    times = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
    depths = [0.0, 10.0, 25.0]
    latitudes = [53.30]
    longitudes = [1.70]
    uo_values = np.zeros((1, 3, 1, 1))
    vo_values = np.zeros((1, 3, 1, 1))
    uo_values[0, 0, 0, 0], vo_values[0, 0, 0, 0] = 0.5, 0.0
    uo_values[0, 1, 0, 0], vo_values[0, 1, 0, 0] = 0.3, 0.0
    uo_values[0, 2, 0, 0], vo_values[0, 2, 0, 0] = 0.2, 0.0  # all 3 depths valid
    ds = _make_primary_current_dataset(
        times=times,
        depths=depths,
        latitudes=latitudes,
        longitudes=longitudes,
        uo_values=uo_values,
        vo_values=vo_values,
    )
    node = evidence.SupportNode(
        node_id="CUR_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )

    result = evidence.normalize_primary_current(
        ds,
        nodes=[node],
        model_bathymetry_by_node_id={node.node_id: 27.0},
        source_dataset="TEST_DATASET",
        evidence_role="PRIMARY_CURRENT",
    )

    row = result.iloc[0]
    assert row["current_sample_depth_m"] == pytest.approx(25.0)  # the deepest valid level
    assert row["depth_level_index"] == 2
    assert row["height_above_model_bed_m"] == pytest.approx(2.0)
    assert bool(row["height_above_model_bed_valid"]) is True


def test_normalize_primary_current_column_schema_matches_constant():
    times = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
    depths = [0.0, 5.0]
    latitudes = [53.30]
    longitudes = [1.70]
    uo_values = np.full((1, 2, 1, 1), 0.1)
    vo_values = np.full((1, 2, 1, 1), 0.1)
    ds = _make_primary_current_dataset(
        times=times,
        depths=depths,
        latitudes=latitudes,
        longitudes=longitudes,
        uo_values=uo_values,
        vo_values=vo_values,
    )
    node = evidence.SupportNode(
        node_id="CUR_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )

    result = evidence.normalize_primary_current(
        ds,
        nodes=[node],
        model_bathymetry_by_node_id={},
        source_dataset="TEST_DATASET",
        evidence_role="PRIMARY_CURRENT",
    )

    assert list(result.columns) == list(evidence.PRIMARY_CURRENT_COLUMNS)


def test_normalize_primary_current_naming_never_calls_it_bottom_or_seabed():
    """Schema/naming regression test (Section 8/19's explicit ticket requirement)."""

    forbidden_substrings = ("bottom_current", "seabed_current", "current_at_seabed")
    columns_lower = [c.lower() for c in evidence.PRIMARY_CURRENT_COLUMNS]
    for forbidden in forbidden_substrings:
        assert not any(forbidden in column for column in columns_lower)


# --- normalize_long_term_surface_current -----------------------------------------


def test_normalize_long_term_surface_current_one_row_per_node_per_timestamp():
    times = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    latitudes = [53.30, 53.31]
    longitudes = [1.70, 1.71]
    uo_values = np.full((3, 2, 2), 0.1)
    vo_values = np.full((3, 2, 2), 0.2)
    ds = _make_long_term_current_dataset(
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        uo_values=uo_values,
        vo_values=vo_values,
    )
    node0 = evidence.SupportNode(
        node_id="LT_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )
    node1 = evidence.SupportNode(
        node_id="LT_0001_0001", grid_i=1, grid_j=1, longitude=1.71, latitude=53.31
    )

    result = evidence.normalize_long_term_surface_current(
        ds,
        nodes=[node0, node1],
        source_dataset="TEST_LT_DATASET",
        evidence_role="LONG_TERM_SURFACE_CURRENT_CONTEXT",
    )

    assert len(result) == 2 * 3
    assert set(result["current_lt_node_id"]) == {node0.node_id, node1.node_id}


def test_normalize_long_term_surface_current_marks_invalid_when_either_component_nan():
    times = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    latitudes = [53.30]
    longitudes = [1.70]
    uo_values = np.zeros((2, 1, 1))
    vo_values = np.zeros((2, 1, 1))
    uo_values[0, 0, 0], vo_values[0, 0, 0] = 3.0, 4.0  # valid timestep
    uo_values[1, 0, 0], vo_values[1, 0, 0] = np.nan, 5.0  # uo missing -> invalid
    ds = _make_long_term_current_dataset(
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        uo_values=uo_values,
        vo_values=vo_values,
    )
    node = evidence.SupportNode(
        node_id="LT_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )

    result = evidence.normalize_long_term_surface_current(
        ds,
        nodes=[node],
        source_dataset="TEST_LT_DATASET",
        evidence_role="LONG_TERM_SURFACE_CURRENT_CONTEXT",
    )

    valid_row = result.iloc[0]
    assert valid_row["surface_current_speed_m_s"] == pytest.approx(5.0)

    invalid_row = result.iloc[1]
    # Column is float64 (mixed with the valid row's real speed), so a masked
    # value lands as NaN rather than a literal `None` object -- pd.isna
    # covers both storage forms without depending on that pandas coercion.
    assert pd.isna(invalid_row["uo_surface_m_s"])
    assert pd.isna(invalid_row["vo_surface_m_s"])
    assert pd.isna(invalid_row["surface_current_speed_m_s"])
    assert pd.isna(invalid_row["surface_current_direction_to_deg"])


# --- normalize_wave ---------------------------------------------------------------


def test_normalize_wave_preserves_from_direction_and_derives_to_direction():
    times = pd.date_range("2025-01-01", periods=2, freq="3h", tz="UTC")
    latitudes = [53.30]
    longitudes = [1.70]
    shape = (2, 1, 1)
    variables = {
        "VHM0": np.full(shape, 1.5),
        "VTPK": np.full(shape, 10.0),
        "VTM02": np.full(shape, 8.0),
        "VTM10": np.full(shape, 9.0),
        "VMDR": np.full(shape, 45.0),
        "VSDX": np.full(shape, 0.1),
        "VSDY": np.full(shape, 0.2),
    }
    ds = _make_wave_dataset(
        times=times, latitudes=latitudes, longitudes=longitudes, variables=variables
    )
    node = evidence.SupportNode(
        node_id="WAVE_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )

    result = evidence.normalize_wave(ds, nodes=[node], source_dataset="TEST_WAVE_DATASET")

    # The FROM direction is preserved, never overwritten by the derived TO.
    assert np.allclose(result["wave_mean_direction_from_deg"].to_numpy(dtype=float), 45.0)
    assert np.allclose(result["wave_mean_direction_to_deg"].to_numpy(dtype=float), 225.0)


def test_normalize_wave_handles_missing_optional_variable_gracefully():
    times = pd.date_range("2025-01-01", periods=2, freq="3h", tz="UTC")
    latitudes = [53.30]
    longitudes = [1.70]
    shape = (2, 1, 1)
    variables = {
        "VHM0": np.full(shape, 1.0),
        "VTPK": np.full(shape, 10.0),
        "VTM02": np.full(shape, 8.0),
        "VTM10": np.full(shape, 9.0),
        "VMDR": np.full(shape, 90.0),
        # VSDX / VSDY deliberately absent from this product.
    }
    ds = _make_wave_dataset(
        times=times, latitudes=latitudes, longitudes=longitudes, variables=variables
    )
    node = evidence.SupportNode(
        node_id="WAVE_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )

    result = evidence.normalize_wave(ds, nodes=[node], source_dataset="TEST_WAVE_DATASET")

    assert "stokes_u_m_s" in result.columns
    assert "stokes_v_m_s" in result.columns
    assert result["stokes_u_m_s"].isna().all()
    assert result["stokes_v_m_s"].isna().all()


def test_normalize_wave_flags_negative_hs_as_invalid():
    times = pd.date_range("2025-01-01", periods=2, freq="3h", tz="UTC")
    latitudes = [53.30]
    longitudes = [1.70]
    shape = (2, 1, 1)
    hs = np.full(shape, 1.2)
    hs[0, 0, 0] = -1.0  # first timestep: a physically invalid negative Hs
    variables = {
        "VHM0": hs,
        "VTPK": np.full(shape, 10.0),
        "VTM02": np.full(shape, 8.0),
        "VTM10": np.full(shape, 9.0),
        "VMDR": np.full(shape, 90.0),
    }
    ds = _make_wave_dataset(
        times=times, latitudes=latitudes, longitudes=longitudes, variables=variables
    )
    node = evidence.SupportNode(
        node_id="WAVE_0000_0000", grid_i=0, grid_j=0, longitude=1.70, latitude=53.30
    )

    result = evidence.normalize_wave(ds, nodes=[node], source_dataset="TEST_WAVE_DATASET")

    negative_row = result.iloc[0]
    # hs_m mixes a real float (row 1) with a masked value (row 0), so the
    # masked cell lands as NaN, not a literal None -- pd.isna is the robust
    # check regardless of that pandas storage coercion.
    assert pd.isna(negative_row["hs_m"])
    assert bool(negative_row["hs_valid"]) is False

    positive_row = result.iloc[1]
    assert positive_row["hs_m"] == pytest.approx(1.2)
    assert bool(positive_row["hs_valid"]) is True


# --- compute_current_node_statistics -----------------------------------------------


def test_compute_current_node_statistics_percentiles_and_completeness():
    hours = [0, 1, 2, 4, 5, 7, 8, 9]  # hours 3 and 6 are deliberately missing
    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    times = [base_time + pd.Timedelta(hours=h) for h in hours]
    speeds = [0.10, 0.52, 0.33, 0.81, 0.24, 0.95, 0.44, 0.67]
    depths = [5.0, 5.0, 5.0, 10.0, 5.0, 5.0, 10.0, 5.0]  # mode is 5.0 (6 vs 2)

    primary_current_df = pd.DataFrame(
        {
            "current_node_id": ["NODE_A"] * len(hours),
            "time_utc": times,
            "current_speed_m_s": speeds,
            "current_sample_depth_m": depths,
        }
    )

    result = evidence.compute_current_node_statistics(primary_current_df)

    assert len(result) == 1
    row = result.iloc[0]
    expected = pd.Series(speeds)
    # Span is hour0..hour9 = 10 expected hours; only 8 hours are present.
    assert row["expected_hourly_count"] == 10
    assert row["valid_hour_count"] == 8
    assert row["completeness_pct"] == pytest.approx(80.0)
    assert row["current_speed_mean_m_s"] == pytest.approx(float(expected.mean()))
    assert row["current_speed_median_m_s"] == pytest.approx(float(expected.median()))
    assert row["current_speed_p90_m_s"] == pytest.approx(float(expected.quantile(0.90)))
    assert row["current_speed_p95_m_s"] == pytest.approx(float(expected.quantile(0.95)))
    assert row["current_speed_p99_m_s"] == pytest.approx(float(expected.quantile(0.99)))
    assert row["current_speed_max_m_s"] == pytest.approx(float(expected.max()))
    assert row["representative_sample_depth_m"] == pytest.approx(5.0)


# --- compute_wave_node_statistics ---------------------------------------------------


def test_compute_wave_node_statistics_hs_and_tp_correct():
    steps = [0, 3, 6, 12, 15]  # 3-hour steps; step 9 is deliberately missing
    base_time = pd.Timestamp("2025-02-01", tz="UTC")
    times = [base_time + pd.Timedelta(hours=h) for h in steps]
    hs_values = [1.0, 2.5, 0.5, 3.0, 1.5]
    tp_values = [8.0, 9.5, 7.0, 10.0, 8.5]

    wave_df = pd.DataFrame(
        {
            "wave_node_id": ["WAVE_A"] * len(steps),
            "time_utc": times,
            "hs_m": hs_values,
            "tp_s": tp_values,
        }
    )

    result = evidence.compute_wave_node_statistics(wave_df)

    assert len(result) == 1
    row = result.iloc[0]
    expected_hs = pd.Series(hs_values)
    expected_tp = pd.Series(tp_values)
    # Span is 0..15h at 3h steps = 6 expected steps (0,3,6,9,12,15); 5 present.
    assert row["expected_3hour_count"] == 6
    assert row["valid_3hour_count"] == 5
    assert row["completeness_pct"] == pytest.approx(100.0 * 5.0 / 6.0)
    assert row["hs_mean_m"] == pytest.approx(float(expected_hs.mean()))
    assert row["hs_median_m"] == pytest.approx(float(expected_hs.median()))
    assert row["hs_p90_m"] == pytest.approx(float(expected_hs.quantile(0.90)))
    assert row["hs_p95_m"] == pytest.approx(float(expected_hs.quantile(0.95)))
    assert row["hs_p99_m"] == pytest.approx(float(expected_hs.quantile(0.99)))
    assert row["hs_max_m"] == pytest.approx(float(expected_hs.max()))
    assert row["tp_median_s"] == pytest.approx(float(expected_tp.median()))
    assert row["tp_p95_s"] == pytest.approx(float(expected_tp.quantile(0.95)))


# --- compute_annual_max_hs -----------------------------------------------------------


def test_compute_annual_max_hs_groups_by_year():
    times = [
        pd.Timestamp("2020-03-01", tz="UTC"),
        pd.Timestamp("2020-06-01", tz="UTC"),
        pd.Timestamp("2020-09-01", tz="UTC"),
        pd.Timestamp("2021-01-15", tz="UTC"),
        pd.Timestamp("2021-05-01", tz="UTC"),
        pd.Timestamp("2021-11-01", tz="UTC"),
    ]
    hs_values = [1.0, 2.0, 1.5, 0.5, 3.0, 2.5]  # 2020 max=2.0, 2021 max=3.0
    wave_df = pd.DataFrame(
        {
            "wave_node_id": ["WAVE_A"] * len(times),
            "time_utc": times,
            "hs_m": hs_values,
        }
    )

    result = evidence.compute_annual_max_hs(wave_df)

    assert len(result) == 2
    by_year = result.set_index("year")["annual_max_hs_m"]
    assert by_year.loc[2020] == pytest.approx(2.0)
    assert by_year.loc[2021] == pytest.approx(3.0)


# --- compute_short_window_surface_context_ratio -----------------------------------


def test_compute_short_window_surface_context_ratio_is_descriptive_only():
    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    hours = list(range(24))
    speeds = [0.1 if 10 <= h <= 14 else (1.0 + 0.05 * h) for h in hours]
    times = [base_time + pd.Timedelta(hours=h) for h in hours]

    long_term_df = pd.DataFrame(
        {
            "current_lt_node_id": ["LT_A"] * len(hours),
            "time_utc": times,
            "surface_current_speed_m_s": speeds,
        }
    )

    primary_current_start = base_time + pd.Timedelta(hours=10)
    primary_current_end = base_time + pd.Timedelta(hours=14)

    result = evidence.compute_short_window_surface_context_ratio(
        long_term_df, primary_current_start, primary_current_end
    )

    assert len(result) == 1
    row = result.iloc[0]

    full = pd.Series(speeds)
    overlap = pd.Series([0.1] * 5)  # hours 10..14 inclusive
    expected_p95_ratio = float(overlap.quantile(0.95) / full.quantile(0.95))
    expected_p99_ratio = float(overlap.quantile(0.99) / full.quantile(0.99))

    assert row["short_window_surface_context_ratio_p95"] == pytest.approx(expected_p95_ratio)
    assert row["short_window_surface_context_ratio_p99"] == pytest.approx(expected_p99_ratio)
    # The calm overlap window is materially different from the busier full
    # period -- proving this reflects a real difference, not a no-op ratio.
    assert row["short_window_surface_context_ratio_p95"] != pytest.approx(1.0)


def test_compute_short_window_surface_context_ratio_null_when_no_overlap():
    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    times = [base_time + pd.Timedelta(hours=h) for h in range(5)]
    speeds = [0.5, 0.6, 0.4, 0.7, 0.55]

    long_term_df = pd.DataFrame(
        {
            "current_lt_node_id": ["LT_A"] * len(times),
            "time_utc": times,
            "surface_current_speed_m_s": speeds,
        }
    )

    # Completely outside the long_term_df's own time range.
    primary_current_start = pd.Timestamp("2030-01-01", tz="UTC")
    primary_current_end = pd.Timestamp("2030-01-02", tz="UTC")

    result = evidence.compute_short_window_surface_context_ratio(
        long_term_df, primary_current_start, primary_current_end
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["short_window_surface_context_ratio_p95"] is None
    assert row["short_window_surface_context_ratio_p99"] is None


# --- build_chainage_metocean_evidence -----------------------------------------------


def test_build_chainage_metocean_evidence_retains_every_station_regardless_of_match():
    route = _make_route(900.0)
    chainage_gdf = _make_chainage_gdf(route, WORKING_CRS, interval_m=100.0)
    assert len(chainage_gdf) == 10  # chainage 0, 100, ..., 900

    # Zero support nodes for every product -- a total acquisition failure.
    empty_mapping = evidence.map_points_to_nearest_node(chainage_gdf.geometry, [], WORKING_CRS)
    empty_current_stats = evidence.compute_current_node_statistics(
        pd.DataFrame(columns=list(evidence.PRIMARY_CURRENT_COLUMNS))
    )
    empty_long_term_stats = evidence.compute_long_term_surface_current_statistics(
        pd.DataFrame(columns=list(evidence.LONG_TERM_SURFACE_CURRENT_COLUMNS))
    )
    empty_wave_stats = evidence.compute_wave_node_statistics(
        pd.DataFrame(columns=list(evidence.WAVE_COLUMNS))
    )

    result = evidence.build_chainage_metocean_evidence(
        chainage_gdf=chainage_gdf,
        canonical_depth_df=None,
        current_mapping=empty_mapping,
        current_stats=empty_current_stats,
        current_node_bathymetry={},
        long_term_mapping=empty_mapping,
        long_term_stats=empty_long_term_stats,
        wave_mapping=empty_mapping,
        wave_stats=empty_wave_stats,
        wave_node_bathymetry={},
    )

    assert len(result) == 10
    assert result["current_node_id"].isna().all()
    assert result["current_lt_node_id"].isna().all()
    assert result["wave_node_id"].isna().all()
    assert result["current_speed_mean_m_s"].isna().all()
    assert result["surface_current_speed_mean_m_s"].isna().all()
    assert result["hs_mean_m"].isna().all()


def test_build_chainage_metocean_evidence_column_schema_matches_constant():
    route = _make_route(200.0)
    chainage_gdf = _make_chainage_gdf(route, WORKING_CRS, interval_m=100.0)
    kwargs = _make_nontrivial_metocean_kwargs(chainage_gdf)

    result = evidence.build_chainage_metocean_evidence(**kwargs)

    assert list(result.columns) == list(evidence.CHAINAGE_METOCEAN_COLUMNS)
    assert len(result) == len(chainage_gdf)
    assert (result["current_node_id"] == "CUR_A").all()


def test_build_chainage_metocean_evidence_joins_canonical_depth_by_station_index():
    route = _make_route(200.0)
    chainage_gdf = _make_chainage_gdf(route, WORKING_CRS, interval_m=100.0)
    assert list(chainage_gdf["station_index"]) == [0, 1, 2]

    # Deliberately scrambled relative to station order, to prove the join
    # happens by station_index value, never by row position.
    canonical_depth_df = pd.DataFrame(
        {"station_index": [2, 0, 1], "depth_lat_m": [32.5, 10.5, 21.5]}
    )

    empty_mapping = evidence.map_points_to_nearest_node(chainage_gdf.geometry, [], WORKING_CRS)
    empty_current_stats = evidence.compute_current_node_statistics(
        pd.DataFrame(columns=list(evidence.PRIMARY_CURRENT_COLUMNS))
    )
    empty_long_term_stats = evidence.compute_long_term_surface_current_statistics(
        pd.DataFrame(columns=list(evidence.LONG_TERM_SURFACE_CURRENT_COLUMNS))
    )
    empty_wave_stats = evidence.compute_wave_node_statistics(
        pd.DataFrame(columns=list(evidence.WAVE_COLUMNS))
    )

    result = evidence.build_chainage_metocean_evidence(
        chainage_gdf=chainage_gdf,
        canonical_depth_df=canonical_depth_df,
        current_mapping=empty_mapping,
        current_stats=empty_current_stats,
        current_node_bathymetry={},
        long_term_mapping=empty_mapping,
        long_term_stats=empty_long_term_stats,
        wave_mapping=empty_mapping,
        wave_stats=empty_wave_stats,
        wave_node_bathymetry={},
    )

    by_station = result.set_index("station_index")["depth_lat_m"]
    assert by_station.loc[0] == pytest.approx(10.5)
    assert by_station.loc[1] == pytest.approx(21.5)
    assert by_station.loc[2] == pytest.approx(32.5)


def test_build_chainage_metocean_evidence_deterministic():
    route = _make_route(200.0)
    chainage_gdf = _make_chainage_gdf(route, WORKING_CRS, interval_m=100.0)
    kwargs = _make_nontrivial_metocean_kwargs(chainage_gdf)

    first = evidence.build_chainage_metocean_evidence(**kwargs)
    second = evidence.build_chainage_metocean_evidence(**kwargs)

    pd.testing.assert_frame_equal(first, second)


# --- print_metocean_evidence_report (smoke) -----------------------------------------


def test_print_metocean_evidence_report_smoke_does_not_crash_on_empty_inputs():
    buffer = io.StringIO()

    evidence.print_metocean_evidence_report(
        primary_current_stats=pd.DataFrame(),
        long_term_stats=pd.DataFrame(),
        wave_stats=pd.DataFrame(),
        short_window_ratios=pd.DataFrame(),
        primary_current_actual_start=None,
        primary_current_actual_end=None,
        long_term_actual_start=None,
        long_term_actual_end=None,
        wave_actual_start=None,
        wave_actual_end=None,
        file=buffer,
    )

    output = buffer.getvalue()
    assert "PL854 Metocean Forcing Evidence Base" in output
