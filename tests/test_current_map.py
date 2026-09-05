"""Offline unit tests for marine_engine.metocean.current_map (MAR-010).

Small synthetic routes/chainage tables only -- never the real PL854 route,
never network access. Uses matplotlib's Agg backend (set by the module
itself) so PNG rendering is deterministic and headless-safe in CI.
"""

from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine.metocean import current_map

WORKING_CRS = "EPSG:32631"


def _attrs(**overrides) -> current_map.NodeReferenceAttributes:
    defaults = {
        "model_bathymetry_m": 25.0,
        "reference_height_m": 5.0,
        "speed_mean_m_s": 0.3,
        "speed_p95_m_s": 0.6,
        "speed_p99_m_s": 0.7,
        "speed_max_m_s": 0.8,
        "sensitivity_p95_min_m_s": 0.2,
        "sensitivity_p95_max_m_s": 0.9,
        "sensitivity_p95_width_m_s": 0.7,
    }
    defaults.update(overrides)
    return current_map.NodeReferenceAttributes(**defaults)


# --- build_current_reference_segments (Section 8, 17-K/L/M) ------------------------


def test_many_stations_sharing_one_node_dissolve_into_one_section():
    """17-K: many chainage stations sharing one node -> exactly one contiguous section."""

    route = LineString([(0.0, 0.0), (1000.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 100.0, 200.0, 300.0, 1000.0],
            "current_node_id": ["A"] * 5,
            "current_node_distance_m": [50.0, 60.0, 55.0, 65.0, 70.0],
        }
    )

    segments = current_map.build_current_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_current_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 1
    row = segments.iloc[0]
    assert row["current_node_id"] == "A"
    assert row["start_chainage_m"] == pytest.approx(0.0)
    assert row["end_chainage_m"] == pytest.approx(1000.0)
    assert row["current_node_distance_min_m"] == pytest.approx(50.0)
    assert row["current_node_distance_median_m"] == pytest.approx(60.0)
    assert row["current_node_distance_max_m"] == pytest.approx(70.0)


def test_node_change_creates_a_segment_boundary():
    """17-L: a node change creates a segment boundary at the midpoint between runs."""

    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 25.0, 50.0, 75.0, 100.0],
            "current_node_id": ["A", "A", "A", "B", "B"],
            "current_node_distance_m": [10.0, 10.0, 10.0, 20.0, 20.0],
        }
    )

    segments = current_map.build_current_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_current_df=chainage_df,
        node_attributes_by_id={"A": _attrs(), "B": _attrs(model_bathymetry_m=30.0)},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 2
    boundary = (50.0 + 75.0) / 2.0
    assert segments.iloc[0]["current_node_id"] == "A"
    assert segments.iloc[0]["start_chainage_m"] == pytest.approx(0.0)
    assert segments.iloc[0]["end_chainage_m"] == pytest.approx(boundary)
    assert segments.iloc[1]["current_node_id"] == "B"
    assert segments.iloc[1]["start_chainage_m"] == pytest.approx(boundary)
    assert segments.iloc[1]["end_chainage_m"] == pytest.approx(100.0)
    # Sections tile the route with no gap/overlap.
    assert segments.iloc[0]["end_chainage_m"] == segments.iloc[1]["start_chainage_m"]


def test_every_station_consumed_exactly_once_across_multiple_runs():
    route = LineString([(0.0, 0.0), (400.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 100.0, 200.0, 300.0, 400.0],
            "current_node_id": ["A", "A", "B", "B", "C"],
            "current_node_distance_m": [10.0] * 5,
        }
    )

    segments = current_map.build_current_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_current_df=chainage_df,
        node_attributes_by_id={"A": _attrs(), "B": _attrs(), "C": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 3
    assert list(segments["current_node_id"]) == ["A", "B", "C"]
    assert segments.iloc[0]["start_chainage_m"] == pytest.approx(0.0)
    assert segments.iloc[-1]["end_chainage_m"] == pytest.approx(400.0)


def test_true_route_geometry_retained_not_a_straight_chord():
    """17-M: an L-shaped route's bend must survive inside a section spanning it."""

    # An L-shaped route: east 100 m, then north 100 m -- total length 200 m.
    route = LineString([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 50.0, 150.0, 200.0],
            "current_node_id": ["A", "A", "A", "A"],
            "current_node_distance_m": [10.0] * 4,
        }
    )

    segments = current_map.build_current_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_current_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 1
    section_geom = segments.iloc[0].geometry
    # A straight chord from (0,0) to (100,100) would have length ~141.4 m;
    # the true route-following length must instead be the full 200 m.
    assert section_geom.length == pytest.approx(200.0, abs=1e-6)
    # The bend vertex (100, 0) must appear in the extracted geometry.
    coords = list(section_geom.coords)
    assert any(pt[0] == pytest.approx(100.0) and pt[1] == pytest.approx(0.0) for pt in coords)


def test_segments_column_schema_matches_constant():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 100.0],
            "current_node_id": ["A", "A"],
            "current_node_distance_m": [1.0, 2.0],
        }
    )

    segments = current_map.build_current_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_current_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert list(segments.columns) == [
        *list(current_map.CURRENT_REFERENCE_SEGMENTS_COLUMNS),
        "geometry",
    ]


def test_segments_handle_unassigned_stations_without_crashing():
    """A run of stations with no node assignment becomes its own null-attribute segment."""

    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 50.0, 100.0],
            "current_node_id": ["A", None, None],
            "current_node_distance_m": [10.0, None, None],
        }
    )

    segments = current_map.build_current_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_current_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 2
    # A None stored alongside other rows' real values can land as NaN rather
    # than a literal None object -- pd.isna covers both storage forms.
    assert pd.isna(segments.iloc[1]["current_node_id"])
    assert pd.isna(segments.iloc[1]["current_reference_speed_p95_m_s"])


# --- render_reference_current_map (17-N) --------------------------------------------


def test_render_reference_current_map_produces_a_nonempty_png(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (500000.0 + 3000.0, 5900000.0 + 4000.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 2500.0, 5000.0],
            "current_node_id": ["A", "B", "B"],
            "current_node_distance_m": [10.0, 20.0, 25.0],
        }
    )
    segments = current_map.build_current_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_current_df=chainage_df,
        node_attributes_by_id={
            "A": _attrs(speed_p95_m_s=0.5),
            "B": _attrs(speed_p95_m_s=0.9),
        },
        working_crs=WORKING_CRS,
    )
    output_path = tmp_path / "map.png"

    result_path = current_map.render_reference_current_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = current_map.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0


def test_render_reference_current_map_handles_missing_background_raster_gracefully(
    tmp_path: Path,
):
    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 1000.0],
            "current_node_id": ["A", "A"],
            "current_node_distance_m": [1.0, 2.0],
        }
    )
    segments = current_map.build_current_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_current_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    result_path = current_map.render_reference_current_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=tmp_path / "map.png",
        background_raster_path=tmp_path / "does_not_exist.tif",
    )

    assert result_path.exists()


# --- No forbidden downstream-physics column names (17-O) ---------------------------


def test_segments_schema_contains_no_forbidden_downstream_terms():
    forbidden = ("bed_shear", "shields", "mobility", "risk")
    columns_lower = [c.lower() for c in current_map.CURRENT_REFERENCE_SEGMENTS_COLUMNS]
    for term in forbidden:
        assert not any(term in column for column in columns_lower)
