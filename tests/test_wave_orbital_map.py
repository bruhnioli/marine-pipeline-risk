"""Offline unit tests for marine_engine.metocean.wave_orbital_map (MAR-011).

Small synthetic routes/chainage tables only -- never the real PL854 route,
never network access. Uses matplotlib's Agg backend (set by the module
itself) so PNG rendering is deterministic and headless-safe in CI.
"""

from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine.metocean import wave_orbital_map

WORKING_CRS = "EPSG:32631"


def _attrs(**overrides) -> wave_orbital_map.WaveNodeReferenceAttributes:
    defaults = {
        "model_bathymetry_m": 25.0,
        "hs_p95_m": 2.0,
        "hs_p99_m": 2.5,
        "hs_max_m": 3.0,
        "tm02_median_s": 6.0,
        "tm02_p95_s": 7.5,
        "orbital_rms_mean_m_s": 0.3,
        "orbital_rms_p95_m_s": 0.6,
        "orbital_rms_p99_m_s": 0.7,
        "orbital_rms_max_m_s": 0.8,
        "orbital_amplitude_p95_m_s": 0.85,
        "orbital_amplitude_p99_m_s": 0.99,
        "orbital_amplitude_max_m_s": 1.13,
    }
    defaults.update(overrides)
    return wave_orbital_map.WaveNodeReferenceAttributes(**defaults)


# --- build_wave_orbital_reference_segments (Section 13, 19-N/O/P) ------------------


def test_many_stations_sharing_one_node_dissolve_into_one_section():
    """19-N: many chainage stations sharing one wave node -> exactly one contiguous section."""

    route = LineString([(0.0, 0.0), (1000.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 100.0, 200.0, 300.0, 1000.0],
            "wave_node_id": ["A"] * 5,
            "wave_node_distance_m": [50.0, 60.0, 55.0, 65.0, 70.0],
        }
    )

    segments = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_wave_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 1
    row = segments.iloc[0]
    assert row["wave_node_id"] == "A"
    assert row["start_chainage_m"] == pytest.approx(0.0)
    assert row["end_chainage_m"] == pytest.approx(1000.0)
    assert row["wave_node_distance_min_m"] == pytest.approx(50.0)
    assert row["wave_node_distance_median_m"] == pytest.approx(60.0)
    assert row["wave_node_distance_max_m"] == pytest.approx(70.0)
    assert row["source_grid_resolution_note"] == wave_orbital_map.SOURCE_GRID_RESOLUTION_NOTE


def test_wave_node_change_creates_a_segment_boundary():
    """19-O: a wave node transition creates a map-section boundary at the run midpoint."""

    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 25.0, 50.0, 75.0, 100.0],
            "wave_node_id": ["A", "A", "A", "B", "B"],
            "wave_node_distance_m": [10.0, 10.0, 10.0, 20.0, 20.0],
        }
    )

    segments = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_wave_df=chainage_df,
        node_attributes_by_id={"A": _attrs(), "B": _attrs(model_bathymetry_m=30.0)},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 2
    boundary = (50.0 + 75.0) / 2.0
    assert segments.iloc[0]["wave_node_id"] == "A"
    assert segments.iloc[0]["end_chainage_m"] == pytest.approx(boundary)
    assert segments.iloc[1]["wave_node_id"] == "B"
    assert segments.iloc[1]["start_chainage_m"] == pytest.approx(boundary)
    assert segments.iloc[0]["end_chainage_m"] == segments.iloc[1]["start_chainage_m"]


def test_true_route_geometry_retained_not_a_straight_chord():
    """19-P: an L-shaped route's bend must survive inside a section spanning it."""

    route = LineString([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 50.0, 150.0, 200.0],
            "wave_node_id": ["A", "A", "A", "A"],
            "wave_node_distance_m": [10.0] * 4,
        }
    )

    segments = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_wave_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 1
    section_geom = segments.iloc[0].geometry
    assert section_geom.length == pytest.approx(200.0, abs=1e-6)
    coords = list(section_geom.coords)
    assert any(pt[0] == pytest.approx(100.0) and pt[1] == pytest.approx(0.0) for pt in coords)


def test_segments_column_schema_matches_constant():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 100.0],
            "wave_node_id": ["A", "A"],
            "wave_node_distance_m": [1.0, 2.0],
        }
    )

    segments = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_wave_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert list(segments.columns) == [
        *list(wave_orbital_map.WAVE_ORBITAL_REFERENCE_SEGMENTS_COLUMNS),
        "geometry",
    ]


def test_segments_handle_unassigned_stations_without_crashing():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 50.0, 100.0],
            "wave_node_id": ["A", None, None],
            "wave_node_distance_m": [10.0, None, None],
        }
    )

    segments = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_wave_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 2
    assert pd.isna(segments.iloc[1]["wave_node_id"])
    assert pd.isna(segments.iloc[1]["orbital_rms_p95_m_s"])


# --- KP label formatting (19-R) ------------------------------------------------------


def test_hotspot_kp_label_uses_km_precision_not_survey_metres():
    """19-R: hotspot labels use e.g. 'KP 2.09-4.11', never 'KP 2+087.50-KP 4+112.50'."""

    label = wave_orbital_map._format_kp_km_range(2087.5, 4112.5)
    assert label == "KP 2.09–4.11"
    assert "+" not in label


# --- render_wave_orbital_map (19-Q) ---------------------------------------------------


def test_render_wave_orbital_map_produces_a_nonempty_landscape_png(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (500000.0 + 6000.0, 5900000.0 + 1500.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 3000.0, 6000.0],
            "wave_node_id": ["A", "B", "B"],
            "wave_node_distance_m": [10.0, 20.0, 25.0],
        }
    )
    segments = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_wave_df=chainage_df,
        node_attributes_by_id={
            "A": _attrs(orbital_rms_p95_m_s=0.4),
            "B": _attrs(orbital_rms_p95_m_s=0.9),
        },
        working_crs=WORKING_CRS,
    )
    output_path = tmp_path / "wave_map.png"

    result_path = wave_orbital_map.render_wave_orbital_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = wave_orbital_map.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0
    assert width_px > height_px  # landscape (Section 15-A)


def test_render_wave_orbital_map_handles_missing_background_raster_gracefully(
    tmp_path: Path,
):
    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    chainage_df = pd.DataFrame(
        {
            "chainage_m": [0.0, 1000.0],
            "wave_node_id": ["A", "A"],
            "wave_node_distance_m": [1.0, 2.0],
        }
    )
    segments = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id="PL854",
        route=route,
        chainage_wave_df=chainage_df,
        node_attributes_by_id={"A": _attrs()},
        working_crs=WORKING_CRS,
    )

    result_path = wave_orbital_map.render_wave_orbital_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=tmp_path / "wave_map.png",
        background_raster_path=tmp_path / "does_not_exist.tif",
    )

    assert result_path.exists()


# --- No forbidden downstream-physics column names (19-S) ---------------------------


def test_segments_schema_contains_no_forbidden_downstream_terms():
    forbidden = ("bed_shear", "shields", "mobility", "risk")
    columns_lower = [c.lower() for c in wave_orbital_map.WAVE_ORBITAL_REFERENCE_SEGMENTS_COLUMNS]
    for term in forbidden:
        assert not any(term in column for column in columns_lower)
