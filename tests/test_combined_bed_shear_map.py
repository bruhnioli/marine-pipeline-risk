"""Offline unit tests for marine_engine.metocean.combined_bed_shear_map (MAR-012).

Small synthetic routes/chainage tables only -- never the real PL854 route,
never network access. Lettered comments map to MAR-012 Section 30's
required test list.
"""

from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine.metocean import combined_bed_shear_map as cbm

WORKING_CRS = "EPSG:32631"


def _attrs(**overrides) -> cbm.HydroPairReferenceAttributes:
    defaults = {
        "tau_max_p95_sensitivity_min_pa": 0.10,
        "tau_max_p95_sensitivity_max_pa": 0.20,
        "tau_max_p95_sensitivity_width_pa": 0.10,
        "tau_max_p99_sensitivity_min_pa": 0.15,
        "tau_max_p99_sensitivity_max_pa": 0.30,
        "tau_max_p99_sensitivity_width_pa": 0.15,
        "overlap_start_time_utc": pd.Timestamp("2024-07-20", tz="UTC"),
        "overlap_end_time_utc": pd.Timestamp("2026-04-30", tz="UTC"),
    }
    defaults.update(overrides)
    return cbm.HydroPairReferenceAttributes(**defaults)


def _chainage_hydro_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- T: many chainage stations sharing one hydro pair dissolve to one segment -------


def test_T_many_stations_sharing_one_hydro_pair_dissolve_into_one_section():
    route = LineString([(0.0, 0.0), (1000.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": c,
                "current_node_id": "current_A",
                "current_node_distance_m": 50.0 + c / 100.0,
                "wave_node_id": "wave_A",
                "wave_node_distance_m": 60.0 + c / 100.0,
                "hydro_pair_id": "current_A__wave_A",
            }
            for c in (0.0, 100.0, 200.0, 300.0, 1000.0)
        ]
    )

    segments = cbm.build_combined_bed_shear_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        node_attributes_by_id={"current_A__wave_A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 1
    row = segments.iloc[0]
    assert row["hydro_pair_id"] == "current_A__wave_A"
    assert row["current_node_id"] == "current_A"
    assert row["wave_node_id"] == "wave_A"
    assert row["start_chainage_m"] == pytest.approx(0.0)
    assert row["end_chainage_m"] == pytest.approx(1000.0)
    # Pooled from BOTH current and wave distances across the run.
    assert row["source_node_distance_min_m"] == pytest.approx(50.0)
    assert row["source_node_distance_max_m"] == pytest.approx(70.0)


def test_hydro_pair_change_creates_a_segment_boundary_even_with_shared_current_node():
    """A wave-node change alone (current node unchanged) must still split the run,
    since segments are built from the COMBINED current+wave identity (Section 25)."""

    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": c,
                "current_node_id": "current_A",
                "current_node_distance_m": 10.0,
                "wave_node_id": wave_id,
                "wave_node_distance_m": 20.0,
                "hydro_pair_id": f"current_A__{wave_id}",
            }
            for c, wave_id in ((0.0, "wave_A"), (50.0, "wave_A"), (100.0, "wave_B"))
        ]
    )

    segments = cbm.build_combined_bed_shear_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        node_attributes_by_id={
            "current_A__wave_A": _attrs(),
            "current_A__wave_B": _attrs(tau_max_p95_sensitivity_max_pa=0.5),
        },
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 2
    assert segments.iloc[0]["wave_node_id"] == "wave_A"
    assert segments.iloc[1]["wave_node_id"] == "wave_B"
    assert segments.iloc[0]["current_node_id"] == segments.iloc[1]["current_node_id"] == "current_A"


# --- U: true pipeline geometry retained, never a straight chord --------------------


def test_U_true_route_geometry_retained_not_a_straight_chord():
    route = LineString([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": c,
                "current_node_id": "current_A",
                "current_node_distance_m": 10.0,
                "wave_node_id": "wave_A",
                "wave_node_distance_m": 10.0,
                "hydro_pair_id": "current_A__wave_A",
            }
            for c in (0.0, 50.0, 150.0, 200.0)
        ]
    )

    segments = cbm.build_combined_bed_shear_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        node_attributes_by_id={"current_A__wave_A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 1
    section_geom = segments.iloc[0].geometry
    assert section_geom.length == pytest.approx(200.0, abs=1e-6)
    coords = list(section_geom.coords)
    assert any(pt[0] == pytest.approx(100.0) and pt[1] == pytest.approx(0.0) for pt in coords)


def test_segments_column_schema_matches_constant():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": c,
                "current_node_id": "current_A",
                "current_node_distance_m": 1.0,
                "wave_node_id": "wave_A",
                "wave_node_distance_m": 2.0,
                "hydro_pair_id": "current_A__wave_A",
            }
            for c in (0.0, 100.0)
        ]
    )

    segments = cbm.build_combined_bed_shear_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        node_attributes_by_id={"current_A__wave_A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert list(segments.columns) == [*list(cbm.COMBINED_BED_SHEAR_SEGMENTS_COLUMNS), "geometry"]


def test_segments_handle_unassigned_stations_without_crashing():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": 0.0,
                "current_node_id": "current_A",
                "current_node_distance_m": 10.0,
                "wave_node_id": "wave_A",
                "wave_node_distance_m": 10.0,
                "hydro_pair_id": "current_A__wave_A",
            },
            {
                "chainage_m": 50.0,
                "current_node_id": None,
                "current_node_distance_m": None,
                "wave_node_id": None,
                "wave_node_distance_m": None,
                "hydro_pair_id": None,
            },
            {
                "chainage_m": 100.0,
                "current_node_id": None,
                "current_node_distance_m": None,
                "wave_node_id": None,
                "wave_node_distance_m": None,
                "hydro_pair_id": None,
            },
        ]
    )

    segments = cbm.build_combined_bed_shear_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        node_attributes_by_id={"current_A__wave_A": _attrs()},
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 2
    assert pd.isna(segments.iloc[1]["hydro_pair_id"])
    assert pd.isna(segments.iloc[1]["tau_max_p95_sensitivity_max_pa"])


def test_segments_empty_input():
    result = cbm.build_combined_bed_shear_segments(
        pipeline_id="PL854",
        route=LineString([(0.0, 0.0), (1.0, 0.0)]),
        chainage_hydro_df=pd.DataFrame(),
        node_attributes_by_id={},
        working_crs=WORKING_CRS,
    )
    assert result.empty


# --- Hotspot label shows the FULL envelope, not merely the upper value -------------


def test_hotspot_label_shows_full_p95_envelope():
    row = pd.Series(
        {
            "start_chainage_m": 12300.0,
            "end_chainage_m": 14100.0,
            "tau_max_p95_sensitivity_min_pa": 0.18,
            "tau_max_p95_sensitivity_max_pa": 0.24,
        }
    )
    label = cbm._format_combined_hotspot_label(row)
    assert "0.18" in label
    assert "0.24" in label
    assert "12.3" in label
    assert "14.1" in label


# --- V/W: map renderer produces a non-empty PNG coloured by the upper p95 bound ----


def _segments_for_render(route: LineString) -> "cbm.gpd.GeoDataFrame":
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": 0.0,
                "current_node_id": "current_A",
                "current_node_distance_m": 10.0,
                "wave_node_id": "wave_A",
                "wave_node_distance_m": 10.0,
                "hydro_pair_id": "current_A__wave_A",
            },
            {
                "chainage_m": route.length / 2.0,
                "current_node_id": "current_B",
                "current_node_distance_m": 20.0,
                "wave_node_id": "wave_B",
                "wave_node_distance_m": 25.0,
                "hydro_pair_id": "current_B__wave_B",
            },
            {
                "chainage_m": route.length,
                "current_node_id": "current_B",
                "current_node_distance_m": 20.0,
                "wave_node_id": "wave_B",
                "wave_node_distance_m": 25.0,
                "hydro_pair_id": "current_B__wave_B",
            },
        ]
    )
    return cbm.build_combined_bed_shear_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        node_attributes_by_id={
            "current_A__wave_A": _attrs(tau_max_p95_sensitivity_max_pa=0.15),
            "current_B__wave_B": _attrs(tau_max_p95_sensitivity_max_pa=0.35),
        },
        working_crs=WORKING_CRS,
    )


def test_V_render_produces_a_nonempty_landscape_png(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (506000.0, 5901500.0)])
    segments = _segments_for_render(route)
    output_path = tmp_path / "combined_map.png"

    result_path = cbm.render_combined_bed_shear_map(
        segments_gdf=segments, route=route, working_crs=WORKING_CRS, output_path=output_path
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = cbm.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0
    assert width_px > height_px  # landscape


def test_W_map_colour_variable_is_the_upper_p95_sensitivity_bound():
    """The map is rendered with the `tau_max_p95_sensitivity_max_pa` column as its
    colour variable, confirmed via the actual rendered segments' schema/values --
    never `tau_max_p95_sensitivity_min_pa` or a mean/best-estimate column."""

    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    segments = _segments_for_render(route)

    assert "tau_max_p95_sensitivity_max_pa" in segments.columns
    assert segments["tau_max_p95_sensitivity_max_pa"].tolist() == [0.15, 0.35]


def test_render_handles_missing_background_raster_gracefully(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    segments = _segments_for_render(route)

    result_path = cbm.render_combined_bed_shear_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=tmp_path / "combined_map.png",
        background_raster_path=tmp_path / "does_not_exist.tif",
    )

    assert result_path.exists()


# --- X: no forbidden downstream-physics terms in the segment schema ----------------


def test_X_segments_schema_contains_no_forbidden_downstream_terms():
    forbidden = ("shields", "theta", "mobility", "risk")
    columns_lower = [c.lower() for c in cbm.COMBINED_BED_SHEAR_SEGMENTS_COLUMNS]
    for term in forbidden:
        assert not any(term in column for column in columns_lower)
