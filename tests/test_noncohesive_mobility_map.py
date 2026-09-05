"""Offline unit tests for marine_engine.sediment.noncohesive_mobility_map (MAR-013).

Small synthetic routes/chainage tables only -- never the real PL854 route,
never network access. Lettered comments map to MAR-013 Section 27's
required test list.
"""

from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from marine_engine.sediment import noncohesive_mobility_map as ncmap

WORKING_CRS = "EPSG:32631"


def _attrs(**overrides) -> ncmap.CapacityAttributes:
    defaults = {
        "largest_tested_d50_with_p90_mobility_ratio_ge_1_mm": 1.0,
        "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": 0.5,
        "largest_tested_d50_with_p99_mobility_ratio_ge_1_mm": 0.25,
        "largest_tested_d50_with_any_exceedance_mm": 2.0,
        "p95_mobility_sequence_monotonic_nonincreasing": True,
        "monotonicity_violation_count": 0,
        "reference_p95_ratios_by_mm": {0.125: 3.0, 0.250: 2.0, 0.500: 1.0, 1.000: 0.6, 2.000: 0.3},
    }
    defaults.update(overrides)
    return ncmap.CapacityAttributes(**defaults)


def _chainage_hydro_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _psa_row(**overrides) -> dict:
    defaults = {
        "psa_data_id": 1,
        "sample_date": "1980-09-17",
        "sample_year": 1980,
        "sample_age_years_at_run": 45,
        "surface_evidence_class": "SURFACE_GRAB",
        "grain_percentile_status": "DERIVED_FROM_NORMALIZED_MASS_BINS",
        "d10_mm": 0.2,
        "d50_mm": 0.35,
        "d90_mm": 0.8,
        "folk_class": "S",
        "gravel": 0.5,
        "sand": 99.0,
        "mud": 0.5,
        "distance_to_pipeline_m": 1200.0,
        "nearest_pipeline_chainage_m": 500.0,
        "nearest_pipeline_kp": "KP 0+500",
        "longitude": 1.7,
        "latitude": 53.37,
    }
    defaults.update(overrides)
    return defaults


def _observed_d50_context_df(psa_df: pd.DataFrame) -> pd.DataFrame:
    from marine_engine.sediment.noncohesive_mobility import build_observed_d50_context

    return build_observed_d50_context(psa_df)


# --- Segment building: dissolve contiguous hydro-pair runs --------------------------


def test_many_stations_sharing_one_hydro_pair_dissolve_into_one_section():
    route = LineString([(0.0, 0.0), (1000.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": c,
                "hydro_pair_id": "current_A__wave_A",
                "mapped_250k_folk_class": "gS",
                "mapped_250k_nominal_scale": 250000,
            }
            for c in (0.0, 100.0, 200.0, 300.0, 1000.0)
        ]
    )
    segments = ncmap.build_noncohesive_mobility_capacity_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        capacity_by_pair_id={"current_A__wave_A": _attrs()},
        observed_d50_context_df=pd.DataFrame(),
        working_crs=WORKING_CRS,
    )

    assert len(segments) == 1
    row = segments.iloc[0]
    assert row["hydro_pair_id"] == "current_A__wave_A"
    assert row["start_chainage_m"] == pytest.approx(0.0)
    assert row["end_chainage_m"] == pytest.approx(1000.0)
    assert row["mapped_250k_folk_class"] == "gS"
    assert row["mobility_ratio_p95_d50_500um"] == pytest.approx(1.0)


def test_segments_column_schema_matches_constant():
    route = LineString([(0.0, 0.0), (100.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": c,
                "hydro_pair_id": "current_A__wave_A",
                "mapped_250k_folk_class": "S",
                "mapped_250k_nominal_scale": 250000,
            }
            for c in (0.0, 100.0)
        ]
    )
    segments = ncmap.build_noncohesive_mobility_capacity_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        capacity_by_pair_id={"current_A__wave_A": _attrs()},
        observed_d50_context_df=pd.DataFrame(),
        working_crs=WORKING_CRS,
    )
    assert list(segments.columns) == [
        *list(ncmap.NONCOHESIVE_MOBILITY_CAPACITY_SEGMENTS_COLUMNS),
        "geometry",
    ]


def test_segments_empty_input():
    result = ncmap.build_noncohesive_mobility_capacity_segments(
        pipeline_id="PL854",
        route=LineString([(0.0, 0.0), (1.0, 0.0)]),
        chainage_hydro_df=pd.DataFrame(),
        capacity_by_pair_id={},
        observed_d50_context_df=pd.DataFrame(),
        working_crs=WORKING_CRS,
    )
    assert result.empty


# --- R/S: PSA points are context only, never propagated into route colour/D50 -------


def test_R_nearest_psa_context_never_changes_capacity_value():
    route = LineString([(0.0, 0.0), (1000.0, 0.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": c,
                "hydro_pair_id": "current_A__wave_A",
                "mapped_250k_folk_class": "gS",
                "mapped_250k_nominal_scale": 250000,
            }
            for c in (0.0, 500.0, 1000.0)
        ]
    )
    psa_df = pd.DataFrame([_psa_row(psa_data_id=99, d50_mm=7.5, nearest_pipeline_chainage_m=500.0)])
    observed_context = _observed_d50_context_df(psa_df)

    segments = ncmap.build_noncohesive_mobility_capacity_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        capacity_by_pair_id={"current_A__wave_A": _attrs()},
        observed_d50_context_df=observed_context,
        working_crs=WORKING_CRS,
    )

    row = segments.iloc[0]
    # The nearest PSA's own d50 (7.5 mm) never overwrites the capacity value
    # (0.5 mm from _attrs()), even though the PSA point is right on the segment.
    assert row["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"] == pytest.approx(0.5)
    assert row["nearest_valid_psa_id"] == 99
    assert row["nearest_valid_psa_d50_mm"] == pytest.approx(7.5)


def test_S_map_route_colour_column_is_capacity_not_psa_d50():
    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": c,
                "hydro_pair_id": "current_A__wave_A",
                "mapped_250k_folk_class": "gS",
                "mapped_250k_nominal_scale": 250000,
            }
            for c in (0.0, 1000.0)
        ]
    )
    segments = ncmap.build_noncohesive_mobility_capacity_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        capacity_by_pair_id={"current_A__wave_A": _attrs()},
        observed_d50_context_df=pd.DataFrame(),
        working_crs=WORKING_CRS,
    )
    # The column used for colouring is the capacity field, a fixed tested
    # value -- never any nearest_valid_psa_* field.
    assert "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm" in segments.columns
    assert segments.iloc[0]["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"] == pytest.approx(
        0.5
    )


# --- Discrete colour scale: no interpolation between scenarios ----------------------


def test_discrete_cmap_maps_each_tested_value_to_a_distinct_colour():
    cmap, norm, values = ncmap._discrete_d50_cmap_and_norm()
    colours = [ncmap._colour_for_value(cmap, norm, v) for v in values]
    assert len(set(colours)) == len(values)  # every tested scenario gets its own colour


def test_discrete_cmap_none_value_gets_neutral_grey():
    cmap, norm, _values = ncmap._discrete_d50_cmap_and_norm()
    colour = ncmap._colour_for_value(cmap, norm, None)
    assert colour == (0.6, 0.6, 0.6, 1.0)


# --- T/U: map and profile renderers produce non-empty PNGs --------------------------


def _segments_for_render(route: LineString) -> "ncmap.gpd.GeoDataFrame":
    chainage_df = _chainage_hydro_df(
        [
            {
                "chainage_m": 0.0,
                "hydro_pair_id": "current_A__wave_A",
                "mapped_250k_folk_class": "gS",
                "mapped_250k_nominal_scale": 250000,
            },
            {
                "chainage_m": route.length / 2.0,
                "hydro_pair_id": "current_B__wave_B",
                "mapped_250k_folk_class": "S",
                "mapped_250k_nominal_scale": 250000,
            },
            {
                "chainage_m": route.length,
                "hydro_pair_id": "current_B__wave_B",
                "mapped_250k_folk_class": "S",
                "mapped_250k_nominal_scale": 250000,
            },
        ]
    )
    return ncmap.build_noncohesive_mobility_capacity_segments(
        pipeline_id="PL854",
        route=route,
        chainage_hydro_df=chainage_df,
        capacity_by_pair_id={
            "current_A__wave_A": _attrs(largest_tested_d50_with_p95_mobility_ratio_ge_1_mm=0.25),
            "current_B__wave_B": _attrs(largest_tested_d50_with_p95_mobility_ratio_ge_1_mm=2.0),
        },
        observed_d50_context_df=pd.DataFrame(),
        working_crs=WORKING_CRS,
    )


def test_T_render_map_produces_a_nonempty_landscape_png(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (506000.0, 5901500.0)])
    segments = _segments_for_render(route)
    psa_df = pd.DataFrame([_psa_row(nearest_pipeline_chainage_m=route.length / 2.0)])
    output_path = tmp_path / "capacity_map.png"

    result_path = ncmap.render_noncohesive_mobility_capacity_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=output_path,
        psa_observations_df=psa_df,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    width_px, height_px = ncmap.read_png_dimensions(output_path)
    assert width_px > 0
    assert height_px > 0
    assert width_px > height_px  # landscape


def test_render_map_handles_missing_background_and_no_psa_gracefully(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    segments = _segments_for_render(route)

    result_path = ncmap.render_noncohesive_mobility_capacity_map(
        segments_gdf=segments,
        route=route,
        working_crs=WORKING_CRS,
        output_path=tmp_path / "capacity_map.png",
        psa_observations_df=None,
        background_raster_path=tmp_path / "does_not_exist.tif",
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_U_render_profile_produces_a_nonempty_png(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (506000.0, 5900000.0)])
    segments = _segments_for_render(route)
    psa_df = pd.DataFrame([_psa_row(nearest_pipeline_chainage_m=route.length / 2.0)])
    output_path = tmp_path / "profile.png"

    result_path = ncmap.render_mobility_capacity_profile(
        segments_gdf=segments,
        psa_observations_df=psa_df,
        working_crs=WORKING_CRS,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_profile_handles_no_psa_gracefully(tmp_path: Path):
    route = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])
    segments = _segments_for_render(route)
    result_path = ncmap.render_mobility_capacity_profile(
        segments_gdf=segments,
        psa_observations_df=None,
        working_crs=WORKING_CRS,
        output_path=tmp_path / "profile.png",
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


# --- V: no forbidden downstream-physics terms in the segment schema ----------------


def test_V_segments_schema_contains_no_forbidden_downstream_terms():
    forbidden = (
        "erosion_rate",
        "bedload_flux",
        "suspended_load",
        "scour_depth",
        "free_span",
        "risk",
    )
    columns_lower = [c.lower() for c in ncmap.NONCOHESIVE_MOBILITY_CAPACITY_SEGMENTS_COLUMNS]
    for term in forbidden:
        assert not any(term in column for column in columns_lower)
