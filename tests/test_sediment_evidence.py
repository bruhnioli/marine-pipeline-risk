"""Offline unit tests for marine_engine.sediment.evidence.

Uses small synthetic geometries -- a short straight route, a buffered AOI
around it, and a handful of hand-placed points/polygons with simple
round-number coordinates in a projected CRS -- never the real PL854 route,
its PSA samples, or any other real project data -- and never touches the
network.
"""

from datetime import UTC, datetime

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, mapping

from marine_engine.preprocessing import chainage
from marine_engine.sediment import evidence

WORKING_CRS = "EPSG:32631"


def _make_route() -> LineString:
    """A 500 m due-east route, entirely synthetic."""

    return LineString([(500000.0, 5900000.0), (500500.0, 5900000.0)])


def _make_aoi(route: LineString, buffer_m: float = 5000.0):
    return route.buffer(buffer_m)


def _make_chainage_gdf(route: LineString, crs: str, interval_m: float = 100.0) -> gpd.GeoDataFrame:
    stations_result = chainage.compute_chainage_stations(route, interval_m)
    return chainage.build_canonical_chainage_gdf(
        pipeline_id="TEST-PL",
        stations=stations_result.stations,
        interval_m=interval_m,
        chainage_origin_basis="source_geometry_start",
        working_crs=crs,
    )


def _feature_from_point_working(point_working: Point, crs: str, properties: dict) -> dict:
    """A GeoJSON-Feature-shaped dict whose coordinates are back-projected to WGS84."""

    point_wgs84 = gpd.GeoSeries([point_working], crs=crs).to_crs("EPSG:4326").iloc[0]
    return {
        "geometry": {"type": "Point", "coordinates": [point_wgs84.x, point_wgs84.y]},
        "properties": properties,
    }


def _polygon_feature_from_working(polygon_working, crs: str, properties: dict) -> dict:
    """A GeoJSON-Feature-shaped dict for a polygon built in the working CRS."""

    polygon_wgs84 = gpd.GeoSeries([polygon_working], crs=crs).to_crs("EPSG:4326").iloc[0]
    return {"geometry": mapping(polygon_wgs84), "properties": properties}


# --- classify_surface_evidence ------------------------------------------------


def test_classify_surface_evidence_grab_sample():
    result = evidence.classify_surface_evidence("Grab: Shipek", 0, 0)
    assert result == evidence.SURFACE_GRAB


def test_classify_surface_evidence_core_interval_at_seabed():
    result = evidence.classify_surface_evidence("Vibrocore", 0.0, 3.5)
    assert result == evidence.SURFACE_CORE_INTERVAL


def test_classify_surface_evidence_subsurface_interval():
    result = evidence.classify_surface_evidence("Vibrocore", 1.2, 3.5)
    assert result == evidence.SUBSURFACE_INTERVAL


def test_classify_surface_evidence_missing_depth_with_grab_equipment():
    result = evidence.classify_surface_evidence("Grab: Day", None, None)
    assert result == evidence.SURFACE_GRAB


def test_classify_surface_evidence_missing_depth_and_ambiguous_equipment():
    result = evidence.classify_surface_evidence("Dredge", None, None)
    assert result == evidence.SURFACE_UNCERTAIN


def test_classify_surface_evidence_totally_unknown():
    result = evidence.classify_surface_evidence(None, None, None)
    assert result == evidence.UNKNOWN_SURFACE_EVIDENCE


# --- normalize_psa_observations -----------------------------------------------


def test_normalize_psa_observations_spatial_relationship():
    route = _make_route()
    aoi = _make_aoi(route)
    # 50 m due north of the route's midpoint (chainage 250 m).
    offset_point_working = Point(500250.0, 5900050.0)
    feature = _feature_from_point_working(
        offset_point_working,
        WORKING_CRS,
        {
            "PSA_DATA_ID": "PSA-1",
            "EQUIPMENT_TYPE": "Grab: Day",
            "DEPTH_TOP": 0.0,
            "DEPTH_BASE": 0.0,
        },
    )

    gdf = evidence.normalize_psa_observations(
        [feature],
        route_working=route,
        working_crs=WORKING_CRS,
        aoi_geometry_working=aoi,
        run_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["distance_to_pipeline_m"] == pytest.approx(50.0, abs=1e-2)
    assert row["nearest_pipeline_chainage_m"] == pytest.approx(250.0, abs=1e-2)
    assert row["nearest_pipeline_kp"].startswith("KP ")


def test_normalize_psa_observations_preserves_subsurface_and_surface_together():
    route = _make_route()
    aoi = _make_aoi(route)
    grab_point = Point(500100.0, 5900010.0)
    deep_point = Point(500200.0, 5900020.0)
    features = [
        _feature_from_point_working(
            grab_point,
            WORKING_CRS,
            {
                "PSA_DATA_ID": "GRAB-1",
                "EQUIPMENT_TYPE": "Grab: Day",
                "DEPTH_TOP": 0.0,
                "DEPTH_BASE": 0.0,
            },
        ),
        _feature_from_point_working(
            deep_point,
            WORKING_CRS,
            {
                "PSA_DATA_ID": "CORE-1",
                "EQUIPMENT_TYPE": "Vibrocore",
                "DEPTH_TOP": 2.0,
                "DEPTH_BASE": 3.5,
            },
        ),
    ]

    gdf = evidence.normalize_psa_observations(
        features,
        route_working=route,
        working_crs=WORKING_CRS,
        aoi_geometry_working=aoi,
        run_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(gdf) == 2
    by_id = gdf.set_index("psa_data_id")
    assert by_id.loc["GRAB-1", "surface_evidence_class"] == evidence.SURFACE_GRAB
    assert by_id.loc["CORE-1", "surface_evidence_class"] == evidence.SUBSURFACE_INTERVAL


def test_gsm_total_flagged_when_materially_inconsistent():
    route = _make_route()
    aoi = _make_aoi(route)
    point = Point(500100.0, 5900010.0)
    feature = _feature_from_point_working(
        point,
        WORKING_CRS,
        {
            "PSA_DATA_ID": "GSM-1",
            "EQUIPMENT_TYPE": "Grab: Day",
            "DEPTH_TOP": 0.0,
            "DEPTH_BASE": 0.0,
            "GRAV": 10.0,
            "SAND": 10.0,
            "MUD": 10.0,
            "GSM_UNITS": "percent",
        },
    )

    gdf = evidence.normalize_psa_observations(
        [feature],
        route_working=route,
        working_crs=WORKING_CRS,
        aoi_geometry_working=aoi,
        run_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    row = gdf.iloc[0]
    assert bool(row["gsm_total_valid"]) is False
    assert row["gsm_total_pct"] == pytest.approx(30.0)


def test_sample_date_independent_of_run_timestamp():
    route = _make_route()
    aoi = _make_aoi(route)
    point = Point(500100.0, 5900010.0)
    sample_moment = datetime(2005, 6, 15, tzinfo=UTC)
    epoch_ms = int(sample_moment.timestamp() * 1000)
    feature = _feature_from_point_working(
        point,
        WORKING_CRS,
        {
            "PSA_DATA_ID": "DATE-1",
            "EQUIPMENT_TYPE": "Grab: Day",
            "DEPTH_TOP": 0.0,
            "DEPTH_BASE": 0.0,
            "EQUIPMENT_START_DATE": epoch_ms,
        },
    )

    gdf_2020 = evidence.normalize_psa_observations(
        [feature],
        route_working=route,
        working_crs=WORKING_CRS,
        aoi_geometry_working=aoi,
        run_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
    )
    gdf_2026 = evidence.normalize_psa_observations(
        [feature],
        route_working=route,
        working_crs=WORKING_CRS,
        aoi_geometry_working=aoi,
        run_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert gdf_2020.iloc[0]["sample_date"] == "2005-06-15"
    assert gdf_2026.iloc[0]["sample_date"] == "2005-06-15"
    assert gdf_2020.iloc[0]["sample_year"] == 2005
    assert gdf_2026.iloc[0]["sample_year"] == 2005
    assert gdf_2020.iloc[0]["sample_age_years_at_run"] == 15
    assert gdf_2026.iloc[0]["sample_age_years_at_run"] == 21


# --- attach_mapped_and_predictive_at_psa_points -------------------------------


def test_attach_mapped_and_predictive_never_overwrites_observed_or_mapped_columns():
    route = _make_route()
    aoi = _make_aoi(route)
    point = Point(500250.0, 5900000.0)  # sits directly on the route
    psa_feature = _feature_from_point_working(
        point,
        WORKING_CRS,
        {
            "PSA_DATA_ID": "PSA-1",
            "EQUIPMENT_TYPE": "Grab: Day",
            "DEPTH_TOP": 0.0,
            "DEPTH_BASE": 0.0,
            "FOLK_CLASS": "mS",
        },
    )
    psa_gdf = evidence.normalize_psa_observations(
        [psa_feature],
        route_working=route,
        working_crs=WORKING_CRS,
        aoi_geometry_working=aoi,
        run_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    polygon = point.buffer(100.0)
    seabed_gdf = evidence.normalize_seabed_sediments_250k(
        [_polygon_feature_from_working(polygon, WORKING_CRS, {"BGS_ID": "B1", "FOLK_S": "sM"})],
        working_crs=WORKING_CRS,
    )
    predictive_gdf = evidence.normalize_predictive_folk_polygons(
        [_polygon_feature_from_working(polygon, WORKING_CRS, {"FOLK_S": "G"})],
        working_crs=WORKING_CRS,
    )

    result = evidence.attach_mapped_and_predictive_at_psa_points(
        psa_gdf, seabed_gdf, predictive_gdf
    )

    row = result.iloc[0]
    assert row["folk_class"] == "mS"
    assert row["mapped_250k_folk_class_at_point"] == "sM"
    assert row["predictive_folk_class_at_point"] == "G"
    assert {
        row["folk_class"],
        row["mapped_250k_folk_class_at_point"],
        row["predictive_folk_class_at_point"],
    } == {"mS", "sM", "G"}


# --- compute_psa_support_counts -----------------------------------------------


def test_compute_psa_support_counts_correct_radii():
    stations = gpd.GeoDataFrame(
        [
            {"pipeline_id": "TEST-PL", "station_index": 0, "chainage_m": 0.0},
            {"pipeline_id": "TEST-PL", "station_index": 1, "chainage_m": 10000.0},
            {"pipeline_id": "TEST-PL", "station_index": 2, "chainage_m": 20000.0},
        ],
        geometry=[
            Point(500000.0, 5900000.0),
            Point(510000.0, 5900000.0),
            Point(520000.0, 5900000.0),
        ],
        crs=WORKING_CRS,
    )
    psa_points = gpd.GeoDataFrame(
        [{"psa_data_id": f"PSA-{i}"} for i in range(4)],
        geometry=[
            Point(500000.0, 5900300.0),  # 300 m from station 0
            Point(500000.0, 5900800.0),  # 800 m from station 0
            Point(500000.0, 5901500.0),  # 1500 m from station 0
            Point(500000.0, 5903000.0),  # 3000 m from station 0
        ],
        crs=WORKING_CRS,
    )

    counts = evidence.compute_psa_support_counts(stations, psa_points)

    assert counts["psa_surface_count_500m"].tolist() == [1, 0, 0]
    assert counts["psa_surface_count_1000m"].tolist() == [2, 0, 0]
    assert counts["psa_surface_count_2000m"].tolist() == [3, 0, 0]


# --- build_chainage_sediment_evidence ------------------------------------------


def test_build_chainage_sediment_evidence_retains_every_station_regardless_of_match():
    route = _make_route()
    aoi = _make_aoi(route)
    chainage_gdf = _make_chainage_gdf(route, WORKING_CRS, interval_m=100.0)

    empty_psa = evidence.normalize_psa_observations(
        [],
        route_working=route,
        working_crs=WORKING_CRS,
        aoi_geometry_working=aoi,
        run_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    empty_seabed = evidence.normalize_seabed_sediments_250k([], working_crs=WORKING_CRS)
    empty_predictive = evidence.normalize_predictive_folk_polygons([], working_crs=WORKING_CRS)

    result = evidence.build_chainage_sediment_evidence(
        chainage_gdf=chainage_gdf,
        psa_gdf_working=empty_psa,
        seabed_250k_gdf_working=empty_seabed,
        predictive_folk_gdf_working=empty_predictive,
        working_crs=WORKING_CRS,
    )

    assert len(result) == len(chainage_gdf)
    assert list(result.columns) == list(evidence.CHAINAGE_SEDIMENT_COLUMNS)
    assert result["mapped_250k_bgs_id"].isna().all()
    assert result["nearest_psa_id"].isna().all()
    assert result["predictive_folk_class"].isna().all()
    assert (result["psa_surface_count_500m"] == 0).all()


def test_determinism_same_inputs_produce_same_chainage_evidence():
    route = _make_route()
    aoi = _make_aoi(route)
    chainage_gdf = _make_chainage_gdf(route, WORKING_CRS, interval_m=100.0)

    psa_point = Point(500250.0, 5900010.0)
    psa_feature = _feature_from_point_working(
        psa_point,
        WORKING_CRS,
        {
            "PSA_DATA_ID": "PSA-1",
            "EQUIPMENT_TYPE": "Grab: Day",
            "DEPTH_TOP": 0.0,
            "DEPTH_BASE": 0.0,
            "FOLK_CLASS": "mS",
        },
    )
    psa_gdf = evidence.normalize_psa_observations(
        [psa_feature],
        route_working=route,
        working_crs=WORKING_CRS,
        aoi_geometry_working=aoi,
        run_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    polygon = psa_point.buffer(150.0)
    seabed_gdf = evidence.normalize_seabed_sediments_250k(
        [_polygon_feature_from_working(polygon, WORKING_CRS, {"BGS_ID": "B1", "FOLK_S": "sM"})],
        working_crs=WORKING_CRS,
    )
    predictive_gdf = evidence.normalize_predictive_folk_polygons(
        [_polygon_feature_from_working(polygon, WORKING_CRS, {"FOLK_S": "G"})],
        working_crs=WORKING_CRS,
    )

    kwargs = {
        "chainage_gdf": chainage_gdf,
        "psa_gdf_working": psa_gdf,
        "seabed_250k_gdf_working": seabed_gdf,
        "predictive_folk_gdf_working": predictive_gdf,
        "working_crs": WORKING_CRS,
    }
    first = evidence.build_chainage_sediment_evidence(**kwargs)
    second = evidence.build_chainage_sediment_evidence(**kwargs)

    pd.testing.assert_frame_equal(first, second)


# --- build_predictive_comparison_table ------------------------------------------


def test_build_predictive_comparison_table_only_includes_surface_evidence():
    psa_with_comparisons = gpd.GeoDataFrame(
        [
            {
                "psa_data_id": "SURF-1",
                "distance_to_pipeline_m": 42.0,
                "surface_evidence_class": evidence.SURFACE_GRAB,
                "folk_class": "mS",
                "mapped_250k_folk_class_at_point": "sM",
                "predictive_folk_class_at_point": "G",
            },
            {
                "psa_data_id": "SUB-1",
                "distance_to_pipeline_m": 55.0,
                "surface_evidence_class": evidence.SUBSURFACE_INTERVAL,
                "folk_class": "G",
                "mapped_250k_folk_class_at_point": "G",
                "predictive_folk_class_at_point": "G",
            },
        ],
        geometry=[Point(0.0, 0.0), Point(1.0, 1.0)],
        crs=WORKING_CRS,
    )

    table = evidence.build_predictive_comparison_table(psa_with_comparisons)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["psa_data_id"] == "SURF-1"
    assert bool(row["circularity_warning"]) is True
    assert row["evidence_role"] == "SECONDARY_MODEL_COMPARISON"


# --- compute_coverage_diagnostics -----------------------------------------------


def test_compute_coverage_diagnostics_counts():
    psa_gdf_working = pd.DataFrame(
        [
            {
                "surface_evidence_class": evidence.SURFACE_GRAB,
                "folk_class": "mS",
                "gravel": 10.0,
                "sand": 80.0,
                "mud": 10.0,
                "d50_mm": 0.2,
                "sample_year": 2000,
                "distance_to_pipeline_m": 100.0,
            },
            {
                "surface_evidence_class": evidence.SURFACE_CORE_INTERVAL,
                "folk_class": None,
                "gravel": None,
                "sand": None,
                "mud": None,
                "d50_mm": None,
                "sample_year": 2005,
                "distance_to_pipeline_m": 600.0,
            },
            {
                "surface_evidence_class": evidence.SUBSURFACE_INTERVAL,
                "folk_class": "G",
                "gravel": 90.0,
                "sand": 5.0,
                "mud": 5.0,
                "d50_mm": 5.0,
                "sample_year": 2010,
                "distance_to_pipeline_m": 50.0,
            },
            {
                "surface_evidence_class": evidence.SURFACE_UNCERTAIN,
                "folk_class": None,
                "gravel": None,
                "sand": None,
                "mud": None,
                "d50_mm": None,
                "sample_year": None,
                "distance_to_pipeline_m": 2500.0,
            },
            {
                "surface_evidence_class": evidence.UNKNOWN_SURFACE_EVIDENCE,
                "folk_class": None,
                "gravel": None,
                "sand": None,
                "mud": None,
                "d50_mm": None,
                "sample_year": 1995,
                "distance_to_pipeline_m": None,
            },
        ]
    )

    coverage = evidence.compute_coverage_diagnostics(psa_gdf_working)

    assert coverage["total_records_in_aoi"] == 5
    assert coverage["surface_evidence_records"] == 2
    assert coverage["subsurface_records"] == 1
    assert coverage["uncertain_records"] == 2
    assert coverage["records_with_folk_class"] == 2
    assert coverage["records_with_gsm_fractions"] == 2
    assert coverage["records_with_usable_d50"] == 2
    assert coverage["sample_year_min"] == 1995
    assert coverage["sample_year_max"] == 2010
    assert coverage["sample_year_median"] == pytest.approx(2002.5)
    assert coverage["distance_to_pipeline_m_min"] == pytest.approx(50.0)
    assert coverage["distance_to_pipeline_m_median"] == pytest.approx(350.0)
    assert coverage["distance_to_pipeline_m_p95"] == pytest.approx(2215.0)
    assert coverage["distance_to_pipeline_m_max"] == pytest.approx(2500.0)
    assert coverage["surface_within_250m"] == 1
    assert coverage["surface_within_500m"] == 1
    assert coverage["surface_within_1000m"] == 2
    assert coverage["surface_within_2000m"] == 2
    assert coverage["surface_within_5000m"] == 2


# --- assess_d50_spatial_support --------------------------------------------------


def test_assess_d50_spatial_support_not_assessable_when_empty():
    chainage_df = pd.DataFrame(columns=["psa_surface_count_1000m"])
    result = evidence.assess_d50_spatial_support(chainage_df, {})
    assert result == evidence.NOT_ASSESSABLE


def test_assess_d50_spatial_support_very_sparse_with_few_usable_d50():
    chainage_df = pd.DataFrame({"psa_surface_count_1000m": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    coverage = {"records_with_usable_d50": 2}

    result = evidence.assess_d50_spatial_support(chainage_df, coverage)

    assert result == evidence.VERY_SPARSE
