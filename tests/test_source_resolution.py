"""Offline unit tests for marine_engine.preprocessing.source_resolution.

Uses small synthetic geometries with simple round-number coordinates in
EPSG:32631 -- never the real PL854 route -- and monkeypatches
`cdi.resolve_cdi_record` so nothing here touches the network. Real CDI
parsing/classification is covered separately in tests/test_cdi.py.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from marine_engine.preprocessing import source_resolution as sr
from marine_engine.providers.bathymetry import cdi
from marine_engine.providers.bathymetry.emodnet import QualityIndexFeature, SourceReferenceFeature
from marine_engine.providers.bathymetry.inventory import SurveyRecord

WORKING_CRS = "EPSG:32631"


def _to_wgs84(geom_working):
    return gpd.GeoSeries([geom_working], crs=WORKING_CRS).to_crs("EPSG:4326").iloc[0]


def _make_cdi_record(source_reference_id: str, **overrides) -> cdi.CdiRecord:
    fields = {
        "source_reference_id": source_reference_id,
        "cdi_record_id": f"CDI-{source_reference_id}",
        "title": f"Synthetic survey {source_reference_id}",
        "description": None,
        "organisation": "Test Hydrographic Office",
        "organisation_edmo_id": 1,
        "data_centre": "Test Distributor",
        "data_centre_edmo_id": 2,
        "survey_name": "TEST-CRUISE",
        "platform": "unknown",
        "acquisition_start": None,
        "acquisition_end": None,
        "acquisition_year": 2000,
        "survey_method": "single-beam echosounders",
        "device": None,
        "horizontal_resolution_note": None,
        "vertical_resolution_note": None,
        "horizontal_crs": "EPSG:4326",
        "vertical_datum": None,
        "geographic_footprint": None,
        "data_format": "Climate and Forecast NetCDF",
        "data_size_mb": 100.0,
        "licence_code": "RS",
        "access_restriction": "by negotiation",
        "access_mechanism": "web data access with registration",
        "metadata_url": f"https://cdi-bathymetry.seadatanet.org/report/edmo/2/{source_reference_id}",
        "data_access_url": None,
        "cdi_import_date": None,
        "cdi_update_date": None,
        "resolution_status": cdi.RESOLUTION_LIVE,
        "notes": None,
    }
    fields.update(overrides)
    return cdi.CdiRecord(**fields)


# --- identify_pl854_source_reference_ids -------------------------------------


def test_identify_source_reference_ids_returns_sorted_unique_non_null():
    df = pd.DataFrame(
        {"source_reference_id": ["B", "A", "A", None, "B", None]},
    )
    assert sr.identify_pl854_source_reference_ids(df) == ["A", "B"]


def test_identify_source_reference_ids_empty_when_all_null():
    df = pd.DataFrame({"source_reference_id": [None, None]})
    assert sr.identify_pl854_source_reference_ids(df) == []


# --- compare_source_reference_and_cdi_footprints (Section 10) ---------------


def test_compare_footprints_neither_available():
    result = sr.compare_source_reference_and_cdi_footprints(None, None)
    assert result.both_available is False
    assert result.overlaps is None


def test_compare_footprints_only_one_available():
    poly = _to_wgs84(box(500000, 5900000, 500100, 5900100))
    result = sr.compare_source_reference_and_cdi_footprints(poly, None)
    assert result.both_available is False
    assert "Only one footprint" in result.notes


def test_compare_footprints_overlapping_not_materially_different():
    emodnet_poly = _to_wgs84(box(500000, 5900000, 500100, 5900100))
    cdi_poly = _to_wgs84(box(499990, 5899990, 500110, 5900110))  # near-identical, slightly larger

    result = sr.compare_source_reference_and_cdi_footprints(emodnet_poly, cdi_poly)

    assert result.both_available is True
    assert result.overlaps is True
    assert result.materially_different is False


def test_compare_footprints_barely_overlapping_is_materially_different():
    emodnet_poly = _to_wgs84(box(500000, 5900000, 500100, 5900100))
    # A CDI bbox far away, touching only a sliver of the EMODnet polygon's corner.
    cdi_poly = _to_wgs84(box(500099, 5900099, 600000, 6000000))

    result = sr.compare_source_reference_and_cdi_footprints(emodnet_poly, cdi_poly)

    assert result.both_available is True
    assert result.materially_different is True


# --- build_source_survey_record ---------------------------------------------


def test_build_source_survey_record_owner_permission_maps_to_manual_download():
    source_ref = SourceReferenceFeature(
        identifier="A",
        source_type="CDI",
        edmo_id=2,
        release="2024",
        date_start=None,
        date_end=None,
        metadata_url="https://x",
        geometry_wgs84=_to_wgs84(box(500000, 5900000, 500100, 5900100)),
    )
    cdi_record = _make_cdi_record("A")

    record = sr.build_source_survey_record(source_ref, None, cdi_record)

    assert record.download_available is False
    assert record.manual_download_required is True
    assert record.access_type == cdi.ACCESS_OWNER_PERMISSION_REQUIRED
    assert record.acquisition_year == 2000
    assert record.product_release_year is None  # this is an individual survey, not an aggregate


# --- resolve_pl854_cdi_sources: full pipeline with synthetic geometry -------


@pytest.fixture
def synthetic_scenario(monkeypatch):
    """Two disjoint source-reference polygons along a 2 km synthetic pipeline."""

    pipeline_geom = LineString([(500000.0, 5900000.0), (502000.0, 5900000.0)])
    pipeline_gdf = gpd.GeoDataFrame(
        {"pipeline_id": ["TEST"]}, geometry=[pipeline_geom], crs=WORKING_CRS
    )

    aoi_geom = box(499800.0, 5899800.0, 502200.0, 5900200.0)
    aoi_gdf = gpd.GeoDataFrame({"study_id": ["TEST"]}, geometry=[aoi_geom], crs=WORKING_CRS)

    station_count = 81  # 0, 25, ..., 2000
    chainage_gdf = gpd.GeoDataFrame(
        {
            "pipeline_id": ["TEST"] * station_count,
            "station_index": list(range(station_count)),
            "chainage_m": [i * 25.0 for i in range(station_count)],
            "kp_label": [f"KP {i * 25.0 / 1000.0:.3f}" for i in range(station_count)],
        },
        geometry=[Point(500000.0 + i * 25.0, 5900000.0) for i in range(station_count)],
        crs=WORKING_CRS,
    )

    poly_a = box(499900.0, 5899900.0, 501000.0, 5900100.0)  # covers chainage 0-1000
    poly_b = box(501000.0, 5899900.0, 502100.0, 5900100.0)  # covers chainage 1000-2000

    source_refs = [
        SourceReferenceFeature(
            identifier="SRC_A",
            source_type="CDI",
            edmo_id=2,
            release="2024",
            date_start=None,
            date_end=None,
            metadata_url="https://x/A",
            geometry_wgs84=_to_wgs84(poly_a),
        ),
        SourceReferenceFeature(
            identifier="SRC_B",
            source_type="CDI",
            edmo_id=2,
            release="2024",
            date_start=None,
            date_end=None,
            metadata_url="https://x/B",
            geometry_wgs84=_to_wgs84(poly_b),
        ),
    ]
    qi_features = [
        QualityIndexFeature(
            identifier="SRC_A",
            source_type="CDI",
            combined=50.0,
            horizontal=3,
            vertical=3,
            age=0,
            purpose=3,
            release="2024",
            geometry_wgs84=_to_wgs84(poly_a),
        ),
        QualityIndexFeature(
            identifier="SRC_B",
            source_type="CDI",
            combined=75.0,
            horizontal=3,
            vertical=4,
            age=0,
            purpose=3,
            release="2024",
            geometry_wgs84=_to_wgs84(poly_b),
        ),
    ]

    chainage_bathymetry = pd.DataFrame(
        {"source_reference_id": ["SRC_A"] * 40 + ["SRC_B"] * 40 + [None]}
    )

    cdi_records = {
        "SRC_A": _make_cdi_record(
            "SRC_A", acquisition_year=1990, survey_method="single-beam echosounders"
        ),
        "SRC_B": _make_cdi_record(
            "SRC_B", acquisition_year=2010, survey_method="multibeam echosounder"
        ),
    }
    monkeypatch.setattr(
        cdi, "resolve_cdi_record", lambda source_id, edmo_id: cdi_records[source_id]
    )

    return {
        "pipeline_gdf": pipeline_gdf,
        "aoi_gdf": aoi_gdf,
        "chainage_gdf": chainage_gdf,
        "chainage_bathymetry": chainage_bathymetry,
        "source_refs": source_refs,
        "qi_features": qi_features,
    }


def test_resolve_pl854_cdi_sources_both_ids_present(synthetic_scenario):
    df, records, overlaps = sr.resolve_pl854_cdi_sources(
        pipeline_gdf=synthetic_scenario["pipeline_gdf"],
        aoi_gdf=synthetic_scenario["aoi_gdf"],
        chainage_gdf=synthetic_scenario["chainage_gdf"],
        chainage_bathymetry=synthetic_scenario["chainage_bathymetry"],
        source_reference_features=synthetic_scenario["source_refs"],
        quality_index_features=synthetic_scenario["qi_features"],
        working_crs=WORKING_CRS,
    )

    assert set(df["source_reference_id"]) == {"SRC_A", "SRC_B"}
    assert list(df.columns) == list(sr.CDI_SOURCES_COLUMNS)


def test_resolve_pl854_cdi_sources_pipeline_and_chainage_coverage(synthetic_scenario):
    df, _, _ = sr.resolve_pl854_cdi_sources(
        pipeline_gdf=synthetic_scenario["pipeline_gdf"],
        aoi_gdf=synthetic_scenario["aoi_gdf"],
        chainage_gdf=synthetic_scenario["chainage_gdf"],
        chainage_bathymetry=synthetic_scenario["chainage_bathymetry"],
        source_reference_features=synthetic_scenario["source_refs"],
        quality_index_features=synthetic_scenario["qi_features"],
        working_crs=WORKING_CRS,
    )

    row_a = df[df["source_reference_id"] == "SRC_A"].iloc[0]
    row_b = df[df["source_reference_id"] == "SRC_B"].iloc[0]

    assert row_a["pipeline_overlap_length_m"] == pytest.approx(1000.0, abs=1.0)
    assert row_b["pipeline_overlap_length_m"] == pytest.approx(1000.0, abs=1.0)
    assert row_a["covered_station_count"] == 41  # stations 0..1000 at 25 m steps inclusive
    assert row_b["covered_station_count"] == 41  # stations 1000..2000 inclusive


def test_resolve_pl854_cdi_sources_survey_age_and_qi_fields(synthetic_scenario):
    df, _, _ = sr.resolve_pl854_cdi_sources(
        pipeline_gdf=synthetic_scenario["pipeline_gdf"],
        aoi_gdf=synthetic_scenario["aoi_gdf"],
        chainage_gdf=synthetic_scenario["chainage_gdf"],
        chainage_bathymetry=synthetic_scenario["chainage_bathymetry"],
        source_reference_features=synthetic_scenario["source_refs"],
        quality_index_features=synthetic_scenario["qi_features"],
        working_crs=WORKING_CRS,
    )

    row_a = df[df["source_reference_id"] == "SRC_A"].iloc[0]
    assert row_a["acquisition_year"] == 1990
    assert row_a["survey_age_at_product_release_year"] == cdi.PRODUCT_RELEASE_YEAR - 1990
    assert row_a["qi_age"] == 0
    assert row_a["qi_vertical"] == 3
    assert row_a["qi_metadata_consistency"] == cdi.CONSISTENCY_CONSISTENT


def test_resolve_pl854_cdi_sources_missing_source_ref_feature_is_skipped_not_fabricated(
    synthetic_scenario,
):
    chainage_bathymetry = pd.DataFrame({"source_reference_id": ["SRC_A", "UNKNOWN_ID"]})

    df, _, _ = sr.resolve_pl854_cdi_sources(
        pipeline_gdf=synthetic_scenario["pipeline_gdf"],
        aoi_gdf=synthetic_scenario["aoi_gdf"],
        chainage_gdf=synthetic_scenario["chainage_gdf"],
        chainage_bathymetry=chainage_bathymetry,
        source_reference_features=synthetic_scenario["source_refs"],
        quality_index_features=synthetic_scenario["qi_features"],
        working_crs=WORKING_CRS,
    )

    assert set(df["source_reference_id"]) == {
        "SRC_A"
    }  # UNKNOWN_ID has no real WFS feature -- skipped


def test_resolve_pl854_cdi_sources_is_deterministic(synthetic_scenario):
    def run():
        return sr.resolve_pl854_cdi_sources(
            pipeline_gdf=synthetic_scenario["pipeline_gdf"],
            aoi_gdf=synthetic_scenario["aoi_gdf"],
            chainage_gdf=synthetic_scenario["chainage_gdf"],
            chainage_bathymetry=synthetic_scenario["chainage_bathymetry"],
            source_reference_features=synthetic_scenario["source_refs"],
            quality_index_features=synthetic_scenario["qi_features"],
            working_crs=WORKING_CRS,
        )

    df_a, _, _ = run()
    df_b, _, _ = run()
    pd.testing.assert_frame_equal(df_a, df_b)


# --- multi-epoch overlap detection (Section 11) -----------------------------


def _survey_record(
    key: str,
    ranges: tuple,
    acquisition_year: int,
    download_available: bool = False,
    manual_download_required: bool = True,
    vertical_datum: str | None = None,
    resolution_status: str = cdi.RESOLUTION_LIVE,
) -> SurveyRecord:
    return SurveyRecord(
        source="EMODnet-CDI",
        source_dataset_id=key,
        acquisition_year=acquisition_year,
        download_available=download_available,
        manual_download_required=manual_download_required,
        vertical_datum=vertical_datum,
        acquisition_status=resolution_status,
        chainage_coverage_ranges=ranges,
        footprint_available=True,
    )


def test_detect_multi_epoch_overlaps_finds_real_overlap_with_different_years():
    records = [
        _survey_record("A", ((0.0, 1000.0),), acquisition_year=1990),
        _survey_record("B", ((500.0, 1500.0),), acquisition_year=2010),
    ]
    cdi_records = {"A": _make_cdi_record("A"), "B": _make_cdi_record("B")}

    overlaps = sr._detect_multi_epoch_overlaps(records, cdi_records)

    assert len(overlaps) == 1
    assert overlaps[0].chainage_start_m == 500.0
    assert overlaps[0].chainage_end_m == 1000.0
    assert set(overlaps[0].acquisition_years) == {1990, 2010}


def test_detect_multi_epoch_overlaps_no_overlap_for_disjoint_ranges():
    records = [
        _survey_record("A", ((0.0, 1000.0),), acquisition_year=1990),
        _survey_record("B", ((1000.0, 2000.0),), acquisition_year=2010),
    ]
    cdi_records = {"A": _make_cdi_record("A"), "B": _make_cdi_record("B")}

    overlaps = sr._detect_multi_epoch_overlaps(records, cdi_records)

    assert overlaps == []


def test_detect_multi_epoch_overlaps_ignores_same_epoch_double_coverage():
    """Two records covering the same segment but the SAME acquisition year
    is not a multi-temporal opportunity -- there is only one real epoch."""

    records = [
        _survey_record("A", ((0.0, 1000.0),), acquisition_year=1991),
        _survey_record("B", ((500.0, 1500.0),), acquisition_year=1991),
    ]
    cdi_records = {"A": _make_cdi_record("A"), "B": _make_cdi_record("B")}

    overlaps = sr._detect_multi_epoch_overlaps(records, cdi_records)

    assert overlaps == []


def test_classify_delta_z_readiness_datum_harmonisation_required_when_datums_differ():
    covering = (
        _survey_record("A", ((0.0, 1000.0),), 1990, vertical_datum="LAT"),
        _survey_record("B", ((500.0, 1500.0),), 2010, vertical_datum=None),
    )
    assert sr._classify_delta_z_readiness(covering) == sr.DELTA_Z_DATUM_HARMONISATION_REQUIRED


def test_classify_delta_z_readiness_source_request_required_when_manual_download_needed():
    covering = (
        _survey_record(
            "A", ((0.0, 1000.0),), 1990, vertical_datum="LAT", manual_download_required=True
        ),
        _survey_record(
            "B", ((500.0, 1500.0),), 2010, vertical_datum="LAT", manual_download_required=True
        ),
    )
    assert sr._classify_delta_z_readiness(covering) == sr.DELTA_Z_SOURCE_REQUEST_REQUIRED


def test_classify_delta_z_readiness_metadata_only_when_neither_downloadable_nor_requestable():
    covering = (
        _survey_record(
            "A",
            ((0.0, 1000.0),),
            1990,
            vertical_datum="LAT",
            download_available=False,
            manual_download_required=False,
        ),
        _survey_record(
            "B",
            ((500.0, 1500.0),),
            2010,
            vertical_datum="LAT",
            download_available=False,
            manual_download_required=False,
        ),
    )
    assert sr._classify_delta_z_readiness(covering) == sr.DELTA_Z_METADATA_ONLY


def test_classify_delta_z_readiness_not_verifiable_when_source_unavailable():
    covering = (
        _survey_record("A", ((0.0, 1000.0),), 1990, resolution_status=cdi.RESOLUTION_UNAVAILABLE),
        _survey_record("B", ((500.0, 1500.0),), 2010),
    )
    assert sr._classify_delta_z_readiness(covering) == sr.DELTA_Z_NOT_VERIFIABLE


# --- output writing ----------------------------------------------------------


def test_write_cdi_sources_parquet_round_trips(tmp_path: Path):
    df = pd.DataFrame([{"source_reference_id": "A"}], columns=list(sr.CDI_SOURCES_COLUMNS))
    out_path = tmp_path / "emodnet_cdi_sources.parquet"

    result_path = sr.write_cdi_sources_parquet(df, out_path)

    assert result_path == out_path
    reloaded = pd.read_parquet(out_path)
    assert reloaded["source_reference_id"].iloc[0] == "A"


def test_write_cdi_sources_gpkg_writes_valid_geometry(tmp_path: Path):
    record = SurveyRecord(
        source="EMODnet-CDI",
        source_dataset_id="A",
        geometry_wgs84=_to_wgs84(box(500000.0, 5900000.0, 500100.0, 5900100.0)),
    )
    out_path = tmp_path / "emodnet_cdi_sources.gpkg"

    result_path = sr.write_cdi_sources_gpkg([record], WORKING_CRS, out_path)

    assert result_path == out_path
    gdf = gpd.read_file(out_path, layer="emodnet_cdi_sources")
    assert len(gdf) == 1
    assert gdf.crs.to_string() == WORKING_CRS


def test_write_cdi_sources_gpkg_returns_none_without_geometry(tmp_path: Path):
    record = SurveyRecord(source="EMODnet-CDI", source_dataset_id="A", geometry_wgs84=None)
    result = sr.write_cdi_sources_gpkg([record], WORKING_CRS, tmp_path / "out.gpkg")
    assert result is None


# --- report printing ----------------------------------------------------------


def test_print_source_resolution_report_includes_lettered_answers(synthetic_scenario):
    import io

    df, _, overlaps = sr.resolve_pl854_cdi_sources(
        pipeline_gdf=synthetic_scenario["pipeline_gdf"],
        aoi_gdf=synthetic_scenario["aoi_gdf"],
        chainage_gdf=synthetic_scenario["chainage_gdf"],
        chainage_bathymetry=synthetic_scenario["chainage_bathymetry"],
        source_reference_features=synthetic_scenario["source_refs"],
        quality_index_features=synthetic_scenario["qi_features"],
        working_crs=WORKING_CRS,
    )

    buffer = io.StringIO()
    sr.print_source_resolution_report(df, overlaps, file=buffer)
    output = buffer.getvalue()

    for letter in ("A.", "B.", "C.", "D.", "E.", "F."):
        assert letter in output
    assert "SRC_A" in output
    assert "SRC_B" in output
