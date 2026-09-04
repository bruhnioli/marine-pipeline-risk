"""Offline unit tests for marine_engine.providers.bathymetry.inventory.

Uses small synthetic line/polygon geometries -- never the real PL854 route
-- and never touches the network.
"""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from marine_engine.providers.bathymetry import inventory as inv

PIPELINE_LINE = LineString([(500000.0, 5900000.0), (501000.0, 5900000.0)])  # 1000 m
AOI_POLYGON = PIPELINE_LINE.buffer(300.0)
WORKING_CRS = "EPSG:32631"


def _chainage_gdf(interval_m: float = 25.0) -> gpd.GeoDataFrame:
    rows = []
    chainage = 0.0
    index = 0
    while chainage <= PIPELINE_LINE.length + 1e-9:
        rows.append(
            {
                "station_index": index,
                "chainage_m": chainage,
                "geometry": PIPELINE_LINE.interpolate(chainage),
            }
        )
        chainage += interval_m
        index += 1
    return gpd.GeoDataFrame(rows, crs=WORKING_CRS)


CHAINAGE_GDF = _chainage_gdf()


def _record(**overrides) -> inv.SurveyRecord:
    fields = {"source": "TEST", "source_dataset_id": "T1"}
    fields.update(overrides)
    return inv.SurveyRecord(**fields)


# --- resolution classification -----------------------------------------------


@pytest.mark.parametrize(
    ("resolution", "expected"),
    [
        (None, "unknown"),
        (0.5, "high"),
        (2.0, "high"),
        (5.0, "medium"),
        (10.0, "low"),
        (115.0, "low"),
    ],
)
def test_classify_resolution(resolution, expected):
    assert inv.classify_resolution(resolution) == expected


def test_resolution_is_preserved_not_altered_by_classification():
    record = _record(nominal_resolution_m=5.0)
    result = inv.calculate_spatial_coverage(record, PIPELINE_LINE, AOI_POLYGON, WORKING_CRS)

    assert result.nominal_resolution_m == 5.0  # unchanged
    assert result.resolution_class == "medium"  # only the derived field changes


# --- null metadata preservation ----------------------------------------------


def test_missing_metadata_stays_none_not_fabricated():
    record = _record()  # everything but source/id defaults to None

    assert record.survey_start_date is None
    assert record.nominal_resolution_m is None
    assert record.vertical_datum is None
    assert record.horizontal_crs is None

    df = inv.build_inventory_dataframe([record])
    row = df.iloc[0]
    assert row["survey_start_date"] is None
    assert row["nominal_resolution_m"] is None
    assert row["vertical_datum"] is None


def test_vertical_datum_preserved_exactly_not_normalized():
    lat_record = _record(source_dataset_id="A", vertical_datum="LAT")
    chart_datum_record = _record(source_dataset_id="B", vertical_datum="Chart Datum")
    unknown_record = _record(source_dataset_id="C", vertical_datum=None)

    # Never coerced to a shared value even though LAT and Chart Datum are
    # often the same thing in practice -- only if the source says so.
    assert lat_record.vertical_datum == "LAT"
    assert chart_datum_record.vertical_datum == "Chart Datum"
    assert unknown_record.vertical_datum is None


def test_temporal_epoch_preserved_through_dataframe():
    record = _record(temporal_epoch="2003")
    df = inv.build_inventory_dataframe([record])
    assert df.iloc[0]["temporal_epoch"] == "2003"


# --- AOI / pipeline spatial intersection -------------------------------------


def test_calculate_spatial_coverage_no_footprint_stays_unverified():
    record = _record(geometry_wgs84=None)
    result = inv.calculate_spatial_coverage(record, PIPELINE_LINE, AOI_POLYGON, WORKING_CRS)

    assert result.footprint_available is False
    assert result.intersects_aoi is None
    assert result.intersects_pipeline is None
    assert result.overlap_aoi_km2 is None
    assert result.pipeline_overlap_length_m is None


def test_calculate_spatial_coverage_full_overlap():
    # A footprint that fully covers the AOI, expressed in WGS84 (as sources report it).
    huge_footprint_wgs84 = (
        gpd.GeoSeries([box(499000, 5899000, 502000, 5901000)], crs=WORKING_CRS)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    record = _record(geometry_wgs84=huge_footprint_wgs84)

    result = inv.calculate_spatial_coverage(record, PIPELINE_LINE, AOI_POLYGON, WORKING_CRS)

    assert result.footprint_available is True
    assert result.intersects_aoi is True
    assert result.overlap_aoi_percent == pytest.approx(100.0, abs=0.5)
    assert result.intersects_pipeline is True
    assert result.pipeline_overlap_length_m == pytest.approx(PIPELINE_LINE.length, rel=0.01)
    assert result.pipeline_overlap_percent == pytest.approx(100.0, abs=0.5)


def test_calculate_spatial_coverage_partial_pipeline_overlap():
    # Covers only the first half of the pipeline (and part of the AOI).
    half_footprint_wgs84 = (
        gpd.GeoSeries([box(499900, 5899900, 500500, 5900100)], crs=WORKING_CRS)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    record = _record(geometry_wgs84=half_footprint_wgs84)

    result = inv.calculate_spatial_coverage(record, PIPELINE_LINE, AOI_POLYGON, WORKING_CRS)

    assert result.intersects_pipeline is True
    assert result.pipeline_overlap_length_m == pytest.approx(500.0, rel=0.01)
    assert result.pipeline_overlap_percent == pytest.approx(50.0, rel=0.02)


def test_calculate_spatial_coverage_never_infers_from_absence_of_geometry():
    # A record with a suggestive title but no geometry must NOT be reported
    # as intersecting -- only "not verified".
    record = _record(title="Right next to the PL854 pipeline!", geometry_wgs84=None)
    result = inv.calculate_spatial_coverage(record, PIPELINE_LINE, AOI_POLYGON, WORKING_CRS)

    assert result.intersects_pipeline is None
    assert result.intersects_aoi is None


def test_calculate_spatial_coverage_disjoint_footprint_reports_no_intersection():
    far_away_wgs84 = gpd.GeoSeries([box(0, 0, 1, 1)], crs="EPSG:4326").iloc[0]
    record = _record(geometry_wgs84=far_away_wgs84)

    result = inv.calculate_spatial_coverage(record, PIPELINE_LINE, AOI_POLYGON, WORKING_CRS)

    assert result.footprint_available is True
    assert result.intersects_aoi is False
    assert result.intersects_pipeline is False


# --- chainage coverage --------------------------------------------------------


def test_calculate_chainage_coverage_single_contiguous_range():
    half_footprint_wgs84 = (
        gpd.GeoSeries([box(499900, 5899900, 500500, 5900100)], crs=WORKING_CRS)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    record = _record(geometry_wgs84=half_footprint_wgs84)

    result = inv.calculate_chainage_coverage(record, CHAINAGE_GDF, WORKING_CRS)

    assert result.covered_chainage_station_count == 21  # 0,25,...,500 inclusive
    assert result.first_covered_chainage_m == 0.0
    assert result.last_covered_chainage_m == 500.0
    assert result.chainage_coverage_ranges == ((0.0, 500.0),)


def test_calculate_chainage_coverage_disjoint_ranges_kept_separate():
    # Two separate boxes covering the start and end of the pipeline, with a
    # gap in the middle -- must be reported as two ranges, not one span.
    box_a = box(499900, 5899950, 500200, 5900050)  # ~chainage 0-200
    box_b = box(500800, 5899950, 501100, 5900050)  # ~chainage 800-1000
    footprint_wgs84 = (
        gpd.GeoSeries([box_a.union(box_b)], crs=WORKING_CRS).to_crs("EPSG:4326").iloc[0]
    )
    record = _record(geometry_wgs84=footprint_wgs84)

    result = inv.calculate_chainage_coverage(record, CHAINAGE_GDF, WORKING_CRS)

    assert len(result.chainage_coverage_ranges) == 2
    first_range, second_range = result.chainage_coverage_ranges
    assert first_range[0] == 0.0
    assert second_range[1] == 1000.0
    assert first_range[1] < second_range[0]  # a real gap between the two ranges


def test_calculate_chainage_coverage_no_footprint_leaves_fields_none():
    record = _record(geometry_wgs84=None)
    result = inv.calculate_chainage_coverage(record, CHAINAGE_GDF, WORKING_CRS)

    assert result.covered_chainage_station_count is None
    assert result.chainage_coverage_ranges == ()


# --- ranking -------------------------------------------------------------------


def test_rank_candidates_prioritizes_pipeline_intersection():
    intersecting = _record(
        source_dataset_id="A", intersects_pipeline=True, pipeline_overlap_length_m=10.0
    )
    non_intersecting = _record(source_dataset_id="B", intersects_pipeline=False)

    ranked = inv.rank_candidates([non_intersecting, intersecting])

    assert ranked[0].source_dataset_id == "A"


def test_rank_candidates_prioritizes_longer_pipeline_coverage():
    short = _record(
        source_dataset_id="SHORT", intersects_pipeline=True, pipeline_overlap_length_m=10.0
    )
    long = _record(
        source_dataset_id="LONG", intersects_pipeline=True, pipeline_overlap_length_m=500.0
    )

    ranked = inv.rank_candidates([short, long])

    assert ranked[0].source_dataset_id == "LONG"


def test_rank_candidates_prioritizes_finer_resolution():
    coarse = _record(
        source_dataset_id="COARSE", intersects_pipeline=True, nominal_resolution_m=100.0
    )
    fine = _record(source_dataset_id="FINE", intersects_pipeline=True, nominal_resolution_m=1.0)

    ranked = inv.rank_candidates([coarse, fine])

    assert ranked[0].source_dataset_id == "FINE"


def test_rank_candidates_is_deterministic_across_repeated_calls():
    records = [
        _record(source_dataset_id="A", intersects_pipeline=True, nominal_resolution_m=5.0),
        _record(source_dataset_id="B", intersects_pipeline=True, nominal_resolution_m=5.0),
        _record(source_dataset_id="C", intersects_pipeline=False),
    ]

    first = [r.source_dataset_id for r in inv.rank_candidates(records)]
    second = [r.source_dataset_id for r in inv.rank_candidates(list(reversed(records)))]

    assert first == second


def test_select_primary_candidate_excludes_emodnet():
    emodnet = _record(
        source="EMODnet",
        source_dataset_id="emodnet__mean",
        intersects_pipeline=True,
        download_available=True,
    )
    historical = _record(
        source="BGS", source_dataset_id="X", intersects_pipeline=True, download_available=True
    )

    ranked = inv.rank_candidates([emodnet, historical])
    primary = inv.select_primary_candidate(ranked)

    assert primary is not None
    assert primary.source == "BGS"


def test_select_primary_candidate_none_when_nothing_qualifies():
    emodnet = _record(source="EMODnet", intersects_pipeline=True, download_available=True)
    metadata_only = _record(
        source="BGS", intersects_pipeline=True, download_available=False, source_dataset_id="X"
    )

    ranked = inv.rank_candidates([emodnet, metadata_only])
    primary = inv.select_primary_candidate(ranked)

    assert primary is None


def test_apply_primary_role_adds_tag_only_to_selected_record():
    a = _record(source_dataset_id="A", intersects_pipeline=True, download_available=True)
    b = _record(source_dataset_id="B", intersects_pipeline=False)

    tagged = inv.apply_primary_role([a, b], a)

    tagged_a = next(r for r in tagged if r.source_dataset_id == "A")
    tagged_b = next(r for r in tagged if r.source_dataset_id == "B")
    assert inv.ROLE_PRIMARY_ANALYSIS_CANDIDATE in tagged_a.potential_role
    assert inv.ROLE_PRIMARY_ANALYSIS_CANDIDATE not in tagged_b.potential_role


# --- temporal overlap detection and classification ---------------------------


def _record_with_ranges(dataset_id: str, ranges: tuple, **overrides) -> inv.SurveyRecord:
    return _record(source_dataset_id=dataset_id, chainage_coverage_ranges=ranges, **overrides)


def test_find_temporal_overlap_segments_detects_real_overlap():
    a = _record_with_ranges("A", ((0.0, 500.0),), vertical_datum="LAT", download_available=True)
    b = _record_with_ranges("B", ((400.0, 1000.0),), vertical_datum="LAT", download_available=True)

    segments = inv.find_temporal_overlap_segments([a, b])

    assert len(segments) == 1
    assert segments[0].chainage_start_m == 400.0
    assert segments[0].chainage_end_m == 500.0
    assert {r.record_key for r in segments[0].records} == {"A", "B"}


def test_find_temporal_overlap_segments_none_when_disjoint():
    a = _record_with_ranges("A", ((0.0, 400.0),))
    b = _record_with_ranges("B", ((500.0, 1000.0),))

    assert inv.find_temporal_overlap_segments([a, b]) == []


def test_find_temporal_overlap_segments_none_with_fewer_than_two_spatial_records():
    a = _record_with_ranges("A", ((0.0, 400.0),))
    b = _record(source_dataset_id="B")  # no footprint/ranges at all

    assert inv.find_temporal_overlap_segments([a, b]) == []


def test_classify_temporal_pair_ready_for_delta_z():
    a = _record(vertical_datum="LAT", download_available=True, manual_download_required=False)
    b = _record(vertical_datum="LAT", download_available=True, manual_download_required=False)

    tags = inv.classify_temporal_pair((a, b))

    assert inv.TAG_READY_FOR_FUTURE_DELTA_Z in tags
    assert inv.TAG_DATUM_HARMONISATION_REQUIRED not in tags


def test_classify_temporal_pair_datum_harmonisation_required():
    a = _record(vertical_datum="LAT")
    b = _record(vertical_datum="Chart Datum")

    tags = inv.classify_temporal_pair((a, b))

    assert inv.TAG_DATUM_HARMONISATION_REQUIRED in tags


def test_classify_temporal_pair_unknown_datum_also_requires_harmonisation():
    a = _record(vertical_datum="LAT")
    b = _record(vertical_datum=None)

    assert inv.TAG_DATUM_HARMONISATION_REQUIRED in inv.classify_temporal_pair((a, b))


def test_classify_temporal_pair_requires_manual_download():
    a = _record(download_available=True, manual_download_required=False, vertical_datum="LAT")
    b = _record(download_available=False, manual_download_required=True, vertical_datum="LAT")

    assert inv.TAG_REQUIRES_MANUAL_DOWNLOAD in inv.classify_temporal_pair((a, b))


def test_classify_temporal_pair_metadata_only():
    a = _record(download_available=False, vertical_datum="LAT")
    b = _record(download_available=False, vertical_datum="LAT")

    assert inv.TAG_METADATA_ONLY in inv.classify_temporal_pair((a, b))


def test_classify_temporal_pair_no_valid_pair_fallback():
    # download unknown (None -> falsy) and datums differ -> at minimum
    # DATUM_HARMONISATION_REQUIRED, never an empty tag tuple.
    a = _record(vertical_datum="LAT", download_available=None)
    b = _record(vertical_datum=None, download_available=None)

    tags = inv.classify_temporal_pair((a, b))
    assert len(tags) >= 1


# --- canonical schema / output writing ---------------------------------------


def test_build_inventory_dataframe_has_all_canonical_columns():
    df = inv.build_inventory_dataframe([_record()])
    assert list(df.columns) == list(inv.CANONICAL_COLUMNS)


def test_build_inventory_gdf_none_when_no_footprints():
    assert inv.build_inventory_gdf([_record(geometry_wgs84=None)], WORKING_CRS) is None


def test_write_inventory_writes_parquet_always_and_gpkg_when_footprint_exists(tmp_path: Path):
    footprint_wgs84 = (
        gpd.GeoSeries([box(499000, 5899000, 502000, 5901000)], crs=WORKING_CRS)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    with_footprint = inv.calculate_spatial_coverage(
        _record(source_dataset_id="A", geometry_wgs84=footprint_wgs84),
        PIPELINE_LINE,
        AOI_POLYGON,
        WORKING_CRS,
    )
    without_footprint = _record(source_dataset_id="B", geometry_wgs84=None)

    parquet_path = tmp_path / "inventory.parquet"
    gpkg_path = tmp_path / "inventory.gpkg"
    gdf = inv.write_inventory(
        [with_footprint, without_footprint], parquet_path, gpkg_path, WORKING_CRS
    )

    assert parquet_path.exists()
    assert gpkg_path.exists()
    assert gdf is not None
    assert len(gdf) == 1  # only the record with a footprint

    import pandas as pd

    df = pd.read_parquet(parquet_path)
    assert len(df) == 2  # both records, footprint or not


def test_write_inventory_skips_gpkg_when_no_footprints(tmp_path: Path):
    parquet_path = tmp_path / "inventory.parquet"
    gpkg_path = tmp_path / "inventory.gpkg"

    gdf = inv.write_inventory([_record(geometry_wgs84=None)], parquet_path, gpkg_path, WORKING_CRS)

    assert parquet_path.exists()
    assert not gpkg_path.exists()
    assert gdf is None


# --- run_discovery end-to-end (source-agnostic orchestration) ---------------


def test_run_discovery_end_to_end(tmp_path: Path):
    footprint_wgs84 = (
        gpd.GeoSeries([box(499000, 5899000, 502000, 5901000)], crs=WORKING_CRS)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    baseline = _record(
        source="EMODnet",
        source_dataset_id="emodnet__mean",
        geometry_wgs84=footprint_wgs84,
        download_available=True,
    )

    report = inv.run_discovery(
        [baseline],
        pipeline_geom_working=PIPELINE_LINE,
        aoi_geom_working=AOI_POLYGON,
        chainage_gdf_working=CHAINAGE_GDF,
        working_crs=WORKING_CRS,
        parquet_path=tmp_path / "inventory.parquet",
        gpkg_path=tmp_path / "inventory.gpkg",
    )

    assert len(report.ranked_records) == 1
    assert report.baseline_candidate is not None
    assert report.baseline_candidate.source == "EMODnet"
    assert report.primary_candidate is None  # EMODnet never qualifies as primary
    assert report.parquet_path.exists()
    assert report.gpkg_path is not None and report.gpkg_path.exists()


def test_print_discovery_report_runs_without_error(tmp_path: Path):
    import io

    report = inv.run_discovery(
        [_record(geometry_wgs84=None)],
        pipeline_geom_working=PIPELINE_LINE,
        aoi_geom_working=AOI_POLYGON,
        chainage_gdf_working=CHAINAGE_GDF,
        working_crs=WORKING_CRS,
        parquet_path=tmp_path / "inventory.parquet",
        gpkg_path=tmp_path / "inventory.gpkg",
    )

    buffer = io.StringIO()
    inv.print_discovery_report(report, file=buffer)

    assert "NONE FOUND" in buffer.getvalue()  # no temporal overlaps with a single record
