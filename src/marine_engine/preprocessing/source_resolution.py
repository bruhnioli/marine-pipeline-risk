"""Resolve PL854's EMODnet CDI source surveys to real acquisition provenance (MAR-006B).

MAR-006 attributed every PL854 chainage station to one EMODnet
`source_references` polygon and its QI classes, but left those polygons as
opaque ids. This module follows each id's own official `metadata_url` to
its SeaDataNet CDI record (`providers/bathymetry/cdi.py`) to recover who
surveyed it, when, with what instrument, under what access terms, and
whether the resulting ~115 m EMODnet composite could be replaced locally by
requesting the original higher-resolution survey.

Reuses `providers/bathymetry/inventory.py`'s pipeline/chainage-overlap and
multi-epoch-segment machinery by wrapping each resolved source as a
`SurveyRecord` -- this is metadata resolution and coverage geometry, not a
new discovery/ranking framework, so no scoring or role-assignment logic
from that module is used here.

Scope: no morphology, no erosion/deposition, no scour/free-span/risk, no
LAT/MSL work, no automatic SeaDataNet data requests -- those are explicitly
out of bounds for this ticket.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from marine_engine.providers.bathymetry import cdi
from marine_engine.providers.bathymetry.emodnet import QualityIndexFeature, SourceReferenceFeature
from marine_engine.providers.bathymetry.inventory import (
    SurveyRecord,
    calculate_chainage_coverage,
    calculate_spatial_coverage,
    find_temporal_overlap_segments,
)

CDI_SOURCES_COLUMNS = (
    "source_reference_id",
    "source_type",
    "product_name",
    "product_release_year",
    "cdi_identifier",
    "title",
    "organisation",
    "edmo_id",
    "survey_name",
    "platform",
    "acquisition_start",
    "acquisition_end",
    "acquisition_year",
    "survey_age_at_product_release_year",
    "survey_method",
    "device",
    "source_resolution_m",
    "horizontal_accuracy_or_resolution",
    "vertical_accuracy_or_resolution",
    "horizontal_crs",
    "vertical_datum",
    "licence",
    "access_class",
    "request_identifier",
    "metadata_url",
    "data_access_url",
    "pipeline_overlap_length_m",
    "pipeline_overlap_percent",
    "covered_station_count",
    "chainage_ranges",
    "qi_age",
    "qi_horizontal",
    "qi_vertical",
    "qi_purpose",
    "qi_combined",
    "qi_metadata_consistency",
    "recovery_potential",
    "notes",
)

PRODUCT_NAME = "EMODnet Digital Bathymetry (DTM 2024)"

# Section 11 (multi-temporal opportunity) classification.
DELTA_Z_READY = "READY_FOR_FUTURE_DELTA_Z"
DELTA_Z_SOURCE_REQUEST_REQUIRED = "SOURCE_DATA_REQUEST_REQUIRED"
DELTA_Z_DATUM_HARMONISATION_REQUIRED = "DATUM_HARMONISATION_REQUIRED"
DELTA_Z_RESOLUTION_INCOMPATIBLE = "RESOLUTION_INCOMPATIBLE"
DELTA_Z_METADATA_ONLY = "METADATA_ONLY"
DELTA_Z_NOT_VERIFIABLE = "NOT_VERIFIABLE"


def identify_pl854_source_reference_ids(chainage_bathymetry: pd.DataFrame) -> list[str]:
    """Which source_reference_id values real PL854 chainage stations actually use.

    Derived from MAR-006's own station-level attribution rather than
    guessed or hard-coded -- a re-run against a changed chainage or AOI
    would pick up whichever ids are real then, not a frozen list.
    """

    ids = chainage_bathymetry["source_reference_id"].dropna().unique().tolist()
    return sorted(str(i) for i in ids)


@dataclass(frozen=True)
class FootprintComparison:
    """Section 10: EMODnet source-reference polygon vs the CDI-stated footprint."""

    both_available: bool
    overlaps: bool | None
    emodnet_area_km2: float | None
    cdi_area_km2: float | None
    intersection_over_emodnet_percent: float | None
    materially_different: bool | None
    notes: str


def compare_source_reference_and_cdi_footprints(
    source_reference_geom_wgs84: BaseGeometry | None,
    cdi_geom_wgs84: BaseGeometry | None,
) -> FootprintComparison:
    """Never assumes the two footprints mean the same thing; only compares what's real.

    The EMODnet source_references polygon identifies which source "wins"
    for the composite DTM at that location; the CDI footprint (where
    available) is only survey provenance -- a coarse bounding box in these
    three records, not a detailed extent, so a partial-overlap result here
    is expected and does not mean either geometry is wrong.
    """

    if source_reference_geom_wgs84 is None or cdi_geom_wgs84 is None:
        available = source_reference_geom_wgs84 is not None or cdi_geom_wgs84 is not None
        return FootprintComparison(
            both_available=False,
            overlaps=None,
            emodnet_area_km2=None,
            cdi_area_km2=None,
            intersection_over_emodnet_percent=None,
            materially_different=None,
            notes=(
                "Only one footprint available; comparison not possible."
                if available
                else "Neither footprint is available."
            ),
        )

    # Rough equal-area-ish comparison in degrees^2 is not meaningful at this
    # latitude -- reproject both to the working metric CRS for area/overlap.
    emodnet_working = (
        gpd.GeoSeries([source_reference_geom_wgs84], crs="EPSG:4326").to_crs("EPSG:32631").iloc[0]
    )
    cdi_working = gpd.GeoSeries([cdi_geom_wgs84], crs="EPSG:4326").to_crs("EPSG:32631").iloc[0]

    intersection = emodnet_working.intersection(cdi_working)
    overlaps = not intersection.is_empty
    emodnet_area_km2 = emodnet_working.area / 1_000_000.0
    cdi_area_km2 = cdi_working.area / 1_000_000.0
    intersection_pct = (
        (intersection.area / emodnet_working.area) * 100.0
        if overlaps and emodnet_working.area
        else 0.0
    )
    # The CDI box is a coarse bounding rectangle around the whole cruise, so
    # it is expected to be larger than -- and only partially matched to --
    # the finer EMODnet source polygon; "materially different" here means
    # they barely overlap at all, not that their areas differ.
    materially_different = intersection_pct < 5.0

    return FootprintComparison(
        both_available=True,
        overlaps=overlaps,
        emodnet_area_km2=emodnet_area_km2,
        cdi_area_km2=cdi_area_km2,
        intersection_over_emodnet_percent=intersection_pct,
        materially_different=materially_different,
        notes=(
            "CDI footprint is a coarse survey bounding box, not a detailed extent; "
            "used only as provenance, never substituted for the EMODnet source-reference "
            "polygon that actually determines DTM composition."
        ),
    )


def _format_ranges(ranges: tuple[tuple[float, float], ...]) -> str:
    return ";".join(f"{start:.2f}-{end:.2f}" for start, end in ranges)


def build_source_survey_record(
    source_ref: SourceReferenceFeature,
    qi: QualityIndexFeature | None,
    cdi_record: cdi.CdiRecord,
) -> SurveyRecord:
    """Wrap one resolved CDI source as a `SurveyRecord` to reuse inventory.py's geometry ops."""

    access_class = cdi.classify_access(cdi_record)
    download_available = access_class == cdi.ACCESS_DIRECT_DOWNLOAD
    manual_download_required = access_class in (
        cdi.ACCESS_SEADATANET_REQUEST,
        cdi.ACCESS_REGISTRATION_REQUIRED,
        cdi.ACCESS_OWNER_PERMISSION_REQUIRED,
    )

    return SurveyRecord(
        source="EMODnet-CDI",
        source_dataset_id=source_ref.identifier,
        source_record_url_or_identifier=cdi_record.metadata_url,
        title=cdi_record.title,
        survey_name=cdi_record.survey_name,
        survey_start_date=cdi_record.acquisition_start.isoformat()
        if cdi_record.acquisition_start
        else None,
        survey_end_date=cdi_record.acquisition_end.isoformat()
        if cdi_record.acquisition_end
        else None,
        acquisition_year=cdi_record.acquisition_year,
        product_release_year=None,  # this IS an individual survey, not an aggregate product
        data_type="source survey (via EMODnet composite)",
        survey_method=cdi_record.survey_method,
        nominal_resolution_m=None,  # never stated by the source; not fabricated
        horizontal_crs=cdi_record.horizontal_crs,
        vertical_datum=cdi_record.vertical_datum,
        licence=cdi_record.licence_code,
        access_type=access_class,
        download_available=download_available,
        manual_download_required=manual_download_required,
        acquisition_status=cdi_record.resolution_status,
        temporal_epoch=str(cdi_record.acquisition_year) if cdi_record.acquisition_year else None,
        notes=cdi_record.notes,
        geometry_wgs84=source_ref.geometry_wgs84,
    )


def resolve_pl854_cdi_sources(
    *,
    pipeline_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    chainage_gdf: gpd.GeoDataFrame,
    chainage_bathymetry: pd.DataFrame,
    source_reference_features: list[SourceReferenceFeature],
    quality_index_features: list[QualityIndexFeature],
    working_crs: str,
) -> tuple[pd.DataFrame, list[SurveyRecord], list["MultiEpochOverlap"]]:
    """End-to-end MAR-006B resolution for the real PL854 route.

    Returns the canonical output DataFrame, the underlying per-source
    `SurveyRecord`s (for callers that also want the coverage geometry), and
    any detected multi-epoch chainage overlaps.
    """

    target_ids = identify_pl854_source_reference_ids(chainage_bathymetry)

    source_ref_by_id = {f.identifier: f for f in source_reference_features if f.identifier}
    qi_by_id = {f.identifier: f for f in quality_index_features if f.identifier}

    pipeline_geom_working = (
        gpd.GeoSeries(pipeline_gdf.geometry, crs=pipeline_gdf.crs).to_crs(working_crs).union_all()
    )
    aoi_geom_working = (
        gpd.GeoSeries(aoi_gdf.geometry, crs=aoi_gdf.crs).to_crs(working_crs).union_all()
    )
    chainage_working = chainage_gdf.to_crs(working_crs)

    records: list[SurveyRecord] = []
    footprint_comparisons: dict[str, FootprintComparison] = {}
    cdi_records: dict[str, cdi.CdiRecord] = {}

    for source_id in target_ids:
        source_ref = source_ref_by_id.get(source_id)
        qi = qi_by_id.get(source_id)
        if source_ref is None:
            continue

        cdi_record = cdi.resolve_cdi_record(source_id, source_ref.edmo_id or 2607)
        cdi_records[source_id] = cdi_record

        footprint_comparisons[source_id] = compare_source_reference_and_cdi_footprints(
            source_ref.geometry_wgs84, cdi_record.geographic_footprint
        )

        record = build_source_survey_record(source_ref, qi, cdi_record)
        record = calculate_spatial_coverage(
            record, pipeline_geom_working, aoi_geom_working, working_crs
        )
        record = calculate_chainage_coverage(record, chainage_working, working_crs)
        records.append(record)

    overlaps = _detect_multi_epoch_overlaps(records, cdi_records)

    df = _build_cdi_sources_dataframe(records, cdi_records, qi_by_id, footprint_comparisons)
    return df, records, overlaps


def _qi_metadata_consistency_for(cdi_record: cdi.CdiRecord, qi: QualityIndexFeature | None) -> str:
    if qi is None:
        return cdi.CONSISTENCY_NOT_VERIFIABLE
    return cdi.classify_qi_metadata_consistency(
        cdi_record,
        qi_age=qi.age,
        qi_horizontal=qi.horizontal,
        qi_vertical=qi.vertical,
        qi_purpose=qi.purpose,
    )


def _build_cdi_sources_dataframe(
    records: list[SurveyRecord],
    cdi_records: dict[str, cdi.CdiRecord],
    qi_by_id: dict[str, QualityIndexFeature],
    footprint_comparisons: dict[str, FootprintComparison],
) -> pd.DataFrame:
    rows = []
    for record in records:
        source_id = record.source_dataset_id
        cdi_record = cdi_records[source_id]
        qi = qi_by_id.get(source_id)
        access_class = cdi.classify_access(cdi_record)
        recovery = cdi.classify_recovery_potential(cdi_record, access_class)
        consistency = _qi_metadata_consistency_for(cdi_record, qi)
        footprint_note = footprint_comparisons[source_id].notes

        rows.append(
            {
                "source_reference_id": source_id,
                "source_type": "CDI",
                "product_name": PRODUCT_NAME,
                "product_release_year": cdi.PRODUCT_RELEASE_YEAR,
                "cdi_identifier": cdi_record.cdi_record_id,
                "title": cdi_record.title,
                "organisation": cdi_record.organisation,
                "edmo_id": cdi_record.organisation_edmo_id,
                "survey_name": cdi_record.survey_name,
                "platform": cdi_record.platform,
                "acquisition_start": cdi_record.acquisition_start.isoformat()
                if cdi_record.acquisition_start
                else None,
                "acquisition_end": cdi_record.acquisition_end.isoformat()
                if cdi_record.acquisition_end
                else None,
                "acquisition_year": cdi_record.acquisition_year,
                "survey_age_at_product_release_year": cdi.calculate_survey_age(
                    cdi_record.acquisition_year, cdi.PRODUCT_RELEASE_YEAR
                ),
                "survey_method": cdi_record.survey_method,
                "device": cdi_record.device,
                "source_resolution_m": None,  # never stated by CDI for these records
                "horizontal_accuracy_or_resolution": cdi_record.horizontal_resolution_note,
                "vertical_accuracy_or_resolution": cdi_record.vertical_resolution_note,
                "horizontal_crs": cdi_record.horizontal_crs,
                "vertical_datum": cdi_record.vertical_datum,
                "licence": cdi_record.licence_code,
                "access_class": access_class,
                "request_identifier": (
                    f"CDI record {cdi_record.cdi_record_id} via {cdi_record.data_centre} "
                    f"({cdi_record.access_mechanism}; {cdi_record.access_restriction})"
                    if cdi_record.cdi_record_id
                    else None
                ),
                "metadata_url": cdi_record.metadata_url,
                "data_access_url": cdi_record.data_access_url,
                "pipeline_overlap_length_m": record.pipeline_overlap_length_m,
                "pipeline_overlap_percent": record.pipeline_overlap_percent,
                "covered_station_count": record.covered_chainage_station_count,
                "chainage_ranges": _format_ranges(record.chainage_coverage_ranges),
                "qi_age": qi.age if qi else None,
                "qi_horizontal": qi.horizontal if qi else None,
                "qi_vertical": qi.vertical if qi else None,
                "qi_purpose": qi.purpose if qi else None,
                "qi_combined": qi.combined if qi else None,
                "qi_metadata_consistency": consistency,
                "recovery_potential": recovery,
                "notes": " | ".join(filter(None, [cdi_record.notes, footprint_note])),
            }
        )
    return pd.DataFrame(rows, columns=list(CDI_SOURCES_COLUMNS))


def write_cdi_sources_parquet(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def write_cdi_sources_gpkg(
    records: list[SurveyRecord], working_crs: str, output_path: Path
) -> Path | None:
    """Optional gpkg of the resolved sources' own EMODnet source-reference polygons."""

    with_geometry = [r for r in records if r.geometry_wgs84 is not None]
    if not with_geometry:
        return None
    geometries = [
        gpd.GeoSeries([r.geometry_wgs84], crs="EPSG:4326").to_crs(working_crs).iloc[0]
        for r in with_geometry
    ]
    gdf = gpd.GeoDataFrame(
        {"source_reference_id": [r.source_dataset_id for r in with_geometry]},
        geometry=geometries,
        crs=working_crs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer="emodnet_cdi_sources")
    return output_path


@dataclass(frozen=True)
class MultiEpochOverlap:
    """Section 11: a chainage segment where >=2 independent acquisition epochs both apply."""

    chainage_start_m: float
    chainage_end_m: float
    source_reference_ids: tuple[str, ...]
    acquisition_years: tuple[int | None, ...]
    classification: str


def _classify_delta_z_readiness(covering: tuple[SurveyRecord, ...]) -> str:
    if any(r.acquisition_status == cdi.RESOLUTION_UNAVAILABLE for r in covering):
        return DELTA_Z_NOT_VERIFIABLE
    if any(r.nominal_resolution_m is None for r in covering):
        # None of these sources state a numeric resolution -- cannot confirm
        # compatibility, but that is different from confirmed incompatible.
        resolution_verdict = None
    else:
        resolution_verdict = "compatible"
    datums = {r.vertical_datum for r in covering}
    if len(datums) > 1 or None in datums:
        return DELTA_Z_DATUM_HARMONISATION_REQUIRED
    if any(r.manual_download_required for r in covering):
        return DELTA_Z_SOURCE_REQUEST_REQUIRED
    if all(not r.download_available and not r.manual_download_required for r in covering):
        return DELTA_Z_METADATA_ONLY
    if resolution_verdict == "compatible" and all(r.download_available for r in covering):
        return DELTA_Z_READY
    return DELTA_Z_NOT_VERIFIABLE


def _detect_multi_epoch_overlaps(
    records: list[SurveyRecord], cdi_records: dict[str, cdi.CdiRecord]
) -> list[MultiEpochOverlap]:
    segments = find_temporal_overlap_segments(records)
    overlaps = []
    for segment in segments:
        years = tuple(r.acquisition_year for r in segment.records)
        if len(set(years)) < 2:
            continue  # same epoch covering the same segment is not a multi-epoch opportunity
        overlaps.append(
            MultiEpochOverlap(
                chainage_start_m=segment.chainage_start_m,
                chainage_end_m=segment.chainage_end_m,
                source_reference_ids=tuple(r.source_dataset_id for r in segment.records),
                acquisition_years=years,
                classification=_classify_delta_z_readiness(segment.records),
            )
        )
    return overlaps


def print_source_resolution_report(
    df: pd.DataFrame,
    overlaps: list[MultiEpochOverlap],
    *,
    file: Any = None,
) -> None:
    """Print the Section 18 required table plus the lettered A-F answers."""

    file = file or sys.stdout
    lines = ["=== PL854 EMODnet CDI Source Resolution ===", ""]

    for _, row in df.iterrows():
        acquired = (
            f"{row['acquisition_start']} to {row['acquisition_end']}"
            if row["acquisition_start"]
            else "unknown"
        )
        lines.append(f"--- {row['source_reference_id']} (CDI {row['cdi_identifier']}) ---")
        lines.append(f"  Title:          {row['title']}")
        lines.append(f"  Organisation:   {row['organisation']}")
        lines.append(f"  Acquired:       {acquired}")
        lines.append(f"  Method/device:  {row['survey_method']}")
        lines.append(
            f"  PL854 chainage: {row['covered_station_count']} stations, "
            f"{row['pipeline_overlap_percent']:.1f}% of route, ranges {row['chainage_ranges']}"
        )
        lines.append(f"  Access:         {row['access_class']}")
        lines.append(f"  Recovery:       {row['recovery_potential']}")
        lines.append("")

    acquisition_years = sorted({int(y) for y in df["acquisition_year"].dropna().unique()})
    lines.append(f"A. Exact source epochs: {acquisition_years or 'none resolved'}")

    mbes_rows = df[df["qi_vertical"] == 4]
    if mbes_rows.empty:
        mbes_answer = "NO"
    else:
        confirmed = mbes_rows[
            mbes_rows["survey_method"]
            .fillna("")
            .str.contains("multibeam|mbes", case=False, regex=True)
        ]
        mbes_answer = "YES" if not confirmed.empty else "NOT VERIFIABLE"
    mbes_ids = ",".join(mbes_rows["source_reference_id"].tolist())
    lines.append(f"B. MBES confirmed: {mbes_answer} (QI_Vertical=4 for: {mbes_ids or 'none'})")

    for _, row in df.iterrows():
        lines.append(
            f"C. {row['source_reference_id']}: access={row['access_class']}, "
            f"recovery={row['recovery_potential']}, request_identifier={row['request_identifier']}"
        )

    multiple_epochs = len(acquisition_years) > 1
    lines.append(
        f"D. Multiple independent source epochs present: {'YES' if multiple_epochs else 'NO'}"
    )

    if overlaps:
        lines.append("E. Multi-temporal overlap: YES")
        for overlap in overlaps:
            lines.append(
                f"   chainage {overlap.chainage_start_m:.2f}-{overlap.chainage_end_m:.2f} m: "
                f"{overlap.source_reference_ids} years={overlap.acquisition_years} "
                f"-> {overlap.classification}"
            )
    else:
        lines.append("E. Multi-temporal overlap: NO")

    old_rows = df[
        df["survey_age_at_product_release_year"].fillna(0) > cdi.QI_AGE_OLD_SURVEY_THRESHOLD_YEARS
    ]
    if old_rows.empty:
        lines.append("F. No resolved source exceeds the 30-year age threshold.")
    else:
        lines.append("F. Portions relying on bathymetry older than 30 years:")
        for _, row in old_rows.iterrows():
            lines.append(
                f"   {row['source_reference_id']} (acquired {row['acquisition_year']}, "
                f"age {row['survey_age_at_product_release_year']}y): "
                f"chainage {row['chainage_ranges']}"
            )

    print("\n".join(lines), file=file)
