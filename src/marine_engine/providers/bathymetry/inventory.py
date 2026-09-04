"""Canonical bathymetry survey inventory: schema, spatial verification, ranking.

Normalizes raw candidate records from each approved source (`ukho.py`,
`bgs.py`, `emodnet.py`) into one flat canonical schema, calculates their
real spatial relationship to the canonical PL854 pipeline/AOI/chainage
(never inferred from a title or approximate coordinates), detects
multi-epoch overlap segments along the pipeline, and produces a
deterministic usefulness ranking.

Scope: MAR-005 discovers and verifies. It does not reproject, mosaic,
harmonise vertical datums, sample bathymetry onto chainage, or compute
seabed change -- those are MAR-006+.
"""

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

# Resolution-class thresholds, for summary display only (never used to alter
# the stored `nominal_resolution_m`). The ticket text only names the boundary
# at exactly 10 m ("= 10 m -> low"); ">10 m" is treated as "low" too, the
# natural extension, rather than left unclassified.
HIGH_RESOLUTION_MAX_M = 2.0
LOW_RESOLUTION_MIN_M = 10.0

ROLE_PRIMARY_ANALYSIS_CANDIDATE = "primary_analysis_candidate"
ROLE_BASELINE = "baseline"
ROLE_VALIDATION_EPOCH_CANDIDATE = "validation_epoch_candidate"
ROLE_METADATA_ONLY_CANDIDATE = "metadata_only_candidate"

TAG_READY_FOR_FUTURE_DELTA_Z = "READY_FOR_FUTURE_DELTA_Z"
TAG_REQUIRES_MANUAL_DOWNLOAD = "REQUIRES_MANUAL_DOWNLOAD"
TAG_DATUM_HARMONISATION_REQUIRED = "DATUM_HARMONISATION_REQUIRED"
TAG_METADATA_ONLY = "METADATA_ONLY"
TAG_NO_VALID_TEMPORAL_PAIR = "NO_VALID_TEMPORAL_PAIR"

CANONICAL_COLUMNS = (
    "source",
    "source_dataset_id",
    "source_record_url_or_identifier",
    "title",
    "survey_name",
    "survey_start_date",
    "survey_end_date",
    "acquisition_year",
    "data_type",
    "survey_method",
    "nominal_resolution_m",
    "resolution_class",
    "horizontal_crs",
    "vertical_datum",
    "licence",
    "access_type",
    "download_available",
    "manual_download_required",
    "acquisition_status",
    "footprint_available",
    "intersects_aoi",
    "overlap_aoi_km2",
    "overlap_aoi_percent",
    "intersects_pipeline",
    "pipeline_overlap_length_m",
    "pipeline_overlap_percent",
    "covered_chainage_station_count",
    "first_covered_chainage_m",
    "last_covered_chainage_m",
    "chainage_coverage_ranges",
    "temporal_epoch",
    "potential_role",
    "notes",
)


@dataclass(frozen=True)
class SurveyRecord:
    """One normalized bathymetry-survey candidate. Missing source facts stay None.

    `geometry_wgs84` is the survey footprint as reported by the source (EPSG:4326),
    kept separate from the flat table columns above -- it is written only to the
    optional GeoPackage output, never fabricated from a title or a single point.
    """

    source: str
    source_dataset_id: str | None = None
    source_record_url_or_identifier: str | None = None
    title: str | None = None
    survey_name: str | None = None
    survey_start_date: str | None = None
    survey_end_date: str | None = None
    acquisition_year: int | None = None
    data_type: str | None = None
    survey_method: str | None = None
    nominal_resolution_m: float | None = None
    horizontal_crs: str | None = None
    vertical_datum: str | None = None
    licence: str | None = None
    access_type: str | None = None
    download_available: bool | None = None
    manual_download_required: bool | None = None
    acquisition_status: str | None = None
    temporal_epoch: str | None = None
    notes: str | None = None
    geometry_wgs84: BaseGeometry | None = None

    # Derived by this module -- never set directly by a source module.
    footprint_available: bool = False
    intersects_aoi: bool | None = None
    overlap_aoi_km2: float | None = None
    overlap_aoi_percent: float | None = None
    intersects_pipeline: bool | None = None
    pipeline_overlap_length_m: float | None = None
    pipeline_overlap_percent: float | None = None
    resolution_class: str | None = None
    covered_chainage_station_count: int | None = None
    first_covered_chainage_m: float | None = None
    last_covered_chainage_m: float | None = None
    chainage_coverage_ranges: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    potential_role: tuple[str, ...] = field(default_factory=tuple)

    @property
    def record_key(self) -> str:
        """A stable identifier for this record, for cross-referencing in reports."""
        return self.source_dataset_id or f"{self.source}:{self.title or 'unknown'}"


@dataclass(frozen=True)
class TemporalOverlapSegment:
    """A chainage range where two or more distinct survey epochs both cover PL854."""

    chainage_start_m: float
    chainage_end_m: float
    records: tuple[SurveyRecord, ...]
    classifications: tuple[str, ...]


def classify_resolution(nominal_resolution_m: float | None) -> str:
    """Bucket a resolution for summary display only; never alters the stored value."""

    if nominal_resolution_m is None:
        return "unknown"
    if nominal_resolution_m <= HIGH_RESOLUTION_MAX_M:
        return "high"
    if nominal_resolution_m < LOW_RESOLUTION_MIN_M:
        return "medium"
    return "low"


def calculate_spatial_coverage(
    record: SurveyRecord,
    pipeline_geom_working: BaseGeometry,
    aoi_geom_working: BaseGeometry,
    working_crs: str,
) -> SurveyRecord:
    """Calculate real AOI/pipeline overlap from the record's own footprint geometry.

    Never infers coverage from a title, installation name, or a bare point.
    If no footprint geometry is available, `footprint_available` stays False
    and all overlap fields stay None -- the caller/report must say coverage
    could not be authoritatively verified, not assume zero or full coverage.
    """

    record = replace(record, resolution_class=classify_resolution(record.nominal_resolution_m))

    if record.geometry_wgs84 is None or record.geometry_wgs84.is_empty:
        return replace(record, footprint_available=False)

    footprint_working = (
        gpd.GeoSeries([record.geometry_wgs84], crs="EPSG:4326").to_crs(working_crs).iloc[0]
    )

    aoi_intersection = footprint_working.intersection(aoi_geom_working)
    intersects_aoi = not aoi_intersection.is_empty
    overlap_aoi_km2 = aoi_intersection.area / 1_000_000.0 if intersects_aoi else None
    overlap_aoi_percent = (
        (aoi_intersection.area / aoi_geom_working.area) * 100.0 if intersects_aoi else None
    )

    pipeline_intersection = pipeline_geom_working.intersection(footprint_working)
    intersects_pipeline = not pipeline_intersection.is_empty
    pipeline_overlap_length_m = pipeline_intersection.length if intersects_pipeline else None
    pipeline_overlap_percent = (
        (pipeline_intersection.length / pipeline_geom_working.length) * 100.0
        if intersects_pipeline
        else None
    )

    return replace(
        record,
        footprint_available=True,
        intersects_aoi=intersects_aoi,
        overlap_aoi_km2=overlap_aoi_km2,
        overlap_aoi_percent=overlap_aoi_percent,
        intersects_pipeline=intersects_pipeline,
        pipeline_overlap_length_m=pipeline_overlap_length_m,
        pipeline_overlap_percent=pipeline_overlap_percent,
    )


def _contiguous_ranges(
    station_indices: list[int], chainages: list[float]
) -> list[tuple[float, float]]:
    """Group a sorted, gap-aware run of station indices into contiguous chainage ranges."""

    if not station_indices:
        return []

    ranges: list[tuple[float, float]] = []
    run_start = 0
    for i in range(1, len(station_indices) + 1):
        if i == len(station_indices) or station_indices[i] != station_indices[i - 1] + 1:
            ranges.append((chainages[run_start], chainages[i - 1]))
            run_start = i
    return ranges


def calculate_chainage_coverage(
    record: SurveyRecord, chainage_gdf_working: gpd.GeoDataFrame, working_crs: str
) -> SurveyRecord:
    """Determine which chainage stations fall inside this record's footprint.

    Represents disjoint intersections as separate contiguous ranges rather
    than assuming one continuous span.
    """

    if record.geometry_wgs84 is None or record.geometry_wgs84.is_empty:
        return record

    footprint_working = (
        gpd.GeoSeries([record.geometry_wgs84], crs="EPSG:4326").to_crs(working_crs).iloc[0]
    )
    covered = chainage_gdf_working[chainage_gdf_working.geometry.intersects(footprint_working)]
    if covered.empty:
        return record

    covered = covered.sort_values("station_index")
    station_indices = covered["station_index"].tolist()
    chainages = covered["chainage_m"].tolist()
    ranges = _contiguous_ranges(station_indices, chainages)

    return replace(
        record,
        covered_chainage_station_count=len(covered),
        first_covered_chainage_m=chainages[0],
        last_covered_chainage_m=chainages[-1],
        chainage_coverage_ranges=tuple(ranges),
    )


def assign_intrinsic_roles(record: SurveyRecord) -> SurveyRecord:
    """Assign the roles derivable from a single record alone (not the whole ranked set).

    `primary_analysis_candidate` is NOT assigned here: it depends on ranking
    across all records and is applied separately by `select_primary_candidate`.
    """

    roles: list[str] = []
    if record.source.upper() == "EMODNET":
        roles.append(ROLE_BASELINE)
    if record.intersects_pipeline:
        roles.append(ROLE_VALIDATION_EPOCH_CANDIDATE)
    if not record.download_available:
        roles.append(ROLE_METADATA_ONLY_CANDIDATE)

    return replace(record, potential_role=tuple(roles))


def _resolution_sort_value(nominal_resolution_m: float | None) -> float:
    return nominal_resolution_m if nominal_resolution_m is not None else float("inf")


def rank_candidates(records: list[SurveyRecord]) -> list[SurveyRecord]:
    """Deterministic ranking by analysis usefulness (see module docstring priorities).

    Priority order: (1) pipeline intersection, (2) pipeline coverage length,
    (3) true spatial resolution (finer first), (4) numerical availability,
    (5) authoritative provenance (a verified footprint over metadata-only),
    (6) vertical datum recorded, (7) horizontal CRS recorded, (8) survey
    method recorded, (9) acquisition year (newest first). Never discards
    older surveys -- this only orders them.
    """

    def sort_key(r: SurveyRecord) -> tuple:
        return (
            0 if r.intersects_pipeline else 1,
            -(r.pipeline_overlap_length_m or 0.0),
            _resolution_sort_value(r.nominal_resolution_m),
            0 if r.download_available else 1,
            0 if r.footprint_available else 1,
            0 if r.vertical_datum else 1,
            0 if r.horizontal_crs else 1,
            0 if r.survey_method else 1,
            -(r.acquisition_year or 0),
            r.record_key,  # final tiebreaker for full determinism
        )

    return sorted(records, key=sort_key)


def select_primary_candidate(ranked_records: list[SurveyRecord]) -> SurveyRecord | None:
    """Pick the best non-EMODnet, pipeline-intersecting, downloadable candidate.

    EMODnet is never eligible here -- it is the mandatory full-AOI baseline,
    not automatically the "high-resolution primary" merely by being available.
    Returns None (and the caller must say so plainly) if nothing qualifies.
    """

    for record in ranked_records:
        if (
            record.source.upper() != "EMODNET"
            and record.intersects_pipeline
            and record.download_available
        ):
            return record
    return None


def apply_primary_role(
    records: list[SurveyRecord], primary: SurveyRecord | None
) -> list[SurveyRecord]:
    """Add `primary_analysis_candidate` to the selected record's roles, if any."""

    if primary is None:
        return records
    return [
        replace(r, potential_role=(*r.potential_role, ROLE_PRIMARY_ANALYSIS_CANDIDATE))
        if r.record_key == primary.record_key
        else r
        for r in records
    ]


def classify_temporal_pair(covering: tuple[SurveyRecord, ...]) -> tuple[str, ...]:
    """Classify a multi-epoch overlap segment. Multiple tags may apply."""

    tags: list[str] = []
    if all(r.download_available for r in covering):
        tags.append(TAG_READY_FOR_FUTURE_DELTA_Z)
    if any(r.manual_download_required for r in covering):
        tags.append(TAG_REQUIRES_MANUAL_DOWNLOAD)

    datums = {r.vertical_datum for r in covering}
    if len(datums) > 1 or None in datums:
        tags.append(TAG_DATUM_HARMONISATION_REQUIRED)
    if all(not r.download_available for r in covering):
        tags.append(TAG_METADATA_ONLY)
    if not tags:
        tags.append(TAG_NO_VALID_TEMPORAL_PAIR)
    return tuple(tags)


def find_temporal_overlap_segments(records: list[SurveyRecord]) -> list[TemporalOverlapSegment]:
    """Find chainage segments covered by >=2 distinct survey epochs.

    Uses a coordinate-compression sweep over every range boundary so that
    three-or-more overlapping epochs at the same segment are also detected,
    not just pairs.
    """

    spatial_records = [r for r in records if r.chainage_coverage_ranges]
    if len(spatial_records) < 2:
        return []

    boundaries: set[float] = set()
    for record in spatial_records:
        for start, end in record.chainage_coverage_ranges:
            boundaries.add(start)
            boundaries.add(end)
    sorted_bounds = sorted(boundaries)

    raw_segments: list[tuple[float, float, tuple[SurveyRecord, ...]]] = []
    for i in range(len(sorted_bounds) - 1):
        seg_start, seg_end = sorted_bounds[i], sorted_bounds[i + 1]
        midpoint = (seg_start + seg_end) / 2
        covering = tuple(
            r
            for r in spatial_records
            if any(start <= midpoint <= end for start, end in r.chainage_coverage_ranges)
        )
        if len(covering) >= 2:
            raw_segments.append((seg_start, seg_end, covering))

    merged: list[list] = []
    for seg_start, seg_end, covering in raw_segments:
        key = tuple(sorted(r.record_key for r in covering))
        if merged and merged[-1][3] == key and merged[-1][1] == seg_start:
            merged[-1][1] = seg_end
        else:
            merged.append([seg_start, seg_end, covering, key])

    return [
        TemporalOverlapSegment(
            chainage_start_m=seg_start,
            chainage_end_m=seg_end,
            records=covering,
            classifications=classify_temporal_pair(covering),
        )
        for seg_start, seg_end, covering, _ in merged
    ]


def _format_ranges(ranges: tuple[tuple[float, float], ...]) -> str:
    return ";".join(f"{start:.2f}-{end:.2f}" for start, end in ranges)


def build_inventory_dataframe(records: list[SurveyRecord]) -> pd.DataFrame:
    """Flatten records into the canonical inventory table (no geometry column)."""

    rows = []
    for r in records:
        rows.append(
            {
                "source": r.source,
                "source_dataset_id": r.source_dataset_id,
                "source_record_url_or_identifier": r.source_record_url_or_identifier,
                "title": r.title,
                "survey_name": r.survey_name,
                "survey_start_date": r.survey_start_date,
                "survey_end_date": r.survey_end_date,
                "acquisition_year": r.acquisition_year,
                "data_type": r.data_type,
                "survey_method": r.survey_method,
                "nominal_resolution_m": r.nominal_resolution_m,
                "resolution_class": r.resolution_class,
                "horizontal_crs": r.horizontal_crs,
                "vertical_datum": r.vertical_datum,
                "licence": r.licence,
                "access_type": r.access_type,
                "download_available": r.download_available,
                "manual_download_required": r.manual_download_required,
                "acquisition_status": r.acquisition_status,
                "footprint_available": r.footprint_available,
                "intersects_aoi": r.intersects_aoi,
                "overlap_aoi_km2": r.overlap_aoi_km2,
                "overlap_aoi_percent": r.overlap_aoi_percent,
                "intersects_pipeline": r.intersects_pipeline,
                "pipeline_overlap_length_m": r.pipeline_overlap_length_m,
                "pipeline_overlap_percent": r.pipeline_overlap_percent,
                "covered_chainage_station_count": r.covered_chainage_station_count,
                "first_covered_chainage_m": r.first_covered_chainage_m,
                "last_covered_chainage_m": r.last_covered_chainage_m,
                "chainage_coverage_ranges": _format_ranges(r.chainage_coverage_ranges),
                "temporal_epoch": r.temporal_epoch,
                "potential_role": ",".join(r.potential_role),
                "notes": r.notes,
            }
        )
    return pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))


def build_inventory_gdf(records: list[SurveyRecord], working_crs: str) -> gpd.GeoDataFrame | None:
    """Build the geometry-bearing subset of the inventory, or None if no footprints exist."""

    with_footprint = [
        r for r in records if r.geometry_wgs84 is not None and not r.geometry_wgs84.is_empty
    ]
    if not with_footprint:
        return None

    df = build_inventory_dataframe(with_footprint)
    geometries = [
        gpd.GeoSeries([r.geometry_wgs84], crs="EPSG:4326").to_crs(working_crs).iloc[0]
        for r in with_footprint
    ]
    return gpd.GeoDataFrame(df, geometry=geometries, crs=working_crs)


def write_inventory(
    records: list[SurveyRecord], parquet_path: Path, gpkg_path: Path, working_crs: str
) -> gpd.GeoDataFrame | None:
    """Write the canonical inventory to Parquet (always) and GeoPackage (if footprints exist)."""

    df = build_inventory_dataframe(records)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    gdf = build_inventory_gdf(records, working_crs)
    if gdf is not None:
        gpkg_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(gpkg_path, driver="GPKG", layer="bathymetry_inventory")
    return gdf


@dataclass(frozen=True)
class DiscoveryReport:
    """Everything needed to summarize a completed discovery run.

    Deliberately source-agnostic (no import of ukho/bgs/emodnet, to avoid a
    circular import): the caller gathers raw records from each source, then
    hands them to `run_discovery`.
    """

    ranked_records: tuple[SurveyRecord, ...]
    primary_candidate: SurveyRecord | None
    baseline_candidate: SurveyRecord | None
    temporal_segments: tuple[TemporalOverlapSegment, ...]
    parquet_path: Path
    gpkg_path: Path | None


def run_discovery(
    raw_records: list[SurveyRecord],
    *,
    pipeline_geom_working: BaseGeometry,
    aoi_geom_working: BaseGeometry,
    chainage_gdf_working: gpd.GeoDataFrame,
    working_crs: str,
    parquet_path: Path,
    gpkg_path: Path,
) -> DiscoveryReport:
    """Calculate spatial/chainage coverage, assign roles, rank, and write the inventory."""

    records = [
        calculate_spatial_coverage(r, pipeline_geom_working, aoi_geom_working, working_crs)
        for r in raw_records
    ]
    records = [calculate_chainage_coverage(r, chainage_gdf_working, working_crs) for r in records]
    records = [assign_intrinsic_roles(r) for r in records]

    ranked = rank_candidates(records)
    primary = select_primary_candidate(ranked)
    ranked = apply_primary_role(ranked, primary)
    baseline = next((r for r in ranked if ROLE_BASELINE in r.potential_role), None)

    temporal_segments = find_temporal_overlap_segments(ranked)

    gdf = write_inventory(ranked, parquet_path, gpkg_path, working_crs)

    return DiscoveryReport(
        ranked_records=tuple(ranked),
        primary_candidate=primary,
        baseline_candidate=baseline,
        temporal_segments=tuple(temporal_segments),
        parquet_path=parquet_path,
        gpkg_path=gpkg_path if gdf is not None else None,
    )


def print_discovery_report(report: DiscoveryReport, *, file: Any = None) -> None:
    """Print a concise, human-readable summary of a discovery run."""

    file = file or sys.stdout
    lines = ["Bathymetry survey inventory:", ""]

    for r in report.ranked_records:
        lines.append(f"- [{r.source}] {r.source_dataset_id or r.title or 'unknown'}")
        lines.append(f"    title: {r.title}")
        lines.append(f"    year: {r.acquisition_year}  status: {r.acquisition_status}")
        lines.append(
            f"    resolution: {r.nominal_resolution_m} m ({r.resolution_class})  "
            f"CRS: {r.horizontal_crs}  vertical datum: {r.vertical_datum}"
        )
        if r.footprint_available:
            lines.append(
                f"    intersects AOI: {r.intersects_aoi} "
                f"(overlap {r.overlap_aoi_km2 and round(r.overlap_aoi_km2, 2)} km2, "
                f"{r.overlap_aoi_percent and round(r.overlap_aoi_percent, 1)}%)"
            )
            overlap_len = r.pipeline_overlap_length_m and round(r.pipeline_overlap_length_m, 1)
            overlap_pct = r.pipeline_overlap_percent and round(r.pipeline_overlap_percent, 1)
            lines.append(
                f"    intersects pipeline: {r.intersects_pipeline} "
                f"(covered {overlap_len} m, {overlap_pct}%)"
            )
            if r.chainage_coverage_ranges:
                ranges_str = "; ".join(
                    f"{start:.1f}-{end:.1f} m" for start, end in r.chainage_coverage_ranges
                )
                station_count = r.covered_chainage_station_count
                lines.append(f"    chainage coverage: {station_count} stations, {ranges_str}")
        else:
            lines.append(
                "    footprint: not available -- spatial overlap not authoritatively verified"
            )
        lines.append(
            f"    download available: {r.download_available}  "
            f"manual download required: {r.manual_download_required}"
        )
        lines.append(f"    roles: {', '.join(r.potential_role) or '(none)'}")
        lines.append(f"    notes: {r.notes}")
        lines.append("")

    lines.append("Multi-temporal overlap segments:")
    if not report.temporal_segments:
        lines.append("  NONE FOUND")
    else:
        for seg in report.temporal_segments:
            survey_ids = ", ".join(r.record_key for r in seg.records)
            lines.append(
                f"  {seg.chainage_start_m:.1f} m -> {seg.chainage_end_m:.1f} m: "
                f"[{survey_ids}] tags={list(seg.classifications)}"
            )
    lines.append("")

    if report.primary_candidate:
        primary_str = report.primary_candidate.record_key
    else:
        primary_str = "NONE -- no automatically downloadable dataset intersects the pipeline"
    baseline_str = report.baseline_candidate.record_key if report.baseline_candidate else "NONE"
    lines.append(f"Primary analysis candidate: {primary_str}")
    lines.append(f"Full-AOI baseline: {baseline_str}")
    lines.append(f"Inventory (Parquet): {report.parquet_path}")
    lines.append(
        f"Inventory (GeoPackage): {report.gpkg_path or '(no footprints available -- not written)'}"
    )

    print("\n".join(lines), file=file)
