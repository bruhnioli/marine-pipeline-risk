"""Corridor Area-of-Interest (AOI) generation from a canonical pipeline.

Takes the canonical pipeline geometry written by a provider (e.g.
`providers/nsta.py`) and derives the study's spatial extent: a metric buffer
around the complete route. This lives in `preprocessing` (not `providers` or
`pipeline`) because it consumes already-ingested canonical data and produces
a derived spatial extent for later stages to query against -- it touches no
external data source and models no physical pipeline behaviour.

The pipeline geometry is the only authoritative spatial source: the AOI is
always derived from it at run time, never hard-coded, so that later stages
depending on geographic bounds (bathymetry/sediment/metocean discovery) get
them reproducibly from `aoi.gpkg` rather than a frozen config copy.
"""

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Broad North Sea basin bounds (not just the Southern North Sea) used only as
# a sanity backstop against a real CRS/units bug -- not a precision check.
NORTH_SEA_LON_RANGE_DEG = (-4.0, 9.0)
NORTH_SEA_LAT_RANGE_DEG = (51.0, 62.0)

# How far the computed buffer distance may drift from the configured value
# before it is treated as a real bug rather than GEOS's polygon-approximation
# of circular arcs at line joints/end caps.
BUFFER_DISTANCE_TOLERANCE_FRACTION = 0.1

MIN_PLAUSIBLE_AREA_KM2 = 1.0
MAX_PLAUSIBLE_AREA_KM2 = 100_000.0


class InvalidPipelineInputError(RuntimeError):
    """The canonical pipeline file is missing, malformed, or has no CRS."""


class InvalidAoiGeometryError(RuntimeError):
    """The generated AOI failed a validity, containment, or sanity check."""


@dataclass(frozen=True)
class AoiSanityMetrics:
    """Diagnostics computed while validating a freshly built AOI."""

    area_km2: float
    min_boundary_distance_m: float
    centroid_wgs84: tuple[float, float]  # (lon, lat)


@dataclass(frozen=True)
class AoiBuildReport:
    """Everything needed to summarize a completed `build_aoi` run."""

    study_id: str
    pipeline_id: str
    corridor_buffer_m: float
    working_crs: str
    pipeline_source_crs: str
    area_km2: float
    min_boundary_distance_m: float
    bounds_working_crs: tuple[float, float, float, float]
    bounds_wgs84: tuple[float, float, float, float]
    centroid_wgs84: tuple[float, float]
    pipeline_gpkg_path: Path
    output_path: Path


def _validate_and_merge_geometries(
    geometries: list[BaseGeometry], pipeline_id: str
) -> BaseGeometry:
    """Validate a set of geometries for one pipeline_id and merge into one.

    Multiple rows sharing the same `pipeline_id` are treated as legitimately
    connected parts of one canonical pipeline and unioned; the current
    canonical writer (MAR-002) always emits exactly one row, so this is a
    defensive fallback rather than the common case.
    """

    if not geometries or any(g is None or g.is_empty for g in geometries):
        raise InvalidPipelineInputError(f"Pipeline '{pipeline_id}' geometry is missing or empty.")
    if not all(g.is_valid for g in geometries):
        raise InvalidPipelineInputError(f"Pipeline '{pipeline_id}' geometry is invalid.")
    if not all(g.geom_type in ("LineString", "MultiLineString") for g in geometries):
        bad_types = {g.geom_type for g in geometries} - {"LineString", "MultiLineString"}
        raise InvalidPipelineInputError(
            f"Pipeline '{pipeline_id}' geometry is not linear (found: {sorted(bad_types)})."
        )

    return geometries[0] if len(geometries) == 1 else unary_union(geometries)


def load_canonical_pipeline(
    pipeline_gpkg_path: Path, pipeline_id: str, layer: str = "pipeline"
) -> tuple[BaseGeometry, str]:
    """Load and validate one pipeline's geometry from a canonical GeoPackage.

    Returns `(geometry, source_crs)`. Raises `InvalidPipelineInputError` for
    any missing file, missing CRS, missing pipeline_id, or malformed
    geometry -- never guesses.
    """

    if not pipeline_gpkg_path.exists():
        raise InvalidPipelineInputError(f"Canonical pipeline file not found: {pipeline_gpkg_path}")

    gdf = gpd.read_file(pipeline_gpkg_path, layer=layer)
    if gdf.crs is None:
        raise InvalidPipelineInputError(f"{pipeline_gpkg_path} has no CRS defined.")

    matches = gdf[gdf["pipeline_id"] == pipeline_id]
    if matches.empty:
        raise InvalidPipelineInputError(
            f"No rows with pipeline_id='{pipeline_id}' in {pipeline_gpkg_path}."
        )

    geometry = _validate_and_merge_geometries(list(matches.geometry), pipeline_id)
    return geometry, gdf.crs.to_string()


def ensure_projected_working_crs(
    geometry: BaseGeometry, source_crs: str, working_crs: str
) -> BaseGeometry:
    """Return `geometry` expressed in `working_crs`, transforming only if needed.

    This is the single choke point that prevents an accidental degrees-based
    buffer: a geographic `working_crs` is rejected outright rather than
    silently used for a metric operation.
    """

    if pyproj.CRS(working_crs).is_geographic:
        raise InvalidPipelineInputError(
            f"working_crs='{working_crs}' is geographic (degree-based); "
            "metric buffering requires a projected CRS."
        )
    if pyproj.CRS(source_crs) == pyproj.CRS(working_crs):
        return geometry
    return gpd.GeoSeries([geometry], crs=source_crs).to_crs(working_crs).iloc[0]


def build_corridor_buffer(geometry_working_crs: BaseGeometry, buffer_m: float) -> BaseGeometry:
    """Buffer a (Multi)LineString by `buffer_m` into one dissolved polygon.

    A single GEOS `.buffer()` call over a (Multi)LineString already returns
    the unioned/dissolved corridor in one operation -- there is no separate
    per-part buffer-then-union step to get wrong, so no artificial internal
    seams or duplicate overlaps arise from multipart input.

    No simplification is applied: PL854's ~1300-vertex line and its buffer
    are cheap to compute and store, so there is no concrete technical need
    to simplify it away.
    """

    if buffer_m <= 0:
        raise ValueError(f"corridor_buffer_m must be positive, got {buffer_m}.")

    aoi = geometry_working_crs.buffer(buffer_m)
    if not aoi.is_valid:
        aoi = aoi.buffer(0)  # standard GEOS self-fix for minor topology noise
    if not aoi.is_valid:
        raise InvalidAoiGeometryError("Buffered AOI geometry is invalid even after repair.")
    if aoi.geom_type not in ("Polygon", "MultiPolygon"):
        raise InvalidAoiGeometryError(f"Expected a (Multi)Polygon AOI, got {aoi.geom_type}.")
    return aoi


def _is_plausible_north_sea_location(lon: float, lat: float) -> bool:
    return (
        NORTH_SEA_LON_RANGE_DEG[0] <= lon <= NORTH_SEA_LON_RANGE_DEG[1]
        and NORTH_SEA_LAT_RANGE_DEG[0] <= lat <= NORTH_SEA_LAT_RANGE_DEG[1]
    )


def run_sanity_checks(
    pipeline_geometry: BaseGeometry,
    aoi_geometry: BaseGeometry,
    buffer_m: float,
    working_crs: str,
) -> AoiSanityMetrics:
    """Validate a freshly built AOI and compute its diagnostic metrics.

    These check OUR OWN computation's correctness (not incoming data
    quality), so violations raise `InvalidAoiGeometryError` rather than
    being merely reported -- e.g. an accidental degrees-based buffer would
    trip the distance/area checks by orders of magnitude.
    """

    if not aoi_geometry.contains(pipeline_geometry):
        raise InvalidAoiGeometryError("AOI does not fully contain the pipeline geometry.")

    min_boundary_distance_m = pipeline_geometry.distance(aoi_geometry.boundary)
    low = buffer_m * (1 - BUFFER_DISTANCE_TOLERANCE_FRACTION)
    high = buffer_m * (1 + BUFFER_DISTANCE_TOLERANCE_FRACTION)
    if not (low <= min_boundary_distance_m <= high):
        raise InvalidAoiGeometryError(
            f"AOI boundary is {min_boundary_distance_m:.1f} m from the pipeline; "
            f"expected close to the configured {buffer_m} m buffer."
        )

    area_km2 = aoi_geometry.area / 1_000_000.0
    if not (MIN_PLAUSIBLE_AREA_KM2 <= area_km2 <= MAX_PLAUSIBLE_AREA_KM2):
        raise InvalidAoiGeometryError(f"AOI area {area_km2:.1f} km^2 is not plausible.")

    centroid_wgs84_point = (
        gpd.GeoSeries([aoi_geometry.centroid], crs=working_crs).to_crs("EPSG:4326").iloc[0]
    )
    centroid_wgs84 = (centroid_wgs84_point.x, centroid_wgs84_point.y)
    if not _is_plausible_north_sea_location(*centroid_wgs84):
        raise InvalidAoiGeometryError(
            f"AOI centroid {centroid_wgs84} (WGS84) is outside the plausible North Sea "
            "bounds -- check for a CRS/units error."
        )

    return AoiSanityMetrics(
        area_km2=area_km2,
        min_boundary_distance_m=min_boundary_distance_m,
        centroid_wgs84=centroid_wgs84,
    )


def build_canonical_aoi_gdf(
    *,
    study_id: str,
    pipeline_id: str,
    buffer_m: float,
    working_crs: str,
    aoi_geometry: BaseGeometry,
    area_km2: float,
    generated_at: datetime,
) -> gpd.GeoDataFrame:
    """Build the one-row canonical study-AOI GeoDataFrame."""

    record = {
        "study_id": study_id,
        "pipeline_id": pipeline_id,
        "corridor_buffer_m": buffer_m,
        "working_crs": working_crs,
        "area_km2": area_km2,
        "generated_at": generated_at.isoformat(),
    }
    return gpd.GeoDataFrame([record], geometry=[aoi_geometry], crs=working_crs)


def write_aoi_gpkg(gdf: gpd.GeoDataFrame, output_path: Path, layer: str = "study_aoi") -> Path:
    """Write the canonical AOI GeoDataFrame to a GeoPackage layer."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


def build_aoi(
    *,
    pipeline_gpkg_path: Path,
    pipeline_id: str,
    study_id: str,
    buffer_m: float,
    working_crs: str,
    output_path: Path,
) -> AoiBuildReport:
    """End-to-end: load -> validate CRS -> buffer -> validate AOI -> write -> report."""

    raw_geometry, source_crs = load_canonical_pipeline(pipeline_gpkg_path, pipeline_id)
    pipeline_geometry = ensure_projected_working_crs(raw_geometry, source_crs, working_crs)

    aoi_geometry = build_corridor_buffer(pipeline_geometry, buffer_m)
    metrics = run_sanity_checks(pipeline_geometry, aoi_geometry, buffer_m, working_crs)

    gdf = build_canonical_aoi_gdf(
        study_id=study_id,
        pipeline_id=pipeline_id,
        buffer_m=buffer_m,
        working_crs=working_crs,
        aoi_geometry=aoi_geometry,
        area_km2=metrics.area_km2,
        generated_at=datetime.now(UTC),
    )
    write_aoi_gpkg(gdf, output_path)

    bounds_wgs84 = tuple(
        float(v)
        for v in gpd.GeoSeries([aoi_geometry], crs=working_crs).to_crs("EPSG:4326").total_bounds
    )

    return AoiBuildReport(
        study_id=study_id,
        pipeline_id=pipeline_id,
        corridor_buffer_m=buffer_m,
        working_crs=working_crs,
        pipeline_source_crs=source_crs,
        area_km2=metrics.area_km2,
        min_boundary_distance_m=metrics.min_boundary_distance_m,
        bounds_working_crs=tuple(aoi_geometry.bounds),
        bounds_wgs84=bounds_wgs84,
        centroid_wgs84=metrics.centroid_wgs84,
        pipeline_gpkg_path=pipeline_gpkg_path,
        output_path=output_path,
    )


def print_aoi_report(report: AoiBuildReport, *, file: Any = None) -> None:
    """Print a concise, human-readable summary of an AOI build run."""

    file = file or sys.stdout
    lines = [
        f"Study:             {report.study_id}",
        f"Pipeline:          {report.pipeline_id}",
        f"Buffer:            {report.corridor_buffer_m:.0f} m",
        f"Working CRS:       {report.working_crs}",
        f"Pipeline src CRS:  {report.pipeline_source_crs}",
        f"AOI area:          {report.area_km2:.2f} km2",
        f"Min pipe->boundary:{report.min_boundary_distance_m:.1f} m",
        f"Projected bounds:  {tuple(round(v, 2) for v in report.bounds_working_crs)}",
        f"WGS84 bounds:      {tuple(round(v, 7) for v in report.bounds_wgs84)}",
        f"Centroid (WGS84):  {tuple(round(v, 7) for v in report.centroid_wgs84)}",
        f"Source pipeline:   {report.pipeline_gpkg_path}",
        f"Output:            {report.output_path}",
    ]
    print("\n".join(lines), file=file)
