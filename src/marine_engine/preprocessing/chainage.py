"""Chainage / KP linear-reference system generation for a canonical pipeline.

Establishes a deterministic linear reference (chainage, in metres, plus a
human-readable KP label) along the canonical pipeline geometry at a
configurable regular interval, always preserving the exact route terminus.
Lives in `preprocessing` alongside `aoi.py`: both derive a spatial structure
from the canonical pipeline for later stages to attach features to, and
neither ingests external data or models physical pipeline behaviour.

Direction honesty
------------------
The canonical pipeline schema (MAR-002) carries no authoritative from/to
installation fields -- only a free-text `pipe_name` (e.g. "LOGGS PP TO
ANGLIA YD GAS LINE"). A name states an intended reading order, not proof
that the LineString's first vertex was digitized at that same physical end;
without an independent installation-coordinate cross-check (out of scope
here), this module never infers which end is "Anglia A" or "LOGGS". Chainage
0 is always just the source geometry's own start vertex, recorded via
`chainage_origin_basis = "source_geometry_start"`.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyproj
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union

# Numerical (not physical) tolerance: guards against floating-point noise in
# GEOS-computed lengths/distances, not real-world survey precision.
NUMERICAL_TOLERANCE_M = 1e-6


class InvalidPipelineRouteError(RuntimeError):
    """The canonical pipeline file is missing, malformed, or not one route."""


class ChainageValidationError(RuntimeError):
    """A generated chainage dataset failed a required invariant."""


@dataclass(frozen=True)
class ChainageOriginDecision:
    """The defensible basis for chainage direction, and why."""

    basis: str
    note: str


@dataclass(frozen=True)
class RouteEndpoints:
    """A route's two endpoints in both the working CRS and WGS84."""

    start_working_crs: tuple[float, float]
    end_working_crs: tuple[float, float]
    start_wgs84: tuple[float, float]  # (lon, lat)
    end_wgs84: tuple[float, float]


@dataclass(frozen=True)
class ChainageStation:
    """One generated station: index, position along the route, and point."""

    station_index: int
    chainage_m: float
    is_terminal: bool
    point: Point


@dataclass(frozen=True)
class ChainageStations:
    """The full generated station sequence plus derived counts."""

    stations: list[ChainageStation]
    regular_station_count: int
    terminal_residual_m: float
    total_length_m: float


@dataclass(frozen=True)
class ChainageValidationReport:
    """Diagnostics computed while validating a freshly built chainage dataset."""

    max_point_to_route_distance_m: float
    all_within_aoi: bool
    station_count: int


@dataclass(frozen=True)
class ChainageBuildReport:
    """Everything needed to summarize a completed `build_chainage` run."""

    study_id: str
    pipeline_id: str
    working_crs: str
    interval_m: float
    pipeline_length_m: float
    regular_station_count: int
    total_station_count: int
    terminal_residual_m: float
    chainage_origin_basis: str
    origin_basis_note: str
    origin_working_crs: tuple[float, float]
    origin_wgs84: tuple[float, float]
    terminus_working_crs: tuple[float, float]
    terminus_wgs84: tuple[float, float]
    max_point_to_route_distance_m: float
    all_within_aoi: bool
    output_path: Path


def _resolve_continuous_route(geometries: list[BaseGeometry], pipeline_id: str) -> LineString:
    """Resolve one or more geometries into a single continuous, orderable route.

    Chainage requires a well-defined start-to-end parametrization, which a
    MultiLineString with disconnected or branching parts cannot provide.
    `shapely.ops.linemerge` succeeds only when parts are endpoint-to-endpoint
    contiguous; anything else fails loudly rather than silently chaining
    disconnected parts in an arbitrary order.
    """

    if not geometries or any(g is None or g.is_empty for g in geometries):
        raise InvalidPipelineRouteError(f"Pipeline '{pipeline_id}' geometry is missing or empty.")
    if not all(g.is_valid for g in geometries):
        raise InvalidPipelineRouteError(f"Pipeline '{pipeline_id}' geometry is invalid.")
    if not all(g.geom_type in ("LineString", "MultiLineString") for g in geometries):
        raise InvalidPipelineRouteError(f"Pipeline '{pipeline_id}' geometry is not linear.")

    combined = geometries[0] if len(geometries) == 1 else unary_union(geometries)
    if combined.geom_type == "LineString":
        return combined

    merged = linemerge(combined)
    if merged.geom_type != "LineString":
        raise InvalidPipelineRouteError(
            f"Pipeline '{pipeline_id}' geometry has disconnected or branching parts and "
            "cannot be represented as one continuous chainage route."
        )
    return merged


def load_pipeline_route(
    pipeline_gpkg_path: Path, pipeline_id: str, layer: str = "pipeline"
) -> tuple[LineString, dict[str, Any], str]:
    """Load, validate, and resolve a canonical pipeline into one chainage route.

    Returns `(route, attributes, source_crs)`, where `attributes` is the
    matched row's non-geometry fields (used only to look for direction-
    provenance hints -- see `determine_chainage_origin`). Raises
    `InvalidPipelineRouteError` for any missing file, missing/non-projected
    CRS, missing pipeline_id, or malformed/disconnected geometry.
    """

    if not pipeline_gpkg_path.exists():
        raise InvalidPipelineRouteError(f"Canonical pipeline file not found: {pipeline_gpkg_path}")

    gdf = gpd.read_file(pipeline_gpkg_path, layer=layer)
    if gdf.crs is None:
        raise InvalidPipelineRouteError(f"{pipeline_gpkg_path} has no CRS defined.")
    if pyproj.CRS(gdf.crs).is_geographic:
        raise InvalidPipelineRouteError(
            f"{pipeline_gpkg_path} CRS ({gdf.crs}) is geographic; chainage requires a "
            "projected/metric CRS."
        )

    matches = gdf[gdf["pipeline_id"] == pipeline_id]
    if matches.empty:
        raise InvalidPipelineRouteError(
            f"No rows with pipeline_id='{pipeline_id}' in {pipeline_gpkg_path}."
        )

    route = _resolve_continuous_route(list(matches.geometry), pipeline_id)
    if route.length <= 0:
        raise InvalidPipelineRouteError(f"Pipeline '{pipeline_id}' route has non-positive length.")

    attributes = matches.iloc[0].drop(labels="geometry").to_dict()
    return route, attributes, gdf.crs.to_string()


def determine_chainage_origin(pipeline_attributes: dict[str, Any]) -> ChainageOriginDecision:
    """Decide the defensible basis for chainage direction and origin.

    Preferred hierarchy: (1) an unambiguous authoritative from/to
    relationship in the canonical attributes, if present; (2) another
    authoritative installation relationship already present in the source
    (none is joined into the current canonical schema); (3) otherwise,
    preserve the source geometry's own vertex order.
    """

    from_field = pipeline_attributes.get("from_installation") or pipeline_attributes.get(
        "upstream_installation"
    )
    to_field = pipeline_attributes.get("to_installation") or pipeline_attributes.get(
        "downstream_installation"
    )
    if from_field and to_field:
        return ChainageOriginDecision(
            basis="source_from_to_metadata",
            note=f"Authoritative route metadata: '{from_field}' -> '{to_field}'.",
        )

    note = "No authoritative from/to installation metadata in the canonical schema."
    pipe_name = pipeline_attributes.get("pipe_name")
    if pipe_name:
        note += (
            f" pipe_name='{pipe_name}' states a direction in its label text, but that is "
            "not proof the geometry's first vertex was digitized at that same physical end."
        )

    return ChainageOriginDecision(basis="source_geometry_start", note=note)


def compute_route_endpoints(route: LineString, working_crs: str) -> RouteEndpoints:
    """Report a route's two endpoints in both the working CRS and WGS84."""

    start = route.coords[0]
    end = route.coords[-1]
    wgs84_points = gpd.GeoSeries([Point(start), Point(end)], crs=working_crs).to_crs("EPSG:4326")

    return RouteEndpoints(
        start_working_crs=(float(start[0]), float(start[1])),
        end_working_crs=(float(end[0]), float(end[1])),
        start_wgs84=(float(wgs84_points.iloc[0].x), float(wgs84_points.iloc[0].y)),
        end_wgs84=(float(wgs84_points.iloc[1].x), float(wgs84_points.iloc[1].y)),
    )


def compute_chainage_stations(
    route: LineString, interval_m: float, tolerance_m: float = NUMERICAL_TOLERANCE_M
) -> ChainageStations:
    """Generate regular `interval_m` stations plus the exact route terminus.

    Uses the route's own linear-referencing (`interpolate`), which walks
    cumulative distance along all segments -- not vertex spacing -- so
    stations land at true along-route distances regardless of how densely
    the source geometry was digitized.
    """

    if interval_m <= 0:
        raise ValueError(f"chainage_interval_m must be positive, got {interval_m}.")

    total_length_m = route.length
    if total_length_m <= 0:
        raise ValueError("Route length must be positive.")

    # +tolerance_m before flooring guards against a total length that is
    # numerically a hair below an exact multiple of interval_m (e.g. GEOS
    # length summation landing on 99.99999999999999 instead of 100.0).
    n_regular = math.floor((total_length_m + tolerance_m) / interval_m) + 1  # includes chainage 0
    regular_chainages = [i * interval_m for i in range(n_regular)]

    terminal_residual_m = total_length_m - regular_chainages[-1]
    chainages = list(regular_chainages)
    if terminal_residual_m > tolerance_m:
        chainages.append(total_length_m)
    else:
        terminal_residual_m = 0.0

    last_index = len(chainages) - 1
    stations = [
        ChainageStation(
            station_index=i,
            chainage_m=chainage,
            is_terminal=(i == last_index),
            point=route.interpolate(chainage),
        )
        for i, chainage in enumerate(chainages)
    ]

    return ChainageStations(
        stations=stations,
        regular_station_count=n_regular,
        terminal_residual_m=terminal_residual_m,
        total_length_m=total_length_m,
    )


def format_kp_label(chainage_m: float) -> str:
    """Format a chainage as a KP label, e.g. 1250 -> "KP 1+250", 480.67 -> "...+480.67".

    `chainage_m` itself is never rounded for this -- only the label's
    fractional-metre display is limited to 2 decimal places.
    """

    km = int(chainage_m // 1000)
    remainder_m = chainage_m - km * 1000
    if math.isclose(remainder_m, round(remainder_m), abs_tol=1e-9):
        return f"KP {km}+{round(remainder_m):03d}"
    return f"KP {km}+{remainder_m:06.2f}"


def build_canonical_chainage_gdf(
    *,
    pipeline_id: str,
    stations: list[ChainageStation],
    interval_m: float,
    chainage_origin_basis: str,
    working_crs: str,
) -> gpd.GeoDataFrame:
    """Build the canonical chainage-points GeoDataFrame."""

    total_length_m = stations[-1].chainage_m
    records = []
    geometries = []
    for station in stations:
        records.append(
            {
                "pipeline_id": pipeline_id,
                "station_index": station.station_index,
                "chainage_m": station.chainage_m,
                "kp_label": format_kp_label(station.chainage_m),
                "chainage_interval_m": interval_m,
                "is_terminal": station.is_terminal,
                "chainage_origin_basis": chainage_origin_basis,
                "easting": station.point.x,
                "northing": station.point.y,
                "fraction_along_route": station.chainage_m / total_length_m,
            }
        )
        geometries.append(station.point)

    return gpd.GeoDataFrame(records, geometry=geometries, crs=working_crs)


def write_chainage_gpkg(
    gdf: gpd.GeoDataFrame, output_path: Path, layer: str = "chainage_points"
) -> Path:
    """Write the canonical chainage GeoDataFrame to a GeoPackage layer."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


def validate_chainage_gdf(
    gdf: gpd.GeoDataFrame,
    route: LineString,
    aoi_geometry: BaseGeometry | None,
    working_crs: str,
    tolerance_m: float = NUMERICAL_TOLERANCE_M,
) -> ChainageValidationReport:
    """Validate every required chainage invariant, failing loudly on any breach."""

    if gdf.crs is None or gdf.crs.to_string() != working_crs:
        raise ChainageValidationError(
            f"Chainage points CRS {gdf.crs} does not match working CRS {working_crs}."
        )

    if gdf["station_index"].duplicated().any():
        raise ChainageValidationError("Duplicate station_index values found.")
    if gdf["chainage_m"].duplicated().any():
        raise ChainageValidationError("Duplicate chainage_m values found.")

    chainage_values = gdf["chainage_m"].to_numpy()
    if not (chainage_values[:-1] < chainage_values[1:]).all():
        raise ChainageValidationError("chainage_m is not strictly monotonically increasing.")

    if not math.isclose(chainage_values[0], 0.0, abs_tol=tolerance_m):
        raise ChainageValidationError(f"First chainage is {chainage_values[0]}, expected 0.")
    if not math.isclose(chainage_values[-1], route.length, abs_tol=tolerance_m):
        raise ChainageValidationError(
            f"Final chainage {chainage_values[-1]} does not match route length {route.length}."
        )

    if not gdf.geometry.is_valid.all():
        raise ChainageValidationError("One or more station points are invalid.")

    distances = gdf.geometry.apply(lambda p: p.distance(route))
    max_distance = float(distances.max())
    if max_distance > tolerance_m:
        raise ChainageValidationError(
            f"A station lies {max_distance:.9f} m from the route; expected effectively 0."
        )

    start_point = Point(route.coords[0])
    end_point = Point(route.coords[-1])
    if gdf.geometry.iloc[0].distance(start_point) > tolerance_m:
        raise ChainageValidationError("First station does not coincide with the route start.")
    if gdf.geometry.iloc[-1].distance(end_point) > tolerance_m:
        raise ChainageValidationError("Final station does not coincide with the route end.")

    all_within_aoi = True
    if aoi_geometry is not None:
        all_within_aoi = bool(gdf.geometry.within(aoi_geometry).all())
        if not all_within_aoi:
            raise ChainageValidationError("One or more stations lie outside the study AOI.")

    return ChainageValidationReport(
        max_point_to_route_distance_m=max_distance,
        all_within_aoi=all_within_aoi,
        station_count=len(gdf),
    )


def build_chainage(
    *,
    pipeline_gpkg_path: Path,
    aoi_gpkg_path: Path | None,
    pipeline_id: str,
    study_id: str,
    interval_m: float,
    working_crs: str,
    output_path: Path,
) -> ChainageBuildReport:
    """End-to-end: load -> determine direction -> generate -> validate -> write -> report."""

    route, attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    if source_crs != working_crs:
        raise InvalidPipelineRouteError(
            f"Pipeline CRS {source_crs} does not match configured working CRS {working_crs}."
        )

    origin_decision = determine_chainage_origin(attributes)
    endpoints = compute_route_endpoints(route, working_crs)
    result = compute_chainage_stations(route, interval_m)

    gdf = build_canonical_chainage_gdf(
        pipeline_id=pipeline_id,
        stations=result.stations,
        interval_m=interval_m,
        chainage_origin_basis=origin_decision.basis,
        working_crs=working_crs,
    )

    aoi_geometry = None
    if aoi_gpkg_path is not None and aoi_gpkg_path.exists():
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        aoi_geometry = unary_union(aoi_gdf.geometry)

    validation = validate_chainage_gdf(gdf, route, aoi_geometry, working_crs)

    write_chainage_gpkg(gdf, output_path)

    return ChainageBuildReport(
        study_id=study_id,
        pipeline_id=pipeline_id,
        working_crs=working_crs,
        interval_m=interval_m,
        pipeline_length_m=result.total_length_m,
        regular_station_count=result.regular_station_count,
        total_station_count=len(result.stations),
        terminal_residual_m=result.terminal_residual_m,
        chainage_origin_basis=origin_decision.basis,
        origin_basis_note=origin_decision.note,
        origin_working_crs=endpoints.start_working_crs,
        origin_wgs84=endpoints.start_wgs84,
        terminus_working_crs=endpoints.end_working_crs,
        terminus_wgs84=endpoints.end_wgs84,
        max_point_to_route_distance_m=validation.max_point_to_route_distance_m,
        all_within_aoi=validation.all_within_aoi,
        output_path=output_path,
    )


def print_chainage_report(report: ChainageBuildReport, *, file: Any = None) -> None:
    """Print a concise, human-readable summary of a chainage build run."""

    file = file or sys.stdout
    semantic_identity = (
        "unresolved"
        if report.chainage_origin_basis == "source_geometry_start"
        else f"resolved via {report.chainage_origin_basis}"
    )
    lines = [
        f"Study:              {report.study_id}",
        f"Working CRS:        {report.working_crs}",
        f"Pipeline length:    {report.pipeline_length_m:,.2f} m",
        f"Interval:           {report.interval_m:.0f} m",
        f"Regular stations:   {report.regular_station_count}",
        f"Terminal residual:  {report.terminal_residual_m:.2f} m",
        f"Total stations:     {report.total_station_count}",
        "Chainage origin:",
        f"  Projected:        easting={report.origin_working_crs[0]:.3f}, "
        f"northing={report.origin_working_crs[1]:.3f}",
        f"  WGS84:            lon={report.origin_wgs84[0]:.7f}, lat={report.origin_wgs84[1]:.7f}",
        "Chainage terminus:",
        f"  Projected:        easting={report.terminus_working_crs[0]:.3f}, "
        f"northing={report.terminus_working_crs[1]:.3f}",
        f"  WGS84:            lon={report.terminus_wgs84[0]:.7f}, "
        f"lat={report.terminus_wgs84[1]:.7f}",
        f"Chainage origin basis: {report.chainage_origin_basis}",
        f"Semantic endpoint identity: {semantic_identity}",
        f"  ({report.origin_basis_note})",
        f"Max point-to-route distance: {report.max_point_to_route_distance_m:.9f} m",
        f"All stations within AOI: {report.all_within_aoi}",
        f"Output:             {report.output_path}",
    ]
    print("\n".join(lines), file=file)
