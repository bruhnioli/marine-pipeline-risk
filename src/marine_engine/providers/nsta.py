"""NSTA (North Sea Transition Authority) offshore pipeline provider.

Ingests a single named pipeline from the authoritative NSTA UKCS offshore
infrastructure dataset and normalizes it into the project's canonical
pipeline representation (see `build_canonical_gdf`).

Source provenance
------------------
NSTA publishes UKCS infrastructure data (reported under Section 34 of the
Energy Act 2016) as public, authoritative ArcGIS Online Feature Services
under the `NSTA_GIS` organisation account. The two services below were
identified via the public ArcGIS Online item-search API
(`https://www.arcgis.com/sharing/rest/search?q=owner:NSTA_GIS ...`), not
guessed or hard-coded from memory, and confirmed reachable/queryable on
2026-09-03:

- "UKCS offshore infrastructure pipeline linear (WGS84)"
  (item 3251baab197d4f53b0797a616a36d380, contentStatus=public_authoritative)
  -- the current/active infrastructure feed.
- "UKCS offshore infrastructure pipeline linear removed (WGS84)"
  (item 425af9bca45e4990a1d2b6b2cc88346e, contentStatus=public_authoritative)
  -- decommissioned/removed infrastructure, published as a separate service.

The active service's layer is itself a *view* filtered to
`(CURR_ROW = 'Y') AND (UPD_TYPE IN ('NO CHANGE','MODIFY','ADD','RECEIVED'))`,
i.e. it already excludes rows NSTA has marked as removed/deleted in their
source system. A pipeline can therefore legitimately be absent from the
active service while still being a real, documented pipeline in the removed
service -- hence the fallback in `fetch_pipeline` below, rather than treating
a zero-match on the active service as "does not exist".

Both services expose layer id 1 (not 0) with identical field schemas, keyed
by `NSTAPIPNO` ("NSTA pipeline number") -- confirmed by inspecting the live
layer schema (`<service>/1?f=json`) rather than assumed from convention.
"""

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge

# --- Provenance: NSTA ArcGIS Online Feature Services -----------------------
# See module docstring for how these were discovered and confirmed.

ACTIVE_PIPELINE_SERVICE_URL = (
    "https://services-eu1.arcgis.com/OZMfUznmLTnWccBc/arcgis/rest/services/"
    "UKCS_offshore_infrastructure_pipeline_linear_(WGS84)/FeatureServer/1"
)
ACTIVE_SOURCE_TITLE = "NSTA UKCS offshore infrastructure pipeline linear (WGS84)"

REMOVED_PIPELINE_SERVICE_URL = (
    "https://services-eu1.arcgis.com/OZMfUznmLTnWccBc/arcgis/rest/services/"
    "UKCS_offshore_infrastructure_pipeline_linear_removed_(WGS84)/FeatureServer/1"
)
REMOVED_SOURCE_TITLE = "NSTA UKCS offshore infrastructure pipeline linear removed (WGS84)"

_SOURCE_TIERS = (
    ("active", ACTIVE_PIPELINE_SERVICE_URL, ACTIVE_SOURCE_TITLE),
    ("removed", REMOVED_PIPELINE_SERVICE_URL, REMOVED_SOURCE_TITLE),
)

PIPELINE_NUMBER_FIELD = "NSTAPIPNO"  # confirmed via <service>/1?f=json field list

SOURCE_CRS = "EPSG:4326"  # both services are published as "(WGS84)"
DEFAULT_WORKING_CRS = "EPSG:32631"  # see `ingest_pipeline` docstring for the reasoning

REQUEST_TIMEOUT_S = 30.0

# Known independent reference facts for PL854 (from decommissioning
# documentation), used only to validate the ingested record -- never as a
# geometry or attribute source.
REFERENCE_PIPELINE_NUMBER = "PL854"
REFERENCE_LENGTH_M = 23_700.0  # "approximate length: 23.7 km"
REFERENCE_DIAMETER_MM = 304.8  # 12 inch nominal
REFERENCE_ROUTE_KEYWORDS = ("ANGLIA", "LOGGS")


class PipelineNotFoundError(RuntimeError):
    """A pipeline number was not found in any known NSTA dataset."""


class AmbiguousPipelineError(RuntimeError):
    """A pipeline number resolved to multiple incompatible records."""


class InvalidGeometryError(RuntimeError):
    """A matched record has missing, invalid, or unexpected geometry."""


@dataclass(frozen=True)
class NstaQueryResult:
    """A single resolved pipeline record plus where it came from."""

    feature: dict[str, Any]
    source_label: str  # "active" | "removed"
    source_title: str
    source_service_url: str
    raw_cache_path: Path


@dataclass(frozen=True)
class ValidationCheck:
    """One reference-vs-source comparison. Informational, not pass/fail."""

    name: str
    reference_value: Any
    source_value: Any
    difference_pct: float | None = None
    note: str = ""


@dataclass(frozen=True)
class IngestionReport:
    """Everything needed to summarize a completed ingestion run."""

    pipeline_number: str
    source_label: str
    source_title: str
    source_service_url: str
    pipeline_number_field: str
    source_feature_id: str | None
    status: str | None
    diameter_mm: float | None
    source_length_m: float | None
    geometry_length_m: float
    source_crs: str
    working_crs: str
    bbox_wgs84: tuple[float, float, float, float]
    output_path: Path
    raw_cache_path: Path
    validation_checks: list[ValidationCheck]


def _query_service(
    service_url: str, pipeline_number: str, timeout: float = REQUEST_TIMEOUT_S
) -> dict[str, Any]:
    """Query one NSTA FeatureServer layer for a pipeline number as GeoJSON.

    Uses the service's documented `/query` REST operation (JSON in, GeoJSON
    out) -- not HTML scraping.
    """

    response = requests.get(
        f"{service_url}/query",
        params={
            "where": f"{PIPELINE_NUMBER_FIELD}='{pipeline_number}'",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if "error" in payload:
        raise RuntimeError(f"NSTA service error querying {service_url}: {payload['error']}")
    return payload


def _cache_raw_response(
    cache_dir: Path, pipeline_number: str, label: str, payload: dict[str, Any]
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{pipeline_number}_{label}.geojson"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def select_pipeline_records(
    feature_collection: dict[str, Any], pipeline_number: str
) -> dict[str, Any]:
    """Pick the single feature matching `pipeline_number` out of a FeatureCollection.

    Re-filters defensively on the real `PIPELINE_NUMBER_FIELD` rather than
    trusting the server-side `where` clause blindly. Raises loudly on zero or
    multiple matches -- never guesses.
    """

    matches = [
        feature
        for feature in feature_collection.get("features", [])
        if feature.get("properties", {}).get(PIPELINE_NUMBER_FIELD) == pipeline_number
    ]

    if not matches:
        raise PipelineNotFoundError(
            f"No records with {PIPELINE_NUMBER_FIELD}='{pipeline_number}' in this dataset."
        )
    if len(matches) > 1:
        raise AmbiguousPipelineError(
            f"{len(matches)} incompatible records found for "
            f"{PIPELINE_NUMBER_FIELD}='{pipeline_number}'; "
            "refusing to guess which is authoritative."
        )
    return matches[0]


def fetch_pipeline(pipeline_number: str, cache_dir: Path) -> NstaQueryResult:
    """Fetch one pipeline's record from NSTA, active dataset first.

    Falls back to the removed/decommissioned dataset only if the active
    dataset has zero matches (see module docstring for why that is not the
    same as "does not exist"). Every response queried is cached to
    `cache_dir`, matched or not, so a failed run still leaves evidence of
    what was checked.
    """

    for label, service_url, title in _SOURCE_TIERS:
        feature_collection = _query_service(service_url, pipeline_number)
        cache_path = _cache_raw_response(cache_dir, pipeline_number, label, feature_collection)

        try:
            feature = select_pipeline_records(feature_collection, pipeline_number)
        except PipelineNotFoundError:
            continue  # not in this tier; the next tier (e.g. removed) may still have it

        return NstaQueryResult(
            feature=feature,
            source_label=label,
            source_title=title,
            source_service_url=service_url,
            raw_cache_path=cache_path,
        )

    checked = ", ".join(label for label, _, _ in _SOURCE_TIERS)
    raise PipelineNotFoundError(
        f"Pipeline '{pipeline_number}' was not found in any known NSTA pipeline dataset "
        f"(checked: {checked})."
    )


def geometry_from_feature(feature: dict[str, Any]) -> BaseGeometry:
    """Parse and validate a GeoJSON feature's geometry as a pipeline polyline.

    Raises `InvalidGeometryError` if geometry is missing, invalid, or not a
    (Multi)LineString. A MultiLineString whose parts are topologically
    contiguous is merged into one LineString; parts that are genuinely
    disjoint are kept as-is (multiple real geometry parts legitimately
    composing one reported pipeline, e.g. separate riser/tie-in segments)
    rather than arbitrarily discarding any of them.
    """

    raw_geometry = feature.get("geometry")
    if not raw_geometry:
        raise InvalidGeometryError("Feature has no geometry.")

    geometry = shape(raw_geometry)
    if geometry.is_empty or not geometry.is_valid:
        raise InvalidGeometryError(f"Feature geometry is empty or invalid ({geometry.geom_type}).")
    if geometry.geom_type not in ("LineString", "MultiLineString"):
        raise InvalidGeometryError(
            f"Expected a (Multi)LineString pipeline geometry, got {geometry.geom_type}."
        )

    if geometry.geom_type == "MultiLineString":
        geometry = linemerge(geometry)

    return geometry


def to_working_crs(
    geometry: BaseGeometry, *, source_crs: str = SOURCE_CRS, working_crs: str = DEFAULT_WORKING_CRS
) -> BaseGeometry:
    """Reproject a single geometry from `source_crs` to `working_crs`."""

    return gpd.GeoSeries([geometry], crs=source_crs).to_crs(working_crs).iloc[0]


def compute_geometry_length_m(geometry_in_projected_crs: BaseGeometry) -> float:
    """Planar length of a geometry already expressed in a projected (metric) CRS."""

    return float(geometry_in_projected_crs.length)


def _pct_diff(value: float | None, reference: float | None) -> float | None:
    if value is None or not reference:
        return None
    return (value - reference) / reference * 100.0


def validate_against_reference(
    properties: dict[str, Any], geometry_length_m: float
) -> list[ValidationCheck]:
    """Compare the ingested record against known independent reference facts.

    Every check is informational: discrepancies are surfaced via
    `difference_pct`/`note`, never raised, per the requirement that source
    and reference lengths need not match exactly.
    """

    checks: list[ValidationCheck] = []

    nstapipno = properties.get(PIPELINE_NUMBER_FIELD)
    checks.append(
        ValidationCheck(
            name="pipeline_number",
            reference_value=REFERENCE_PIPELINE_NUMBER,
            source_value=nstapipno,
            note="match" if nstapipno == REFERENCE_PIPELINE_NUMBER else "MISMATCH",
        )
    )

    source_length_m = properties.get("LENGTH_M")
    checks.append(
        ValidationCheck(
            name="source_length_m (LENGTH_M)",
            reference_value=REFERENCE_LENGTH_M,
            source_value=source_length_m,
            difference_pct=_pct_diff(source_length_m, REFERENCE_LENGTH_M),
            note="" if source_length_m is not None else "not reported by source",
        )
    )
    checks.append(
        ValidationCheck(
            name="geometry_length_m (derived)",
            reference_value=REFERENCE_LENGTH_M,
            source_value=geometry_length_m,
            difference_pct=_pct_diff(geometry_length_m, REFERENCE_LENGTH_M),
        )
    )

    diameter_mm = properties.get("DIAMETERMM")
    checks.append(
        ValidationCheck(
            name="diameter_mm",
            reference_value=REFERENCE_DIAMETER_MM,
            source_value=diameter_mm,
            difference_pct=_pct_diff(diameter_mm, REFERENCE_DIAMETER_MM),
        )
    )

    haystack = " ".join(
        str(properties.get(field) or "") for field in ("PIPE_NAME", "DESCRIPTIO", "PIPE_SYS")
    ).upper()
    found = tuple(kw for kw in REFERENCE_ROUTE_KEYWORDS if kw in haystack)
    checks.append(
        ValidationCheck(
            name="anglia_loggs_relationship",
            reference_value=REFERENCE_ROUTE_KEYWORDS,
            source_value=found,
            note="both keywords present"
            if len(found) == len(REFERENCE_ROUTE_KEYWORDS)
            else "incomplete match",
        )
    )

    return checks


def build_canonical_gdf(
    *,
    feature: dict[str, Any],
    geometry_working_crs: BaseGeometry,
    working_crs: str,
    source_crs: str,
    source_title: str,
    geometry_length_m: float,
    retrieved_at: datetime,
) -> gpd.GeoDataFrame:
    """Build the one-row canonical pipeline GeoDataFrame for a matched feature."""

    props = feature.get("properties", {})
    record = {
        "pipeline_id": props.get(PIPELINE_NUMBER_FIELD),
        "source": source_title,
        "source_feature_id": props.get("FEATURE_ID"),
        "source_crs": source_crs,
        "working_crs": working_crs,
        "source_length_m": props.get("LENGTH_M"),
        "geometry_length_m": geometry_length_m,
        "diameter_mm": props.get("DIAMETERMM"),
        "status": props.get("STATUS"),
        "pipe_name": props.get("PIPE_NAME"),
        "fluid": props.get("FLUID"),
        "operator": props.get("REP_GROUP"),
        "retrieved_at": retrieved_at.isoformat(),
    }
    return gpd.GeoDataFrame([record], geometry=[geometry_working_crs], crs=working_crs)


def write_canonical_gpkg(gdf: gpd.GeoDataFrame, output_path: Path, layer: str = "pipeline") -> Path:
    """Write the canonical GeoDataFrame to a GeoPackage layer."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


def ingest_pipeline(
    pipeline_number: str,
    *,
    cache_dir: Path,
    output_path: Path,
    working_crs: str = DEFAULT_WORKING_CRS,
) -> IngestionReport:
    """End-to-end: acquire -> locate -> validate -> normalize -> write -> report.

    CRS choice: the source services publish geometry in EPSG:4326 (WGS84).
    EPSG:32631 (WGS84 / UTM zone 31N) is used as the working CRS because (a)
    the pipeline's own longitude range falls inside zone 31N's 0-6 E span,
    and (b) the source's own `CRS_CODE`/`CRS_NAME` attributes record the
    original survey CRS as EPSG:23031 (ED50 / UTM zone 31N) -- i.e. NSTA's
    own provenance independently confirms zone 31N is the natural projected
    zone for this route. EPSG:32631 (the WGS84 realisation of that same
    zone) is used rather than EPSG:23031 because our input geometry is
    already WGS84: pairing WGS84 coordinates with an ED50-based projection
    would silently introduce a real positional offset. Source geometry
    (EPSG:4326) is never overwritten -- only the canonical output is
    reprojected.
    """

    query_result = fetch_pipeline(pipeline_number, cache_dir=cache_dir)
    feature = query_result.feature

    geometry_wgs84 = geometry_from_feature(feature)
    bbox_wgs84 = geometry_wgs84.bounds  # (minx, miny, maxx, maxy) in EPSG:4326

    geometry_working = to_working_crs(
        geometry_wgs84, source_crs=SOURCE_CRS, working_crs=working_crs
    )
    geometry_length_m = compute_geometry_length_m(geometry_working)

    retrieved_at = datetime.now(UTC)
    gdf = build_canonical_gdf(
        feature=feature,
        geometry_working_crs=geometry_working,
        working_crs=working_crs,
        source_crs=SOURCE_CRS,
        source_title=query_result.source_title,
        geometry_length_m=geometry_length_m,
        retrieved_at=retrieved_at,
    )
    write_canonical_gpkg(gdf, output_path)

    props = feature.get("properties", {})
    checks = validate_against_reference(props, geometry_length_m)

    return IngestionReport(
        pipeline_number=pipeline_number,
        source_label=query_result.source_label,
        source_title=query_result.source_title,
        source_service_url=query_result.source_service_url,
        pipeline_number_field=PIPELINE_NUMBER_FIELD,
        source_feature_id=props.get("FEATURE_ID"),
        status=props.get("STATUS"),
        diameter_mm=props.get("DIAMETERMM"),
        source_length_m=props.get("LENGTH_M"),
        geometry_length_m=geometry_length_m,
        source_crs=SOURCE_CRS,
        working_crs=working_crs,
        bbox_wgs84=bbox_wgs84,
        output_path=output_path,
        raw_cache_path=query_result.raw_cache_path,
        validation_checks=checks,
    )


def print_ingestion_report(report: IngestionReport, *, file: Any = None) -> None:
    """Print a concise, human-readable summary of an ingestion run."""

    file = file or sys.stdout
    lines = [
        f"Pipeline:          {report.pipeline_number}",
        f"Source dataset:    {report.source_title} [{report.source_label}]",
        f"Source service:    {report.source_service_url}",
        f"Identified via:    {report.pipeline_number_field} = '{report.pipeline_number}'",
        f"Source feature id: {report.source_feature_id}",
        f"Status:            {report.status}",
        f"Diameter:          {report.diameter_mm} mm",
        f"Source length:     {report.source_length_m} m",
        f"Geometry length:   {report.geometry_length_m:.1f} m (computed in {report.working_crs})",
        f"Source CRS:        {report.source_crs}",
        f"Working CRS:       {report.working_crs}",
        f"BBox (WGS84):      {tuple(round(v, 7) for v in report.bbox_wgs84)}",
        f"Raw cache:         {report.raw_cache_path}",
        f"Output:            {report.output_path}",
        "Validation:",
    ]
    for check in report.validation_checks:
        pct = f" ({check.difference_pct:+.2f}%)" if check.difference_pct is not None else ""
        note = f" -- {check.note}" if check.note else ""
        lines.append(
            f"  - {check.name}: reference={check.reference_value} "
            f"source={check.source_value}{pct}{note}"
        )

    print("\n".join(lines), file=file)
