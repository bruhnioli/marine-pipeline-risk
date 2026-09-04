"""PL854 seabed sediment/substrate evidence base (MAR-008).

Scope and interpretation (mandatory reading before touching this module)
--------------------------------------------------------------------------
Three evidence tiers, kept as three separate, never-blended facts per
observation/station:

- Tier 1 -- observed PSA (BGS "Offshore samples: particle size analysis"):
  primary observational evidence. Includes both genuine seabed surface
  grabs AND downhole/core subsamples -- `surface_evidence_class`
  distinguishes them; only surface evidence should ever be read as
  "present-day seabed sediment at this location".
- Tier 2 -- regional mapped substrate (BGS Seabed Sediments 250k): a
  1:250,000 regional geological mapping product, never site-specific
  ground truth. `mapped_250k_*` field names say "mapped", never
  "observed", to keep this distinction visible downstream.
- Tier 3 -- predictive/model-derived substrate (BGS Predictive Seabed
  Sediments UK): `evidence_role = SECONDARY_MODEL_COMPARISON` always.
  This is a Distributional Random Forest trained on ~38,000 observations
  using bathymetry/morphometry/currents/tides as covariates -- it must
  NEVER overwrite an observed or mapped value, fill a missing observed
  D50, or be treated as ground truth for validating a future model that
  uses the same kind of predictors. Every predictive-field record carries
  an explicit `circularity_warning`.

This module never resolves the three tiers into one "best class" or a
sediment confidence score -- disagreement between tiers is itself a
reportable fact (see `compute_agreement_diagnostics`), not something to
average away.

No sediment mobility, Shields parameter, critical shear stress,
bedload/suspended transport, erosion/deposition, cohesive/noncohesive
classification, or any mud-threshold-based physical judgement is computed
anywhere in this module -- that is explicitly out of scope for MAR-008 and
left to later, separate scientific work.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString

from marine_engine.preprocessing.chainage import format_kp_label, project_point_to_route
from marine_engine.sediment import grain_size

# --- Surface-evidence classification (Section 6) ----------------------------

SURFACE_GRAB = "SURFACE_GRAB"
SURFACE_CORE_INTERVAL = "SURFACE_CORE_INTERVAL"
SUBSURFACE_INTERVAL = "SUBSURFACE_INTERVAL"
SURFACE_UNCERTAIN = "SURFACE_UNCERTAIN"
UNKNOWN_SURFACE_EVIDENCE = "UNKNOWN"

SURFACE_EVIDENCE_CLASSES: tuple[str, ...] = (
    SURFACE_GRAB,
    SURFACE_CORE_INTERVAL,
    SUBSURFACE_INTERVAL,
    SURFACE_UNCERTAIN,
    UNKNOWN_SURFACE_EVIDENCE,
)

# A numerical (floating-point representation) tolerance only -- never a
# real-world eligibility threshold (Section 6 explicitly forbids inventing
# e.g. an arbitrary 0.1 m band).
DEPTH_TOP_ZERO_TOLERANCE_M = 1e-6

# --- Distance-support / coverage scales --------------------------------------

SUPPORT_RADII_M: tuple[float, ...] = (500.0, 1000.0, 2000.0)
COVERAGE_DISTANCE_BANDS_M: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 5000.0)

# Project heuristic for planning only -- how many percentage points a
# reported gravel+sand+mud total may deviate from 100 before being flagged
# as materially inconsistent (Section 5).
GSM_TOTAL_TOLERANCE_PCT = 2.0

_PERCENT_UNIT_NAMES = ("percent", "%")


def classify_surface_evidence(
    equipment_type: str | None, depth_top: float | None, depth_base: float | None
) -> str:
    """Classify one PSA record's applicability as present-day surface evidence.

    Uses only source fields (equipment type, depth top/base) -- never
    invents a depth or a real-world eligibility band. `depth_base` is
    accepted for signature symmetry with the source schema and to make the
    call site self-documenting, but is not itself part of the decision
    (Section 6 only requires it stay explicit in the record, not that it
    change the classification).
    """

    del depth_base  # not part of the classification decision -- see docstring
    equipment_lower = (equipment_type or "").lower()
    is_grab = "grab" in equipment_lower

    if depth_top is not None and depth_top > DEPTH_TOP_ZERO_TOLERANCE_M:
        return SUBSURFACE_INTERVAL

    if depth_top is not None and abs(depth_top) <= DEPTH_TOP_ZERO_TOLERANCE_M:
        return SURFACE_GRAB if is_grab else SURFACE_CORE_INTERVAL

    # depth_top missing/None -- never invented.
    if is_grab:
        return SURFACE_GRAB
    if equipment_type:
        return SURFACE_UNCERTAIN
    return UNKNOWN_SURFACE_EVIDENCE


def is_surface_evidence_class(value: Any) -> bool:
    """True for the two classes eligible to represent present-day surface sediment."""

    return value in (SURFACE_GRAB, SURFACE_CORE_INTERVAL)


# --- Small shared helpers -----------------------------------------------------


def _epoch_ms_to_date_and_year(value: Any) -> tuple[str | None, int | None]:
    """An Esri Date field (epoch milliseconds) to an ISO date string and year."""

    if value is None:
        return None, None
    try:
        moment = datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return None, None
    return moment.date().isoformat(), moment.year


def _validate_gsm_total(
    gravel: float | None, sand: float | None, mud: float | None, gsm_units: str | None
) -> tuple[float | None, bool | None]:
    """gravel+sand+mud ~= 100, within GSM_TOTAL_TOLERANCE_PCT -- never silently renormalized."""

    if gravel is None or sand is None or mud is None:
        return None, None
    if not gsm_units or gsm_units.strip().lower() not in _PERCENT_UNIT_NAMES:
        return None, None
    total = float(gravel) + float(sand) + float(mud)
    return total, abs(total - 100.0) <= GSM_TOTAL_TOLERANCE_PCT


def _iter_coords(geometry: BaseGeometry) -> list[tuple[float, float]]:
    """Every vertex coordinate of an arbitrary (possibly multi-part) geometry."""

    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [(float(geometry.x), float(geometry.y))]
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [(float(x), float(y)) for x, y in geometry.coords]
    if hasattr(geometry, "geoms"):
        coords: list[tuple[float, float]] = []
        for part in geometry.geoms:
            coords.extend(_iter_coords(part))
        return coords
    return []


# --- Tier 1: PSA observation normalization (Sections 4-13) -------------------

PSA_OBSERVATION_COLUMNS = (
    # Identity/provenance
    "psa_data_id",
    "activity_id",
    "sample_name",
    "sample_alias",
    "sample_source",
    "client",
    "contractor",
    "equipment_type",
    "sample_date",
    "sample_year",
    "sample_age_years_at_run",
    "surface_evidence_class",
    # Location
    "longitude",
    "latitude",
    "working_easting",
    "working_northing",
    "distance_to_pipeline_m",
    "nearest_pipeline_chainage_m",
    "nearest_pipeline_kp",
    "inside_aoi",
    "nearest_chainage_station_index",
    "nearest_chainage_station_distance_m",
    # Sediment observations
    "folk_class",
    "folk_description",
    "gravel",
    "sand",
    "mud",
    "gsm_units",
    "gsm_total_pct",
    "gsm_total_valid",
    # Grain percentiles
    "d10_mm",
    "d50_mm",
    "d90_mm",
    "grain_percentile_status",
    # Sample interval
    "depth_top",
    "depth_base",
    "terminal_depth",
    "depth_units",
    "depth_datum",
    # QA/provenance
    "phi_units",
    "phi_bin_scheme",
    "phi_bin_count",
    "phi_total",
    "confidentiality",
    "accessuse_restric",
    "terms_of_use",
    "terms_of_use_url",
    "additional_info",
    "raw_attributes_json",
)


def _normalize_one_psa_record(
    properties: dict[str, Any],
    point_wgs84: Point,
    point_working: Point,
    route_working: LineString,
    aoi_geometry_working: BaseGeometry,
    run_timestamp: datetime,
) -> dict[str, Any]:
    equipment_type = properties.get("EQUIPMENT_TYPE")
    depth_top = properties.get("DEPTH_TOP")
    depth_base = properties.get("DEPTH_BASE")
    surface_class = classify_surface_evidence(equipment_type, depth_top, depth_base)

    sample_date, sample_year = _epoch_ms_to_date_and_year(properties.get("EQUIPMENT_START_DATE"))
    sample_age_years_at_run = (run_timestamp.year - sample_year) if sample_year else None

    projection = project_point_to_route(route_working, point_working)
    inside_aoi = bool(aoi_geometry_working.intersects(point_working))

    gravel, sand, mud = properties.get("GRAV"), properties.get("SAND"), properties.get("MUD")
    gsm_units = properties.get("GSM_UNITS")
    gsm_total_pct, gsm_total_valid = _validate_gsm_total(gravel, sand, mud, gsm_units)

    percentile_result = grain_size.derive_grain_percentiles(
        raw_properties=properties,
        phi_units=properties.get("PHI_UNITS"),
        gravel_pct=gravel,
        sand_pct=sand,
        mud_pct=mud,
        gsm_units=gsm_units,
        weight=properties.get("WEIGHT"),
        weight_units=properties.get("WEIGHT_UNITS"),
    )

    return {
        "psa_data_id": properties.get("PSA_DATA_ID"),
        "activity_id": properties.get("ACTIVITY_ID"),
        "sample_name": properties.get("SAMPLE_NAME"),
        "sample_alias": properties.get("SAMPLE_ALIAS"),
        "sample_source": properties.get("SAMPLE_SOURCE"),
        "client": properties.get("CLIENT"),
        "contractor": properties.get("CONTRACTOR"),
        "equipment_type": equipment_type,
        "sample_date": sample_date,
        "sample_year": sample_year,
        "sample_age_years_at_run": sample_age_years_at_run,
        "surface_evidence_class": surface_class,
        "longitude": float(point_wgs84.x),
        "latitude": float(point_wgs84.y),
        "working_easting": float(point_working.x),
        "working_northing": float(point_working.y),
        "distance_to_pipeline_m": float(projection.distance_m),
        "nearest_pipeline_chainage_m": float(projection.chainage_m),
        "nearest_pipeline_kp": format_kp_label(projection.chainage_m),
        "inside_aoi": inside_aoi,
        "nearest_chainage_station_index": None,
        "nearest_chainage_station_distance_m": None,
        "folk_class": properties.get("FOLK_CLASS"),
        "folk_description": properties.get("FOLK"),
        "gravel": gravel,
        "sand": sand,
        "mud": mud,
        "gsm_units": gsm_units,
        "gsm_total_pct": gsm_total_pct,
        "gsm_total_valid": gsm_total_valid,
        "d10_mm": percentile_result.d10_mm,
        "d50_mm": percentile_result.d50_mm,
        "d90_mm": percentile_result.d90_mm,
        "grain_percentile_status": percentile_result.status,
        "depth_top": depth_top,
        "depth_base": depth_base,
        "terminal_depth": properties.get("TERMINAL_DEPTH"),
        "depth_units": properties.get("DEPTH_UNITS"),
        "depth_datum": properties.get("DEPTH_DATUM"),
        "phi_units": properties.get("PHI_UNITS"),
        "phi_bin_scheme": percentile_result.phi_bin_scheme,
        "phi_bin_count": percentile_result.phi_bin_count,
        "phi_total": percentile_result.phi_total_before_normalization,
        "confidentiality": properties.get("CONFIDENTIALITY"),
        "accessuse_restric": properties.get("ACCESSUSE_RESTRIC"),
        "terms_of_use": properties.get("TERMS_OF_USE"),
        "terms_of_use_url": properties.get("TERMS_OF_USE_URL"),
        "additional_info": properties.get("ADDITIONAL_INFO"),
        "raw_attributes_json": json.dumps(properties, default=str),
    }


def normalize_psa_observations(
    features: list[dict[str, Any]],
    *,
    route_working: LineString,
    working_crs: str,
    aoi_geometry_working: BaseGeometry,
    run_timestamp: datetime,
) -> gpd.GeoDataFrame:
    """Every PSA GeoJSON feature -> one normalized, provenance-preserving row.

    A feature with missing/non-point geometry is skipped defensively
    (never given a fabricated location); everything else is preserved,
    including subsurface intervals (Section 4: "do not silently drop").
    """

    raw_properties: list[dict[str, Any]] = []
    points_wgs84: list[Point] = []
    for feature in features:
        geometry = feature.get("geometry")
        if not geometry or geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates")
        if not coordinates or len(coordinates) < 2:
            continue
        points_wgs84.append(Point(float(coordinates[0]), float(coordinates[1])))
        raw_properties.append(feature.get("properties") or {})

    if not points_wgs84:
        empty = pd.DataFrame(columns=list(PSA_OBSERVATION_COLUMNS))
        return gpd.GeoDataFrame(empty, geometry=[], crs=working_crs)

    points_working = gpd.GeoSeries(points_wgs84, crs="EPSG:4326").to_crs(working_crs)

    records = [
        _normalize_one_psa_record(
            properties,
            point_wgs84,
            point_working,
            route_working,
            aoi_geometry_working,
            run_timestamp,
        )
        for properties, point_wgs84, point_working in zip(
            raw_properties, points_wgs84, points_working, strict=True
        )
    ]

    df = pd.DataFrame(records, columns=list(PSA_OBSERVATION_COLUMNS))
    return gpd.GeoDataFrame(df, geometry=list(points_working), crs=working_crs)


def attach_nearest_chainage_station(
    psa_gdf_working: gpd.GeoDataFrame, chainage_gdf_working: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """The nearest of the 941 discrete chainage stations to each PSA point (Section 8, optional)."""

    result = psa_gdf_working.copy()
    if psa_gdf_working.empty or chainage_gdf_working.empty:
        return result

    station_coords = np.column_stack(
        [chainage_gdf_working.geometry.x.to_numpy(), chainage_gdf_working.geometry.y.to_numpy()]
    )
    psa_coords = np.column_stack(
        [psa_gdf_working.geometry.x.to_numpy(), psa_gdf_working.geometry.y.to_numpy()]
    )
    tree = cKDTree(station_coords)
    distances, indices = tree.query(psa_coords)

    station_indices = chainage_gdf_working["station_index"].to_numpy()
    result["nearest_chainage_station_index"] = station_indices[indices]
    result["nearest_chainage_station_distance_m"] = distances
    return result


# --- Tier 2: BGS Seabed Sediments 250k normalization (Section 14) -----------

SEABED_250K_COLUMNS = (
    "bgs_id",
    "folk_s",
    "folk_d50_text",
    "lex_rcs_d",
    "version",
    "released",
    "released_year",
    "nom_scale",
    "aoi_intersection_area_km2",
    "pipeline_intersection_length_m",
    "chainage_range_start_m",
    "chainage_range_end_m",
)


def normalize_seabed_sediments_250k(
    features: list[dict[str, Any]], *, working_crs: str
) -> gpd.GeoDataFrame:
    """BGS Seabed Sediments 250k polygon features -> a normalized GeoDataFrame.

    `folk_d50_text` is `FOLK_D50` preserved verbatim as source TEXT -- the
    field name is misleading (it is not a numeric median grain diameter);
    it is never parsed or converted to a number here (Section 14/2).
    """

    records: list[dict[str, Any]] = []
    geometries_wgs84 = []
    for feature in features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        properties = feature.get("properties") or {}
        released, released_year = _epoch_ms_to_date_and_year(properties.get("RELEASED"))
        geometries_wgs84.append(geom)
        records.append(
            {
                "bgs_id": properties.get("BGS_ID"),
                "folk_s": properties.get("FOLK_S"),
                "folk_d50_text": properties.get("FOLK_D50"),
                "lex_rcs_d": properties.get("LEX_RCS_D"),
                "version": properties.get("VERSION"),
                "released": released,
                "released_year": released_year,
                "nom_scale": properties.get("NOM_SCALE"),
            }
        )

    if not geometries_wgs84:
        empty = pd.DataFrame(columns=list(SEABED_250K_COLUMNS))
        return gpd.GeoDataFrame(empty, geometry=[], crs=working_crs)

    gdf_wgs84 = gpd.GeoDataFrame(records, geometry=geometries_wgs84, crs="EPSG:4326")
    return gdf_wgs84.to_crs(working_crs)


def compute_250k_intersections(
    gdf_working: gpd.GeoDataFrame,
    *,
    aoi_geometry_working: BaseGeometry,
    route_working: LineString,
) -> gpd.GeoDataFrame:
    """AOI intersection area and pipeline intersection length/chainage range per polygon."""

    result = gdf_working.copy()
    aoi_areas_km2 = []
    pipe_lengths_m = []
    chainage_starts_m = []
    chainage_ends_m = []
    for geom in result.geometry:
        aoi_part = geom.intersection(aoi_geometry_working)
        aoi_areas_km2.append(0.0 if aoi_part.is_empty else aoi_part.area / 1_000_000.0)

        pipe_part = route_working.intersection(geom)
        if pipe_part.is_empty:
            pipe_lengths_m.append(0.0)
            chainage_starts_m.append(None)
            chainage_ends_m.append(None)
            continue

        pipe_lengths_m.append(float(pipe_part.length))
        chainages = [route_working.project(Point(c)) for c in _iter_coords(pipe_part)]
        chainage_starts_m.append(min(chainages) if chainages else None)
        chainage_ends_m.append(max(chainages) if chainages else None)

    result["aoi_intersection_area_km2"] = aoi_areas_km2
    result["pipeline_intersection_length_m"] = pipe_lengths_m
    result["chainage_range_start_m"] = chainage_starts_m
    result["chainage_range_end_m"] = chainage_ends_m
    # Selecting columns without "geometry" would silently downgrade this from a
    # GeoDataFrame to a plain DataFrame -- keep it explicit.
    return result[[*SEABED_250K_COLUMNS, "geometry"]]


# --- Tier 3: BGS Predictive Seabed Sediments normalization (Sections 16-18) --


def normalize_predictive_folk_polygons(
    features: list[dict[str, Any]], *, working_crs: str
) -> gpd.GeoDataFrame:
    """Predictive Folk-class grid-cell polygons -> a normalized GeoDataFrame.

    Field names carry `predictive_` throughout so this can never be
    mistaken downstream for observed or mapped substrate (Section 16).
    """

    records: list[dict[str, Any]] = []
    geometries_wgs84 = []
    for feature in features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        properties = feature.get("properties") or {}
        geometries_wgs84.append(geom)
        records.append(
            {
                "predictive_folk_class": properties.get("FOLK_S"),
                "predictive_folk_class_description": properties.get("FOLK_CLASS"),
                "dataset": properties.get("DATASET"),
                "version": properties.get("VERSION"),
            }
        )

    columns = ("predictive_folk_class", "predictive_folk_class_description", "dataset", "version")
    if not geometries_wgs84:
        empty = pd.DataFrame(columns=list(columns))
        return gpd.GeoDataFrame(empty, geometry=[], crs=working_crs)

    gdf_wgs84 = gpd.GeoDataFrame(records, geometry=geometries_wgs84, crs="EPSG:4326")
    return gdf_wgs84.to_crs(working_crs)


# --- Spatial point-in-polygon attribute joins (Sections 15, 23) ------------


def join_polygon_attributes_at_points(
    points_gdf_working: gpd.GeoDataFrame,
    polygons_gdf_working: gpd.GeoDataFrame,
    column_map: dict[str, str],
) -> pd.DataFrame:
    """For each point, the named attributes of whichever polygon (if any) contains it.

    A point matching more than one polygon (e.g. exactly on a shared edge)
    takes the first spatial match found -- overlapping regional/predictive
    polygons are not expected in this data; this is a defensive tie-break,
    never a scientific claim about which polygon is "more correct".
    """

    n = len(points_gdf_working)
    result: dict[str, list[Any]] = {out_col: [None] * n for out_col in column_map.values()}
    if polygons_gdf_working.empty or n == 0:
        return pd.DataFrame(result)

    sindex = polygons_gdf_working.sindex
    for i, point in enumerate(points_gdf_working.geometry):
        candidates = list(sindex.query(point, predicate="intersects"))
        if not candidates:
            continue
        row = polygons_gdf_working.iloc[candidates[0]]
        for source_col, out_col in column_map.items():
            result[out_col][i] = row[source_col]
    return pd.DataFrame(result)


def attach_mapped_and_predictive_at_psa_points(
    psa_gdf_working: gpd.GeoDataFrame,
    seabed_250k_gdf_working: gpd.GeoDataFrame,
    predictive_folk_gdf_working: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """The mapped-250k and predictive Folk class AT each PSA point (for Section 23 agreement only).

    Never merged back into the canonical PSA observation table/output
    (Section 18/21) -- this is a separate, comparison-only attachment.
    """

    result = psa_gdf_working.reset_index(drop=True).copy()
    mapped = join_polygon_attributes_at_points(
        result, seabed_250k_gdf_working, column_map={"folk_s": "mapped_250k_folk_class_at_point"}
    )
    predictive = join_polygon_attributes_at_points(
        result,
        predictive_folk_gdf_working,
        column_map={"predictive_folk_class": "predictive_folk_class_at_point"},
    )
    result["mapped_250k_folk_class_at_point"] = mapped["mapped_250k_folk_class_at_point"].to_numpy()
    result["predictive_folk_class_at_point"] = predictive[
        "predictive_folk_class_at_point"
    ].to_numpy()
    return result


# --- Chainage-level support diagnostics and nearest-surface-PSA (Sections 9, 22) --


def compute_psa_support_counts(
    chainage_gdf_working: gpd.GeoDataFrame,
    psa_surface_gdf_working: gpd.GeoDataFrame,
    radii_m: tuple[float, ...] = SUPPORT_RADII_M,
) -> pd.DataFrame:
    """Surface-PSA-sample counts within each radius of each station -- diagnostics only.

    Never a confidence score, interpolation radius, or proof the sample's
    sediment occurs at the pipe (Section 9).
    """

    columns = {f"psa_surface_count_{int(r)}m": [0] * len(chainage_gdf_working) for r in radii_m}
    if psa_surface_gdf_working.empty or chainage_gdf_working.empty:
        return pd.DataFrame(columns)

    station_coords = np.column_stack(
        [chainage_gdf_working.geometry.x.to_numpy(), chainage_gdf_working.geometry.y.to_numpy()]
    )
    psa_coords = np.column_stack(
        [
            psa_surface_gdf_working.geometry.x.to_numpy(),
            psa_surface_gdf_working.geometry.y.to_numpy(),
        ]
    )
    tree = cKDTree(psa_coords)
    for r in radii_m:
        counts = tree.query_ball_point(station_coords, r, return_length=True)
        columns[f"psa_surface_count_{int(r)}m"] = list(np.asarray(counts, dtype=int))
    return pd.DataFrame(columns)


_NEAREST_PSA_COLUMNS = (
    "nearest_psa_id",
    "nearest_psa_distance_m",
    "nearest_psa_sample_date",
    "nearest_psa_sample_year",
    "nearest_psa_folk_class",
    "nearest_psa_gravel_pct",
    "nearest_psa_sand_pct",
    "nearest_psa_mud_pct",
    "nearest_psa_d10_mm",
    "nearest_psa_d50_mm",
    "nearest_psa_d90_mm",
    "nearest_psa_percentile_status",
)

_NEAREST_PSA_SOURCE_COLUMNS = (
    "psa_data_id",
    None,  # distance is computed, not copied
    "sample_date",
    "sample_year",
    "folk_class",
    "gravel",
    "sand",
    "mud",
    "d10_mm",
    "d50_mm",
    "d90_mm",
    "grain_percentile_status",
)


def attach_nearest_surface_psa(
    chainage_gdf_working: gpd.GeoDataFrame, psa_surface_gdf_working: gpd.GeoDataFrame
) -> pd.DataFrame:
    """The single nearest SURFACE PSA observation's identity/values at each station.

    Every output field name retains "nearest" so downstream code cannot
    mistake this for a measurement taken at the pipeline itself (Section 9).
    """

    n = len(chainage_gdf_working)
    result: dict[str, list[Any]] = {col: [None] * n for col in _NEAREST_PSA_COLUMNS}
    if psa_surface_gdf_working.empty or n == 0:
        return pd.DataFrame(result)

    station_coords = np.column_stack(
        [chainage_gdf_working.geometry.x.to_numpy(), chainage_gdf_working.geometry.y.to_numpy()]
    )
    psa_coords = np.column_stack(
        [
            psa_surface_gdf_working.geometry.x.to_numpy(),
            psa_surface_gdf_working.geometry.y.to_numpy(),
        ]
    )
    tree = cKDTree(psa_coords)
    distances, indices = tree.query(station_coords)

    psa_reset = psa_surface_gdf_working.reset_index(drop=True)
    result["nearest_psa_distance_m"] = [float(d) for d in distances]
    for out_col, source_col in zip(_NEAREST_PSA_COLUMNS, _NEAREST_PSA_SOURCE_COLUMNS, strict=True):
        if source_col is None:
            continue
        source_values = psa_reset[source_col].to_numpy()
        result[out_col] = [source_values[idx] for idx in indices]
    return pd.DataFrame(result)


# --- Full chainage-level sediment evidence assembly (Section 22) -----------

CHAINAGE_SEDIMENT_COLUMNS = (
    "pipeline_id",
    "station_index",
    "chainage_m",
    "kp_label",
    "mapped_250k_bgs_id",
    "mapped_250k_folk_class",
    "mapped_250k_folk_d50_text",
    "mapped_250k_release",
    "mapped_250k_nominal_scale",
    "psa_surface_count_500m",
    "psa_surface_count_1000m",
    "psa_surface_count_2000m",
    "nearest_psa_id",
    "nearest_psa_distance_m",
    "nearest_psa_sample_date",
    "nearest_psa_sample_year",
    "nearest_psa_folk_class",
    "nearest_psa_gravel_pct",
    "nearest_psa_sand_pct",
    "nearest_psa_mud_pct",
    "nearest_psa_d10_mm",
    "nearest_psa_d50_mm",
    "nearest_psa_d90_mm",
    "nearest_psa_percentile_status",
    "predictive_folk_class",
    "predictive_gravel_pct",
    "predictive_sand_pct",
    "predictive_mud_pct",
    "predictive_evidence_role",
    "predictive_circularity_warning",
)


def build_chainage_sediment_evidence(
    *,
    chainage_gdf: gpd.GeoDataFrame,
    psa_gdf_working: gpd.GeoDataFrame,
    seabed_250k_gdf_working: gpd.GeoDataFrame,
    predictive_folk_gdf_working: gpd.GeoDataFrame,
    working_crs: str,
) -> pd.DataFrame:
    """Assemble the 941-station chainage sediment evidence table.

    Every one of the 941 stations is retained regardless of match (same
    invariant as MAR-007's `join_source_provenance`); no `canonical_d50` is
    created here (Section 22 -- that decision is deferred to a later
    ticket). Predictive percentage fields are left null at chainage-station
    density -- see the metadata's `predictive_percentage_chainage_note` for
    why (Section 16's own "if not safely queryable, do not fabricate").
    """

    chainage_working = (
        chainage_gdf.to_crs(working_crs).sort_values("station_index").reset_index(drop=True)
    )

    mapped_250k = join_polygon_attributes_at_points(
        chainage_working,
        seabed_250k_gdf_working,
        column_map={
            "bgs_id": "mapped_250k_bgs_id",
            "folk_s": "mapped_250k_folk_class",
            "folk_d50_text": "mapped_250k_folk_d50_text",
            "released": "mapped_250k_release",
            "nom_scale": "mapped_250k_nominal_scale",
        },
    )

    predictive_folk = join_polygon_attributes_at_points(
        chainage_working,
        predictive_folk_gdf_working,
        column_map={"predictive_folk_class": "predictive_folk_class"},
    )
    n = len(chainage_working)
    predictive_folk["predictive_gravel_pct"] = [None] * n
    predictive_folk["predictive_sand_pct"] = [None] * n
    predictive_folk["predictive_mud_pct"] = [None] * n
    predictive_folk["predictive_evidence_role"] = ["SECONDARY_MODEL_COMPARISON"] * n
    predictive_folk["predictive_circularity_warning"] = [True] * n

    is_surface = psa_gdf_working["surface_evidence_class"].isin(
        (SURFACE_GRAB, SURFACE_CORE_INTERVAL)
    )
    psa_surface = psa_gdf_working[is_surface]

    support_counts = compute_psa_support_counts(chainage_working, psa_surface)
    nearest_psa = attach_nearest_surface_psa(chainage_working, psa_surface)

    chainage_df = pd.concat(
        [
            chainage_working[
                ["pipeline_id", "station_index", "chainage_m", "kp_label"]
            ].reset_index(drop=True),
            mapped_250k.reset_index(drop=True),
            support_counts.reset_index(drop=True),
            nearest_psa.reset_index(drop=True),
            predictive_folk.reset_index(drop=True),
        ],
        axis=1,
    )
    return chainage_df[list(CHAINAGE_SEDIMENT_COLUMNS)]


# --- Predictive comparison output (Sections 17, 21) -------------------------

PREDICTIVE_COMPARISON_COLUMNS = (
    "psa_data_id",
    "distance_to_pipeline_m",
    "surface_evidence_class",
    "observed_folk_class",
    "mapped_250k_folk_class",
    "predictive_folk_class",
    "predictive_gravel_pct",
    "predictive_sand_pct",
    "predictive_mud_pct",
    "provider",
    "dataset",
    "evidence_role",
    "model_type",
    "model_covariates_include_bathymetry",
    "model_covariates_include_morphometry",
    "model_covariates_include_currents",
    "model_covariates_include_tides",
    "circularity_warning",
    "circularity_note",
)

_CIRCULARITY_NOTE = (
    "This product must not be treated as independent ground truth for a downstream "
    "model that uses overlapping bathymetry, morphology, or hydrodynamic predictors."
)


def build_predictive_comparison_table(
    psa_with_comparisons: gpd.GeoDataFrame,
    *,
    predictive_percentages_by_psa_id: dict[Any, dict[str, float | None]] | None = None,
) -> pd.DataFrame:
    """One row per SURFACE PSA observation, comparing it against the predictive product only.

    Never merged into the canonical PSA observation table (Section 18/21).
    """

    is_surface = psa_with_comparisons["surface_evidence_class"].isin(
        (SURFACE_GRAB, SURFACE_CORE_INTERVAL)
    )
    surface = psa_with_comparisons[is_surface].reset_index(drop=True)
    n = len(surface)
    predictive_percentages_by_psa_id = predictive_percentages_by_psa_id or {}

    def _percentages(key: str) -> list[float | None]:
        return [
            (predictive_percentages_by_psa_id.get(psa_id) or {}).get(key)
            for psa_id in surface["psa_data_id"]
        ]

    records = {
        "psa_data_id": surface["psa_data_id"],
        "distance_to_pipeline_m": surface["distance_to_pipeline_m"],
        "surface_evidence_class": surface["surface_evidence_class"],
        "observed_folk_class": surface["folk_class"],
        "mapped_250k_folk_class": surface["mapped_250k_folk_class_at_point"],
        "predictive_folk_class": surface["predictive_folk_class_at_point"],
        "predictive_gravel_pct": _percentages("gravel"),
        "predictive_sand_pct": _percentages("sand"),
        "predictive_mud_pct": _percentages("mud"),
        "provider": ["BGS"] * n,
        "dataset": ["Predictive Seabed Sediments UK"] * n,
        "evidence_role": ["SECONDARY_MODEL_COMPARISON"] * n,
        "model_type": ["Distributional Random Forest"] * n,
        "model_covariates_include_bathymetry": [True] * n,
        "model_covariates_include_morphometry": [True] * n,
        "model_covariates_include_currents": [True] * n,
        "model_covariates_include_tides": [True] * n,
        "circularity_warning": [True] * n,
        "circularity_note": [_CIRCULARITY_NOTE] * n,
    }
    return pd.DataFrame(records, columns=list(PREDICTIVE_COMPARISON_COLUMNS))


# --- Agreement diagnostics (Section 23) -- reported, never scored ----------


def _compare_classes(df: pd.DataFrame, column_a: str, column_b: str) -> dict[str, int]:
    both = df[[column_a, column_b]].dropna()
    same = int((both[column_a] == both[column_b]).sum())
    different = int(len(both) - same)
    missing = int(len(df) - len(both))
    return {
        "same_class_count": same,
        "different_class_count": different,
        "missing_mapped_class_count": missing,
    }


def compute_agreement_diagnostics(
    psa_with_comparisons: gpd.GeoDataFrame, chainage_df: pd.DataFrame
) -> dict[str, Any]:
    """Report (never score) class agreement across tiers.

    A point disagreement is never treated as proof "the map is wrong".
    """

    is_surface = psa_with_comparisons["surface_evidence_class"].isin(
        (SURFACE_GRAB, SURFACE_CORE_INTERVAL)
    )
    surface = psa_with_comparisons[is_surface]

    return {
        "observed_vs_mapped_250k": _compare_classes(
            surface, "folk_class", "mapped_250k_folk_class_at_point"
        ),
        "observed_vs_predictive": _compare_classes(
            surface, "folk_class", "predictive_folk_class_at_point"
        ),
        "mapped_250k_vs_predictive_along_chainage": _compare_classes(
            chainage_df, "mapped_250k_folk_class", "predictive_folk_class"
        ),
        "note": (
            "Exact string comparison of each source's own short Folk-class code. The "
            "predictive layer's short-code notation uses parenthetical modifiers "
            "(e.g. '(g)S') where PSA/250k use unparenthesized notation (e.g. 'gS'); this "
            "under-counts true semantic agreement involving the predictive source -- a "
            "known notational limitation, not corrected here. No agreement figure here is "
            "a confidence score, and no single disagreement is treated as proof the "
            "regional map is 'wrong'."
        ),
    }


# --- Coverage diagnostics (Section 24) ---------------------------------------


def compute_coverage_diagnostics(psa_gdf_working: gpd.GeoDataFrame) -> dict[str, Any]:
    total = len(psa_gdf_working)
    is_surface = psa_gdf_working["surface_evidence_class"].isin(
        (SURFACE_GRAB, SURFACE_CORE_INTERVAL)
    )
    is_subsurface = psa_gdf_working["surface_evidence_class"] == SUBSURFACE_INTERVAL
    is_uncertain = psa_gdf_working["surface_evidence_class"].isin(
        [SURFACE_UNCERTAIN, UNKNOWN_SURFACE_EVIDENCE]
    )

    has_folk = psa_gdf_working["folk_class"].notna()
    has_gsm = psa_gdf_working[["gravel", "sand", "mud"]].notna().all(axis=1)
    has_d50 = psa_gdf_working["d50_mm"].notna()

    sample_years = pd.to_numeric(psa_gdf_working["sample_year"], errors="coerce").dropna()
    distances = pd.to_numeric(psa_gdf_working["distance_to_pipeline_m"], errors="coerce").dropna()
    surface_distances = pd.to_numeric(
        psa_gdf_working.loc[is_surface, "distance_to_pipeline_m"], errors="coerce"
    ).dropna()

    band_counts = {
        f"surface_within_{int(r)}m": int((surface_distances <= r).sum())
        for r in COVERAGE_DISTANCE_BANDS_M
    }

    return {
        "total_records_in_aoi": total,
        "surface_evidence_records": int(is_surface.sum()),
        "subsurface_records": int(is_subsurface.sum()),
        "uncertain_records": int(is_uncertain.sum()),
        "records_with_folk_class": int(has_folk.sum()),
        "records_with_gsm_fractions": int(has_gsm.sum()),
        "records_with_usable_d50": int(has_d50.sum()),
        "sample_year_min": int(sample_years.min()) if not sample_years.empty else None,
        "sample_year_max": int(sample_years.max()) if not sample_years.empty else None,
        "sample_year_median": float(sample_years.median()) if not sample_years.empty else None,
        "distance_to_pipeline_m_min": float(distances.min()) if not distances.empty else None,
        "distance_to_pipeline_m_median": float(distances.median()) if not distances.empty else None,
        "distance_to_pipeline_m_p95": float(distances.quantile(0.95))
        if not distances.empty
        else None,
        "distance_to_pipeline_m_max": float(distances.max()) if not distances.empty else None,
        **band_counts,
    }


def compute_chainage_support_proportions(chainage_df: pd.DataFrame) -> dict[str, float]:
    """The proportion of PL854 chainage with >=1 surface PSA sample within each radius."""

    total = len(chainage_df)
    if total == 0:
        return {}
    return {
        f"chainage_proportion_with_surface_psa_within_{int(r)}m": float(
            (chainage_df[f"psa_surface_count_{int(r)}m"] >= 1).sum() / total
        )
        for r in SUPPORT_RADII_M
    }


# --- Section 25: descriptive-only D50 spatial-support assessment -----------

PROMISING = "PROMISING"
SPARSE = "SPARSE"
VERY_SPARSE = "VERY_SPARSE"
NOT_ASSESSABLE = "NOT_ASSESSABLE"

# Project heuristic for planning only -- never a physical/statistical
# threshold, and must never feed a future physical model directly. The
# external scientific reviewer makes the actual go/no-go decision from the
# reported metrics; this classification is a convenience label only.
_PROMISING_MIN_CHAINAGE_PROPORTION_1000M = 0.80
_PROMISING_MIN_USABLE_D50_COUNT = 20
_SPARSE_MIN_CHAINAGE_PROPORTION_1000M = 0.40
_SPARSE_MIN_USABLE_D50_COUNT = 5


def assess_d50_spatial_support(chainage_df: pd.DataFrame, coverage: dict[str, Any]) -> str:
    """PROMISING / SPARSE / VERY_SPARSE / NOT_ASSESSABLE -- descriptive only.

    See the module docstring for the full rationale.
    """

    total = len(chainage_df)
    if total == 0:
        return NOT_ASSESSABLE

    proportion_1000m = float((chainage_df["psa_surface_count_1000m"] >= 1).sum() / total)
    usable_d50_count = coverage.get("records_with_usable_d50", 0)

    if (
        proportion_1000m >= _PROMISING_MIN_CHAINAGE_PROPORTION_1000M
        and usable_d50_count >= _PROMISING_MIN_USABLE_D50_COUNT
    ):
        return PROMISING
    if (
        proportion_1000m >= _SPARSE_MIN_CHAINAGE_PROPORTION_1000M
        and usable_d50_count >= _SPARSE_MIN_USABLE_D50_COUNT
    ):
        return SPARSE
    if usable_d50_count > 0:
        return VERY_SPARSE
    return NOT_ASSESSABLE


# --- Output writing (Sections 19-22, 26) ------------------------------------


def write_psa_observations(
    gdf: gpd.GeoDataFrame, parquet_path: Path, gpkg_path: Path | None = None
) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gdf.drop(columns="geometry")).to_parquet(parquet_path, index=False)
    if gpkg_path is not None:
        gpkg_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(gpkg_path, driver="GPKG", layer="psa_observations")


def write_seabed_sediments_250k(
    gdf: gpd.GeoDataFrame, gpkg_path: Path, parquet_path: Path | None = None
) -> None:
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(gpkg_path, driver="GPKG", layer="seabed_sediments_250k")
    if parquet_path is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(gdf.drop(columns="geometry")).to_parquet(parquet_path, index=False)


def write_predictive_comparison(df: pd.DataFrame, parquet_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)


def write_chainage_sediment_evidence(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def write_sediment_evidence_metadata(
    *,
    providers: dict[str, Any],
    query_timestamp: datetime,
    aoi_identifier: str,
    coverage: dict[str, Any],
    chainage_support: dict[str, Any],
    agreement: dict[str, Any],
    d50_assessment: str,
    outputs: dict[str, Path],
    output_path: Path,
) -> Path:
    metadata = {
        "providers": providers,
        "evidence_hierarchy": [
            "Tier 1 -- observed PSA (BGS Offshore samples: particle size analysis); primary "
            "observational evidence, includes both surface grabs and downhole/core subsamples.",
            "Tier 2 -- regional mapped substrate (BGS Seabed Sediments 250k, 1:250,000); "
            "regional geological mapping, never site-specific ground truth.",
            "Tier 3 -- predictive/model-derived substrate (BGS Predictive Seabed Sediments UK); "
            "SECONDARY_MODEL_COMPARISON only, never blended with the other two tiers.",
        ],
        "tiers_never_blended": True,
        "surface_classification_logic": {
            "classes": list(SURFACE_EVIDENCE_CLASSES),
            "rule": (
                "DEPTH_TOP > tolerance -> SUBSURFACE_INTERVAL (always, regardless of "
                "equipment). DEPTH_TOP ~= 0 (within a floating-point tolerance of "
                f"{DEPTH_TOP_ZERO_TOLERANCE_M} m, never a real-world eligibility band) -> "
                "SURFACE_GRAB if EQUIPMENT_TYPE names a grab sampler, else "
                "SURFACE_CORE_INTERVAL. DEPTH_TOP missing -> SURFACE_GRAB if a grab "
                "sampler is named (equipment-defensible), else SURFACE_UNCERTAIN if some "
                "other equipment is named, else UNKNOWN."
            ),
        },
        "psa_unit_handling": {
            "gsm_units_seen": "percent",
            "gsm_total_tolerance_pct": GSM_TOTAL_TOLERANCE_PCT,
            "gsm_total_tolerance_note": "project heuristic for planning only",
            "phi_units_seen": ("percent", "grams"),
            "unknown_or_missing_units": "never used to derive a numeric percentile",
        },
        "grain_percentile_algorithm": {
            "phi_definition": "phi = -log2(d_mm); d_mm = 2 ** (-phi)",
            "bin_deduplication": (
                "PHI_X and PHI_X_0 are the same phi position under two field-name "
                "aliases; populated aliases that materially disagree make the record "
                "AMBIGUOUS_BIN_SCHEME rather than guessed at."
            ),
            "whole_sample_coverage_guard": (
                "a materially present (>1%, project heuristic) gravel/sand/mud fraction "
                "with zero phi bins in its Wentworth/Udden phi sub-range, or a bin total "
                "materially inconsistent with the record's own stated WEIGHT, marks the "
                "record INSUFFICIENT_BINS/INVALID_TOTAL rather than computing a "
                "partial-fraction percentile mislabeled as whole-sample."
            ),
            "interpolation": (
                "linear interpolation of the cumulative-mass curve in phi space "
                "(coarsest bin outward); D10 is the small-mm/high-phi crossing, D90 the "
                "large-mm/low-phi crossing, per the standard percent-finer convention."
            ),
            "never_derived_from": [
                "Folk class or a class-lookup table",
                "gravel/sand/mud percentages alone",
                "the BGS predictive product",
            ],
        },
        "seabed_sediments_250k_scale": (
            "1:250,000 regional geological mapping -- not pipeline-scale ground truth. "
            "FOLK_D50 is preserved as source TEXT (the field name is misleading; it is "
            "not a numeric median grain diameter)."
        ),
        "predictive_model_leakage_warning": _CIRCULARITY_NOTE,
        "predictive_percentage_chainage_note": (
            "Predictive sand/gravel/mud percentages (layers 9/10/11 of the Predictive "
            "Seabed Sediments service) are raster layers with no bulk attribute-query "
            "mechanism -- only a per-point 'identify' call (confirmed live). Sampling "
            "this at all 941 chainage stations would require ~2800 individual network "
            "requests, which is not 'straightforward'/'safely queryable' at that "
            "density (Section 16); these percentages are therefore sampled only at "
            "surface PSA observation points (see bgs_predictive_sediment_comparison), "
            "and left null in the chainage evidence table rather than fabricated."
        ),
        "query_timestamp": query_timestamp.isoformat(),
        "aoi_identifier": aoi_identifier,
        "coverage_diagnostics": coverage,
        "chainage_support_proportions": chainage_support,
        "agreement_diagnostics": agreement,
        "d50_spatial_support_assessment": d50_assessment,
        "d50_spatial_support_assessment_note": (
            "Descriptive classification only, using an explicit project heuristic for "
            "planning purposes -- never a physical/statistical threshold, and must "
            "never feed a future physical model directly."
        ),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "scientific_limitations": [
            "No sediment mobility, Shields parameter, critical shear stress, "
            "bedload/suspended transport, erosion/deposition, or cohesive/noncohesive "
            "classification is computed anywhere in this ticket.",
            "A sample's presence inside the AOI, or its distance to the pipeline, is "
            "never treated as proof its sediment occurs AT the pipeline.",
            "The predictive product's short-code Folk notation differs from PSA/250k's "
            "own notation, which under-counts true agreement in the diagnostics above.",
            "No canonical_d50 or single 'best' sediment class is produced -- observed, "
            "mapped, and predictive evidence remain three separate facts throughout.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_path


# --- Report printing (Sections 24, 25, 31) ----------------------------------


def print_sediment_evidence_report(
    *,
    coverage: dict[str, Any],
    chainage_support: dict[str, Any],
    agreement: dict[str, Any],
    d50_assessment: str,
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 Seabed Sediment Evidence Base (MAR-008) ===", ""]

    lines.append("PSA inventory:")
    lines.append(f"  Total records in AOI:        {coverage['total_records_in_aoi']}")
    lines.append(f"  Surface evidence records:    {coverage['surface_evidence_records']}")
    lines.append(f"  Subsurface records:          {coverage['subsurface_records']}")
    lines.append(f"  Uncertain records:           {coverage['uncertain_records']}")
    lines.append(f"  Records with Folk class:     {coverage['records_with_folk_class']}")
    lines.append(f"  Records with GSM fractions:  {coverage['records_with_gsm_fractions']}")
    lines.append(f"  Records with usable D50:     {coverage['records_with_usable_d50']}")
    lines.append(
        "  Sample year range: "
        f"{coverage['sample_year_min']}-{coverage['sample_year_max']} "
        f"(median {coverage['sample_year_median']})"
    )
    lines.append("")

    lines.append("Pipeline proximity (surface evidence):")
    lines.append(
        "  Distance to pipeline (m): "
        f"min={coverage['distance_to_pipeline_m_min']} "
        f"median={coverage['distance_to_pipeline_m_median']} "
        f"p95={coverage['distance_to_pipeline_m_p95']} "
        f"max={coverage['distance_to_pipeline_m_max']}"
    )
    for band in COVERAGE_DISTANCE_BANDS_M:
        lines.append(
            f"  Surface samples within {int(band)} m: {coverage[f'surface_within_{int(band)}m']}"
        )
    lines.append("")

    lines.append("Chainage support (surface PSA within radius, proportion of 941 stations):")
    for key, value in chainage_support.items():
        lines.append(f"  {key}: {value:.1%}")
    lines.append("")

    lines.append("Agreement diagnostics (reported, not scored):")
    for comparison_name in (
        "observed_vs_mapped_250k",
        "observed_vs_predictive",
        "mapped_250k_vs_predictive_along_chainage",
    ):
        comparison = agreement[comparison_name]
        lines.append(
            f"  {comparison_name}: same={comparison['same_class_count']} "
            f"different={comparison['different_class_count']} "
            f"missing={comparison['missing_mapped_class_count']}"
        )
    lines.append(f"  Note: {agreement['note']}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("D50 SPATIAL SUPPORT ASSESSMENT")
    lines.append(f"  {d50_assessment}")
    lines.append(
        "  Descriptive only, using an explicit project heuristic for planning "
        "purposes -- not a physical/statistical threshold. The external scientific "
        "reviewer decides whether to build a continuous pipeline D50 field from these "
        "metrics."
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "Critical conclusion: this is observation + regional interpretation + "
        "predictive estimate -- never collapsed into a single 'PL854 sediment is X' claim."
    )

    print("\n".join(lines), file=file)
