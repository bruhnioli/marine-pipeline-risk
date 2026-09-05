"""PL854 metocean forcing evidence base assembly (MAR-009/MAR-009A).

Scope and interpretation (mandatory reading before touching this module)
--------------------------------------------------------------------------
Three separate, never-blended metocean evidence products, each mapped from
PL854's 941 chainage stations onto a much SMALLER set of real Copernicus
Marine model grid cells ("support nodes") -- never 941 fabricated
independent time series (Section 4 of the ticket):

- Primary current (`current_node_id`): 1.5 km 3D hourly current, deepest
  PHYSICALLY ELIGIBLE standard level only (see `metocean/current.py` --
  never the model's native bottom cell, never called "bottom current").
- Long-term surface current context (`current_lt_node_id`): 7 km hourly 2D
  current, 1993 onward -- `LONG_TERM_SURFACE_CURRENT_CONTEXT` role only.
  Never used to fill a missing primary-current value, never downscaled,
  never a source of vertical current profiles.
- Wave climate (`wave_node_id`): 3-hourly WAVEWATCH III reanalysis, 1980
  onward, surface parameters only.

Support nodes are the actual wet model grid cells nearest each chainage
station -- nearest-neighbour assignment (never bilinear interpolation of
data or masks) preserves real model-cell provenance rather than fabricating
pipeline-resolution forcing (Section 5). Only nodes ACTUALLY assigned to at
least one chainage station are ever normalized into a time series
(MAR-009A) -- the real PL854 acquisition confirmed that normalizing every
wet cell in the request bbox silently inflated wave/long-term-current
"support node" counts (330 / 18) far past the real route-used count (14 /
4); `normalize_wave`/`normalize_long_term_surface_current`/
`normalize_primary_current` must always be called with an
already-filtered, chainage-used node list.

Model bathymetry (`deptho` on each product's own static dataset) and the
canonical MAR-006 `depth_lat_m` (positive-down relative to LAT) are DIFFERENT
vertical datums and are never subtracted or compared as an "error" here
(Section 9) -- both are carried side by side, and downstream metadata
states `canonical_model_bathymetry_vertical_datums_not_harmonised = true`.

Static vs dynamic grid-index reconciliation (MAR-009A, Section 6)
--------------------------------------------------------------------
A support node's `(grid_i, grid_j)` is identified from a PRODUCT'S OWN
STATIC dataset. Reusing those same integer indices directly against a
DIFFERENT (dynamic) xarray Dataset object assumes the two datasets share
an identical local array origin/order merely because they represent the
same underlying model grid -- an assumption this module never makes.
`reconcile_node_grid_indices` re-resolves each node's canonical
longitude/latitude against the ACTUAL dynamic dataset's own coordinate
arrays before every dynamic sampling operation, and refuses to guess (skips
the node, flagged) if no sufficiently close cell exists there.

No bed shear stress, Shields parameter, sediment mobility, wave orbital
velocity, erosion/deposition, scour, free-span, or risk scoring is
computed anywhere in this module -- this ticket stops at forcing evidence.
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from shapely.geometry import Point

from marine_engine.metocean import current as current_module
from marine_engine.metocean import wave as wave_module

# --- Support nodes (Section 4-5) ---------------------------------------------


@dataclass(frozen=True)
class SupportNode:
    """One real, wet Copernicus Marine model grid cell used as a source support node."""

    node_id: str
    grid_i: int
    grid_j: int
    longitude: float
    latitude: float


def identify_wet_grid_cells(
    longitude: np.ndarray, latitude: np.ndarray, wet_mask_2d: np.ndarray, node_id_prefix: str
) -> list[SupportNode]:
    """Every wet cell of a regular lon/lat grid (`wet_mask_2d` indexed `[lat, lon]`).

    A stable node id is derived from the cell's own (grid_j, grid_i)
    indices -- never from an artificial per-chainage-station counter, so
    the same real model cell always gets the same id across runs.
    """

    nodes = []
    for j in range(len(latitude)):
        for i in range(len(longitude)):
            if wet_mask_2d[j, i]:
                nodes.append(
                    SupportNode(
                        node_id=f"{node_id_prefix}_{j:04d}_{i:04d}",
                        grid_i=i,
                        grid_j=j,
                        longitude=float(longitude[i]),
                        latitude=float(latitude[j]),
                    )
                )
    return nodes


def map_points_to_nearest_node(
    points_working: gpd.GeoSeries, nodes: list[SupportNode], working_crs: str
) -> pd.DataFrame:
    """The nearest support node id + distance (m, in `working_crs`) for each point.

    This is the mechanism that collapses 941 dense chainage points onto a
    much smaller number of real model cells -- never one fabricated
    time series per station.
    """

    n = len(points_working)
    if not nodes:
        return pd.DataFrame({"node_id": [None] * n, "distance_m": [None] * n})

    node_points_working = gpd.GeoSeries(
        [Point(node.longitude, node.latitude) for node in nodes], crs="EPSG:4326"
    ).to_crs(working_crs)
    node_coords = np.column_stack(
        [node_points_working.x.to_numpy(), node_points_working.y.to_numpy()]
    )
    tree = cKDTree(node_coords)

    point_coords = np.column_stack([points_working.x.to_numpy(), points_working.y.to_numpy()])
    distances, indices = tree.query(point_coords)

    node_ids = [node.node_id for node in nodes]
    return pd.DataFrame(
        {"node_id": [node_ids[i] for i in indices], "distance_m": [float(d) for d in distances]}
    )


def build_support_node_table(
    nodes: list[SupportNode],
    assigned_node_ids: pd.Series,
    assigned_distances_m: pd.Series,
    *,
    model_bathymetry_by_node_id: dict[str, float] | None = None,
    deptho_lev_by_node_id: dict[str, float] | None = None,
    source_product: str,
    source_dataset: str,
    evidence_role: str,
) -> pd.DataFrame:
    """One row per node ACTUALLY used by at least one chainage station.

    `station_count_assigned`/min/max distance are computed from the real
    assignment, never assumed.
    """

    used_ids = set(assigned_node_ids.dropna())
    model_bathymetry_by_node_id = model_bathymetry_by_node_id or {}
    deptho_lev_by_node_id = deptho_lev_by_node_id or {}

    records = []
    for node in nodes:
        if node.node_id not in used_ids:
            continue
        distances = assigned_distances_m[assigned_node_ids == node.node_id]
        records.append(
            {
                "node_id": node.node_id,
                "grid_i": node.grid_i,
                "grid_j": node.grid_j,
                "longitude": node.longitude,
                "latitude": node.latitude,
                "model_bathymetry_m": model_bathymetry_by_node_id.get(node.node_id),
                "deptho_lev_interp": deptho_lev_by_node_id.get(node.node_id),
                "station_count_assigned": int(len(distances)),
                "min_chainage_distance_to_node_m": float(distances.min())
                if len(distances)
                else None,
                "max_chainage_distance_to_node_m": float(distances.max())
                if len(distances)
                else None,
                "source_product": source_product,
                "source_dataset": source_dataset,
                "evidence_role": evidence_role,
            }
        )
    return pd.DataFrame(records)


# --- Static/dynamic grid reconciliation (MAR-009A, Section 6) ---------------

# Project data-QA heuristic, not a physical constant: how close (as a
# fraction of the coordinate array's own median spacing) a node's canonical
# longitude/latitude must land to a dynamic dataset's own grid cell to be
# treated as "the same real cell" rather than an unreconcilable mismatch.
GRID_RECONCILIATION_TOLERANCE_FRACTION = 0.1


def check_depth_coordinate_alignment(
    static_depth_m: np.ndarray, dynamic_depth_m: np.ndarray
) -> bool:
    """Whether a static dataset's depth coordinate exactly matches a dynamic one's.

    Only an EXACT match (same length, same values) is treated as
    "alignable" -- never guessed at (Section 3). When this is False, the
    static mask must not be used as an eligibility condition; the
    bathymetry-depth constraint alone still applies.
    """

    static_depth_m = np.asarray(static_depth_m)
    dynamic_depth_m = np.asarray(dynamic_depth_m)
    if static_depth_m.shape != dynamic_depth_m.shape:
        return False
    return bool(np.array_equal(static_depth_m, dynamic_depth_m))


def reconcile_node_grid_indices(
    node: SupportNode,
    dynamic_longitude: np.ndarray,
    dynamic_latitude: np.ndarray,
    *,
    tolerance_fraction: float = GRID_RECONCILIATION_TOLERANCE_FRACTION,
) -> tuple[int, int] | None:
    """Re-resolve a node's canonical lon/lat against a DIFFERENT dataset's own grid.

    Never reuses the node's own `(grid_i, grid_j)` (identified from
    whichever dataset the node itself came from) against an unrelated
    xarray Dataset object. Returns `None` -- never a guessed index -- if no
    cell in `dynamic_longitude`/`dynamic_latitude` lands within
    `tolerance_fraction` of that axis's own median grid spacing.
    """

    dynamic_longitude = np.asarray(dynamic_longitude)
    dynamic_latitude = np.asarray(dynamic_latitude)

    lon_idx = int(np.argmin(np.abs(dynamic_longitude - node.longitude)))
    lat_idx = int(np.argmin(np.abs(dynamic_latitude - node.latitude)))

    lon_spacing = (
        float(np.median(np.abs(np.diff(np.sort(dynamic_longitude)))))
        if len(dynamic_longitude) > 1
        else 0.0
    )
    lat_spacing = (
        float(np.median(np.abs(np.diff(np.sort(dynamic_latitude)))))
        if len(dynamic_latitude) > 1
        else 0.0
    )

    lon_diff = abs(float(dynamic_longitude[lon_idx]) - node.longitude)
    lat_diff = abs(float(dynamic_latitude[lat_idx]) - node.latitude)

    if lon_spacing and lon_diff > tolerance_fraction * lon_spacing:
        return None
    if lat_spacing and lat_diff > tolerance_fraction * lat_spacing:
        return None
    # Zero-length axes (a single-cell subset) have no spacing to compare
    # against -- fall back to an exact-value check so a genuine mismatch
    # is still caught rather than silently accepted.
    if not lon_spacing and lon_diff > 0:
        return None
    if not lat_spacing and lat_diff > 0:
        return None

    return lon_idx, lat_idx


# --- Time coordinate helpers --------------------------------------------------


def _to_utc_timestamps(time_values: np.ndarray) -> pd.DatetimeIndex:
    index = pd.to_datetime(time_values)
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


# --- Primary current normalization (Sections 8, 9, 19) -----------------------

PRIMARY_CURRENT_COLUMNS = (
    "current_node_id",
    "time_utc",
    "uo_m_s",
    "vo_m_s",
    "current_speed_m_s",
    "current_direction_to_deg",
    "current_sample_depth_m",
    "model_bathymetry_m",
    "height_above_model_bed_m",
    "height_above_model_bed_valid",
    "depth_level_index",
    "below_bed_finite_candidate_count",
    "max_below_bed_candidate_depth_m",
    "source_dataset",
    "temporal_role",
)


def normalize_primary_current(
    current_ds: xr.Dataset,
    *,
    nodes: list[SupportNode],
    model_bathymetry_by_node_id: dict[str, float],
    static_mask_by_node_id: dict[str, np.ndarray] | None,
    source_dataset: str,
    evidence_role: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Per (support node, hour): the deepest PHYSICALLY ELIGIBLE standard-level current only.

    Never duplicated across 941 chainage stations -- one row per real
    support node per timestamp (Section 19). `static_mask_by_node_id` (keyed
    by node_id, each value a mask array aligned to `current_ds["depth"]`) is
    `None` when the static mask could not be unambiguously aligned to the
    dynamic depth coordinate (Section 3) -- the bathymetry-depth
    constraint alone still applies in that case.

    Each node's `(grid_i, grid_j)` is re-resolved against `current_ds`'s
    OWN longitude/latitude before sampling (MAR-009A, Section 6) -- never
    the static dataset's indices reused blindly. A node whose coordinate
    cannot be reconciled within tolerance is skipped entirely and returned
    in the second tuple element, never silently sampled from the wrong
    cell.
    """

    if not nodes:
        return pd.DataFrame(columns=list(PRIMARY_CURRENT_COLUMNS)), []

    depths_m = current_ds["depth"].to_numpy()
    dynamic_longitude = current_ds["longitude"].to_numpy()
    dynamic_latitude = current_ds["latitude"].to_numpy()
    times = _to_utc_timestamps(current_ds["time"].to_numpy())
    static_mask_by_node_id = static_mask_by_node_id or {}

    records: list[dict[str, Any]] = []
    unreconciled_node_ids: list[str] = []
    for node in nodes:
        resolved = reconcile_node_grid_indices(node, dynamic_longitude, dynamic_latitude)
        if resolved is None:
            unreconciled_node_ids.append(node.node_id)
            continue
        dyn_lon_idx, dyn_lat_idx = resolved

        uo_node = current_ds["uo"].isel(latitude=dyn_lat_idx, longitude=dyn_lon_idx).to_numpy()
        vo_node = current_ds["vo"].isel(latitude=dyn_lat_idx, longitude=dyn_lon_idx).to_numpy()
        model_bathymetry_m = model_bathymetry_by_node_id.get(node.node_id)
        mask_at_depths = static_mask_by_node_id.get(node.node_id)

        for t_index, time_value in enumerate(times):
            selection = current_module.select_deepest_valid_standard_level(
                depths_m,
                uo_node[t_index],
                vo_node[t_index],
                model_bathymetry_m=model_bathymetry_m,
                mask_at_depths=mask_at_depths,
            )
            height_m, height_valid = current_module.compute_height_above_model_bed_m(
                model_bathymetry_m, selection.depth_m
            )
            speed_m_s = None
            direction_to_deg = None
            if selection.uo_m_s is not None:
                speed_m_s = float(
                    current_module.compute_current_speed_m_s(
                        np.array([selection.uo_m_s]), np.array([selection.vo_m_s])
                    )[0]
                )
                direction_to_deg = float(
                    current_module.compute_current_direction_to_deg(
                        np.array([selection.uo_m_s]), np.array([selection.vo_m_s])
                    )[0]
                )
            records.append(
                {
                    "current_node_id": node.node_id,
                    "time_utc": time_value,
                    "uo_m_s": selection.uo_m_s,
                    "vo_m_s": selection.vo_m_s,
                    "current_speed_m_s": speed_m_s,
                    "current_direction_to_deg": direction_to_deg,
                    "current_sample_depth_m": selection.depth_m,
                    "model_bathymetry_m": model_bathymetry_m,
                    "height_above_model_bed_m": height_m,
                    "height_above_model_bed_valid": height_valid,
                    "depth_level_index": selection.depth_index,
                    "below_bed_finite_candidate_count": selection.below_bed_finite_candidate_count,
                    "max_below_bed_candidate_depth_m": selection.max_below_bed_candidate_depth_m,
                    "source_dataset": source_dataset,
                    "temporal_role": evidence_role,
                }
            )

    return pd.DataFrame(records, columns=list(PRIMARY_CURRENT_COLUMNS)), unreconciled_node_ids


def compute_below_bed_diagnostics(primary_current_df: pd.DataFrame) -> pd.DataFrame:
    """Per current_node_id: how many below-bed finite candidates were excluded, and when.

    QA diagnostics only (Section 4) -- these never enter the canonical
    forcing statistics; they only explain what the physical eligibility
    filter removed.
    """

    columns = (
        "current_node_id",
        "below_model_bed_finite_candidate_count",
        "timestamps_with_below_bed_finite_candidates",
        "max_below_bed_candidate_depth_m",
    )
    if primary_current_df.empty:
        return pd.DataFrame(columns=list(columns))

    records = []
    for node_id, group in primary_current_df.groupby("current_node_id"):
        counts = group["below_bed_finite_candidate_count"].fillna(0)
        max_depths = group["max_below_bed_candidate_depth_m"].dropna()
        records.append(
            {
                "current_node_id": node_id,
                "below_model_bed_finite_candidate_count": int(counts.sum()),
                "timestamps_with_below_bed_finite_candidates": int((counts > 0).sum()),
                "max_below_bed_candidate_depth_m": float(max_depths.max())
                if len(max_depths)
                else None,
            }
        )
    return pd.DataFrame(records, columns=list(columns))


# --- Long-term surface current normalization (Sections 11, 20) --------------

LONG_TERM_SURFACE_CURRENT_COLUMNS = (
    "current_lt_node_id",
    "time_utc",
    "uo_surface_m_s",
    "vo_surface_m_s",
    "surface_current_speed_m_s",
    "surface_current_direction_to_deg",
    "source_dataset",
    "evidence_role",
)


def normalize_long_term_surface_current(
    current_ds: xr.Dataset, *, nodes: list[SupportNode], source_dataset: str, evidence_role: str
) -> tuple[pd.DataFrame, list[str]]:
    """Per (long-term support node, hour): 2D surface current only -- context, never bottom.

    `nodes` must already be filtered to chainage-used nodes only (MAR-009A,
    Section 7) -- this function does not itself filter by usage. Each
    node's grid indices are re-resolved against `current_ds`'s own
    coordinate arrays before sampling (Section 6); an unreconcilable node
    is skipped and returned in the second tuple element.
    """

    if not nodes:
        return pd.DataFrame(columns=list(LONG_TERM_SURFACE_CURRENT_COLUMNS)), []

    dynamic_longitude = current_ds["longitude"].to_numpy()
    dynamic_latitude = current_ds["latitude"].to_numpy()
    times = _to_utc_timestamps(current_ds["time"].to_numpy())
    records: list[dict[str, Any]] = []
    unreconciled_node_ids: list[str] = []
    for node in nodes:
        resolved = reconcile_node_grid_indices(node, dynamic_longitude, dynamic_latitude)
        if resolved is None:
            unreconciled_node_ids.append(node.node_id)
            continue
        dyn_lon_idx, dyn_lat_idx = resolved

        uo_node = current_ds["uo"].isel(latitude=dyn_lat_idx, longitude=dyn_lon_idx).to_numpy()
        vo_node = current_ds["vo"].isel(latitude=dyn_lat_idx, longitude=dyn_lon_idx).to_numpy()
        speed = current_module.compute_current_speed_m_s(uo_node, vo_node)
        direction_to = current_module.compute_current_direction_to_deg(uo_node, vo_node)

        for t_index, time_value in enumerate(times):
            is_valid = bool(np.isfinite(uo_node[t_index]) and np.isfinite(vo_node[t_index]))
            records.append(
                {
                    "current_lt_node_id": node.node_id,
                    "time_utc": time_value,
                    "uo_surface_m_s": float(uo_node[t_index]) if is_valid else None,
                    "vo_surface_m_s": float(vo_node[t_index]) if is_valid else None,
                    "surface_current_speed_m_s": float(speed[t_index]) if is_valid else None,
                    "surface_current_direction_to_deg": float(direction_to[t_index])
                    if is_valid
                    else None,
                    "source_dataset": source_dataset,
                    "evidence_role": evidence_role,
                }
            )

    return pd.DataFrame(
        records, columns=list(LONG_TERM_SURFACE_CURRENT_COLUMNS)
    ), unreconciled_node_ids


# --- Wave normalization (Sections 12, 13, 22) --------------------------------

WAVE_COLUMNS = (
    "wave_node_id",
    "time_utc",
    "hs_m",
    "hs_valid",
    "tp_s",
    "tm02_s",
    "tm10_s",
    "wave_mean_direction_from_deg",
    "wave_mean_direction_to_deg",
    "stokes_u_m_s",
    "stokes_v_m_s",
    "source_dataset",
)


def _optional_variable(
    ds: xr.Dataset, name: str, lat_idx: int, lon_idx: int, length: int
) -> np.ndarray:
    if name not in ds.variables:
        return np.full(length, np.nan)
    return ds[name].isel(latitude=lat_idx, longitude=lon_idx).to_numpy()


def normalize_wave(
    wave_ds: xr.Dataset, *, nodes: list[SupportNode], source_dataset: str
) -> tuple[pd.DataFrame, list[str]]:
    """Per (wave support node, 3-hour step): Hs/Tp/Tm02/Tm10/direction, QA'd, never a bed force.

    `nodes` must already be filtered to chainage-used nodes only (MAR-009A,
    Section 7). Each node's grid indices are re-resolved against
    `wave_ds`'s own coordinate arrays before sampling (Section 6); an
    unreconcilable node is skipped and returned in the second tuple element.
    """

    if not nodes:
        return pd.DataFrame(columns=list(WAVE_COLUMNS)), []

    dynamic_longitude = wave_ds["longitude"].to_numpy()
    dynamic_latitude = wave_ds["latitude"].to_numpy()
    times = _to_utc_timestamps(wave_ds["time"].to_numpy())
    records: list[dict[str, Any]] = []
    unreconciled_node_ids: list[str] = []
    for node in nodes:
        resolved = reconcile_node_grid_indices(node, dynamic_longitude, dynamic_latitude)
        if resolved is None:
            unreconciled_node_ids.append(node.node_id)
            continue
        dyn_lon_idx, dyn_lat_idx = resolved

        hs = _optional_variable(wave_ds, "VHM0", dyn_lat_idx, dyn_lon_idx, len(times))
        tp = _optional_variable(wave_ds, "VTPK", dyn_lat_idx, dyn_lon_idx, len(times))
        tm02 = _optional_variable(wave_ds, "VTM02", dyn_lat_idx, dyn_lon_idx, len(times))
        tm10 = _optional_variable(wave_ds, "VTM10", dyn_lat_idx, dyn_lon_idx, len(times))
        vmdr = _optional_variable(wave_ds, "VMDR", dyn_lat_idx, dyn_lon_idx, len(times))
        stokes_u = _optional_variable(wave_ds, "VSDX", dyn_lat_idx, dyn_lon_idx, len(times))
        stokes_v = _optional_variable(wave_ds, "VSDY", dyn_lat_idx, dyn_lon_idx, len(times))

        hs_valid_mask = wave_module.validate_significant_wave_height(hs)
        direction_from = wave_module.normalize_direction_deg(vmdr)
        direction_to = wave_module.derive_wave_direction_to_deg(vmdr)

        for t_index, time_value in enumerate(times):
            records.append(
                {
                    "wave_node_id": node.node_id,
                    "time_utc": time_value,
                    "hs_m": float(hs[t_index]) if hs_valid_mask[t_index] else None,
                    "hs_valid": bool(hs_valid_mask[t_index]) if np.isfinite(hs[t_index]) else None,
                    "tp_s": float(tp[t_index]) if np.isfinite(tp[t_index]) else None,
                    "tm02_s": float(tm02[t_index]) if np.isfinite(tm02[t_index]) else None,
                    "tm10_s": float(tm10[t_index]) if np.isfinite(tm10[t_index]) else None,
                    "wave_mean_direction_from_deg": float(direction_from[t_index])
                    if np.isfinite(vmdr[t_index])
                    else None,
                    "wave_mean_direction_to_deg": float(direction_to[t_index])
                    if np.isfinite(vmdr[t_index])
                    else None,
                    "stokes_u_m_s": float(stokes_u[t_index])
                    if np.isfinite(stokes_u[t_index])
                    else None,
                    "stokes_v_m_s": float(stokes_v[t_index])
                    if np.isfinite(stokes_v[t_index])
                    else None,
                    "source_dataset": source_dataset,
                }
            )

    return pd.DataFrame(records, columns=list(WAVE_COLUMNS)), unreconciled_node_ids


# --- Descriptive statistics (Sections 24, 25) --------------------------------


def compute_current_node_statistics(primary_current_df: pd.DataFrame) -> pd.DataFrame:
    """Per current_node_id: temporal support + speed percentiles.

    No EVT fitting, no return periods.
    """

    columns = (
        "current_node_id",
        "start_time_utc",
        "end_time_utc",
        "expected_hourly_count",
        "valid_hour_count",
        "completeness_pct",
        "current_speed_mean_m_s",
        "current_speed_median_m_s",
        "current_speed_p90_m_s",
        "current_speed_p95_m_s",
        "current_speed_p99_m_s",
        "current_speed_max_m_s",
        "representative_sample_depth_m",
    )
    if primary_current_df.empty:
        return pd.DataFrame(columns=list(columns))

    records = []
    for node_id, group in primary_current_df.groupby("current_node_id"):
        valid_speed = group["current_speed_m_s"].dropna()
        start = group["time_utc"].min()
        end = group["time_utc"].max()
        expected_hours = (
            int(round((end - start).total_seconds() / 3600.0)) + 1 if pd.notna(start) else 0
        )
        mode_depth = group["current_sample_depth_m"].dropna().mode()
        records.append(
            {
                "current_node_id": node_id,
                "start_time_utc": start,
                "end_time_utc": end,
                "expected_hourly_count": expected_hours,
                "valid_hour_count": int(len(valid_speed)),
                "completeness_pct": (100.0 * len(valid_speed) / expected_hours)
                if expected_hours
                else None,
                "current_speed_mean_m_s": float(valid_speed.mean()) if len(valid_speed) else None,
                "current_speed_median_m_s": float(valid_speed.median())
                if len(valid_speed)
                else None,
                "current_speed_p90_m_s": float(valid_speed.quantile(0.90))
                if len(valid_speed)
                else None,
                "current_speed_p95_m_s": float(valid_speed.quantile(0.95))
                if len(valid_speed)
                else None,
                "current_speed_p99_m_s": float(valid_speed.quantile(0.99))
                if len(valid_speed)
                else None,
                "current_speed_max_m_s": float(valid_speed.max()) if len(valid_speed) else None,
                "representative_sample_depth_m": float(mode_depth.iloc[0])
                if len(mode_depth)
                else None,
            }
        )
    return pd.DataFrame(records, columns=list(columns))


def compute_long_term_surface_current_statistics(long_term_df: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "current_lt_node_id",
        "start_time_utc",
        "end_time_utc",
        "valid_hour_count",
        "surface_current_speed_mean_m_s",
        "surface_current_speed_p95_m_s",
        "surface_current_speed_p99_m_s",
        "surface_current_speed_max_m_s",
    )
    if long_term_df.empty:
        return pd.DataFrame(columns=list(columns))

    records = []
    for node_id, group in long_term_df.groupby("current_lt_node_id"):
        valid_speed = group["surface_current_speed_m_s"].dropna()
        records.append(
            {
                "current_lt_node_id": node_id,
                "start_time_utc": group["time_utc"].min(),
                "end_time_utc": group["time_utc"].max(),
                "valid_hour_count": int(len(valid_speed)),
                "surface_current_speed_mean_m_s": float(valid_speed.mean())
                if len(valid_speed)
                else None,
                "surface_current_speed_p95_m_s": float(valid_speed.quantile(0.95))
                if len(valid_speed)
                else None,
                "surface_current_speed_p99_m_s": float(valid_speed.quantile(0.99))
                if len(valid_speed)
                else None,
                "surface_current_speed_max_m_s": float(valid_speed.max())
                if len(valid_speed)
                else None,
            }
        )
    return pd.DataFrame(records, columns=list(columns))


def compute_wave_node_statistics(wave_df: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "wave_node_id",
        "start_time_utc",
        "end_time_utc",
        "expected_3hour_count",
        "valid_3hour_count",
        "completeness_pct",
        "hs_mean_m",
        "hs_median_m",
        "hs_p90_m",
        "hs_p95_m",
        "hs_p99_m",
        "hs_max_m",
        "tp_median_s",
        "tp_p95_s",
    )
    if wave_df.empty:
        return pd.DataFrame(columns=list(columns))

    records = []
    for node_id, group in wave_df.groupby("wave_node_id"):
        valid_hs = group["hs_m"].dropna()
        valid_tp = group["tp_s"].dropna()
        start = group["time_utc"].min()
        end = group["time_utc"].max()
        expected_steps = (
            int(round((end - start).total_seconds() / (3 * 3600.0))) + 1 if pd.notna(start) else 0
        )
        records.append(
            {
                "wave_node_id": node_id,
                "start_time_utc": start,
                "end_time_utc": end,
                "expected_3hour_count": expected_steps,
                "valid_3hour_count": int(len(valid_hs)),
                "completeness_pct": (100.0 * len(valid_hs) / expected_steps)
                if expected_steps
                else None,
                "hs_mean_m": float(valid_hs.mean()) if len(valid_hs) else None,
                "hs_median_m": float(valid_hs.median()) if len(valid_hs) else None,
                "hs_p90_m": float(valid_hs.quantile(0.90)) if len(valid_hs) else None,
                "hs_p95_m": float(valid_hs.quantile(0.95)) if len(valid_hs) else None,
                "hs_p99_m": float(valid_hs.quantile(0.99)) if len(valid_hs) else None,
                "hs_max_m": float(valid_hs.max()) if len(valid_hs) else None,
                "tp_median_s": float(valid_tp.median()) if len(valid_tp) else None,
                "tp_p95_s": float(valid_tp.quantile(0.95)) if len(valid_tp) else None,
            }
        )
    return pd.DataFrame(records, columns=list(columns))


def compute_annual_max_hs(wave_df: pd.DataFrame) -> pd.DataFrame:
    """Annual maximum Hs per wave node -- an observational diagnostic only.

    Never fit to GEV/Gumbel here; no 10/50/100-year return levels are
    computed anywhere in this module (Section 25).
    """

    if wave_df.empty:
        return pd.DataFrame(columns=["wave_node_id", "year", "annual_max_hs_m"])

    working = wave_df.dropna(subset=["hs_m"]).copy()
    working["year"] = working["time_utc"].dt.year
    grouped = (
        working.groupby(["wave_node_id", "year"])["hs_m"].max().reset_index(name="annual_max_hs_m")
    )
    return grouped


def compute_short_window_surface_context_ratio(
    long_term_df: pd.DataFrame,
    primary_current_start: pd.Timestamp | None,
    primary_current_end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Descriptive-only ratio of overlap-period vs full-period p95/p99 surface speed.

    Never a scale factor, bias correction, or physics coefficient
    (Section 27). Null when the comparison cannot be formed cleanly.
    """

    columns = (
        "current_lt_node_id",
        "short_window_surface_context_ratio_p95",
        "short_window_surface_context_ratio_p99",
    )
    if long_term_df.empty or primary_current_start is None or primary_current_end is None:
        return pd.DataFrame(columns=list(columns))

    records = []
    for node_id, group in long_term_df.groupby("current_lt_node_id"):
        full = group["surface_current_speed_m_s"].dropna()
        overlap = group.loc[
            (group["time_utc"] >= primary_current_start)
            & (group["time_utc"] <= primary_current_end),
            "surface_current_speed_m_s",
        ].dropna()
        if full.empty or overlap.empty:
            records.append(
                {
                    "current_lt_node_id": node_id,
                    "short_window_surface_context_ratio_p95": None,
                    "short_window_surface_context_ratio_p99": None,
                }
            )
            continue
        full_p95, full_p99 = full.quantile(0.95), full.quantile(0.99)
        overlap_p95, overlap_p99 = overlap.quantile(0.95), overlap.quantile(0.99)
        records.append(
            {
                "current_lt_node_id": node_id,
                "short_window_surface_context_ratio_p95": float(overlap_p95 / full_p95)
                if full_p95
                else None,
                "short_window_surface_context_ratio_p99": float(overlap_p99 / full_p99)
                if full_p99
                else None,
            }
        )
    return pd.DataFrame(records, columns=list(columns))


def compute_primary_current_route_summary(primary_current_df: pd.DataFrame) -> dict[str, Any]:
    """Route-wide (across all nodes/hours) primary-current integrity summary (Section 19).

    `all_within_water_column` is the single required pass/fail statement:
    True only if every row where a canonical current sample WAS selected
    has `height_above_model_bed_valid == True` -- a row with no eligible
    depth at all (nothing selected) is never counted as a violation.
    """

    empty = {
        "model_bathymetry_m_min": None,
        "model_bathymetry_m_median": None,
        "model_bathymetry_m_max": None,
        "current_sample_depth_m_min": None,
        "current_sample_depth_m_median": None,
        "current_sample_depth_m_max": None,
        "height_above_model_bed_m_min": None,
        "height_above_model_bed_m_median": None,
        "height_above_model_bed_m_p95": None,
        "height_above_model_bed_m_max": None,
        "all_within_water_column": None,
    }
    if primary_current_df.empty:
        return empty

    bathymetry = primary_current_df["model_bathymetry_m"].dropna()
    depth = primary_current_df["current_sample_depth_m"].dropna()
    height = primary_current_df["height_above_model_bed_m"].dropna()
    all_within = bool(primary_current_df["height_above_model_bed_valid"].fillna(True).all())

    return {
        "model_bathymetry_m_min": float(bathymetry.min()) if len(bathymetry) else None,
        "model_bathymetry_m_median": float(bathymetry.median()) if len(bathymetry) else None,
        "model_bathymetry_m_max": float(bathymetry.max()) if len(bathymetry) else None,
        "current_sample_depth_m_min": float(depth.min()) if len(depth) else None,
        "current_sample_depth_m_median": float(depth.median()) if len(depth) else None,
        "current_sample_depth_m_max": float(depth.max()) if len(depth) else None,
        "height_above_model_bed_m_min": float(height.min()) if len(height) else None,
        "height_above_model_bed_m_median": float(height.median()) if len(height) else None,
        "height_above_model_bed_m_p95": float(height.quantile(0.95)) if len(height) else None,
        "height_above_model_bed_m_max": float(height.max()) if len(height) else None,
        "all_within_water_column": all_within,
    }


def compute_distance_diagnostics(mapping_df: pd.DataFrame) -> dict[str, float | None]:
    """min/median/p95/max station-to-node distance (m) across all chainage stations.

    A spatial-support diagnostic only (Section 10) -- never converted into
    a confidence score. `mapping_df` is the per-station nearest-node
    mapping (one row per chainage station, as returned by
    `map_points_to_nearest_node`), not the support-node table.
    """

    if "distance_m" not in mapping_df.columns:
        return {"min_m": None, "median_m": None, "p95_m": None, "max_m": None}
    distances = pd.to_numeric(mapping_df["distance_m"], errors="coerce").dropna()
    if distances.empty:
        return {"min_m": None, "median_m": None, "p95_m": None, "max_m": None}
    return {
        "min_m": float(distances.min()),
        "median_m": float(distances.median()),
        "p95_m": float(distances.quantile(0.95)),
        "max_m": float(distances.max()),
    }


# --- Chainage evidence assembly (Section 23) ---------------------------------

CHAINAGE_METOCEAN_COLUMNS = (
    "pipeline_id",
    "station_index",
    "chainage_m",
    "kp_label",
    "depth_lat_m",
    "current_node_id",
    "current_node_distance_m",
    "current_model_bathymetry_m",
    "current_sample_depth_m",
    "current_height_above_model_bed_m",
    "current_speed_mean_m_s",
    "current_speed_median_m_s",
    "current_speed_p90_m_s",
    "current_speed_p95_m_s",
    "current_speed_p99_m_s",
    "current_speed_max_m_s",
    "current_valid_hour_count",
    "current_lt_node_id",
    "current_lt_node_distance_m",
    "surface_current_speed_mean_m_s",
    "surface_current_speed_p95_m_s",
    "surface_current_speed_p99_m_s",
    "surface_current_speed_max_m_s",
    "wave_node_id",
    "wave_node_distance_m",
    "wave_model_bathymetry_m",
    "hs_mean_m",
    "hs_median_m",
    "hs_p90_m",
    "hs_p95_m",
    "hs_p99_m",
    "hs_max_m",
    "tp_median_s",
    "tp_p95_s",
    "wave_valid_3hour_count",
)


def build_chainage_metocean_evidence(
    *,
    chainage_gdf: gpd.GeoDataFrame,
    canonical_depth_df: pd.DataFrame | None,
    current_mapping: pd.DataFrame,
    current_stats: pd.DataFrame,
    current_node_bathymetry: dict[str, float],
    long_term_mapping: pd.DataFrame,
    long_term_stats: pd.DataFrame,
    wave_mapping: pd.DataFrame,
    wave_stats: pd.DataFrame,
    wave_node_bathymetry: dict[str, float],
) -> pd.DataFrame:
    """Assemble the 941-station chainage metocean evidence table.

    Every station is retained regardless of match (same invariant as
    MAR-007/MAR-008); no time-series is duplicated into this table, only
    per-node descriptive statistics (Section 23).
    """

    base = chainage_gdf[["pipeline_id", "station_index", "chainage_m", "kp_label"]].reset_index(
        drop=True
    )
    n = len(base)

    if canonical_depth_df is not None and "station_index" in canonical_depth_df.columns:
        depth_lookup = canonical_depth_df.set_index("station_index")["depth_lat_m"]
        base["depth_lat_m"] = base["station_index"].map(depth_lookup)
    else:
        base["depth_lat_m"] = [None] * n

    current_part = current_mapping.rename(
        columns={"node_id": "current_node_id", "distance_m": "current_node_distance_m"}
    ).reset_index(drop=True)
    current_part["current_model_bathymetry_m"] = current_part["current_node_id"].map(
        current_node_bathymetry
    )
    current_part = current_part.merge(
        current_stats.rename(
            columns={
                "representative_sample_depth_m": "current_sample_depth_m",
                "valid_hour_count": "current_valid_hour_count",
            }
        )[
            [
                "current_node_id",
                "current_sample_depth_m",
                "current_speed_mean_m_s",
                "current_speed_median_m_s",
                "current_speed_p90_m_s",
                "current_speed_p95_m_s",
                "current_speed_p99_m_s",
                "current_speed_max_m_s",
                "current_valid_hour_count",
            ]
        ],
        on="current_node_id",
        how="left",
    )
    current_part["current_height_above_model_bed_m"] = current_part.apply(
        lambda row: (
            row["current_model_bathymetry_m"] - row["current_sample_depth_m"]
            if pd.notna(row["current_model_bathymetry_m"])
            and pd.notna(row["current_sample_depth_m"])
            else None
        ),
        axis=1,
    )

    long_term_part = long_term_mapping.rename(
        columns={"node_id": "current_lt_node_id", "distance_m": "current_lt_node_distance_m"}
    ).reset_index(drop=True)
    long_term_part = long_term_part.merge(
        long_term_stats[
            [
                "current_lt_node_id",
                "surface_current_speed_mean_m_s",
                "surface_current_speed_p95_m_s",
                "surface_current_speed_p99_m_s",
                "surface_current_speed_max_m_s",
            ]
        ],
        on="current_lt_node_id",
        how="left",
    )

    wave_part = wave_mapping.rename(
        columns={"node_id": "wave_node_id", "distance_m": "wave_node_distance_m"}
    ).reset_index(drop=True)
    wave_part["wave_model_bathymetry_m"] = wave_part["wave_node_id"].map(wave_node_bathymetry)
    wave_part = wave_part.merge(
        wave_stats.rename(columns={"valid_3hour_count": "wave_valid_3hour_count"})[
            [
                "wave_node_id",
                "hs_mean_m",
                "hs_median_m",
                "hs_p90_m",
                "hs_p95_m",
                "hs_p99_m",
                "hs_max_m",
                "tp_median_s",
                "tp_p95_s",
                "wave_valid_3hour_count",
            ]
        ],
        on="wave_node_id",
        how="left",
    )

    result = pd.concat(
        [
            base.reset_index(drop=True),
            current_part.reset_index(drop=True),
            long_term_part.reset_index(drop=True),
            wave_part.reset_index(drop=True),
        ],
        axis=1,
    )
    return result[list(CHAINAGE_METOCEAN_COLUMNS)]


# --- Output writing -----------------------------------------------------------


def write_parquet(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def write_metocean_evidence_metadata(*, metadata: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_path


# --- Report printing (Section 37) --------------------------------------------


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def print_metocean_evidence_report(
    *,
    primary_current_stats: pd.DataFrame,
    primary_current_route_summary: dict[str, Any],
    below_bed_diagnostics: pd.DataFrame,
    primary_distance_diagnostics: dict[str, float | None],
    long_term_stats: pd.DataFrame,
    long_term_distance_diagnostics: dict[str, float | None],
    short_window_ratios: pd.DataFrame,
    wave_stats: pd.DataFrame,
    wave_distance_diagnostics: dict[str, float | None],
    primary_current_actual_start: datetime | None,
    primary_current_actual_end: datetime | None,
    long_term_actual_start: datetime | None,
    long_term_actual_end: datetime | None,
    wave_actual_start: datetime | None,
    wave_actual_end: datetime | None,
    old_vs_new_comparison: dict[str, Any] | None = None,
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 Metocean Forcing Evidence Base (MAR-009/MAR-009A) ===", ""]

    # --- Primary current integrity (Section 19) -----------------------------
    lines.append("## Primary current integrity")
    lines.append(f"  Route-used support node count: {len(primary_current_stats)}")
    lines.append(
        f"  Actual interval: {primary_current_actual_start} .. {primary_current_actual_end}"
    )
    summary = primary_current_route_summary
    lines.append(
        "  Model bathymetry (m): min="
        f"{_fmt(summary['model_bathymetry_m_min'])} "
        f"median={_fmt(summary['model_bathymetry_m_median'])} "
        f"max={_fmt(summary['model_bathymetry_m_max'])}"
    )
    lines.append(
        "  Selected current depth (m): min="
        f"{_fmt(summary['current_sample_depth_m_min'])} "
        f"median={_fmt(summary['current_sample_depth_m_median'])} "
        f"max={_fmt(summary['current_sample_depth_m_max'])}"
    )
    lines.append(
        "  Height above model bed (m): min="
        f"{_fmt(summary['height_above_model_bed_m_min'])} "
        f"median={_fmt(summary['height_above_model_bed_m_median'])} "
        f"p95={_fmt(summary['height_above_model_bed_m_p95'])} "
        f"max={_fmt(summary['height_above_model_bed_m_max'])}"
    )
    if not below_bed_diagnostics.empty:
        lines.append(
            "  Below-bed finite candidates excluded: "
            f"{int(below_bed_diagnostics['below_model_bed_finite_candidate_count'].sum())} "
            f"(timestamps affected: "
            f"{int(below_bed_diagnostics['timestamps_with_below_bed_finite_candidates'].sum())}, "
            "max below-bed candidate depth: "
            f"{_fmt(below_bed_diagnostics['max_below_bed_candidate_depth_m'].max())} m)"
        )
    if not primary_current_stats.empty:
        lines.append(
            "  Current speed (m/s) across nodes: mean="
            f"{primary_current_stats['current_speed_mean_m_s'].mean():.3f} "
            f"p95={primary_current_stats['current_speed_p95_m_s'].max():.3f} "
            f"p99={primary_current_stats['current_speed_p99_m_s'].max():.3f} "
            f"max={primary_current_stats['current_speed_max_m_s'].max():.3f}"
        )
        lines.append(
            "  Completeness %: min="
            f"{_fmt(primary_current_stats['completeness_pct'].min(), '.1f')} "
            f"median={_fmt(primary_current_stats['completeness_pct'].median(), '.1f')}"
        )
    lines.append(
        "  Station-to-node distance (m): min="
        f"{_fmt(primary_distance_diagnostics['min_m'])} "
        f"median={_fmt(primary_distance_diagnostics['median_m'])} "
        f"p95={_fmt(primary_distance_diagnostics['p95_m'])} "
        f"max={_fmt(primary_distance_diagnostics['max_m'])}"
    )
    lines.append(
        "  NOT NATIVE BOTTOM-CELL CURRENT -- deepest physically eligible standard level only."
    )
    if summary.get("all_within_water_column"):
        lines.append("  ALL CANONICAL CURRENT SAMPLES ARE WITHIN THE COPERNICUS MODEL WATER COLUMN")
    elif summary.get("all_within_water_column") is False:
        lines.append(
            "  FAILURE: one or more canonical current samples are below the model water column"
        )
    lines.append("")

    # --- Long-term surface current (Section 19) -----------------------------
    lines.append("## Long-term surface current")
    lines.append(f"  Route-used support nodes: {len(long_term_stats)}")
    lines.append(f"  Actual interval: {long_term_actual_start} .. {long_term_actual_end}")
    if not long_term_stats.empty:
        lines.append(
            "  Surface speed (m/s): "
            f"p95={long_term_stats['surface_current_speed_p95_m_s'].max():.3f} "
            f"p99={long_term_stats['surface_current_speed_p99_m_s'].max():.3f} "
            f"max={long_term_stats['surface_current_speed_max_m_s'].max():.3f}"
        )
    lines.append(
        "  Station-to-node distance (m): min="
        f"{_fmt(long_term_distance_diagnostics['min_m'])} "
        f"median={_fmt(long_term_distance_diagnostics['median_m'])} "
        f"p95={_fmt(long_term_distance_diagnostics['p95_m'])} "
        f"max={_fmt(long_term_distance_diagnostics['max_m'])}"
    )
    if not short_window_ratios.empty:
        lines.append(
            "  short_window_surface_context_ratio (p95): "
            f"{short_window_ratios['short_window_surface_context_ratio_p95'].tolist()}"
        )
    lines.append("  SURFACE CURRENT CONTEXT ONLY.")
    lines.append("")

    # --- Waves (Section 19) -------------------------------------------------
    lines.append("## Waves")
    lines.append(f"  Route-used support nodes: {len(wave_stats)}")
    lines.append(f"  Actual interval: {wave_actual_start} .. {wave_actual_end}")
    if not wave_stats.empty:
        lines.append(
            "  Hs (m): mean="
            f"{wave_stats['hs_mean_m'].mean():.3f} p95={wave_stats['hs_p95_m'].max():.3f} "
            f"p99={wave_stats['hs_p99_m'].max():.3f} max={wave_stats['hs_max_m'].max():.3f}"
        )
        lines.append(
            f"  Tp (s): median={wave_stats['tp_median_s'].median():.3f} "
            f"p95={wave_stats['tp_p95_s'].max():.3f}"
        )
        lines.append(
            "  Completeness %: min="
            f"{_fmt(wave_stats['completeness_pct'].min(), '.1f')} "
            f"median={_fmt(wave_stats['completeness_pct'].median(), '.1f')}"
        )
    lines.append(
        "  Station-to-node distance (m): min="
        f"{_fmt(wave_distance_diagnostics['min_m'])} "
        f"median={_fmt(wave_distance_diagnostics['median_m'])} "
        f"p95={_fmt(wave_distance_diagnostics['p95_m'])} "
        f"max={_fmt(wave_distance_diagnostics['max_m'])}"
    )
    lines.append("")

    # --- Old-vs-corrected comparison (Sections 12, 13, 19) -------------------
    if old_vs_new_comparison:
        lines.append("## Old-vs-corrected comparison")
        for label, (old_value, new_value) in old_vs_new_comparison.items():
            changed = "CHANGED" if old_value != new_value else "unchanged"
            lines.append(f"  {label}: old={_fmt(old_value)} -> new={_fmt(new_value)} ({changed})")
        lines.append("")

    lines.append(
        "No bed shear stress, Shields parameter, sediment mobility, or risk scoring computed."
    )

    print("\n".join(lines), file=file)
