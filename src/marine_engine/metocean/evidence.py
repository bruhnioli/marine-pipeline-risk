"""PL854 metocean forcing evidence base assembly (MAR-009).

Scope and interpretation (mandatory reading before touching this module)
--------------------------------------------------------------------------
Three separate, never-blended metocean evidence products, each mapped from
PL854's 941 chainage stations onto a much SMALLER set of real Copernicus
Marine model grid cells ("support nodes") -- never 941 fabricated
independent time series (Section 4 of the ticket):

- Primary current (`current_node_id`): 1.5 km 3D hourly current, deepest
  VALID STANDARD LEVEL only (see `metocean/current.py` -- never the
  model's native bottom cell, never called "bottom current").
- Long-term surface current context (`current_lt_node_id`): 7 km hourly 2D
  current, 1993 onward -- `LONG_TERM_SURFACE_CURRENT_CONTEXT` role only.
  Never used to fill a missing primary-current value, never downscaled,
  never a source of vertical current profiles.
- Wave climate (`wave_node_id`): 3-hourly WAVEWATCH III reanalysis, 1980
  onward, surface parameters only.

Support nodes are the actual wet model grid cells nearest each chainage
station -- nearest-neighbour assignment (never bilinear interpolation of
data or masks) preserves real model-cell provenance rather than fabricating
pipeline-resolution forcing (Section 5).

Model bathymetry (`deptho` on each product's own static dataset) and the
canonical MAR-006 `depth_lat_m` (positive-down relative to LAT) are DIFFERENT
vertical datums and are never subtracted or compared as an "error" here
(Section 9) -- both are carried side by side, and downstream metadata
states `canonical_model_bathymetry_vertical_datums_not_harmonised = true`.

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
    "source_dataset",
    "temporal_role",
)


def normalize_primary_current(
    current_ds: xr.Dataset,
    *,
    nodes: list[SupportNode],
    model_bathymetry_by_node_id: dict[str, float],
    source_dataset: str,
    evidence_role: str,
) -> pd.DataFrame:
    """Per (support node, hour): the deepest-valid-standard-level current only.

    Never duplicated across 941 chainage stations -- one row per real
    support node per timestamp (Section 19).
    """

    if not nodes:
        return pd.DataFrame(columns=list(PRIMARY_CURRENT_COLUMNS))

    depths_m = current_ds["depth"].to_numpy()
    times = _to_utc_timestamps(current_ds["time"].to_numpy())

    records: list[dict[str, Any]] = []
    for node in nodes:
        uo_node = current_ds["uo"].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()
        vo_node = current_ds["vo"].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()
        model_bathymetry_m = model_bathymetry_by_node_id.get(node.node_id)

        for t_index, time_value in enumerate(times):
            selection = current_module.select_deepest_valid_standard_level(
                depths_m, uo_node[t_index], vo_node[t_index]
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
                    "source_dataset": source_dataset,
                    "temporal_role": evidence_role,
                }
            )

    return pd.DataFrame(records, columns=list(PRIMARY_CURRENT_COLUMNS))


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
) -> pd.DataFrame:
    """Per (long-term support node, hour): 2D surface current only -- context, never bottom."""

    if not nodes:
        return pd.DataFrame(columns=list(LONG_TERM_SURFACE_CURRENT_COLUMNS))

    times = _to_utc_timestamps(current_ds["time"].to_numpy())
    records: list[dict[str, Any]] = []
    for node in nodes:
        uo_node = current_ds["uo"].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()
        vo_node = current_ds["vo"].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()
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

    return pd.DataFrame(records, columns=list(LONG_TERM_SURFACE_CURRENT_COLUMNS))


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


def _optional_variable(ds: xr.Dataset, name: str, node: SupportNode, length: int) -> np.ndarray:
    if name not in ds.variables:
        return np.full(length, np.nan)
    return ds[name].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()


def normalize_wave(
    wave_ds: xr.Dataset, *, nodes: list[SupportNode], source_dataset: str
) -> pd.DataFrame:
    """Per (wave support node, 3-hour step): Hs/Tp/Tm02/Tm10/direction, QA'd, never a bed force."""

    if not nodes:
        return pd.DataFrame(columns=list(WAVE_COLUMNS))

    times = _to_utc_timestamps(wave_ds["time"].to_numpy())
    records: list[dict[str, Any]] = []
    for node in nodes:
        hs = _optional_variable(wave_ds, "VHM0", node, len(times))
        tp = _optional_variable(wave_ds, "VTPK", node, len(times))
        tm02 = _optional_variable(wave_ds, "VTM02", node, len(times))
        tm10 = _optional_variable(wave_ds, "VTM10", node, len(times))
        vmdr = _optional_variable(wave_ds, "VMDR", node, len(times))
        stokes_u = _optional_variable(wave_ds, "VSDX", node, len(times))
        stokes_v = _optional_variable(wave_ds, "VSDY", node, len(times))

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

    return pd.DataFrame(records, columns=list(WAVE_COLUMNS))


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


def print_metocean_evidence_report(
    *,
    primary_current_stats: pd.DataFrame,
    long_term_stats: pd.DataFrame,
    wave_stats: pd.DataFrame,
    short_window_ratios: pd.DataFrame,
    primary_current_actual_start: datetime | None,
    primary_current_actual_end: datetime | None,
    long_term_actual_start: datetime | None,
    long_term_actual_end: datetime | None,
    wave_actual_start: datetime | None,
    wave_actual_end: datetime | None,
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 Metocean Forcing Evidence Base (MAR-009) ===", ""]

    lines.append("## Primary current")
    lines.append(
        f"  Actual interval: {primary_current_actual_start} .. {primary_current_actual_end}"
    )
    lines.append(f"  Support nodes: {len(primary_current_stats)}")
    if not primary_current_stats.empty:
        lines.append(
            "  Speed (m/s) across nodes: mean="
            f"{primary_current_stats['current_speed_mean_m_s'].mean():.3f} "
            f"p95={primary_current_stats['current_speed_p95_m_s'].max():.3f} "
            f"p99={primary_current_stats['current_speed_p99_m_s'].max():.3f} "
            f"max={primary_current_stats['current_speed_max_m_s'].max():.3f}"
        )
        lines.append(
            "  Representative sample depth (m): "
            f"min={primary_current_stats['representative_sample_depth_m'].min()} "
            f"max={primary_current_stats['representative_sample_depth_m'].max()}"
        )
    lines.append("  NOT NATIVE BOTTOM-CELL CURRENT -- deepest valid standard level only.")
    lines.append("")

    lines.append("## Long-term surface current")
    lines.append(f"  Actual interval: {long_term_actual_start} .. {long_term_actual_end}")
    lines.append(f"  Support nodes: {len(long_term_stats)}")
    if not long_term_stats.empty:
        lines.append(
            "  Surface speed (m/s): "
            f"p95={long_term_stats['surface_current_speed_p95_m_s'].max():.3f} "
            f"p99={long_term_stats['surface_current_speed_p99_m_s'].max():.3f} "
            f"max={long_term_stats['surface_current_speed_max_m_s'].max():.3f}"
        )
    if not short_window_ratios.empty:
        lines.append(
            "  short_window_surface_context_ratio (p95): "
            f"{short_window_ratios['short_window_surface_context_ratio_p95'].tolist()}"
        )
    lines.append("  SURFACE CURRENT CONTEXT ONLY.")
    lines.append("")

    lines.append("## Waves")
    lines.append(f"  Actual interval: {wave_actual_start} .. {wave_actual_end}")
    lines.append(f"  Support nodes: {len(wave_stats)}")
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
    lines.append("")

    lines.append(
        "No bed shear stress, Shields parameter, sediment mobility, or risk scoring computed."
    )

    print("\n".join(lines), file=file)
