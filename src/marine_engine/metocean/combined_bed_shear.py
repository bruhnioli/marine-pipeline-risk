"""Soulsby algebraic wave-current bed shear stress sensitivity (MAR-012).

Scope -- read before touching this module
--------------------------------------------
Combines the accepted MAR-010 current-only 1 m log-profile normalization
and the accepted MAR-011A spectral wave orbital velocity into a
contemporaneous wave-current bed-shear-stress sensitivity product. The
canonical role name for every output of this module is
`SOULSBY_ALGEBRAIC_WAVE_CURRENT_BED_SHEAR_SENSITIVITY`. This ticket
calculates current-only bed shear stress, wave-only bed shear stress, the
Soulsby mean combined stress, and the Soulsby maximum combined stress
during the representative wave cycle -- and STOPS there. No Shields
parameter, critical shear stress, sediment mobility, erosion/deposition,
scour, free-span, pipeline loading, fatigue, or risk scoring is computed
anywhere in this module.

Scientific interpretation -- do not substitute another model
-------------------------------------------------------------------
This is the Soulsby ALGEBRAIC wave-current interaction approximation
(`interaction_model = SOULSBY_ALGEBRAIC_WAVE_CURRENT_BED_SHEAR`). It is NOT
a full Grant-Madsen (1979) iterative wave-current bottom-boundary-layer
solution (`full_grant_madsen_bbl_applied = false`), does NOT iterate an
apparent wave-enhanced current roughness, and does NOT model wave-current
effects on wave dispersion. The formulation is fixed by the ticket; this
module never performs independent literature research or model
substitution.

Roughness is reused, never re-chosen (Section 4)
-----------------------------------------------------
`ROUGHNESS_SCENARIOS_M` is imported directly from
`current_normalization` -- the exact same five fixed sensitivity scenarios,
reused so current and wave stress in a given scenario row always share the
SAME z0. `ks_m = 30 * z0_m` is a diagnostic only, never reinterpreted as an
observed grain size, never used to re-derive z0 from D50.

Fixed reference fluid constants (Section 5)
------------------------------------------------
`RHO_WATER_KG_M3`/`KINEMATIC_VISCOSITY_M2_S`/`VON_KARMAN_KAPPA` are fixed
project reference seawater properties for this research calculation --
never claimed as PL854 in-situ measurements, never a temperature/salinity
dependent viscosity model.

Current-only bed shear via log-profile inversion (Section 6)
--------------------------------------------------------------------
`compute_current_friction_velocity_m_s` inverts MAR-010's own log-profile
formulation at the SAME z0/target height already used to produce
`current_only_1m_speed_m_s` -- this module never recomputes the current
normalization differently and never uses the raw standard-depth current
directly. For `U_1m = 0`, the plain formula already yields `u_star_c = 0`
(zero divided by a finite, nonzero logarithm) with no special-casing
required; a missing/NaN `U_1m` propagates to NaN, never a fabricated zero.

Wave friction: two competing branches, never averaged (Sections 8-13)
-------------------------------------------------------------------------------
The wave semi-orbital excursion, Reynolds number, and BOTH the
smooth/laminar and rough friction-factor branches are computed; the
canonical `wave_friction_factor` is the MAX of the two branches (never
their average), with `wave_friction_controlling_branch` naming which one
won. For a genuinely calm sea state (`Uw == 0`, not merely small), the
excursion, Reynolds number, both friction branches, and the wave stress
are all explicitly zero/null -- this module never forces a friction factor
for water that is not moving. This explicit calm-sea branch exists because
`0 * NaN` is `NaN` in IEEE 754, not `0`; relying on bare arithmetic
propagation here would silently corrupt an otherwise-valid zero.

Wave-current axis angle: 180-degree symmetry (Section 14)
--------------------------------------------------------------
Oscillatory wave motion reverses every half-cycle, so the wave axis has
180-degree symmetry with the current's TO-direction. `fold_wave_current_axis_angle_deg`
computes the minimal 0..180 angular difference, then folds it around 90,
producing an acute angle in `[0, 90]`. For zero current the direction (and
therefore phi) is explicitly null (never the spurious `atan2(0, 0) == 0`
current direction) -- the combined-stress special-casing below exists
precisely to keep the general formula correct despite that null.

Soulsby mean/max combined stress: explicit special-casing (Sections 15-16)
-------------------------------------------------------------------------------
The general algebraic formulas are mathematically self-consistent with the
required special cases (`tau_w=0 -> tau_max=tau_c`; `tau_c=0 ->
tau_max=tau_w`; both zero -> `tau_max=0`) PROVIDED phi is finite. Since
phi is deliberately null when current is zero, `compute_soulsby_max_combined_stress_pa`
explicitly branches on `tau_c == 0` / `tau_w == 0` rather than trusting the
raw trigonometric formula to self-resolve through a null phi.

No Shields/mobility yet (Section 17)
-----------------------------------------
This module never computes theta, theta_cr, tau_cr, D*, a mobility ratio,
threshold exceedance, or erosion likelihood. MAR-012 ends at bed shear
stress.

Temporal alignment: exact-timestamp inner join only (Section 18)
------------------------------------------------------------------------
Primary current is hourly (~2024-07 to 2026-09); wave is 3-hourly (through
2026-04). The combined series uses ONLY real contemporaneous overlap via
an exact-UTC-timestamp inner join performed AFTER spatial node pairing --
never combining independent current/wave percentile statistics, never
pairing arbitrary timestamps, never interpolating. Because the hourly
current series already contains every 3-hour wave timestamp, a plain
inner merge on `time_utc` is sufficient and exact.

Spatial node pairing: verify, never assume (Section 19)
----------------------------------------------------------
`current_node_id` and `wave_node_id` are never assumed to refer to the
same physical grid cell merely by construction -- `build_hydro_pairs`
verifies this by projecting each product's own stored longitude/latitude
into the working CRS and matching nearest cells, with a strict tolerance
derived from a fraction of the current grid's own median nearest-neighbour
spacing (mirroring `evidence.GRID_RECONCILIATION_TOLERANCE_FRACTION`).
A current node with no wave node within that tolerance raises
`UnreconciledHydroNodeError` -- this module never silently nearest-neighbours
two unrelated cells.
"""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

from marine_engine.metocean import current as current_module
from marine_engine.metocean.current_normalization import (
    ROUGHNESS_SCENARIOS_M,
    TARGET_HEIGHT_ABOVE_MODEL_BED_M,
)

# --- Fixed scientific constants (Sections 4, 5, 10-16 -- do not change) -------------

RHO_WATER_KG_M3 = 1027.0
KINEMATIC_VISCOSITY_M2_S = 1.36e-6
VON_KARMAN_KAPPA = 0.40

SCIENTIFIC_ROLE = "SOULSBY_ALGEBRAIC_WAVE_CURRENT_BED_SHEAR_SENSITIVITY"
INTERACTION_MODEL = "SOULSBY_ALGEBRAIC_WAVE_CURRENT_BED_SHEAR"

WAVE_REYNOLDS_TRANSITION = 5e5
LAMINAR_BRANCH = "LAMINAR_BRANCH"
SMOOTH_TURBULENT_BRANCH = "SMOOTH_TURBULENT_BRANCH"
SMOOTH_OR_LAMINAR = "SMOOTH_OR_LAMINAR"
ROUGH = "ROUGH"

# Project data-QA heuristic (Section 19), mirroring
# `evidence.GRID_RECONCILIATION_TOLERANCE_FRACTION`: how close (as a
# fraction of the current grid's own median nearest-neighbour spacing) a
# current node's coordinate must land to a wave node's coordinate to be
# treated as "the same real cell" rather than an unreconcilable mismatch.
HYDRO_PAIR_TOLERANCE_FRACTION = 0.1
# Fallback absolute tolerance (m) only used when fewer than two current
# nodes exist, so a median spacing cannot be estimated.
HYDRO_PAIR_FALLBACK_TOLERANCE_M = 10.0


class UnreconciledHydroNodeError(Exception):
    """A current support node has no wave support node within the strict coordinate tolerance.

    Raised rather than silently nearest-neighbouring two unrelated grid
    cells (Section 19) -- a genuine defect in the assumption that current
    and wave products share the same support grid must stop the run, not
    be masked.
    """


class CombinedBedShearCompletenessError(Exception):
    """More matched combined timestamps exist than the expected regular-cadence count allows.

    Mirrors the project's other `*CompletenessError` classes: a properly
    exact-timestamp-joined series can never legitimately exceed 100%
    completeness against its own expected 3-hourly cadence.
    """


def _completeness_pct(valid_count: int, expected_count: int) -> float | None:
    if not expected_count:
        return None
    if valid_count > expected_count:
        raise CombinedBedShearCompletenessError(
            f"{valid_count} matched combined timestamps exceeds the expected regular-cadence "
            f"count of {expected_count} -- completeness must never exceed 100%"
        )
    return 100.0 * valid_count / expected_count


# --- Roughness diagnostics (Section 4) ------------------------------------------------


def compute_ks_m(z0_m: float | np.ndarray) -> np.ndarray:
    """`ks_m = 30 * z0_m` -- a diagnostic only, never reinterpreted as observed grain size."""

    return 30.0 * np.asarray(z0_m, dtype=float)


# --- Spatial node pairing (Section 19) ------------------------------------------------

HYDRO_PAIRS_COLUMNS = (
    "hydro_pair_id",
    "current_node_id",
    "wave_node_id",
    "current_longitude",
    "current_latitude",
    "wave_longitude",
    "wave_latitude",
    "coordinate_separation_m",
)


def _median_nearest_neighbour_spacing_m(coords: np.ndarray) -> float | None:
    """Median nearest-neighbour distance among a set of projected points, or None if <2 points."""

    if len(coords) < 2:
        return None
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    return float(np.median(distances[:, 1]))


def build_hydro_pairs(
    current_nodes_df: pd.DataFrame,
    wave_nodes_df: pd.DataFrame,
    *,
    working_crs: str,
    tolerance_fraction: float = HYDRO_PAIR_TOLERANCE_FRACTION,
) -> pd.DataFrame:
    """Pair each current support node to its nearest wave support node BY COORDINATE.

    Never assumes `current_node_id`/`wave_node_id` share a naming
    convention (Section 19) -- both products' own stored longitude/latitude
    are projected into `working_crs` and matched by nearest distance. The
    tolerance is a fraction of the current grid's own median
    nearest-neighbour spacing (real identical AMM15 cells match to
    numerical precision, far inside this tolerance); a current node with no
    wave node within tolerance raises `UnreconciledHydroNodeError` rather
    than silently pairing it with the nearest-but-too-distant cell.
    """

    if current_nodes_df.empty or wave_nodes_df.empty:
        return pd.DataFrame(columns=list(HYDRO_PAIRS_COLUMNS))

    current_nodes_df = current_nodes_df.reset_index(drop=True)
    wave_nodes_df = wave_nodes_df.reset_index(drop=True)

    current_points = gpd.GeoSeries(
        [
            Point(lon, lat)
            for lon, lat in zip(
                current_nodes_df["longitude"], current_nodes_df["latitude"], strict=True
            )
        ],
        crs="EPSG:4326",
    ).to_crs(working_crs)
    wave_points = gpd.GeoSeries(
        [
            Point(lon, lat)
            for lon, lat in zip(wave_nodes_df["longitude"], wave_nodes_df["latitude"], strict=True)
        ],
        crs="EPSG:4326",
    ).to_crs(working_crs)

    current_coords = np.column_stack([current_points.x.to_numpy(), current_points.y.to_numpy()])
    wave_coords = np.column_stack([wave_points.x.to_numpy(), wave_points.y.to_numpy()])

    median_spacing_m = _median_nearest_neighbour_spacing_m(current_coords)
    tolerance_m = (
        tolerance_fraction * median_spacing_m
        if median_spacing_m is not None
        else HYDRO_PAIR_FALLBACK_TOLERANCE_M
    )

    tree = cKDTree(wave_coords)
    distances, indices = tree.query(current_coords)

    records = []
    for row_index in range(len(current_nodes_df)):
        distance_m = float(distances[row_index])
        current_row = current_nodes_df.iloc[row_index]
        if distance_m > tolerance_m:
            raise UnreconciledHydroNodeError(
                f"current node '{current_row['node_id']}' has no wave support node within "
                f"the strict coordinate tolerance ({tolerance_m:.3f} m; nearest is "
                f"{distance_m:.3f} m) -- refusing to pair unrelated grid cells"
            )
        wave_row = wave_nodes_df.iloc[int(indices[row_index])]
        records.append(
            {
                "hydro_pair_id": f"{current_row['node_id']}__{wave_row['node_id']}",
                "current_node_id": current_row["node_id"],
                "wave_node_id": wave_row["node_id"],
                "current_longitude": float(current_row["longitude"]),
                "current_latitude": float(current_row["latitude"]),
                "wave_longitude": float(wave_row["longitude"]),
                "wave_latitude": float(wave_row["latitude"]),
                "coordinate_separation_m": distance_m,
            }
        )
    return pd.DataFrame(records, columns=list(HYDRO_PAIRS_COLUMNS))


# --- Current-only bed shear stress (Section 6) ----------------------------------------


def compute_current_friction_velocity_m_s(
    u_1m_speed_m_s: np.ndarray,
    z0_m: float | np.ndarray,
    *,
    target_height_m: float = TARGET_HEIGHT_ABOVE_MODEL_BED_M,
    kappa: float = VON_KARMAN_KAPPA,
) -> np.ndarray:
    """Invert MAR-010's log-profile at `z0_m`/`target_height_m`: `kappa*U_1m / ln((z_t+z0)/z0)`.

    For `U_1m = 0` this already yields 0 with no special-casing (division
    by a finite, nonzero logarithm); a missing `U_1m` propagates to NaN.
    """

    u = np.asarray(u_1m_speed_m_s, dtype=float)
    z0 = np.asarray(z0_m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return kappa * u / np.log((target_height_m + z0) / z0)


def compute_current_bed_shear_stress_pa(
    u_star_c: np.ndarray, rho: float = RHO_WATER_KG_M3
) -> np.ndarray:
    """`tau_c_pa = rho * u_star_c^2`."""

    return rho * np.asarray(u_star_c, dtype=float) ** 2


# --- Wave semi-orbital excursion + Reynolds number (Sections 8-9) --------------------


def compute_wave_semi_orbital_excursion_m(uw: np.ndarray, t_rep: np.ndarray) -> np.ndarray:
    """`A_wave_m = Uw * T_rep / (2*pi)`. For Uw == 0 (calm): A_wave_m = 0, explicitly.

    A missing/NaN `Uw` (genuinely invalid input) is never confused with a
    calm sea state: only an exact `Uw == 0` forces the explicit zero
    branch, so `0 * NaN` (which is `NaN`, not `0`) never corrupts a
    genuine calm-sea result.
    """

    uw = np.asarray(uw, dtype=float)
    t_rep = np.asarray(t_rep, dtype=float)
    is_calm = uw == 0
    with np.errstate(invalid="ignore"):
        general = uw * t_rep / (2.0 * np.pi)
    return np.where(is_calm, 0.0, general)


def compute_wave_reynolds_number(
    uw: np.ndarray, a_wave: np.ndarray, nu: float = KINEMATIC_VISCOSITY_M2_S
) -> np.ndarray:
    """`Rw = Uw * A_wave / nu`, defined only for Uw > 0 AND A_wave > 0 (Section 9)."""

    uw = np.asarray(uw, dtype=float)
    a_wave = np.asarray(a_wave, dtype=float)
    eligible = (uw > 0) & (a_wave > 0)
    rw = np.full(uw.shape, np.nan)
    rw[eligible] = uw[eligible] * a_wave[eligible] / nu
    return rw


# --- Wave friction factor: smooth/laminar vs rough branch (Sections 10-12) ----------


def classify_wave_reynolds_regime(rw: np.ndarray) -> np.ndarray:
    """`LAMINAR_BRANCH` (Rw<=5e5) / `SMOOTH_TURBULENT_BRANCH` (Rw>5e5); null where Rw undefined."""

    rw = np.asarray(rw, dtype=float)
    eligible = np.isfinite(rw)
    regime = np.full(rw.shape, None, dtype=object)
    regime[eligible & (rw <= WAVE_REYNOLDS_TRANSITION)] = LAMINAR_BRANCH
    regime[eligible & (rw > WAVE_REYNOLDS_TRANSITION)] = SMOOTH_TURBULENT_BRANCH
    return regime


def compute_wave_friction_smooth_branch(rw: np.ndarray) -> np.ndarray:
    """`f_ws = B*Rw^(-N)`: B=2.0,N=0.5 for Rw<=5e5; B=0.0521,N=0.187 for Rw>5e5."""

    rw = np.asarray(rw, dtype=float)
    eligible = np.isfinite(rw)
    laminar = eligible & (rw <= WAVE_REYNOLDS_TRANSITION)
    turbulent = eligible & (rw > WAVE_REYNOLDS_TRANSITION)
    f_ws = np.full(rw.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_ws[laminar] = 2.0 * np.power(rw[laminar], -0.5)
        f_ws[turbulent] = 0.0521 * np.power(rw[turbulent], -0.187)
    return f_ws


def compute_wave_friction_rough_branch(
    a_wave: np.ndarray,
    z0_m: float | np.ndarray,
) -> np.ndarray:
    """`f_wr = 1.39*(A_wave/z0)^(-0.52)`, defined only for A_wave > 0. Never derives z0 from D50."""

    a_wave = np.asarray(a_wave, dtype=float)
    eligible = a_wave > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f_wr = 1.39 * np.power(np.where(eligible, a_wave, np.nan) / z0_m, -0.52)
    return f_wr


def compute_wave_friction_factor(
    f_ws: np.ndarray, f_wr: np.ndarray, uw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`f_w = max(f_ws, f_wr)` for Uw > 0; both the factor and which branch controlled it.

    Never averages the two branches. For a calm sea state (Uw == 0), both
    the friction factor and the controlling-branch label are explicitly
    null -- a friction factor is never forced for water that is not moving.
    A tie is reported as `SMOOTH_OR_LAMINAR` (an arbitrary but deterministic
    choice, since the numeric factor is identical either way).
    """

    f_ws = np.asarray(f_ws, dtype=float)
    f_wr = np.asarray(f_wr, dtype=float)
    uw = np.asarray(uw, dtype=float)
    eligible = uw > 0

    with np.errstate(invalid="ignore"):
        f_w = np.where(eligible, np.maximum(f_ws, f_wr), np.nan)
        rough_wins = eligible & (f_wr > f_ws)

    controlling_branch = np.full(uw.shape, None, dtype=object)
    controlling_branch[eligible] = SMOOTH_OR_LAMINAR
    controlling_branch[rough_wins] = ROUGH
    return f_w, controlling_branch


def compute_wave_bed_shear_stress_pa(
    uw: np.ndarray, f_w: np.ndarray, rho: float = RHO_WATER_KG_M3
) -> np.ndarray:
    """`tau_w_pa = 0.5 * rho * f_w * Uw^2`. For Uw == 0 (calm): tau_w_pa = 0, explicitly."""

    uw = np.asarray(uw, dtype=float)
    f_w = np.asarray(f_w, dtype=float)
    is_calm = uw == 0
    with np.errstate(invalid="ignore"):
        general = 0.5 * rho * f_w * uw**2
    return np.where(is_calm, 0.0, general)


# --- Wave/current direction angle (Section 14) ----------------------------------------


def compute_current_direction_to_deg_or_null(
    current_only_1m_uo_m_s: np.ndarray,
    current_only_1m_vo_m_s: np.ndarray,
    current_only_1m_speed_m_s: np.ndarray,
) -> np.ndarray:
    """Current TO-direction from the MAR-010 normalized vector; null for zero/missing current.

    `atan2(0, 0)` conventionally returns 0, which would misrepresent a
    stationary current as having a true direction -- explicitly nulled
    instead (Section 14: "For zero current: direction/phi may be null").
    """

    uo = np.asarray(current_only_1m_uo_m_s, dtype=float)
    vo = np.asarray(current_only_1m_vo_m_s, dtype=float)
    speed = np.asarray(current_only_1m_speed_m_s, dtype=float)
    direction = current_module.compute_current_direction_to_deg(uo, vo)
    return np.where(np.isfinite(speed) & (speed > 0), direction, np.nan)


def fold_wave_current_axis_angle_deg(
    current_direction_to_deg: np.ndarray, wave_direction_to_deg: np.ndarray
) -> np.ndarray:
    """Acute wave-current axis angle phi in [0, 90] (Section 14).

    Computes the minimal 0..180 angular difference between the two axes,
    then folds it around 90 -- exploiting oscillatory wave motion's
    180-degree symmetry. A null input (e.g. zero/missing current
    direction) propagates to a null phi.
    """

    current_dir = np.asarray(current_direction_to_deg, dtype=float)
    wave_dir = np.asarray(wave_direction_to_deg, dtype=float)
    with np.errstate(invalid="ignore"):
        raw_diff = np.mod(np.abs(current_dir - wave_dir), 360.0)
        minimal_diff = np.minimum(raw_diff, 360.0 - raw_diff)
        phi = np.where(minimal_diff <= 90.0, minimal_diff, 180.0 - minimal_diff)
    return phi


# --- Soulsby mean/max combined stress (Sections 15-16) ---------------------------------


def compute_soulsby_mean_combined_stress_pa(
    tau_c_pa: np.ndarray, tau_w_pa: np.ndarray
) -> np.ndarray:
    """`tau_m_pa = tau_c * [1 + 1.2*(tau_w/(tau_c+tau_w))^3.2]`; both zero -> tau_m = 0."""

    tau_c = np.asarray(tau_c_pa, dtype=float)
    tau_w = np.asarray(tau_w_pa, dtype=float)
    total = tau_c + tau_w
    with np.errstate(invalid="ignore", divide="ignore"):
        general = tau_c * (1.0 + 1.2 * np.power(np.where(total > 0, tau_w / total, np.nan), 3.2))
    return np.where(total > 0, general, 0.0)


def compute_soulsby_max_combined_stress_pa(
    tau_c_pa: np.ndarray,
    tau_w_pa: np.ndarray,
    tau_m_pa: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    """`tau_max_pa = sqrt((tau_m+tau_w*cos(phi))^2 + (tau_w*sin(phi))^2)`, phi in radians.

    Explicitly branches on `tau_w == 0` (-> tau_c) and `tau_c == 0` (->
    tau_w) rather than trusting the raw trigonometric formula alone: phi
    is deliberately null whenever current is zero (Section 14), and
    `0 * NaN` is `NaN` in IEEE 754 -- without this explicit override the
    general formula would incorrectly propagate NaN through the very
    special case it is required to satisfy exactly.
    """

    tau_c = np.asarray(tau_c_pa, dtype=float)
    tau_w = np.asarray(tau_w_pa, dtype=float)
    tau_m = np.asarray(tau_m_pa, dtype=float)
    phi_rad = np.radians(np.asarray(phi_deg, dtype=float))

    with np.errstate(invalid="ignore"):
        general = np.sqrt((tau_m + tau_w * np.cos(phi_rad)) ** 2 + (tau_w * np.sin(phi_rad)) ** 2)

    result = np.where(tau_w == 0, tau_c, general)
    result = np.where(tau_c == 0, tau_w, result)
    return result


# --- Canonical combined 3-hourly output (Section 20) -----------------------------------

COMBINED_BED_SHEAR_3HOURLY_COLUMNS = (
    # Identity
    "hydro_pair_id",
    "current_node_id",
    "wave_node_id",
    "time_utc",
    # Scenario
    "roughness_scenario",
    "z0_m",
    "ks_m",
    # Fluid
    "rho_water_kg_m3",
    "kinematic_viscosity_m2_s",
    "von_karman_kappa",
    # Current
    "current_only_1m_uo_m_s",
    "current_only_1m_vo_m_s",
    "current_only_1m_speed_m_s",
    "current_direction_to_deg",
    "current_friction_velocity_m_s",
    "tau_current_pa",
    # Wave
    "wave_orbital_velocity_rms_m_s",
    "wave_orbital_velocity_equivalent_amplitude_m_s",
    "representative_wave_period_s",
    "observed_tp_s",
    "wave_direction_to_deg",
    "wave_semi_orbital_excursion_m",
    "wave_reynolds_number",
    "wave_reynolds_regime",
    "wave_friction_smooth_branch",
    "wave_friction_rough_branch",
    "wave_friction_controlling_branch",
    "wave_friction_factor",
    "tau_wave_pa",
    # Interaction
    "wave_current_axis_angle_deg",
    "tau_mean_combined_pa",
    "tau_max_combined_pa",
    # Provenance
    "scientific_role",
)

_WAVE_SIDE_COLUMNS_RENAME = {
    "wave_orbital_velocity_rms_near_bed_m_s": "wave_orbital_velocity_rms_m_s",
    "tp_s": "observed_tp_s",
    "equivalent_peak_period_from_tz_s": "representative_wave_period_s",
    "wave_mean_direction_to_deg": "wave_direction_to_deg",
}


def build_combined_bed_shear_3hourly(
    current_hourly_df: pd.DataFrame,
    wave_3hourly_df: pd.DataFrame,
    hydro_pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """LONG format: one row per `hydro_pair_id x time_utc x roughness_scenario` (Section 20).

    Joins the current side (already 5 scenario rows per node/hour, from
    MAR-010) to its paired wave node's 3-hourly series via an EXACT
    timestamp inner merge -- never combining independent statistics, never
    interpolating, never fanning out across chainage stations. Because the
    wave side carries exactly one row per (wave_node_id, time_utc), the
    merge naturally broadcasts to all 5 scenario rows without duplicating
    the wave side.
    """

    if current_hourly_df.empty or wave_3hourly_df.empty or hydro_pairs_df.empty:
        return pd.DataFrame(columns=list(COMBINED_BED_SHEAR_3HOURLY_COLUMNS))

    current_side = current_hourly_df.merge(
        hydro_pairs_df[["current_node_id", "wave_node_id", "hydro_pair_id"]],
        on="current_node_id",
        how="inner",
    )
    wave_side = wave_3hourly_df.rename(columns=_WAVE_SIDE_COLUMNS_RENAME)[
        [
            "wave_node_id",
            "time_utc",
            "wave_orbital_velocity_rms_m_s",
            "wave_orbital_velocity_equivalent_amplitude_m_s",
            "representative_wave_period_s",
            "observed_tp_s",
            "wave_direction_to_deg",
        ]
    ]

    merged = current_side.merge(wave_side, on=["wave_node_id", "time_utc"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=list(COMBINED_BED_SHEAR_3HOURLY_COLUMNS))

    z0 = merged["z0_m"].to_numpy(dtype=float)
    u1m = merged["current_only_1m_speed_m_s"].to_numpy(dtype=float)
    uo_1m = merged["current_only_1m_uo_m_s"].to_numpy(dtype=float)
    vo_1m = merged["current_only_1m_vo_m_s"].to_numpy(dtype=float)

    u_star_c = compute_current_friction_velocity_m_s(u1m, z0)
    tau_c = compute_current_bed_shear_stress_pa(u_star_c)
    current_dir = compute_current_direction_to_deg_or_null(uo_1m, vo_1m, u1m)

    uw = merged["wave_orbital_velocity_equivalent_amplitude_m_s"].to_numpy(dtype=float)
    t_rep = merged["representative_wave_period_s"].to_numpy(dtype=float)
    a_wave = compute_wave_semi_orbital_excursion_m(uw, t_rep)

    rw = compute_wave_reynolds_number(uw, a_wave)
    regime = classify_wave_reynolds_regime(rw)
    f_ws = compute_wave_friction_smooth_branch(rw)
    f_wr = compute_wave_friction_rough_branch(a_wave, z0)
    f_w, controlling_branch = compute_wave_friction_factor(f_ws, f_wr, uw)
    tau_w = compute_wave_bed_shear_stress_pa(uw, f_w)

    wave_dir = merged["wave_direction_to_deg"].to_numpy(dtype=float)
    phi = fold_wave_current_axis_angle_deg(current_dir, wave_dir)

    tau_m = compute_soulsby_mean_combined_stress_pa(tau_c, tau_w)
    tau_max = compute_soulsby_max_combined_stress_pa(tau_c, tau_w, tau_m, phi)

    result = pd.DataFrame(
        {
            "hydro_pair_id": merged["hydro_pair_id"].to_numpy(),
            "current_node_id": merged["current_node_id"].to_numpy(),
            "wave_node_id": merged["wave_node_id"].to_numpy(),
            "time_utc": merged["time_utc"].to_numpy(),
            "roughness_scenario": merged["roughness_scenario"].to_numpy(),
            "z0_m": z0,
            "ks_m": compute_ks_m(z0),
            "rho_water_kg_m3": RHO_WATER_KG_M3,
            "kinematic_viscosity_m2_s": KINEMATIC_VISCOSITY_M2_S,
            "von_karman_kappa": VON_KARMAN_KAPPA,
            "current_only_1m_uo_m_s": uo_1m,
            "current_only_1m_vo_m_s": vo_1m,
            "current_only_1m_speed_m_s": u1m,
            "current_direction_to_deg": current_dir,
            "current_friction_velocity_m_s": u_star_c,
            "tau_current_pa": tau_c,
            "wave_orbital_velocity_rms_m_s": merged["wave_orbital_velocity_rms_m_s"].to_numpy(
                dtype=float
            ),
            "wave_orbital_velocity_equivalent_amplitude_m_s": uw,
            "representative_wave_period_s": t_rep,
            "observed_tp_s": merged["observed_tp_s"].to_numpy(dtype=float),
            "wave_direction_to_deg": wave_dir,
            "wave_semi_orbital_excursion_m": a_wave,
            "wave_reynolds_number": rw,
            "wave_reynolds_regime": regime,
            "wave_friction_smooth_branch": f_ws,
            "wave_friction_rough_branch": f_wr,
            "wave_friction_controlling_branch": controlling_branch,
            "wave_friction_factor": f_w,
            "tau_wave_pa": tau_w,
            "wave_current_axis_angle_deg": phi,
            "tau_mean_combined_pa": tau_m,
            "tau_max_combined_pa": tau_max,
            "scientific_role": SCIENTIFIC_ROLE,
        }
    )
    return result[list(COMBINED_BED_SHEAR_3HOURLY_COLUMNS)]


# --- Temporal alignment summary (Section 18) -------------------------------------------


def compute_temporal_alignment_summary(combined_df: pd.DataFrame) -> dict[str, Any]:
    """Route-wide overlap start/end, expected/matched 3-hour timestamps, completeness.

    Computed from the DISTINCT matched `time_utc` values in the combined
    table (never summed across nodes/scenarios, which would inflate the
    count) -- a single, honest route-wide alignment figure (Section 18/32).
    """

    empty = {
        "overlap_start_time_utc": None,
        "overlap_end_time_utc": None,
        "expected_3hour_count": 0,
        "matched_timestamp_count": 0,
        "completeness_pct": None,
    }
    if combined_df.empty:
        return empty

    distinct_times = combined_df["time_utc"].drop_duplicates().sort_values()
    start, end = distinct_times.iloc[0], distinct_times.iloc[-1]
    expected_count = int(round((end - start).total_seconds() / (3 * 3600.0))) + 1
    matched_count = int(len(distinct_times))

    return {
        "overlap_start_time_utc": start,
        "overlap_end_time_utc": end,
        "expected_3hour_count": expected_count,
        "matched_timestamp_count": matched_count,
        "completeness_pct": _completeness_pct(matched_count, expected_count),
    }


# --- Long-term wave-only shear context (Section 21) ------------------------------------

WAVE_ONLY_BED_SHEAR_LONG_TERM_STATS_COLUMNS = (
    "wave_node_id",
    "roughness_scenario",
    "z0_m",
    "long_term_start_time_utc",
    "long_term_end_time_utc",
    "long_term_valid_count",
    "long_term_tau_wave_mean_pa",
    "long_term_tau_wave_p90_pa",
    "long_term_tau_wave_p95_pa",
    "long_term_tau_wave_p99_pa",
    "long_term_tau_wave_max_pa",
    "overlap_valid_count",
    "overlap_tau_wave_mean_pa",
    "overlap_tau_wave_p90_pa",
    "overlap_tau_wave_p95_pa",
    "overlap_tau_wave_p99_pa",
    "overlap_tau_wave_max_pa",
    "overlap_to_long_term_tau_wave_p95_ratio",
)


def compute_wave_only_bed_shear_long_term_stats(
    wave_3hourly_df: pd.DataFrame, *, overlap_keys_df: pd.DataFrame
) -> pd.DataFrame:
    """Per `wave_node_id x roughness_scenario`: full-record and overlap-subset wave-only tau stats.

    Computed from the COMPLETE MAR-011A wave record (never limited to the
    current-current overlap) so long-term context remains available even
    though the combined series itself is overlap-limited (Section 21).
    `overlap_keys_df` (distinct `wave_node_id, time_utc` pairs actually
    matched into the combined table) selects the overlap subset used for
    `overlap_wave_tau_p95 / long_term_wave_tau_p95` -- a REPRESENTATIVENESS
    CONTEXT ratio only, never a confidence score.
    """

    if wave_3hourly_df.empty:
        return pd.DataFrame(columns=list(WAVE_ONLY_BED_SHEAR_LONG_TERM_STATS_COLUMNS))

    uw_full = wave_3hourly_df["wave_orbital_velocity_equivalent_amplitude_m_s"].to_numpy(
        dtype=float
    )
    t_rep_full = wave_3hourly_df["equivalent_peak_period_from_tz_s"].to_numpy(dtype=float)
    a_wave_full = compute_wave_semi_orbital_excursion_m(uw_full, t_rep_full)
    rw_full = compute_wave_reynolds_number(uw_full, a_wave_full)
    f_ws_full = compute_wave_friction_smooth_branch(rw_full)

    if overlap_keys_df.empty:
        is_overlap = np.zeros(len(wave_3hourly_df), dtype=bool)
    else:
        overlap_marker = (
            overlap_keys_df[["wave_node_id", "time_utc"]].drop_duplicates().assign(_is_overlap=True)
        )
        merged_flag = wave_3hourly_df[["wave_node_id", "time_utc"]].merge(
            overlap_marker, on=["wave_node_id", "time_utc"], how="left"
        )
        is_overlap = merged_flag["_is_overlap"].fillna(False).to_numpy()

    records: list[dict[str, Any]] = []
    for scenario_name, z0 in ROUGHNESS_SCENARIOS_M:
        f_wr = compute_wave_friction_rough_branch(a_wave_full, z0)
        f_w, _controlling_branch = compute_wave_friction_factor(f_ws_full, f_wr, uw_full)
        tau_w = compute_wave_bed_shear_stress_pa(uw_full, f_w)

        working = pd.DataFrame(
            {
                "wave_node_id": wave_3hourly_df["wave_node_id"].to_numpy(),
                "time_utc": wave_3hourly_df["time_utc"].to_numpy(),
                "tau_wave_pa": tau_w,
                "is_overlap": is_overlap,
            }
        )
        for node_id, group in working.groupby("wave_node_id"):
            full_valid = group["tau_wave_pa"].dropna()
            overlap_valid = group.loc[group["is_overlap"], "tau_wave_pa"].dropna()
            long_term_p95 = float(full_valid.quantile(0.95)) if len(full_valid) else None
            overlap_p95 = float(overlap_valid.quantile(0.95)) if len(overlap_valid) else None
            ratio = (
                overlap_p95 / long_term_p95 if overlap_p95 is not None and long_term_p95 else None
            )
            records.append(
                {
                    "wave_node_id": node_id,
                    "roughness_scenario": scenario_name,
                    "z0_m": z0,
                    "long_term_start_time_utc": group["time_utc"].min(),
                    "long_term_end_time_utc": group["time_utc"].max(),
                    "long_term_valid_count": int(len(full_valid)),
                    "long_term_tau_wave_mean_pa": float(full_valid.mean())
                    if len(full_valid)
                    else None,
                    "long_term_tau_wave_p90_pa": float(full_valid.quantile(0.90))
                    if len(full_valid)
                    else None,
                    "long_term_tau_wave_p95_pa": long_term_p95,
                    "long_term_tau_wave_p99_pa": float(full_valid.quantile(0.99))
                    if len(full_valid)
                    else None,
                    "long_term_tau_wave_max_pa": float(full_valid.max())
                    if len(full_valid)
                    else None,
                    "overlap_valid_count": int(len(overlap_valid)),
                    "overlap_tau_wave_mean_pa": float(overlap_valid.mean())
                    if len(overlap_valid)
                    else None,
                    "overlap_tau_wave_p90_pa": float(overlap_valid.quantile(0.90))
                    if len(overlap_valid)
                    else None,
                    "overlap_tau_wave_p95_pa": overlap_p95,
                    "overlap_tau_wave_p99_pa": float(overlap_valid.quantile(0.99))
                    if len(overlap_valid)
                    else None,
                    "overlap_tau_wave_max_pa": float(overlap_valid.max())
                    if len(overlap_valid)
                    else None,
                    "overlap_to_long_term_tau_wave_p95_ratio": ratio,
                }
            )
    return pd.DataFrame(records, columns=list(WAVE_ONLY_BED_SHEAR_LONG_TERM_STATS_COLUMNS))


# --- Combined node statistics (Section 22) ----------------------------------------------

COMBINED_BED_SHEAR_STATS_COLUMNS = (
    "hydro_pair_id",
    "roughness_scenario",
    "overlap_start_time_utc",
    "overlap_end_time_utc",
    "expected_3hour_count",
    "matched_count",
    "completeness_pct",
    "tau_current_mean_pa",
    "tau_current_p95_pa",
    "tau_current_p99_pa",
    "tau_current_max_pa",
    "tau_wave_mean_pa",
    "tau_wave_p95_pa",
    "tau_wave_p99_pa",
    "tau_wave_max_pa",
    "tau_mean_combined_mean_pa",
    "tau_mean_combined_p95_pa",
    "tau_mean_combined_p99_pa",
    "tau_mean_combined_max_pa",
    "tau_max_combined_mean_pa",
    "tau_max_combined_p90_pa",
    "tau_max_combined_p95_pa",
    "tau_max_combined_p99_pa",
    "tau_max_combined_max_pa",
    "wave_current_axis_angle_median_deg",
    "wave_current_axis_angle_p05_deg",
    "wave_current_axis_angle_p95_deg",
    "tau_max_to_max_single_component_median_ratio",
    "tau_max_to_max_single_component_p95_ratio",
)


def compute_combined_bed_shear_stats(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Per `hydro_pair_id x roughness_scenario` descriptive statistics (Section 22).

    No risk score.
    """

    if combined_df.empty:
        return pd.DataFrame(columns=list(COMBINED_BED_SHEAR_STATS_COLUMNS))

    records = []
    for (pair_id, scenario), group in combined_df.groupby(["hydro_pair_id", "roughness_scenario"]):
        start = group["time_utc"].min()
        end = group["time_utc"].max()
        expected_count = int(round((end - start).total_seconds() / (3 * 3600.0))) + 1
        matched_count = int(len(group))

        tau_current = group["tau_current_pa"].dropna()
        tau_wave = group["tau_wave_pa"].dropna()
        tau_mean_combined = group["tau_mean_combined_pa"].dropna()
        tau_max_combined = group["tau_max_combined_pa"].dropna()
        angle = group["wave_current_axis_angle_deg"].dropna()

        max_single_component = np.maximum(
            group["tau_current_pa"].to_numpy(dtype=float),
            group["tau_wave_pa"].to_numpy(dtype=float),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                max_single_component > 0,
                group["tau_max_combined_pa"].to_numpy(dtype=float) / max_single_component,
                np.nan,
            )
        ratio_valid = pd.Series(ratio).dropna()

        records.append(
            {
                "hydro_pair_id": pair_id,
                "roughness_scenario": scenario,
                "overlap_start_time_utc": start,
                "overlap_end_time_utc": end,
                "expected_3hour_count": expected_count,
                "matched_count": matched_count,
                "completeness_pct": _completeness_pct(matched_count, expected_count),
                "tau_current_mean_pa": float(tau_current.mean()) if len(tau_current) else None,
                "tau_current_p95_pa": float(tau_current.quantile(0.95))
                if len(tau_current)
                else None,
                "tau_current_p99_pa": float(tau_current.quantile(0.99))
                if len(tau_current)
                else None,
                "tau_current_max_pa": float(tau_current.max()) if len(tau_current) else None,
                "tau_wave_mean_pa": float(tau_wave.mean()) if len(tau_wave) else None,
                "tau_wave_p95_pa": float(tau_wave.quantile(0.95)) if len(tau_wave) else None,
                "tau_wave_p99_pa": float(tau_wave.quantile(0.99)) if len(tau_wave) else None,
                "tau_wave_max_pa": float(tau_wave.max()) if len(tau_wave) else None,
                "tau_mean_combined_mean_pa": float(tau_mean_combined.mean())
                if len(tau_mean_combined)
                else None,
                "tau_mean_combined_p95_pa": float(tau_mean_combined.quantile(0.95))
                if len(tau_mean_combined)
                else None,
                "tau_mean_combined_p99_pa": float(tau_mean_combined.quantile(0.99))
                if len(tau_mean_combined)
                else None,
                "tau_mean_combined_max_pa": float(tau_mean_combined.max())
                if len(tau_mean_combined)
                else None,
                "tau_max_combined_mean_pa": float(tau_max_combined.mean())
                if len(tau_max_combined)
                else None,
                "tau_max_combined_p90_pa": float(tau_max_combined.quantile(0.90))
                if len(tau_max_combined)
                else None,
                "tau_max_combined_p95_pa": float(tau_max_combined.quantile(0.95))
                if len(tau_max_combined)
                else None,
                "tau_max_combined_p99_pa": float(tau_max_combined.quantile(0.99))
                if len(tau_max_combined)
                else None,
                "tau_max_combined_max_pa": float(tau_max_combined.max())
                if len(tau_max_combined)
                else None,
                "wave_current_axis_angle_median_deg": float(angle.median()) if len(angle) else None,
                "wave_current_axis_angle_p05_deg": float(angle.quantile(0.05))
                if len(angle)
                else None,
                "wave_current_axis_angle_p95_deg": float(angle.quantile(0.95))
                if len(angle)
                else None,
                "tau_max_to_max_single_component_median_ratio": float(ratio_valid.median())
                if len(ratio_valid)
                else None,
                "tau_max_to_max_single_component_p95_ratio": float(ratio_valid.quantile(0.95))
                if len(ratio_valid)
                else None,
            }
        )
    return pd.DataFrame(records, columns=list(COMBINED_BED_SHEAR_STATS_COLUMNS))


# --- Roughness sensitivity envelope (Section 23) ----------------------------------------

SENSITIVITY_ENVELOPE_COLUMNS = (
    "hydro_pair_id",
    "tau_max_p95_sensitivity_min_pa",
    "tau_max_p95_sensitivity_max_pa",
    "tau_max_p95_sensitivity_width_pa",
    "tau_max_p99_sensitivity_min_pa",
    "tau_max_p99_sensitivity_max_pa",
    "tau_max_p99_sensitivity_width_pa",
)


def compute_sensitivity_envelope(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Per hydro_pair_id: min/max/width of `tau_max_combined_p95/p99_pa` across the FIVE scenarios.

    Never a roughness-averaged "best estimate", never a sixth canonical
    scenario (Section 23) -- the envelope IS the output.
    """

    if stats_df.empty:
        return pd.DataFrame(columns=list(SENSITIVITY_ENVELOPE_COLUMNS))

    records = []
    for pair_id, group in stats_df.groupby("hydro_pair_id"):
        p95_values = group["tau_max_combined_p95_pa"].dropna()
        p99_values = group["tau_max_combined_p99_pa"].dropna()
        p95_min = float(p95_values.min()) if len(p95_values) else None
        p95_max = float(p95_values.max()) if len(p95_values) else None
        p99_min = float(p99_values.min()) if len(p99_values) else None
        p99_max = float(p99_values.max()) if len(p99_values) else None
        records.append(
            {
                "hydro_pair_id": pair_id,
                "tau_max_p95_sensitivity_min_pa": p95_min,
                "tau_max_p95_sensitivity_max_pa": p95_max,
                "tau_max_p95_sensitivity_width_pa": (
                    p95_max - p95_min if p95_min is not None and p95_max is not None else None
                ),
                "tau_max_p99_sensitivity_min_pa": p99_min,
                "tau_max_p99_sensitivity_max_pa": p99_max,
                "tau_max_p99_sensitivity_width_pa": (
                    p99_max - p99_min if p99_min is not None and p99_max is not None else None
                ),
            }
        )
    return pd.DataFrame(records, columns=list(SENSITIVITY_ENVELOPE_COLUMNS))


# --- Representativeness warning (Section 24, verbatim) ---------------------------------

REPRESENTATIVENESS_WARNING = (
    "COMBINED BED-SHEAR STATISTICS ARE BASED ON THE CONTEMPORANEOUS PRIMARY-CURRENT / WAVE "
    "OVERLAP, NOT THE FULL 1980–2026 WAVE RECORD AND NOT A 25-YEAR RETURN-PERIOD ANALYSIS."
)
