"""Noncohesive sediment mobility capacity (MAR-013).

Scope -- read before touching this module
--------------------------------------------
Builds the first sediment-RESPONSE product, but PL854 has no defensible
continuous D50 field, so this module never claims "the actual seabed
sediment is mobile here" along the route. Instead it calculates
`NONCOHESIVE_SEDIMENT_MOBILITY_CAPACITY`: for a FIXED set of hypothetical
noncohesive grain-size test scenarios, it computes physically
self-consistent grain-related skin-friction stress, the Soulsby-Whitehouse
critical Shields stress, a mobility ratio, and threshold exceedance -- then
reports which TESTED grain sizes the real hydrodynamic forcing is capable
of mobilising. This is a forcing-CAPACITY product, never a continuous
site-specific sediment-truth product. No bedload transport rate, suspended
load, Exner morphology change, erosion/deposition prediction, scour depth,
pipeline exposure, free-span, fatigue, or risk scoring is computed
anywhere in this module.

Critical scientific rule: consistent D50, never MAR-012's roughness (Section 2)
-------------------------------------------------------------------------------------
MAR-012's `tau_max_p95_sensitivity_max_pa` is never divided by a
D50-derived threshold here -- MAR-012's five roughness scenarios are
independent SENSITIVITY dimensions, not candidate grain sizes. For Shields
initiation-of-motion analysis, the grain-related skin friction and the
critical threshold must be calculated CONSISTENTLY from the SAME candidate
D50: `z0_skin_m = d50_m / 12`, used to recompute BOTH the grain-related
current/wave skin friction (this module) AND compared against the
Soulsby-Whitehouse critical stress for that SAME D50. MAR-012 remains valid
as hydrodynamic bed-stress sensitivity and map context; this is a distinct
calculation.

Nine fixed test scenarios, never a site assignment (Section 3)
--------------------------------------------------------------------
`TESTED_D50_SCENARIOS_MM` are nine FIXED noncohesive grain-size TEST
scenarios (`TESTED_NONCOHESIVE_GRAIN_SIZE_SCENARIOS_NOT_SITE_SPECIFIC_D50`)
-- never a preferred/default D50, never averaged, never assigned from a
BGS Folk class, never interpolated from the five observed PSA D50 records
into a route field. The lower boundary (0.063 mm) is the lower edge of
this noncohesive framework; it is never extended into mud/clay/cohesive
thresholds.

Grain-related skin friction reuses MAR-012's validated formulation
(Sections 5, 8-9)
--------------------------------------------------------------------------------
The current-side friction-velocity inversion and the wave-side
smooth/laminar-vs-rough friction-factor branches and Soulsby algebraic
combined-stress formulas are IDENTICAL in structure to MAR-012's own
already-validated functions (imported directly from `combined_bed_shear`,
per the ticket's own explicit "use the same ... already validated in
MAR-012" instruction) -- only the roughness length changes, from MAR-010's
fixed sensitivity z0 to this module's `z0_skin_m = d50_m / 12`. The current
side uses the RAW MAR-009B canonical reference sample's own
`height_above_model_bed_m` as the log-profile reference height (never
MAR-010's fixed 1 m target), so `compute_current_friction_velocity_m_s` is
called with an ARRAY `target_height_m` here rather than the scalar 1.0 m
default -- the same formula, generalised to any reference height.

Soulsby-Whitehouse critical Shields threshold (Section 10)
----------------------------------------------------------------
`theta_cr = 0.30/(1+1.2*D*) + 0.055*(1-exp(-0.020*D*))`, where `D*` is the
dimensionless grain size. `tau_cr_pa = theta_cr * (rho_sediment -
rho_water) * g * d50_m`. `rho_sediment_kg_m3 = 2650` is a quartz/mineral
reference assumption for this test-scenario analysis, never a measured
PL854 sediment mineral density.

Mobility ratio and incipient motion (Section 11)
-----------------------------------------------------
`mobility_ratio = tau_max_grain_skin_pa / tau_cr_pa`; `>= 1` is
`ABOVE_OR_AT_NONCOHESIVE_INCIPIENT_MOTION_THRESHOLD`, else
`BELOW_NONCOHESIVE_INCIPIENT_MOTION_THRESHOLD`. This is never called
erosion, transport rate, scour, or pipeline risk -- threshold crossing
indicates initiation-of-motion POTENTIAL under the model assumptions only.

Capacity is a discrete, verified-not-assumed property of the tested set
(Sections 14-15)
-------------------------------------------------------------------------------
`largest_tested_d50_with_*_mobility_ratio_ge_1_mm` is the largest TESTED
scenario passing a given percentile threshold -- never an interpolated
continuous critical grain size, and never reported above the largest
tested scenario (a passing largest-tested scenario instead flags
`CAPACITY_EXCEEDS_TESTED_GRAIN_SIZE_RANGE`). Monotonicity of
`mobility_ratio_p95` across the nine scenarios is VERIFIED per hydro pair,
never assumed -- a non-monotonic sequence is recorded explicitly
(`monotonicity_violation_count`), and even then the discrete largest-
passing-scenario field may still be reported, but never described as "all
grains smaller than X are mobile" unless the tested sequence is actually
monotonic.

BGS sediment evidence: strict, non-circular use (Sections 16-17)
------------------------------------------------------------------------
Regional BGS 250k Folk-class context (`mapped_250k_folk_class`) is never
converted into a numeric D50 -- `mapped_250k_folk_d50_text` is source TEXT,
never parsed as a quantity (mirrors MAR-008's own established discipline).
The BGS predictive sediment product never enters this module's physics
(`SECONDARY_MODEL_COMPARISON` only, per MAR-008). The five valid observed
PSA D50 points are surfaced as point context only
(`POINT_OBSERVATION_NOT_INTERPOLATED_TO_PIPELINE`) -- never interpolated,
Voronoi/IDW/kriging-assigned, or nearest-neighbour-propagated into a route
D50 field.
"""

import numpy as np
import pandas as pd

from marine_engine.metocean.combined_bed_shear import (
    compute_current_bed_shear_stress_pa,
    compute_current_friction_velocity_m_s,
    compute_soulsby_max_combined_stress_pa,
    compute_soulsby_mean_combined_stress_pa,
    compute_wave_bed_shear_stress_pa,
    compute_wave_friction_factor,
    compute_wave_friction_rough_branch,
    compute_wave_friction_smooth_branch,
    compute_wave_reynolds_number,
    compute_wave_semi_orbital_excursion_m,
    fold_wave_current_axis_angle_deg,
)

# --- Fixed scientific constants (Sections 3-4, 10 -- do not change) ----------------

GRAVITY_M_S2 = 9.80665
RHO_WATER_KG_M3 = 1027.0
RHO_SEDIMENT_KG_M3 = 2650.0
KINEMATIC_VISCOSITY_M2_S = 1.36e-6
VON_KARMAN_KAPPA = 0.40

TESTED_D50_SCENARIOS_MM: tuple[float, ...] = (
    0.063,
    0.125,
    0.250,
    0.500,
    1.000,
    2.000,
    4.000,
    8.000,
    16.000,
)
GRAIN_SIZE_SCENARIO_SEMANTICS = "TESTED_NONCOHESIVE_GRAIN_SIZE_SCENARIOS_NOT_SITE_SPECIFIC_D50"

SCIENTIFIC_ROLE = "NONCOHESIVE_SEDIMENT_MOBILITY_CAPACITY"

ABOVE_OR_AT_THRESHOLD = "ABOVE_OR_AT_NONCOHESIVE_INCIPIENT_MOTION_THRESHOLD"
BELOW_THRESHOLD = "BELOW_NONCOHESIVE_INCIPIENT_MOTION_THRESHOLD"

NO_TESTED_SCENARIO_PASSES_THRESHOLD = "NO_TESTED_SCENARIO_PASSES_THRESHOLD"
CAPACITY_WITHIN_TESTED_RANGE = "CAPACITY_WITHIN_TESTED_RANGE"
CAPACITY_EXCEEDS_TESTED_GRAIN_SIZE_RANGE = "CAPACITY_EXCEEDS_TESTED_GRAIN_SIZE_RANGE"

# Reference D50 scenarios (mm) surfaced on the map segments purely for
# convenient cross-checking -- never used to select or colour anything.
REFERENCE_D50_SCENARIOS_MM: tuple[float, ...] = (0.125, 0.250, 0.500, 1.000, 2.000)


class NoncohesiveMobilityCompletenessError(Exception):
    """More valid matched timestamps exist than the expected regular-cadence count allows."""


def _completeness_pct(valid_count: int, expected_count: int) -> float | None:
    if not expected_count:
        return None
    if valid_count > expected_count:
        raise NoncohesiveMobilityCompletenessError(
            f"{valid_count} valid matched timestamps exceeds the expected regular-cadence "
            f"count of {expected_count} -- completeness must never exceed 100%"
        )
    return 100.0 * valid_count / expected_count


# --- Grain-size scenario helpers (Section 5) ----------------------------------------


def compute_z0_skin_m(d50_m: float | np.ndarray) -> np.ndarray:
    """`z0_skin = d50 / 12` -- the grain-related (skin) roughness length."""

    return np.asarray(d50_m, dtype=float) / 12.0


# --- Soulsby-Whitehouse critical Shields threshold (Section 10) ---------------------


def compute_dimensionless_grain_size(
    d50_m: float | np.ndarray,
    *,
    g: float = GRAVITY_M_S2,
    rho_sediment: float = RHO_SEDIMENT_KG_M3,
    rho_water: float = RHO_WATER_KG_M3,
    nu: float = KINEMATIC_VISCOSITY_M2_S,
) -> np.ndarray:
    """`D* = d50 * [g*(s-1)/nu^2]^(1/3)`, `s = rho_sediment / rho_water`."""

    d50_m = np.asarray(d50_m, dtype=float)
    s = rho_sediment / rho_water
    return d50_m * (g * (s - 1.0) / nu**2) ** (1.0 / 3.0)


def compute_soulsby_whitehouse_critical_shields_parameter(d_star: float | np.ndarray) -> np.ndarray:
    """`theta_cr = 0.30/(1+1.2*D*) + 0.055*[1 - exp(-0.020*D*)]`."""

    d_star = np.asarray(d_star, dtype=float)
    return 0.30 / (1.0 + 1.2 * d_star) + 0.055 * (1.0 - np.exp(-0.020 * d_star))


def compute_critical_shear_stress_pa(
    theta_cr: float | np.ndarray,
    d50_m: float | np.ndarray,
    *,
    rho_sediment: float = RHO_SEDIMENT_KG_M3,
    rho_water: float = RHO_WATER_KG_M3,
    g: float = GRAVITY_M_S2,
) -> np.ndarray:
    """`tau_cr_pa = theta_cr * (rho_sediment - rho_water) * g * d50_m`."""

    theta_cr = np.asarray(theta_cr, dtype=float)
    d50_m = np.asarray(d50_m, dtype=float)
    return theta_cr * (rho_sediment - rho_water) * g * d50_m


# --- Mobility ratio and incipient motion (Section 11) --------------------------------


def compute_mobility_ratio(
    tau_max_grain_skin_pa: np.ndarray, tau_critical_pa: np.ndarray
) -> np.ndarray:
    """`mobility_ratio = tau_max_grain_skin / tau_cr`; undefined (NaN) for `tau_cr <= 0`."""

    tau_max = np.asarray(tau_max_grain_skin_pa, dtype=float)
    tau_cr = np.asarray(tau_critical_pa, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tau_cr > 0, tau_max / tau_cr, np.nan)


def classify_incipient_motion_status(mobility_ratio: np.ndarray) -> np.ndarray:
    """`ABOVE_OR_AT_.../BELOW_...` per `mobility_ratio >= 1`; null where the ratio is undefined."""

    mobility_ratio = np.asarray(mobility_ratio, dtype=float)
    status = np.full(mobility_ratio.shape, None, dtype=object)
    valid = np.isfinite(mobility_ratio)
    status[valid & (mobility_ratio >= 1.0)] = ABOVE_OR_AT_THRESHOLD
    status[valid & (mobility_ratio < 1.0)] = BELOW_THRESHOLD
    return status


# --- Canonical 3-hourly output (Section 12) -------------------------------------------

NONCOHESIVE_MOBILITY_3HOURLY_COLUMNS = (
    # Identity
    "hydro_pair_id",
    "current_node_id",
    "wave_node_id",
    "time_utc",
    # Grain scenario
    "tested_d50_mm",
    "tested_d50_m",
    "z0_skin_m",
    "rho_sediment_kg_m3",
    # Current
    "reference_current_speed_m_s",
    "reference_height_above_model_bed_m",
    "current_direction_to_deg",
    "current_skin_friction_velocity_m_s",
    "tau_current_skin_pa",
    # Wave
    "wave_orbital_rms_m_s",
    "wave_orbital_amplitude_m_s",
    "representative_wave_period_s",
    "wave_direction_to_deg",
    "wave_semi_orbital_excursion_m",
    "wave_reynolds_number",
    "wave_skin_friction_factor",
    "tau_wave_skin_pa",
    # Combined
    "wave_current_axis_angle_deg",
    "tau_mean_grain_skin_pa",
    "tau_max_grain_skin_pa",
    # Threshold
    "dimensionless_grain_size_dstar",
    "critical_shields_parameter",
    "tau_critical_pa",
    "mobility_ratio",
    "incipient_motion_status",
    # Provenance
    "scientific_role",
)

_WAVE_SIDE_RENAME = {
    "wave_orbital_velocity_rms_near_bed_m_s": "wave_orbital_rms_m_s",
    "wave_orbital_velocity_equivalent_amplitude_m_s": "wave_orbital_amplitude_m_s",
    "equivalent_peak_period_from_tz_s": "representative_wave_period_s",
    "wave_mean_direction_to_deg": "wave_direction_to_deg",
}


def build_noncohesive_mobility_3hourly(
    current_hourly_df: pd.DataFrame,
    wave_3hourly_df: pd.DataFrame,
    hydro_pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """LONG format: one row per `hydro_pair_id x time_utc x tested_d50_scenario` (Section 12).

    The current/wave sides are joined ONCE on the exact matched timestamp
    (never per-scenario, since neither raw input carries a grain-size
    dimension); the nine D50 scenarios are then fanned out explicitly, each
    recomputing the grain-related skin friction/critical stress
    consistently from that SAME candidate D50 (Section 2). Never fans out
    across chainage stations.
    """

    if current_hourly_df.empty or wave_3hourly_df.empty or hydro_pairs_df.empty:
        return pd.DataFrame(columns=list(NONCOHESIVE_MOBILITY_3HOURLY_COLUMNS))

    current_side = current_hourly_df.merge(
        hydro_pairs_df[["current_node_id", "wave_node_id", "hydro_pair_id"]],
        on="current_node_id",
        how="inner",
    )
    wave_side = wave_3hourly_df.rename(columns=_WAVE_SIDE_RENAME)[
        [
            "wave_node_id",
            "time_utc",
            "wave_orbital_rms_m_s",
            "wave_orbital_amplitude_m_s",
            "representative_wave_period_s",
            "wave_direction_to_deg",
        ]
    ]

    base = current_side.merge(wave_side, on=["wave_node_id", "time_utc"], how="inner")
    if base.empty:
        return pd.DataFrame(columns=list(NONCOHESIVE_MOBILITY_3HOURLY_COLUMNS))

    u_ref = base["current_speed_m_s"].to_numpy(dtype=float)
    height_valid = base["height_above_model_bed_valid"].fillna(False).to_numpy(dtype=bool)
    z_r_raw = base["height_above_model_bed_m"].to_numpy(dtype=float)
    # A sample exactly AT the model bed (z_r == 0) would divide-by-zero
    # inside the log-profile ratio -- masked to NaN, never propagated as inf.
    z_r_eligible = height_valid & np.isfinite(z_r_raw) & (z_r_raw > 0)
    z_r = np.where(z_r_eligible, z_r_raw, np.nan)

    raw_direction = base["current_direction_to_deg"].to_numpy(dtype=float)
    speed_valid = np.isfinite(u_ref) & (u_ref > 0)
    current_dir = np.where(speed_valid, raw_direction, np.nan)

    uw = base["wave_orbital_amplitude_m_s"].to_numpy(dtype=float)
    t_rep = base["representative_wave_period_s"].to_numpy(dtype=float)
    wave_dir = base["wave_direction_to_deg"].to_numpy(dtype=float)

    # Scenario-independent wave quantities (do not depend on z0_skin) --
    # computed once and reused across all nine D50 scenarios.
    a_wave = compute_wave_semi_orbital_excursion_m(uw, t_rep)
    rw = compute_wave_reynolds_number(uw, a_wave)
    f_ws = compute_wave_friction_smooth_branch(rw)
    phi = fold_wave_current_axis_angle_deg(current_dir, wave_dir)

    identity = {
        "hydro_pair_id": base["hydro_pair_id"].to_numpy(),
        "current_node_id": base["current_node_id"].to_numpy(),
        "wave_node_id": base["wave_node_id"].to_numpy(),
        "time_utc": base["time_utc"].to_numpy(),
        "reference_current_speed_m_s": u_ref,
        "reference_height_above_model_bed_m": z_r_raw,
        "current_direction_to_deg": current_dir,
        "wave_orbital_rms_m_s": base["wave_orbital_rms_m_s"].to_numpy(dtype=float),
        "wave_orbital_amplitude_m_s": uw,
        "representative_wave_period_s": t_rep,
        "wave_direction_to_deg": wave_dir,
        "wave_semi_orbital_excursion_m": a_wave,
        "wave_reynolds_number": rw,
        "wave_current_axis_angle_deg": phi,
    }

    scenario_frames = []
    for d50_mm in TESTED_D50_SCENARIOS_MM:
        d50_m = d50_mm / 1000.0
        z0_skin = compute_z0_skin_m(d50_m)

        u_star_c = compute_current_friction_velocity_m_s(u_ref, z0_skin, target_height_m=z_r)
        tau_c = compute_current_bed_shear_stress_pa(u_star_c)

        f_wr = compute_wave_friction_rough_branch(a_wave, z0_skin)
        f_w, _controlling_branch = compute_wave_friction_factor(f_ws, f_wr, uw)
        tau_w = compute_wave_bed_shear_stress_pa(uw, f_w)

        tau_m = compute_soulsby_mean_combined_stress_pa(tau_c, tau_w)
        tau_max = compute_soulsby_max_combined_stress_pa(tau_c, tau_w, tau_m, phi)

        d_star = compute_dimensionless_grain_size(d50_m)
        theta_cr = compute_soulsby_whitehouse_critical_shields_parameter(d_star)
        tau_cr = compute_critical_shear_stress_pa(theta_cr, d50_m)
        mobility_ratio = compute_mobility_ratio(tau_max, np.full(len(base), tau_cr))
        status = classify_incipient_motion_status(mobility_ratio)

        scenario_frames.append(
            pd.DataFrame(
                {
                    **identity,
                    "tested_d50_mm": d50_mm,
                    "tested_d50_m": d50_m,
                    "z0_skin_m": z0_skin,
                    "rho_sediment_kg_m3": RHO_SEDIMENT_KG_M3,
                    "current_skin_friction_velocity_m_s": u_star_c,
                    "tau_current_skin_pa": tau_c,
                    "wave_skin_friction_factor": f_w,
                    "tau_wave_skin_pa": tau_w,
                    "tau_mean_grain_skin_pa": tau_m,
                    "tau_max_grain_skin_pa": tau_max,
                    "dimensionless_grain_size_dstar": d_star,
                    "critical_shields_parameter": theta_cr,
                    "tau_critical_pa": tau_cr,
                    "mobility_ratio": mobility_ratio,
                    "incipient_motion_status": status,
                    "scientific_role": SCIENTIFIC_ROLE,
                }
            )
        )

    result = pd.concat(scenario_frames, ignore_index=True)
    return result[list(NONCOHESIVE_MOBILITY_3HOURLY_COLUMNS)]


# --- Per-pair / per-D50 statistics (Section 13) ---------------------------------------

NONCOHESIVE_MOBILITY_STATS_COLUMNS = (
    "hydro_pair_id",
    "tested_d50_mm",
    "overlap_start_time_utc",
    "overlap_end_time_utc",
    "valid_count",
    "completeness_pct",
    "tau_max_skin_mean_pa",
    "tau_max_skin_p90_pa",
    "tau_max_skin_p95_pa",
    "tau_max_skin_p99_pa",
    "tau_max_skin_max_pa",
    "tau_critical_pa",
    "mobility_ratio_mean",
    "mobility_ratio_p50",
    "mobility_ratio_p90",
    "mobility_ratio_p95",
    "mobility_ratio_p99",
    "mobility_ratio_max",
    "threshold_exceedance_count",
    "threshold_exceedance_fraction",
    "threshold_exceedance_pct",
)


def compute_noncohesive_mobility_stats(mobility_3hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Per `hydro_pair_id x tested_d50_mm` descriptive statistics (Section 13).

    `mobility_ratio_p90/p95/p99` are computed from the TIME-RESOLVED ratio
    series itself, never as a ratio of independently-computed percentile
    stresses (Section 30-O) -- the two are not mathematically equivalent in
    general (percentile of a ratio != ratio of percentiles).
    """

    if mobility_3hourly_df.empty:
        return pd.DataFrame(columns=list(NONCOHESIVE_MOBILITY_STATS_COLUMNS))

    records = []
    for (pair_id, d50_mm), group in mobility_3hourly_df.groupby(["hydro_pair_id", "tested_d50_mm"]):
        start = group["time_utc"].min()
        end = group["time_utc"].max()
        expected_count = int(round((end - start).total_seconds() / (3 * 3600.0))) + 1

        tau_max_skin = group["tau_max_grain_skin_pa"].dropna()
        ratio = group["mobility_ratio"].dropna()
        valid_count = int(len(ratio))
        exceedance_count = int((ratio >= 1.0).sum())

        records.append(
            {
                "hydro_pair_id": pair_id,
                "tested_d50_mm": d50_mm,
                "overlap_start_time_utc": start,
                "overlap_end_time_utc": end,
                "valid_count": valid_count,
                "completeness_pct": _completeness_pct(valid_count, expected_count),
                "tau_max_skin_mean_pa": float(tau_max_skin.mean()) if len(tau_max_skin) else None,
                "tau_max_skin_p90_pa": float(tau_max_skin.quantile(0.90))
                if len(tau_max_skin)
                else None,
                "tau_max_skin_p95_pa": float(tau_max_skin.quantile(0.95))
                if len(tau_max_skin)
                else None,
                "tau_max_skin_p99_pa": float(tau_max_skin.quantile(0.99))
                if len(tau_max_skin)
                else None,
                "tau_max_skin_max_pa": float(tau_max_skin.max()) if len(tau_max_skin) else None,
                "tau_critical_pa": float(group["tau_critical_pa"].iloc[0]) if len(group) else None,
                "mobility_ratio_mean": float(ratio.mean()) if len(ratio) else None,
                "mobility_ratio_p50": float(ratio.quantile(0.50)) if len(ratio) else None,
                "mobility_ratio_p90": float(ratio.quantile(0.90)) if len(ratio) else None,
                "mobility_ratio_p95": float(ratio.quantile(0.95)) if len(ratio) else None,
                "mobility_ratio_p99": float(ratio.quantile(0.99)) if len(ratio) else None,
                "mobility_ratio_max": float(ratio.max()) if len(ratio) else None,
                "threshold_exceedance_count": exceedance_count,
                "threshold_exceedance_fraction": (
                    exceedance_count / valid_count if valid_count else None
                ),
                "threshold_exceedance_pct": (
                    100.0 * exceedance_count / valid_count if valid_count else None
                ),
            }
        )
    return pd.DataFrame(records, columns=list(NONCOHESIVE_MOBILITY_STATS_COLUMNS))


# --- Mobility capacity summary + monotonicity QA (Sections 14-15) --------------------

MOBILITY_CAPACITY_COLUMNS = (
    "hydro_pair_id",
    "largest_tested_d50_with_p90_mobility_ratio_ge_1_mm",
    "largest_tested_d50_with_p90_mobility_ratio_ge_1_status",
    "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm",
    "largest_tested_d50_with_p95_mobility_ratio_ge_1_status",
    "largest_tested_d50_with_p99_mobility_ratio_ge_1_mm",
    "largest_tested_d50_with_p99_mobility_ratio_ge_1_status",
    "largest_tested_d50_with_any_exceedance_mm",
    "largest_tested_d50_with_any_exceedance_status",
    "p95_mobility_sequence_monotonic_nonincreasing",
    "monotonicity_violation_count",
)


def _largest_passing_d50(
    sorted_d50_mm: list[float], passes_by_d50: dict[float, bool]
) -> tuple[float | None, str]:
    """The largest TESTED scenario passing, plus a status flag (Sections 14-15).

    Never reports a value above the largest tested scenario -- a passing
    largest-tested scenario instead flags
    `CAPACITY_EXCEEDS_TESTED_GRAIN_SIZE_RANGE` (the true capacity ceiling
    may lie beyond what was tested, but that is never guessed at here).
    """

    passing = [d for d in sorted_d50_mm if passes_by_d50.get(d, False)]
    if not passing:
        return None, NO_TESTED_SCENARIO_PASSES_THRESHOLD
    largest = max(passing)
    if largest == max(sorted_d50_mm):
        return largest, CAPACITY_EXCEEDS_TESTED_GRAIN_SIZE_RANGE
    return largest, CAPACITY_WITHIN_TESTED_RANGE


def compute_mobility_capacity_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Per hydro_pair_id: discrete tested-scenario capacities + monotonicity QA.

    `mobility_ratio_p95` is NEVER assumed to decrease monotonically with
    D50 (Section 15) -- the ordering across the nine scenarios is checked
    explicitly for every hydro pair.
    """

    if stats_df.empty:
        return pd.DataFrame(columns=list(MOBILITY_CAPACITY_COLUMNS))

    sorted_d50_mm = sorted(TESTED_D50_SCENARIOS_MM)
    records = []
    for pair_id, group in stats_df.groupby("hydro_pair_id"):
        by_d50 = group.set_index("tested_d50_mm")

        def _passes(column: str, d50: float, *, by_d50: pd.DataFrame = by_d50) -> bool:
            if d50 not in by_d50.index:
                return False
            value = by_d50.loc[d50, column]
            return bool(pd.notna(value) and value >= 1.0)

        p90_value, p90_status = _largest_passing_d50(
            sorted_d50_mm, {d: _passes("mobility_ratio_p90", d) for d in sorted_d50_mm}
        )
        p95_value, p95_status = _largest_passing_d50(
            sorted_d50_mm, {d: _passes("mobility_ratio_p95", d) for d in sorted_d50_mm}
        )
        p99_value, p99_status = _largest_passing_d50(
            sorted_d50_mm, {d: _passes("mobility_ratio_p99", d) for d in sorted_d50_mm}
        )
        any_value, any_status = _largest_passing_d50(
            sorted_d50_mm, {d: _passes("mobility_ratio_max", d) for d in sorted_d50_mm}
        )

        p95_sequence = [
            float(by_d50.loc[d, "mobility_ratio_p95"])
            if d in by_d50.index and pd.notna(by_d50.loc[d, "mobility_ratio_p95"])
            else None
            for d in sorted_d50_mm
        ]
        violations = 0
        for prev, nxt in zip(p95_sequence, p95_sequence[1:], strict=False):
            if prev is not None and nxt is not None and nxt > prev:
                violations += 1
        has_full_sequence = all(v is not None for v in p95_sequence)

        records.append(
            {
                "hydro_pair_id": pair_id,
                "largest_tested_d50_with_p90_mobility_ratio_ge_1_mm": p90_value,
                "largest_tested_d50_with_p90_mobility_ratio_ge_1_status": p90_status,
                "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": p95_value,
                "largest_tested_d50_with_p95_mobility_ratio_ge_1_status": p95_status,
                "largest_tested_d50_with_p99_mobility_ratio_ge_1_mm": p99_value,
                "largest_tested_d50_with_p99_mobility_ratio_ge_1_status": p99_status,
                "largest_tested_d50_with_any_exceedance_mm": any_value,
                "largest_tested_d50_with_any_exceedance_status": any_status,
                "p95_mobility_sequence_monotonic_nonincreasing": (
                    (violations == 0) if has_full_sequence else None
                ),
                "monotonicity_violation_count": violations if has_full_sequence else None,
            }
        )
    return pd.DataFrame(records, columns=list(MOBILITY_CAPACITY_COLUMNS))


# --- Observed D50 point context (Section 17) ------------------------------------------

OBSERVED_D50_CONTEXT_COLUMNS = (
    "psa_data_id",
    "sample_date",
    "sample_year",
    "sample_age_years_at_run",
    "d50_mm",
    "d10_mm",
    "d90_mm",
    "folk_class",
    "gravel_pct",
    "sand_pct",
    "mud_pct",
    "distance_to_pipeline_m",
    "nearest_pipeline_chainage_m",
    "nearest_pipeline_kp",
    "evidence_role",
    "interpretation",
)

_VALID_GRAIN_PERCENTILE_STATUSES = (
    "DERIVED_FROM_PERCENT_BINS",
    "DERIVED_FROM_NORMALIZED_MASS_BINS",
)

PRIMARY_OBSERVATIONAL = "PRIMARY_OBSERVATIONAL"
POINT_OBSERVATION_NOT_INTERPOLATED = "POINT_OBSERVATION_NOT_INTERPOLATED_TO_PIPELINE"


def build_observed_d50_context(psa_observations_df: pd.DataFrame) -> pd.DataFrame:
    """Valid observed surface PSA D50 points -- POINT CONTEXT ONLY (Section 17).

    Retains only records with surface evidence, a valid grain-percentile
    derivation status, and a finite `d50_mm` -- never interpolated,
    Voronoi/IDW/kriging-assigned, or nearest-neighbour-propagated onto the
    pipeline. The count is always derived from the real data, never
    hard-coded.
    """

    if psa_observations_df.empty:
        return pd.DataFrame(columns=list(OBSERVED_D50_CONTEXT_COLUMNS))

    is_surface = psa_observations_df["surface_evidence_class"].isin(
        ("SURFACE_GRAB", "SURFACE_CORE_INTERVAL")
    )
    is_valid_status = psa_observations_df["grain_percentile_status"].isin(
        _VALID_GRAIN_PERCENTILE_STATUSES
    )
    is_finite_d50 = psa_observations_df["d50_mm"].notna()
    valid = psa_observations_df.loc[is_surface & is_valid_status & is_finite_d50].reset_index(
        drop=True
    )
    if valid.empty:
        return pd.DataFrame(columns=list(OBSERVED_D50_CONTEXT_COLUMNS))

    n = len(valid)
    result = pd.DataFrame(
        {
            "psa_data_id": valid["psa_data_id"],
            "sample_date": valid["sample_date"],
            "sample_year": valid["sample_year"],
            "sample_age_years_at_run": valid["sample_age_years_at_run"],
            "d50_mm": valid["d50_mm"],
            "d10_mm": valid["d10_mm"],
            "d90_mm": valid["d90_mm"],
            "folk_class": valid["folk_class"],
            "gravel_pct": valid["gravel"],
            "sand_pct": valid["sand"],
            "mud_pct": valid["mud"],
            "distance_to_pipeline_m": valid["distance_to_pipeline_m"],
            "nearest_pipeline_chainage_m": valid["nearest_pipeline_chainage_m"],
            "nearest_pipeline_kp": valid["nearest_pipeline_kp"],
            "evidence_role": [PRIMARY_OBSERVATIONAL] * n,
            "interpretation": [POINT_OBSERVATION_NOT_INTERPOLATED] * n,
        }
    )
    return result[list(OBSERVED_D50_CONTEXT_COLUMNS)]
