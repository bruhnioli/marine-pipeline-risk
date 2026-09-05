"""Current-only near-bed log-profile normalization sensitivity (MAR-010).

Scope -- read before touching this module
--------------------------------------------
Normalizes the MAR-009B corrected primary-current reference sample
(`deepest_valid_standard_level_current`, `height_above_model_bed_m` above
the Copernicus model's own bed) to a standard height of 1.0 m above that
SAME model bed, using a current-only logarithmic velocity-profile ratio.
The canonical role name for every output of this module is
`CURRENT_ONLY_LOG_PROFILE_SENSITIVITY` -- never "bed current"/"seabed
current"/"combined near-bed current" anywhere in this codebase. This is
NOT native bottom-cell current, NOT combined wave-current velocity, NOT
bed shear stress, NOT pipeline loading, NOT sediment mobility. No wave
parameter, no BGS-predictive sediment, and no pipeline directionality is
ever read or used here; `z_r`/model bathymetry stay wholly within the
Copernicus model's own vertical reference (never the canonical MAR-006 LAT
bathymetry).

Formula (fixed by the ticket -- do not change; do not substitute another
boundary-layer model or perform independent literature research here)
-----------------------------------------------------------------------------
    S(z_t, z_r, z0) = [ln(z_t + z0) - ln(z0)] / [ln(z_r + z0) - ln(z0)]
    uo_1m = S * uo_ref;  vo_1m = S * vo_ref;  speed_1m = sqrt(uo_1m^2 + vo_1m^2)

`S` is a single positive scalar shared by both vector components, so
current direction is always preserved exactly (`atan2(S*uo, S*vo) ==
atan2(uo, vo)` for `S > 0`).

Roughness is a sensitivity dimension, never bed truth
-----------------------------------------------------------
`ROUGHNESS_SCENARIOS_M` are five FIXED sensitivity scenarios representing
long-standing marine/offshore engineering roughness classes -- never a
canonical PL854 seabed roughness choice, never a BGS-Folk-to-scenario
mapping, never a D50-derived continuous roughness field, and never
averaged into a pseudo-best estimate. The sensitivity ENVELOPE (min/max
across the five scenarios) is itself the output; there is deliberately no
sixth "canonical" scenario.

Vertical-domain screen (Section 4) -- a project data-QA heuristic, not a
physical law
-------------------------------------------------------------------------------
`VERTICAL_DOMAIN_SCREEN_FRACTION` (0.30) is a conservative validity SCREEN
for applying this simple current-only log-profile formulation at all --
never a universal physical threshold, always named
`log_profile_vertical_domain_status` (`WITHIN_.../OUTSIDE_...`) so it is
never confused with a hard physical limit. A row outside the screen is
never silently normalized: its five scenario rows still exist (so the
screen's own exclusion is visible), but every normalized value is null.

Current-wave semantics (Section 5)
--------------------------------------
This module is CURRENT ONLY. It never reads wave parameters, never
modifies roughness based on waves, and never computes an apparent
wave-current roughness or combined velocity -- that interaction is
explicitly deferred (see the `current_wave_interaction_applied = false`
metadata flag written by the CLI layer). Nothing here may be called
"combined near-bed current"/"bed current"/"seabed current".
"""

from typing import Any

import numpy as np
import pandas as pd

# --- Fixed scientific constants (Section 2/3/4 of the ticket -- do not change) ------

TARGET_HEIGHT_ABOVE_MODEL_BED_M = 1.0

# Five fixed sensitivity scenarios (Section 3). Order matters only for
# deterministic output row ordering, never for "picking a default".
ROUGHNESS_SCENARIOS_M: tuple[tuple[str, float], ...] = (
    ("SILT", 5e-6),
    ("FINE_SAND", 1e-5),
    ("MEDIUM_SAND", 4e-5),
    ("COARSE_SAND", 1e-4),
    ("GRAVEL", 3e-4),
)

# Project data-QA heuristic (Section 4), never a universal physical
# threshold -- always surfaced under an explicitly-named status string,
# never a bare boolean that could be mistaken for a physical limit.
VERTICAL_DOMAIN_SCREEN_FRACTION = 0.30
WITHIN_VERTICAL_SCREEN = "WITHIN_CONSERVATIVE_VERTICAL_SCREEN"
OUTSIDE_VERTICAL_SCREEN = "OUTSIDE_CONSERVATIVE_VERTICAL_SCREEN"

SCIENTIFIC_ROLE = "CURRENT_ONLY_LOG_PROFILE_SENSITIVITY"


class NormalizationCompletenessError(Exception):
    """More valid 1 m-equivalent samples exist than the expected regular-cadence count allows.

    Mirrors MAR-009B's `TemporalCompletenessError` for this derived
    product: since the hourly sensitivity table is built strictly from the
    already-deduplicated MAR-009B canonical hourly series, completeness
    here can never legitimately exceed 100% either.
    """


def compute_log_profile_scale_factor(
    z_t: float | np.ndarray, z_r: np.ndarray, z0: float | np.ndarray
) -> np.ndarray:
    """S(z_t, z_r, z0) -- see module docstring. Callers must pre-validate z_t,z_r,z0 > 0.

    Returns the scalar ratio only; multiply `uo_ref`/`vo_ref` by it
    directly to preserve direction exactly -- never recompute direction
    from the scaled components via a different path.
    """

    return (np.log(z_t + z0) - np.log(z0)) / (np.log(z_r + z0) - np.log(z0))


def is_log_profile_input_valid(z_t: float, z_r: np.ndarray, z0: float) -> np.ndarray:
    """z_r > 0 AND z0 > 0 AND z_t > 0 -- required for the log ratio to be defined (Section 4).

    `z_r` may be a non-finite/non-positive array (e.g. a missing or
    exactly-at-bed reference sample); `z0`/`z_t` are scalars but are still
    checked explicitly rather than assumed positive by construction, so a
    caller passing a contrived invalid scenario is still safely rejected.
    """

    z_r = np.asarray(z_r, dtype=float)
    return np.isfinite(z_r) & (z_r > 0) & (z0 > 0) & (z_t > 0)


def classify_vertical_domain_status(z_r_over_h: np.ndarray) -> np.ndarray:
    """WITHIN/OUTSIDE the project's conservative vertical-applicability screen (Section 4)."""

    z_r_over_h = np.asarray(z_r_over_h, dtype=float)
    return np.where(
        np.isfinite(z_r_over_h) & (z_r_over_h <= VERTICAL_DOMAIN_SCREEN_FRACTION),
        WITHIN_VERTICAL_SCREEN,
        OUTSIDE_VERTICAL_SCREEN,
    )


# --- Hourly sensitivity time series (Section 6) -------------------------------------

CURRENT_ONLY_1M_SENSITIVITY_COLUMNS = (
    "current_node_id",
    "time_utc",
    "reference_uo_m_s",
    "reference_vo_m_s",
    "reference_speed_m_s",
    "reference_height_above_model_bed_m",
    "model_bathymetry_m",
    "z_r_over_h_model",
    "log_profile_vertical_domain_status",
    "roughness_scenario",
    "z0_m",
    "target_height_above_model_bed_m",
    "log_profile_scale_factor",
    "current_only_1m_uo_m_s",
    "current_only_1m_vo_m_s",
    "current_only_1m_speed_m_s",
    "source_dataset",
    "scientific_role",
)


def build_current_only_1m_sensitivity_hourly(primary_current_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format: one row per (real support node, hour, roughness scenario).

    Only rows that HAD a MAR-009A/B canonical reference sample at all are
    included (Section 6 -- "260,400 valid primary-current rows x 5
    scenarios", never a fabricated row for a timestamp with no reference
    current). Rows outside the vertical-domain screen are still emitted
    (so the screen's own exclusion remains visible per node/time/scenario)
    but every normalized value is null (Section 4) -- never silently
    normalized anyway.
    """

    if primary_current_df.empty:
        return pd.DataFrame(columns=list(CURRENT_ONLY_1M_SENSITIVITY_COLUMNS))

    has_reference = (
        primary_current_df["current_sample_depth_m"].notna()
        & primary_current_df["height_above_model_bed_valid"].fillna(False)
        & primary_current_df["uo_m_s"].notna()
        & primary_current_df["vo_m_s"].notna()
    )
    base = primary_current_df.loc[has_reference].reset_index(drop=True)
    if base.empty:
        return pd.DataFrame(columns=list(CURRENT_ONLY_1M_SENSITIVITY_COLUMNS))

    z_r = base["height_above_model_bed_m"].to_numpy(dtype=float)
    h_model = base["model_bathymetry_m"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_r_over_h = np.where(h_model > 0, z_r / h_model, np.nan)
    domain_status = classify_vertical_domain_status(z_r_over_h)
    within_screen = domain_status == WITHIN_VERTICAL_SCREEN

    uo_ref = base["uo_m_s"].to_numpy(dtype=float)
    vo_ref = base["vo_m_s"].to_numpy(dtype=float)
    speed_ref = np.sqrt(uo_ref**2 + vo_ref**2)

    node_ids = base["current_node_id"].to_numpy()
    times = base["time_utc"].to_numpy()
    source_dataset = base["source_dataset"].to_numpy() if "source_dataset" in base else None

    scenario_frames = []
    for scenario_name, z0 in ROUGHNESS_SCENARIOS_M:
        eligible = within_screen & is_log_profile_input_valid(
            TARGET_HEIGHT_ABOVE_MODEL_BED_M, z_r, z0
        )
        scale = np.full(len(base), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale[eligible] = compute_log_profile_scale_factor(
                TARGET_HEIGHT_ABOVE_MODEL_BED_M, z_r[eligible], z0
            )
        uo_1m = np.where(eligible, scale * uo_ref, np.nan)
        vo_1m = np.where(eligible, scale * vo_ref, np.nan)
        speed_1m = np.where(eligible, np.sqrt(uo_1m**2 + vo_1m**2), np.nan)

        scenario_frames.append(
            pd.DataFrame(
                {
                    "current_node_id": node_ids,
                    "time_utc": times,
                    "reference_uo_m_s": uo_ref,
                    "reference_vo_m_s": vo_ref,
                    "reference_speed_m_s": speed_ref,
                    "reference_height_above_model_bed_m": z_r,
                    "model_bathymetry_m": h_model,
                    "z_r_over_h_model": z_r_over_h,
                    "log_profile_vertical_domain_status": domain_status,
                    "roughness_scenario": scenario_name,
                    "z0_m": z0,
                    "target_height_above_model_bed_m": TARGET_HEIGHT_ABOVE_MODEL_BED_M,
                    "log_profile_scale_factor": scale,
                    "current_only_1m_uo_m_s": uo_1m,
                    "current_only_1m_vo_m_s": vo_1m,
                    "current_only_1m_speed_m_s": speed_1m,
                    "source_dataset": source_dataset,
                    "scientific_role": SCIENTIFIC_ROLE,
                }
            )
        )

    result = pd.concat(scenario_frames, ignore_index=True)
    return result[list(CURRENT_ONLY_1M_SENSITIVITY_COLUMNS)]


# --- Vertical-domain QA summary (Section 4, 15) -------------------------------------


def compute_vertical_domain_summary(hourly_sensitivity_df: pd.DataFrame) -> dict[str, Any]:
    """Route-wide z_r/model-bathymetry/z_r-over-h stats + count outside the screen.

    De-duplicates the five-scenarios-per-row long format back down to one
    row per (node, time) first -- these fields are identical across all
    five scenario rows of the same reference sample.
    """

    empty = {
        "z_r_min": None,
        "z_r_median": None,
        "z_r_p95": None,
        "z_r_max": None,
        "model_bathymetry_m_min": None,
        "model_bathymetry_m_median": None,
        "model_bathymetry_m_max": None,
        "z_r_over_h_min": None,
        "z_r_over_h_median": None,
        "z_r_over_h_p95": None,
        "z_r_over_h_max": None,
        "rows_outside_screen": 0,
        "total_reference_rows": 0,
    }
    if hourly_sensitivity_df.empty:
        return empty

    unique_rows = hourly_sensitivity_df.drop_duplicates(subset=["current_node_id", "time_utc"])
    z_r = unique_rows["reference_height_above_model_bed_m"].dropna()
    bathymetry = unique_rows["model_bathymetry_m"].dropna()
    z_r_over_h = unique_rows["z_r_over_h_model"].dropna()
    outside_count = int(
        (unique_rows["log_profile_vertical_domain_status"] == OUTSIDE_VERTICAL_SCREEN).sum()
    )

    return {
        "z_r_min": float(z_r.min()) if len(z_r) else None,
        "z_r_median": float(z_r.median()) if len(z_r) else None,
        "z_r_p95": float(z_r.quantile(0.95)) if len(z_r) else None,
        "z_r_max": float(z_r.max()) if len(z_r) else None,
        "model_bathymetry_m_min": float(bathymetry.min()) if len(bathymetry) else None,
        "model_bathymetry_m_median": float(bathymetry.median()) if len(bathymetry) else None,
        "model_bathymetry_m_max": float(bathymetry.max()) if len(bathymetry) else None,
        "z_r_over_h_min": float(z_r_over_h.min()) if len(z_r_over_h) else None,
        "z_r_over_h_median": float(z_r_over_h.median()) if len(z_r_over_h) else None,
        "z_r_over_h_p95": float(z_r_over_h.quantile(0.95)) if len(z_r_over_h) else None,
        "z_r_over_h_max": float(z_r_over_h.max()) if len(z_r_over_h) else None,
        "rows_outside_screen": outside_count,
        "total_reference_rows": int(len(unique_rows)),
    }


# --- Node x scenario statistics + per-node sensitivity envelope (Section 7) ---------

CURRENT_ONLY_1M_SENSITIVITY_STATS_COLUMNS = (
    "current_node_id",
    "roughness_scenario",
    "z0_m",
    "valid_hour_count",
    "completeness_pct",
    "scale_factor_min",
    "scale_factor_median",
    "scale_factor_p95",
    "scale_factor_max",
    "speed_1m_mean_m_s",
    "speed_1m_median_m_s",
    "speed_1m_p90_m_s",
    "speed_1m_p95_m_s",
    "speed_1m_p99_m_s",
    "speed_1m_max_m_s",
)

CURRENT_ONLY_1M_SENSITIVITY_ENVELOPE_COLUMNS = (
    "current_node_id",
    "speed_1m_p95_sensitivity_min_m_s",
    "speed_1m_p95_sensitivity_max_m_s",
    "speed_1m_p95_sensitivity_width_m_s",
)


def _completeness_pct(valid_count: int, expected_count: int) -> float | None:
    if not expected_count:
        return None
    if valid_count > expected_count:
        raise NormalizationCompletenessError(
            f"{valid_count} valid 1 m-equivalent samples exceeds the expected regular-cadence "
            f"count of {expected_count} -- completeness must never exceed 100%"
        )
    return 100.0 * valid_count / expected_count


def compute_current_only_1m_sensitivity_stats(hourly_sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """Per (current_node_id x roughness_scenario): completeness + scale-factor + speed stats.

    No mean/midpoint "best estimate" roughness row is ever produced --
    exactly the five fixed scenarios, one row each, per node.
    """

    if hourly_sensitivity_df.empty:
        return pd.DataFrame(columns=list(CURRENT_ONLY_1M_SENSITIVITY_STATS_COLUMNS))

    records = []
    for (node_id, scenario), group in hourly_sensitivity_df.groupby(
        ["current_node_id", "roughness_scenario"]
    ):
        start = group["time_utc"].min()
        end = group["time_utc"].max()
        expected_hours = (
            int(round((end - start).total_seconds() / 3600.0)) + 1 if pd.notna(start) else 0
        )
        valid_speed = group["current_only_1m_speed_m_s"].dropna()
        valid_scale = group["log_profile_scale_factor"].dropna()
        records.append(
            {
                "current_node_id": node_id,
                "roughness_scenario": scenario,
                "z0_m": float(group["z0_m"].iloc[0]),
                "valid_hour_count": int(len(valid_speed)),
                "completeness_pct": _completeness_pct(len(valid_speed), expected_hours),
                "scale_factor_min": float(valid_scale.min()) if len(valid_scale) else None,
                "scale_factor_median": float(valid_scale.median()) if len(valid_scale) else None,
                "scale_factor_p95": float(valid_scale.quantile(0.95)) if len(valid_scale) else None,
                "scale_factor_max": float(valid_scale.max()) if len(valid_scale) else None,
                "speed_1m_mean_m_s": float(valid_speed.mean()) if len(valid_speed) else None,
                "speed_1m_median_m_s": float(valid_speed.median()) if len(valid_speed) else None,
                "speed_1m_p90_m_s": float(valid_speed.quantile(0.90)) if len(valid_speed) else None,
                "speed_1m_p95_m_s": float(valid_speed.quantile(0.95)) if len(valid_speed) else None,
                "speed_1m_p99_m_s": float(valid_speed.quantile(0.99)) if len(valid_speed) else None,
                "speed_1m_max_m_s": float(valid_speed.max()) if len(valid_speed) else None,
            }
        )
    return pd.DataFrame(records, columns=list(CURRENT_ONLY_1M_SENSITIVITY_STATS_COLUMNS))


def compute_current_only_1m_sensitivity_envelope(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Per node: min/max/width of the FIVE scenarios' own p95 speed.

    Never an average across scenarios (Section 7) -- the envelope IS the
    output.
    """

    if stats_df.empty:
        return pd.DataFrame(columns=list(CURRENT_ONLY_1M_SENSITIVITY_ENVELOPE_COLUMNS))

    records = []
    for node_id, group in stats_df.groupby("current_node_id"):
        p95_values = group["speed_1m_p95_m_s"].dropna()
        if p95_values.empty:
            records.append(
                {
                    "current_node_id": node_id,
                    "speed_1m_p95_sensitivity_min_m_s": None,
                    "speed_1m_p95_sensitivity_max_m_s": None,
                    "speed_1m_p95_sensitivity_width_m_s": None,
                }
            )
            continue
        low, high = float(p95_values.min()), float(p95_values.max())
        records.append(
            {
                "current_node_id": node_id,
                "speed_1m_p95_sensitivity_min_m_s": low,
                "speed_1m_p95_sensitivity_max_m_s": high,
                "speed_1m_p95_sensitivity_width_m_s": high - low,
            }
        )
    return pd.DataFrame(records, columns=list(CURRENT_ONLY_1M_SENSITIVITY_ENVELOPE_COLUMNS))
