"""Wave-only spectral near-bed orbital velocity (MAR-011).

Scope -- read before touching this module
--------------------------------------------
Converts the MAR-009B canonical wave evidence (`hs_m`/`tm02_s`/`tp_s`/
`tm10_s` from Copernicus `VHM0`/`VTM02`/`VTPK`/`VTM10`) into a scalar,
wave-induced near-bed orbital-velocity forcing product using the Soulsby &
Smallman irregular-wave spectral approximation. The canonical role name
for every output of this module is
`WAVE_ONLY_SPECTRAL_NEAR_BED_ORBITAL_VELOCITY` -- never "bed current"/
"seabed current"/"combined wave-current velocity"/"bed shear stress"
anywhere in this codebase. This is wave-induced motion only, immediately
above the thin wave boundary layer near the model bed, derived from
total-spectrum wave parameters -- NOT water-particle velocity inside the
sediment, NOT combined wave-current velocity, NOT bed shear stress, NOT
pipeline loading, NOT sediment mobility.

Why VTM02 is canonical (Section 3)
--------------------------------------
`Tz` in the formula below is always `tm02_s` (Copernicus `VTM02`, the
spectral m0/m2 zero-crossing period) -- never `tp_s` (`VTPK`, preserved
only as an observed diagnostic) and never `tm10_s` (`VTM10`, preserved
only as energy-period context). Changing `VTPK`/`VTM10` alone must never
change the canonical orbital velocity.

Formula (fixed by the ticket -- do not change; do not substitute another
wave formulation or perform independent literature research here)
-----------------------------------------------------------------------------
    Tn = sqrt(h / g)                          natural scaling period
    t  = Tn / Tz                               dimensionless spectral parameter
    A  = [6500 + (0.56 + 15.54*t)^6]^(1/6)
    Urms = 0.25*Hs / [Tn * (1 + A*t^2)^3]      Soulsby & Smallman (Soulsby 2006)

`h` is the Copernicus WAVE PRODUCT's OWN static `deptho` at that same real
wave support node -- never the canonical MAR-006 EMODnet LAT depth, never
a current-product bathymetry substitute.

Calibration domain (Section 6) -- a method accuracy domain, not a
physical law
-------------------------------------------------------------------------------
The approximation is accepted for `0 <= t <= 0.54`
(`CALIBRATION_DOMAIN_MAX_T`) -- never called a universal wave-physics
threshold, always surfaced under the explicitly-named
`orbital_velocity_method_status` (`WITHIN_.../OUTSIDE_...`). A row outside
the domain (or with invalid Hs/Tz/depth) keeps its raw Hs/Tm02/Tp/depth
values and its status, but the canonical Urms/equivalent-amplitude are
null -- never a silent out-of-domain extrapolation.

Non-breaking assumption (Section 7) -- `hs_over_model_depth` is reported
as a QA diagnostic only; this module never invents a breaking-wave cutoff
and never rejects a row on an Hs/h threshold.

Current / wave-current interaction (Section 8, 9)
------------------------------------------------------
This module is WAVE ONLY. It never reads current data, never applies a
Doppler/current-modified dispersion correction, never computes an
apparent wave-current roughness, and never applies a directional-
spreading correction to Urms -- direction is preserved unchanged for
downstream use, never used to adjust the scalar RMS orbital velocity here.
"""

from typing import Any

import numpy as np
import pandas as pd

# --- Fixed scientific constants (Section 4, 6 of the ticket -- do not change) -------

GRAVITY_M_S2 = 9.80665

TZ_SOURCE = "VTM02"
EQUIVALENT_PEAK_PERIOD_FACTOR = 1.28  # Section 5: equivalent_peak_period = 1.28 * Tz

# Project method-accuracy domain (Section 6), never a universal physical
# threshold -- always surfaced under an explicitly-named status string.
CALIBRATION_DOMAIN_MAX_T = 0.54
WITHIN_CALIBRATION_DOMAIN = "WITHIN_SOULSBY_SMALLMAN_CALIBRATION_RANGE"
OUTSIDE_CALIBRATION_DOMAIN = "OUTSIDE_SOULSBY_SMALLMAN_CALIBRATION_RANGE"

SCIENTIFIC_ROLE = "WAVE_ONLY_SPECTRAL_NEAR_BED_ORBITAL_VELOCITY"


class OrbitalVelocityCompletenessError(Exception):
    """More valid orbital samples exist than the expected regular-cadence count allows.

    Mirrors MAR-009B's `TemporalCompletenessError`/MAR-010's
    `NormalizationCompletenessError` for this derived product: since the
    3-hourly orbital table is built strictly from the already-deduplicated
    MAR-009B canonical wave series, completeness here can never
    legitimately exceed 100% either.
    """


def compute_natural_scaling_period_s(
    model_bathymetry_m: np.ndarray, g: float = GRAVITY_M_S2
) -> np.ndarray:
    """Tn = sqrt(h / g). Non-positive/non-finite `h` propagates to NaN, never a crash."""

    h = np.asarray(model_bathymetry_m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(h / g)


def compute_soulsby_smallman_t_parameter(tn_s: np.ndarray, tz_s: np.ndarray) -> np.ndarray:
    """t = Tn / Tz. Non-positive/non-finite `Tz` propagates to NaN/inf, never a crash."""

    tn = np.asarray(tn_s, dtype=float)
    tz = np.asarray(tz_s, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tn / tz


def compute_soulsby_smallman_a(t: np.ndarray) -> np.ndarray:
    """A = [6500 + (0.56 + 15.54*t)^6]^(1/6)."""

    t = np.asarray(t, dtype=float)
    with np.errstate(invalid="ignore"):
        return (6500.0 + (0.56 + 15.54 * t) ** 6) ** (1.0 / 6.0)


def compute_orbital_velocity_rms_m_s(
    hs_m: np.ndarray, tn_s: np.ndarray, a: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Urms = 0.25*Hs / [Tn * (1 + A*t^2)^3] -- the Soulsby & Smallman approximation.

    Callers must pre-screen eligibility (`is_orbital_velocity_input_valid`
    + `classify_calibration_domain_status`); this function computes the
    formula only and does not itself null out-of-domain rows.
    """

    hs = np.asarray(hs_m, dtype=float)
    tn = np.asarray(tn_s, dtype=float)
    a = np.asarray(a, dtype=float)
    t = np.asarray(t, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (0.25 * hs) / (tn * (1.0 + a * t**2) ** 3)


def compute_equivalent_amplitude_m_s(urms_m_s: np.ndarray) -> np.ndarray:
    """sqrt(2) * Urms -- a downstream helper, never more canonical than Urms itself."""

    return np.sqrt(2.0) * np.asarray(urms_m_s, dtype=float)


def compute_equivalent_peak_period_from_tz_s(tz_s: np.ndarray) -> np.ndarray:
    """1.28 * Tz -- a JONSWAP-representative diagnostic, never a substitute for observed Tp."""

    return EQUIVALENT_PEAK_PERIOD_FACTOR * np.asarray(tz_s, dtype=float)


def is_orbital_velocity_input_valid(
    hs_m: np.ndarray, tz_s: np.ndarray, model_bathymetry_m: np.ndarray
) -> np.ndarray:
    """Finite Hs>=0 AND finite Tz>0 AND finite model depth>0 (Section 6).

    `Hs == 0` is explicitly valid (a flat calm sea state) and must yield
    `Urms == 0`, never be rejected.
    """

    hs = np.asarray(hs_m, dtype=float)
    tz = np.asarray(tz_s, dtype=float)
    h = np.asarray(model_bathymetry_m, dtype=float)
    return np.isfinite(hs) & (hs >= 0) & np.isfinite(tz) & (tz > 0) & np.isfinite(h) & (h > 0)


def is_depth_and_period_valid(tz_s: np.ndarray, model_bathymetry_m: np.ndarray) -> np.ndarray:
    """Finite Tz>0 AND finite model depth>0 -- the subset of Section 6's validity
    conditions that `t`/`A`/`orbital_velocity_method_status` themselves depend on.

    `h == 0` is a real edge case worth naming explicitly: `Tn = sqrt(0/g) == 0`
    is finite (not NaN), so a naive finite-only check on `t` would otherwise
    let an invalid zero depth slip through as spuriously "within domain".
    """

    tz = np.asarray(tz_s, dtype=float)
    h = np.asarray(model_bathymetry_m, dtype=float)
    return np.isfinite(tz) & (tz > 0) & np.isfinite(h) & (h > 0)


def classify_calibration_domain_status(t: np.ndarray) -> np.ndarray:
    """WITHIN/OUTSIDE the Soulsby & Smallman calibration domain (Section 6).

    A non-finite `t` (e.g. from an invalid depth or Tz) is always OUTSIDE
    -- never silently treated as within range.
    """

    t = np.asarray(t, dtype=float)
    return np.where(
        np.isfinite(t) & (t >= 0.0) & (t <= CALIBRATION_DOMAIN_MAX_T),
        WITHIN_CALIBRATION_DOMAIN,
        OUTSIDE_CALIBRATION_DOMAIN,
    )


# --- 3-hourly canonical orbital velocity table (Section 10) -------------------------

WAVE_ORBITAL_VELOCITY_COLUMNS = (
    "wave_node_id",
    "time_utc",
    "hs_m",
    "tm02_s",
    "tp_s",
    "tm10_s",
    "wave_mean_direction_from_deg",
    "wave_mean_direction_to_deg",
    "model_bathymetry_m",
    "natural_scaling_period_tn_s",
    "tz_source",
    "tz_s",
    "soulsby_smallman_t_parameter",
    "soulsby_smallman_A",
    "hs_over_model_depth",
    "orbital_velocity_method_status",
    "wave_orbital_velocity_rms_near_bed_m_s",
    "wave_orbital_velocity_equivalent_amplitude_m_s",
    "equivalent_peak_period_from_tz_s",
    "observed_to_equivalent_peak_period_ratio",
    "source_dataset",
    "scientific_role",
)


def _optional_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return df[name].to_numpy()


def build_wave_orbital_velocity_3hourly(wave_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (real wave support node, 3-hour step) -- never per chainage station.

    `wave_df` must already carry a `model_bathymetry_m` column (joined by
    the caller from the wave product's own static support-node table,
    Section 2) -- this function performs NO fan-out and drops NO input
    row: every `(wave_node_id, time_utc)` pair from the input survives,
    with canonical values null wherever inputs are invalid or the
    calibration domain does not apply (Section 6).
    """

    if wave_df.empty:
        return pd.DataFrame(columns=list(WAVE_ORBITAL_VELOCITY_COLUMNS))

    hs = wave_df["hs_m"].to_numpy(dtype=float)
    tz = wave_df["tm02_s"].to_numpy(dtype=float)
    tp = wave_df["tp_s"].to_numpy(dtype=float)
    h = wave_df["model_bathymetry_m"].to_numpy(dtype=float)

    depth_period_valid = is_depth_and_period_valid(tz, h)
    tn = compute_natural_scaling_period_s(h)
    t_raw = compute_soulsby_smallman_t_parameter(tn, tz)
    # An exactly-zero depth yields a finite (not NaN) Tn/t via plain propagation
    # -- explicitly force t to NaN wherever depth/Tz are invalid so the
    # calibration-domain status can never read a spuriously "within domain" t.
    t = np.where(depth_period_valid, t_raw, np.nan)
    a = compute_soulsby_smallman_a(t)
    status = classify_calibration_domain_status(t)

    eligible = is_orbital_velocity_input_valid(hs, tz, h) & (status == WITHIN_CALIBRATION_DOMAIN)
    urms_raw = compute_orbital_velocity_rms_m_s(hs, tn, a, t)
    urms = np.where(eligible, urms_raw, np.nan)
    amplitude = compute_equivalent_amplitude_m_s(urms)

    depth_ok = np.isfinite(h) & (h > 0) & np.isfinite(hs)
    with np.errstate(invalid="ignore", divide="ignore"):
        hs_over_model_depth = np.where(depth_ok, hs / h, np.nan)

    tz_ok = np.isfinite(tz) & (tz > 0)
    equivalent_peak_period = np.where(tz_ok, compute_equivalent_peak_period_from_tz_s(tz), np.nan)
    tp_ok = np.isfinite(tp) & (tp > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            tp_ok & np.isfinite(equivalent_peak_period), tp / equivalent_peak_period, np.nan
        )

    result = pd.DataFrame(
        {
            "wave_node_id": wave_df["wave_node_id"].to_numpy(),
            "time_utc": wave_df["time_utc"].to_numpy(),
            "hs_m": hs,
            "tm02_s": tz,
            "tp_s": tp,
            "tm10_s": _optional_column(wave_df, "tm10_s"),
            "wave_mean_direction_from_deg": _optional_column(
                wave_df, "wave_mean_direction_from_deg"
            ),
            "wave_mean_direction_to_deg": _optional_column(wave_df, "wave_mean_direction_to_deg"),
            "model_bathymetry_m": h,
            "natural_scaling_period_tn_s": tn,
            "tz_source": TZ_SOURCE,
            "tz_s": tz,
            "soulsby_smallman_t_parameter": t,
            "soulsby_smallman_A": a,
            "hs_over_model_depth": hs_over_model_depth,
            "orbital_velocity_method_status": status,
            "wave_orbital_velocity_rms_near_bed_m_s": urms,
            "wave_orbital_velocity_equivalent_amplitude_m_s": amplitude,
            "equivalent_peak_period_from_tz_s": equivalent_peak_period,
            "observed_to_equivalent_peak_period_ratio": ratio,
            "source_dataset": wave_df["source_dataset"].to_numpy()
            if "source_dataset" in wave_df.columns
            else None,
            "scientific_role": SCIENTIFIC_ROLE,
        }
    )
    return result[list(WAVE_ORBITAL_VELOCITY_COLUMNS)]


# --- Route-wide wave-input / calibration-domain QA summary (Section 21) ------------


def compute_wave_orbital_domain_summary(hourly_df: pd.DataFrame) -> dict[str, Any]:
    """Route-wide Hs/depth/Tm02/Tp/Hs-over-depth/t stats + count outside calibration domain."""

    empty = {
        "model_bathymetry_m_min": None,
        "model_bathymetry_m_median": None,
        "model_bathymetry_m_max": None,
        "hs_m_mean": None,
        "hs_m_p95": None,
        "hs_m_p99": None,
        "hs_m_max": None,
        "tm02_s_median": None,
        "tm02_s_p95": None,
        "tp_s_median": None,
        "tp_s_p95": None,
        "hs_over_model_depth_min": None,
        "hs_over_model_depth_median": None,
        "hs_over_model_depth_p95": None,
        "hs_over_model_depth_p99": None,
        "hs_over_model_depth_max": None,
        "t_parameter_min": None,
        "t_parameter_median": None,
        "t_parameter_p95": None,
        "t_parameter_max": None,
        "rows_outside_calibration_domain": 0,
        "total_rows": 0,
    }
    if hourly_df.empty:
        return empty

    bathymetry = hourly_df["model_bathymetry_m"].dropna()
    hs = hourly_df["hs_m"].dropna()
    tm02 = hourly_df["tm02_s"].dropna()
    tp = hourly_df["tp_s"].dropna()
    hs_over_depth = hourly_df["hs_over_model_depth"].dropna()
    t_values = hourly_df["soulsby_smallman_t_parameter"].dropna()
    outside_count = int(
        (hourly_df["orbital_velocity_method_status"] == OUTSIDE_CALIBRATION_DOMAIN).sum()
    )

    return {
        "model_bathymetry_m_min": float(bathymetry.min()) if len(bathymetry) else None,
        "model_bathymetry_m_median": float(bathymetry.median()) if len(bathymetry) else None,
        "model_bathymetry_m_max": float(bathymetry.max()) if len(bathymetry) else None,
        "hs_m_mean": float(hs.mean()) if len(hs) else None,
        "hs_m_p95": float(hs.quantile(0.95)) if len(hs) else None,
        "hs_m_p99": float(hs.quantile(0.99)) if len(hs) else None,
        "hs_m_max": float(hs.max()) if len(hs) else None,
        "tm02_s_median": float(tm02.median()) if len(tm02) else None,
        "tm02_s_p95": float(tm02.quantile(0.95)) if len(tm02) else None,
        "tp_s_median": float(tp.median()) if len(tp) else None,
        "tp_s_p95": float(tp.quantile(0.95)) if len(tp) else None,
        "hs_over_model_depth_min": float(hs_over_depth.min()) if len(hs_over_depth) else None,
        "hs_over_model_depth_median": float(hs_over_depth.median()) if len(hs_over_depth) else None,
        "hs_over_model_depth_p95": float(hs_over_depth.quantile(0.95))
        if len(hs_over_depth)
        else None,
        "hs_over_model_depth_p99": float(hs_over_depth.quantile(0.99))
        if len(hs_over_depth)
        else None,
        "hs_over_model_depth_max": float(hs_over_depth.max()) if len(hs_over_depth) else None,
        "t_parameter_min": float(t_values.min()) if len(t_values) else None,
        "t_parameter_median": float(t_values.median()) if len(t_values) else None,
        "t_parameter_p95": float(t_values.quantile(0.95)) if len(t_values) else None,
        "t_parameter_max": float(t_values.max()) if len(t_values) else None,
        "rows_outside_calibration_domain": outside_count,
        "total_rows": int(len(hourly_df)),
    }


# --- Per-node statistics (Section 12) ------------------------------------------------

WAVE_ORBITAL_VELOCITY_STATS_COLUMNS = (
    "wave_node_id",
    "start_time_utc",
    "end_time_utc",
    "expected_3hour_count",
    "valid_orbital_count",
    "completeness_pct",
    "model_bathymetry_m",
    "hs_mean_m",
    "hs_p95_m",
    "hs_p99_m",
    "hs_max_m",
    "tm02_median_s",
    "tm02_p95_s",
    "tp_median_s",
    "tp_p95_s",
    "t_parameter_min",
    "t_parameter_median",
    "t_parameter_p95",
    "t_parameter_max",
    "rows_outside_calibration_range",
    "orbital_rms_mean_m_s",
    "orbital_rms_median_m_s",
    "orbital_rms_p90_m_s",
    "orbital_rms_p95_m_s",
    "orbital_rms_p99_m_s",
    "orbital_rms_max_m_s",
    "orbital_amplitude_mean_m_s",
    "orbital_amplitude_p95_m_s",
    "orbital_amplitude_p99_m_s",
    "orbital_amplitude_max_m_s",
    "tp_observed_to_equivalent_median_ratio",
    "tp_observed_to_equivalent_p05_ratio",
    "tp_observed_to_equivalent_p95_ratio",
)


def _completeness_pct(valid_count: int, expected_count: int) -> float | None:
    if not expected_count:
        return None
    if valid_count > expected_count:
        raise OrbitalVelocityCompletenessError(
            f"{valid_count} valid orbital samples exceeds the expected regular-cadence count "
            f"of {expected_count} -- completeness must never exceed 100%"
        )
    return 100.0 * valid_count / expected_count


def compute_wave_orbital_velocity_stats(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per real route-used wave node: completeness, wave-input, method-QA, orbital stats."""

    if hourly_df.empty:
        return pd.DataFrame(columns=list(WAVE_ORBITAL_VELOCITY_STATS_COLUMNS))

    records = []
    for node_id, group in hourly_df.groupby("wave_node_id"):
        start = group["time_utc"].min()
        end = group["time_utc"].max()
        expected_steps = (
            int(round((end - start).total_seconds() / (3 * 3600.0))) + 1 if pd.notna(start) else 0
        )
        valid_orbital = group["wave_orbital_velocity_rms_near_bed_m_s"].dropna()
        valid_amplitude = group["wave_orbital_velocity_equivalent_amplitude_m_s"].dropna()
        valid_hs = group["hs_m"].dropna()
        valid_tm02 = group["tm02_s"].dropna()
        valid_tp = group["tp_s"].dropna()
        valid_t = group["soulsby_smallman_t_parameter"].dropna()
        valid_ratio = group["observed_to_equivalent_peak_period_ratio"].dropna()
        bathymetry_values = group["model_bathymetry_m"].dropna()
        outside_count = int(
            (group["orbital_velocity_method_status"] == OUTSIDE_CALIBRATION_DOMAIN).sum()
        )

        records.append(
            {
                "wave_node_id": node_id,
                "start_time_utc": start,
                "end_time_utc": end,
                "expected_3hour_count": expected_steps,
                "valid_orbital_count": int(len(valid_orbital)),
                "completeness_pct": _completeness_pct(len(valid_orbital), expected_steps),
                "model_bathymetry_m": float(bathymetry_values.iloc[0])
                if len(bathymetry_values)
                else None,
                "hs_mean_m": float(valid_hs.mean()) if len(valid_hs) else None,
                "hs_p95_m": float(valid_hs.quantile(0.95)) if len(valid_hs) else None,
                "hs_p99_m": float(valid_hs.quantile(0.99)) if len(valid_hs) else None,
                "hs_max_m": float(valid_hs.max()) if len(valid_hs) else None,
                "tm02_median_s": float(valid_tm02.median()) if len(valid_tm02) else None,
                "tm02_p95_s": float(valid_tm02.quantile(0.95)) if len(valid_tm02) else None,
                "tp_median_s": float(valid_tp.median()) if len(valid_tp) else None,
                "tp_p95_s": float(valid_tp.quantile(0.95)) if len(valid_tp) else None,
                "t_parameter_min": float(valid_t.min()) if len(valid_t) else None,
                "t_parameter_median": float(valid_t.median()) if len(valid_t) else None,
                "t_parameter_p95": float(valid_t.quantile(0.95)) if len(valid_t) else None,
                "t_parameter_max": float(valid_t.max()) if len(valid_t) else None,
                "rows_outside_calibration_range": outside_count,
                "orbital_rms_mean_m_s": float(valid_orbital.mean()) if len(valid_orbital) else None,
                "orbital_rms_median_m_s": float(valid_orbital.median())
                if len(valid_orbital)
                else None,
                "orbital_rms_p90_m_s": float(valid_orbital.quantile(0.90))
                if len(valid_orbital)
                else None,
                "orbital_rms_p95_m_s": float(valid_orbital.quantile(0.95))
                if len(valid_orbital)
                else None,
                "orbital_rms_p99_m_s": float(valid_orbital.quantile(0.99))
                if len(valid_orbital)
                else None,
                "orbital_rms_max_m_s": float(valid_orbital.max()) if len(valid_orbital) else None,
                "orbital_amplitude_mean_m_s": float(valid_amplitude.mean())
                if len(valid_amplitude)
                else None,
                "orbital_amplitude_p95_m_s": float(valid_amplitude.quantile(0.95))
                if len(valid_amplitude)
                else None,
                "orbital_amplitude_p99_m_s": float(valid_amplitude.quantile(0.99))
                if len(valid_amplitude)
                else None,
                "orbital_amplitude_max_m_s": float(valid_amplitude.max())
                if len(valid_amplitude)
                else None,
                "tp_observed_to_equivalent_median_ratio": float(valid_ratio.median())
                if len(valid_ratio)
                else None,
                "tp_observed_to_equivalent_p05_ratio": float(valid_ratio.quantile(0.05))
                if len(valid_ratio)
                else None,
                "tp_observed_to_equivalent_p95_ratio": float(valid_ratio.quantile(0.95))
                if len(valid_ratio)
                else None,
            }
        )
    return pd.DataFrame(records, columns=list(WAVE_ORBITAL_VELOCITY_STATS_COLUMNS))
