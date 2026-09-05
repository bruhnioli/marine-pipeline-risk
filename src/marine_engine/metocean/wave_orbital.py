"""Wave-only spectral near-bed orbital velocity (MAR-011/MAR-011A).

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

Accuracy-qualification range, NOT a validity boundary (MAR-011A correction)
-------------------------------------------------------------------------------
MAR-011 originally (and INCORRECTLY) treated `0 <= t <= 0.54` as a hard
validity boundary, nulling the canonical Urms for ~14.5% of the real
PL854 record solely because `t > 0.54`. Soulsby (2006), HR Wallingford
Report TR155, Section 3.1, actually states only that the approximation
fits the exact computed spectral value to BETTER THAN 1% for
`0 <= Tn/Tz <= 0.54`, and that orbital velocities are very small for
`Tn/Tz > 0.54` -- an ACCURACY QUALIFICATION, never "the method is invalid
above 0.54". `REPORTED_1PCT_ACCURACY_MAX_T` (0.54) is therefore surfaced
under `soulsby_smallman_accuracy_status`
(`WITHIN_REPORTED_BETTER_THAN_1PCT_ACCURACY_RANGE` /
`OUTSIDE_REPORTED_BETTER_THAN_1PCT_ACCURACY_RANGE`) -- Urms is computed
and reported for EVERY physically valid (finite Hs>=0, finite Tz>0,
finite depth>0) row regardless of `t`; only genuinely invalid Hs/Tz/depth
inputs null the canonical Urms, never `t > 0.54` alone. OUTSIDE means
"TR155 does not provide the same <1% accuracy guarantee here" -- it never
means "not computed" and never claims a quantified accuracy outside the
range.

Non-breaking assumption (Section 7 of MAR-011) -- `hs_over_model_depth` is
reported as a QA diagnostic only; this module never invents a
breaking-wave cutoff and never rejects a row on an Hs/h threshold.

Current / wave-current interaction (Section 8, 9 of MAR-011)
------------------------------------------------------------------
This module is WAVE ONLY. It never reads current data, never applies a
Doppler/current-modified dispersion correction, never computes an
apparent wave-current roughness, and never applies a directional-
spreading correction to Urms -- direction is preserved unchanged for
downstream use, never used to adjust the scalar RMS orbital velocity here.
"""

from typing import Any

import numpy as np
import pandas as pd

# --- Fixed scientific constants (do not change) -------------------------------------

GRAVITY_M_S2 = 9.80665

TZ_SOURCE = "VTM02"
EQUIVALENT_PEAK_PERIOD_FACTOR = 1.28  # Section 5: equivalent_peak_period = 1.28 * Tz

# TR155's own reported accuracy-qualification boundary (Soulsby 2006, Section
# 3.1: "fits the exact computed values to better than 1% for 0 <= Tn/Tz <=
# 0.54 ... orbital velocities are very small for Tn/Tz > 0.54"). This is an
# ACCURACY QUALIFICATION, never a validity/calibration boundary -- Urms is
# still computed and reported above it (MAR-011A).
REPORTED_1PCT_ACCURACY_MAX_T = 0.54
WITHIN_REPORTED_1PCT_ACCURACY_RANGE = "WITHIN_REPORTED_BETTER_THAN_1PCT_ACCURACY_RANGE"
OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE = "OUTSIDE_REPORTED_BETTER_THAN_1PCT_ACCURACY_RANGE"

SCIENTIFIC_ROLE = "WAVE_ONLY_SPECTRAL_NEAR_BED_ORBITAL_VELOCITY"


class OrbitalVelocityCompletenessError(Exception):
    """More valid orbital samples exist than the expected regular-cadence count allows.

    Mirrors MAR-009B's `TemporalCompletenessError`/MAR-010's
    `NormalizationCompletenessError` for this derived product: since the
    3-hourly orbital table is built strictly from the already-deduplicated
    MAR-009B canonical wave series, completeness here can never
    legitimately exceed 100% either. Unaffected by MAR-011A -- this is
    about genuinely invalid/missing input data, never about `t > 0.54`.
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

    Computed for every physically valid row regardless of `t` (MAR-011A) --
    callers pre-screen only genuine Hs/Tz/depth validity
    (`is_orbital_velocity_input_valid`), never `t <= 0.54`, before using
    this result as the canonical Urms.
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
    """Finite Hs>=0 AND finite Tz>0 AND finite model depth>0.

    This is the ONLY gate for the canonical Urms (MAR-011A) -- `t > 0.54`
    never rejects a row. `Hs == 0` is explicitly valid (a flat calm sea
    state) and must yield `Urms == 0`, never be rejected.
    """

    hs = np.asarray(hs_m, dtype=float)
    tz = np.asarray(tz_s, dtype=float)
    h = np.asarray(model_bathymetry_m, dtype=float)
    return np.isfinite(hs) & (hs >= 0) & np.isfinite(tz) & (tz > 0) & np.isfinite(h) & (h > 0)


def is_depth_and_period_valid(tz_s: np.ndarray, model_bathymetry_m: np.ndarray) -> np.ndarray:
    """Finite Tz>0 AND finite model depth>0 -- the subset of validity conditions
    that `t`/`A`/`soulsby_smallman_accuracy_status` themselves depend on.

    `h == 0` is a real edge case worth naming explicitly: `Tn = sqrt(0/g) == 0`
    is finite (not NaN), so a naive finite-only check on `t` would otherwise
    let an invalid zero depth slip through as spuriously within the
    accuracy-qualified range.
    """

    tz = np.asarray(tz_s, dtype=float)
    h = np.asarray(model_bathymetry_m, dtype=float)
    return np.isfinite(tz) & (tz > 0) & np.isfinite(h) & (h > 0)


def classify_soulsby_smallman_accuracy_status(t: np.ndarray) -> np.ndarray:
    """WITHIN/OUTSIDE TR155's own reported better-than-1%-accuracy range.

    An ACCURACY QUALIFICATION, never a validity boundary (MAR-011A): a row
    classified OUTSIDE still has a computed canonical Urms elsewhere in the
    pipeline -- this function only labels how much accuracy TR155 itself
    claims for that estimate. A non-finite `t` (e.g. from an invalid depth
    or Tz) is always OUTSIDE -- it is never silently treated as within the
    reported-accuracy range, though genuinely invalid inputs are rejected
    separately via `is_orbital_velocity_input_valid`.
    """

    t = np.asarray(t, dtype=float)
    return np.where(
        np.isfinite(t) & (t >= 0.0) & (t <= REPORTED_1PCT_ACCURACY_MAX_T),
        WITHIN_REPORTED_1PCT_ACCURACY_RANGE,
        OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE,
    )


# --- 3-hourly canonical orbital velocity table (Section 10 of MAR-011) --------------

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
    "soulsby_smallman_accuracy_status",
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
    Section 2 of MAR-011) -- this function performs NO fan-out and drops
    NO input row: every `(wave_node_id, time_utc)` pair from the input
    survives. The canonical Urms/equivalent-amplitude are null ONLY for
    genuinely invalid Hs/Tz/depth (MAR-011A) -- `t > 0.54` alone never
    nulls them, it only changes the reported `soulsby_smallman_accuracy_status`.
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
    # accuracy status can never read a spuriously "within range" t.
    t = np.where(depth_period_valid, t_raw, np.nan)
    a = compute_soulsby_smallman_a(t)
    status = classify_soulsby_smallman_accuracy_status(t)

    # MAR-011A: eligibility for the canonical Urms is genuine Hs/Tz/depth
    # validity ONLY -- never gated by `status`/`t <= 0.54`.
    eligible = is_orbital_velocity_input_valid(hs, tz, h)
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
            "soulsby_smallman_accuracy_status": status,
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


# --- Route-wide wave-input / accuracy-range QA summary ------------------------------


def compute_wave_orbital_domain_summary(hourly_df: pd.DataFrame) -> dict[str, Any]:
    """Route-wide Hs/depth/Tm02/Tp/Hs-over-depth/t stats + accuracy-range counts.

    `within_reported_1pct_accuracy_count`/`outside_reported_1pct_accuracy_count`
    are restricted to rows with a genuinely valid (non-null canonical Urms)
    input -- a row rejected for invalid Hs/Tz/depth is counted in neither
    (MAR-011A Section 4): the accuracy-range breakdown describes only the
    valid sea states, never conflated with missing/invalid data.
    """

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
        "input_valid_count": 0,
        "within_reported_1pct_accuracy_count": 0,
        "outside_reported_1pct_accuracy_count": 0,
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

    valid_mask = hourly_df["wave_orbital_velocity_rms_near_bed_m_s"].notna()
    within_mask = valid_mask & (
        hourly_df["soulsby_smallman_accuracy_status"] == WITHIN_REPORTED_1PCT_ACCURACY_RANGE
    )
    outside_mask = valid_mask & (
        hourly_df["soulsby_smallman_accuracy_status"] == OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE
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
        "input_valid_count": int(valid_mask.sum()),
        "within_reported_1pct_accuracy_count": int(within_mask.sum()),
        "outside_reported_1pct_accuracy_count": int(outside_mask.sum()),
        "total_rows": int(len(hourly_df)),
    }


# --- Per-node statistics (Section 12 of MAR-011, Sections 4-5 of MAR-011A) ----------

WAVE_ORBITAL_VELOCITY_STATS_COLUMNS = (
    "wave_node_id",
    "start_time_utc",
    "end_time_utc",
    "expected_3hour_count",
    "input_valid_count",
    "input_data_completeness_pct",
    "within_reported_1pct_accuracy_count",
    "within_reported_1pct_accuracy_pct",
    "outside_reported_1pct_accuracy_count",
    "outside_reported_1pct_accuracy_pct",
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
    "orbital_rms_mean_m_s",
    "orbital_rms_median_m_s",
    "orbital_rms_p90_m_s",
    "orbital_rms_p95_m_s",
    "orbital_rms_p99_m_s",
    "orbital_rms_max_m_s",
    "orbital_rms_p95_within_reported_1pct_accuracy_range_m_s",
    "orbital_amplitude_mean_m_s",
    "orbital_amplitude_p95_m_s",
    "orbital_amplitude_p99_m_s",
    "orbital_amplitude_max_m_s",
    "tp_observed_to_equivalent_median_ratio",
    "tp_observed_to_equivalent_p05_ratio",
    "tp_observed_to_equivalent_p95_ratio",
)


def _completeness_pct(valid_count: int, expected_count: int) -> float | None:
    """Genuine data completeness only -- never reduced by `t > 0.54` (MAR-011A Section 4)."""

    if not expected_count:
        return None
    if valid_count > expected_count:
        raise OrbitalVelocityCompletenessError(
            f"{valid_count} valid orbital samples exceeds the expected regular-cadence count "
            f"of {expected_count} -- completeness must never exceed 100%"
        )
    return 100.0 * valid_count / expected_count


def _pct_of(count: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return 100.0 * count / denominator


def compute_wave_orbital_velocity_stats(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per real route-used wave node.

    Canonical `orbital_rms_*`/`orbital_amplitude_*` statistics are computed
    over ALL genuinely valid rows (MAR-011A Section 5) -- the full
    physically valid record, never the `t <= 0.54` subset alone.
    `orbital_rms_p95_within_reported_1pct_accuracy_range_m_s` is a
    separate, explicitly-named secondary QA statistic over that subset --
    never substituted for the canonical full-record p95.
    """

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

        valid_mask = group["wave_orbital_velocity_rms_near_bed_m_s"].notna()
        within_mask = valid_mask & (
            group["soulsby_smallman_accuracy_status"] == WITHIN_REPORTED_1PCT_ACCURACY_RANGE
        )
        outside_mask = valid_mask & (
            group["soulsby_smallman_accuracy_status"] == OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE
        )
        input_valid_count = int(valid_mask.sum())
        within_count = int(within_mask.sum())
        outside_count = int(outside_mask.sum())
        within_p95 = group.loc[within_mask, "wave_orbital_velocity_rms_near_bed_m_s"]

        records.append(
            {
                "wave_node_id": node_id,
                "start_time_utc": start,
                "end_time_utc": end,
                "expected_3hour_count": expected_steps,
                "input_valid_count": input_valid_count,
                "input_data_completeness_pct": _completeness_pct(input_valid_count, expected_steps),
                "within_reported_1pct_accuracy_count": within_count,
                "within_reported_1pct_accuracy_pct": _pct_of(within_count, input_valid_count),
                "outside_reported_1pct_accuracy_count": outside_count,
                "outside_reported_1pct_accuracy_pct": _pct_of(outside_count, input_valid_count),
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
                "orbital_rms_p95_within_reported_1pct_accuracy_range_m_s": float(
                    within_p95.quantile(0.95)
                )
                if len(within_p95)
                else None,
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
