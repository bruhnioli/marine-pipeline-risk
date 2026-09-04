"""Wave direction semantics and QA (MAR-009).

Direction semantics (Section 13)
-----------------------------------
`VMDR` is a wave FROM direction (the direction waves are travelling FROM,
the standard oceanographic/meteorological convention for wave/wind
direction) -- opposite in sense to the current module's TO-direction
convention (`current.py`). The raw FROM direction is always preserved as
`wave_mean_direction_from_deg`; a TO-direction is only ever derived for
convenience as `wave_mean_direction_to_deg = (from_deg + 180) % 360`,
alongside the original, never replacing it. Never arithmetic-average
compass angles -- `circular_mean_deg` uses proper circular statistics
(mean of unit vectors) so e.g. 359 deg and 1 deg average to ~0, not 180.
"""

import numpy as np


def normalize_direction_deg(direction_deg: np.ndarray) -> np.ndarray:
    """Normalize any real-valued bearing to `[0, 360)`."""

    return np.mod(direction_deg, 360.0)


def derive_wave_direction_to_deg(wave_mean_direction_from_deg: np.ndarray) -> np.ndarray:
    """`(from_deg + 180) % 360` -- derived for convenience, the FROM value is always kept too."""

    return normalize_direction_deg(wave_mean_direction_from_deg + 180.0)


def circular_mean_deg(direction_deg: np.ndarray) -> float | None:
    """The circular mean of a set of compass bearings (degrees), via unit-vector averaging.

    Never a plain arithmetic mean: e.g. [359, 1] circularly averages to
    ~0 deg, not 180 deg. Returns None if there are no finite values to
    average (never fabricates a direction).
    """

    direction_deg = np.asarray(direction_deg, dtype=float)
    finite = direction_deg[np.isfinite(direction_deg)]
    if finite.size == 0:
        return None

    radians = np.radians(finite)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    if abs(mean_sin) <= 1e-12 and abs(mean_cos) <= 1e-12:
        return None  # perfectly cancelling directions -- no well-defined mean
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)


def validate_significant_wave_height(hs_m: np.ndarray) -> np.ndarray:
    """True where `hs_m` is a physically valid significant wave height (finite and >= 0).

    A negative Hs is a data-quality problem, never silently accepted as
    valid (Section 28) -- callers must check this mask before using Hs.
    """

    hs_m = np.asarray(hs_m, dtype=float)
    return np.isfinite(hs_m) & (hs_m >= 0.0)


def validate_wave_period_s(period_s: np.ndarray) -> np.ndarray:
    """True where `period_s` is a physically valid period (finite and > 0)."""

    period_s = np.asarray(period_s, dtype=float)
    return np.isfinite(period_s) & (period_s > 0.0)
