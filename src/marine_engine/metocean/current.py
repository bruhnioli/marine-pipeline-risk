"""Current vector semantics and deepest-valid-standard-level selection (MAR-009).

Vector semantics (Section 10)
-------------------------------
`uo` (eastward) and `vo` (northward) give `current_speed_m_s = sqrt(uo^2 +
vo^2)` and an optional `current_direction_to_deg`: degrees clockwise from
true north, where the vector points TOWARD that bearing (0 = north, 90 =
east) -- the word "to" is always explicit in the name to avoid confusion
with the wave module's FROM-direction convention (`wave.py`). Never called
plain "direction".

Deepest valid standard level (Section 8)
-------------------------------------------
The 1.5 km 3D current product is distributed on standard geopotential
depths (surface, 3, 5, 10, ... m), not the model's native terrain-following
bottom cell. `select_deepest_valid_standard_level` finds, per (node, time),
the deepest standard depth with BOTH `uo` and `vo` finite -- this is
explicitly NOT the same thing as the model's true bottom cell, and must
never be called `bottom_current`/`seabed_current`/`near-bed current`
(unqualified)/`current_at_seabed` anywhere in this codebase. The only
canonical name is `deepest_valid_standard_level_current`.
"""

from dataclasses import dataclass

import numpy as np

# A numerical (floating-point representation) tolerance only -- never a
# physical threshold. `height_above_model_bed_m` must be >= 0 within this
# tolerance; a materially negative value (the "sample" deeper than the
# model's own seafloor) is flagged, not silently accepted (Section 8/28).
HEIGHT_ABOVE_BED_TOLERANCE_M = 1e-6


def compute_current_speed_m_s(uo: np.ndarray, vo: np.ndarray) -> np.ndarray:
    """`sqrt(uo^2 + vo^2)`."""

    return np.sqrt(np.square(uo) + np.square(vo))


def compute_current_direction_to_deg(uo: np.ndarray, vo: np.ndarray) -> np.ndarray:
    """Degrees clockwise from true north that the current vector points TOWARD.

    0 = north, 90 = east. `uo=1, vo=0` (pure eastward) -> 90;
    `uo=0, vo=1` (pure northward) -> 0.
    """

    return np.degrees(np.arctan2(uo, vo)) % 360.0


@dataclass(frozen=True)
class DeepestValidLevel:
    """One (node, time) pair's deepest-valid-standard-level current sample."""

    depth_m: float | None
    uo_m_s: float | None
    vo_m_s: float | None
    depth_index: int | None


def select_deepest_valid_standard_level(
    depths_m: np.ndarray, uo_at_depths: np.ndarray, vo_at_depths: np.ndarray
) -> DeepestValidLevel:
    """The deepest standard depth (any order) with both `uo`/`vo` finite.

    Never selects an invalid deeper cell: filters to finite (uo, vo) pairs
    FIRST, then picks the maximum depth value among only those -- an
    invalid deeper level is simply excluded from consideration, never
    accidentally preferred over a shallower valid one.
    """

    valid = np.isfinite(uo_at_depths) & np.isfinite(vo_at_depths)
    if not np.any(valid):
        return DeepestValidLevel(None, None, None, None)

    valid_indices = np.flatnonzero(valid)
    deepest_index = int(valid_indices[np.argmax(depths_m[valid_indices])])
    return DeepestValidLevel(
        depth_m=float(depths_m[deepest_index]),
        uo_m_s=float(uo_at_depths[deepest_index]),
        vo_m_s=float(vo_at_depths[deepest_index]),
        depth_index=deepest_index,
    )


def compute_height_above_model_bed_m(
    model_bathymetry_m: float | None,
    current_sample_depth_m: float | None,
    *,
    tolerance_m: float = HEIGHT_ABOVE_BED_TOLERANCE_M,
) -> tuple[float | None, bool]:
    """`model_bathymetry_m - current_sample_depth_m`, and whether it is valid (>= 0).

    Only computed when both inputs are valid in the model's own vertical
    reference (Section 8) -- returns `(None, True)` when either input is
    missing (nothing to flag), and `(height, False)` when the sample
    is materially deeper than the model's own seafloor.
    """

    if model_bathymetry_m is None or current_sample_depth_m is None:
        return None, True

    height = model_bathymetry_m - current_sample_depth_m
    is_valid = height >= -tolerance_m
    return height, is_valid
