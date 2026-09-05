"""Current vector semantics and deepest-valid-standard-level selection (MAR-009/MAR-009A).

Vector semantics (Section 10)
-------------------------------
`uo` (eastward) and `vo` (northward) give `current_speed_m_s = sqrt(uo^2 +
vo^2)` and an optional `current_direction_to_deg`: degrees clockwise from
true north, where the vector points TOWARD that bearing (0 = north, 90 =
east) -- the word "to" is always explicit in the name to avoid confusion
with the wave module's FROM-direction convention (`wave.py`). Never called
plain "direction".

Deepest valid standard level, and why finite alone is not enough (MAR-009A)
-----------------------------------------------------------------------------
The 1.5 km 3D current product is distributed on standard geopotential
depths (surface, 3, 5, 10, ... m), not the model's native terrain-following
bottom cell. The REAL PL854 acquisition (MAR-009) confirmed that treating
"`uo`/`vo` both finite" as sufficient is not safe: at every one of the 14
real support nodes, standard levels well below that cell's own
`model_bathymetry_m` (e.g. depths of 60-75 m against a bathymetry of
~23-30 m) still carried finite `uo`/`vo` values -- inspection showed those
values IDENTICALLY REPEATED across several of the deepest standard levels
(a "held/padded" fill pattern below the point where the data stops varying
meaningfully), not genuine distinct measurements. A depth candidate is
therefore only ELIGIBLE when ALL applicable conditions hold:

1. `uo` is finite
2. `vo` is finite
3. the standard depth is physically within the model's own water column:
   `depth_m <= model_bathymetry_m + tolerance`
4. where the static 3D `mask` can be unambiguously aligned to the same
   standard-depth coordinate, the corresponding cell must be wet

Condition 3 uses the SAME numerical tolerance as
`compute_height_above_model_bed_m`'s own validity check
(`HEIGHT_ABOVE_BED_TOLERANCE_M`), so a depth that passes selection can
never simultaneously fail the height-above-bed validity check downstream --
the two are two views of the same physical constraint and must never
disagree. `model_bathymetry_m` is a REQUIRED keyword-only argument (never
defaulted away) precisely so a caller cannot accidentally revert to the
old finite-only behaviour by omitting it.

`select_deepest_valid_standard_level` finds, per (node, time), the deepest
ELIGIBLE standard depth -- this is explicitly NOT the same thing as the
model's true bottom cell, and must never be called `bottom_current`/
`seabed_current`/`near-bed current` (unqualified)/`current_at_seabed`
anywhere in this codebase. The only canonical name is
`deepest_valid_standard_level_current`.

Every candidate that is finite but excluded specifically by condition 3
(below the model's own bathymetry) is counted in `DeepestValidLevel`'s
`below_bed_finite_candidate_count`/`max_below_bed_candidate_depth_m` --
QA diagnostics only, never folded into the canonical forcing statistics.
"""

from dataclasses import dataclass

import numpy as np

# A numerical (floating-point representation) tolerance only -- never a
# physical threshold. `height_above_model_bed_m` must be >= 0 within this
# tolerance; a materially negative value (the "sample" deeper than the
# model's own seafloor) is flagged, not silently accepted (Section 8/28).
# The SAME constant gates the water-column eligibility condition in
# `select_deepest_valid_standard_level`, so the two checks can never
# disagree with each other (MAR-009A).
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
    """One (node, time) pair's deepest-PHYSICALLY-ELIGIBLE-standard-level current sample.

    `below_bed_finite_candidate_count`/`max_below_bed_candidate_depth_m`
    describe candidates that were finite but excluded specifically for
    being deeper than the model's own bathymetry -- QA diagnostics only,
    never part of the canonical selection itself.
    """

    depth_m: float | None
    uo_m_s: float | None
    vo_m_s: float | None
    depth_index: int | None
    below_bed_finite_candidate_count: int = 0
    max_below_bed_candidate_depth_m: float | None = None


def select_deepest_valid_standard_level(
    depths_m: np.ndarray,
    uo_at_depths: np.ndarray,
    vo_at_depths: np.ndarray,
    *,
    model_bathymetry_m: float | None,
    mask_at_depths: np.ndarray | None = None,
    bathymetry_tolerance_m: float = HEIGHT_ABOVE_BED_TOLERANCE_M,
) -> DeepestValidLevel:
    """The deepest standard depth (any order) satisfying every applicable eligibility condition.

    Never selects an invalid deeper cell: computes the full eligibility
    mask (finite AND within the model's own water column AND, where
    alignable, mask-wet) FIRST, then picks the maximum depth value among
    only the eligible ones.

    `model_bathymetry_m=None` means no water-column constraint is
    available for this node (never silently assumed -- the caller must
    pass `None` deliberately, not omit the argument). `mask_at_depths=None`
    means the static mask could not be unambiguously aligned to
    `depths_m` for this product (Section 3); the water-column depth
    constraint alone is still applied.
    """

    finite = np.isfinite(uo_at_depths) & np.isfinite(vo_at_depths)

    if model_bathymetry_m is not None:
        within_water_column = depths_m <= (model_bathymetry_m + bathymetry_tolerance_m)
    else:
        within_water_column = np.ones_like(depths_m, dtype=bool)

    below_bed_finite = finite & ~within_water_column
    below_bed_finite_candidate_count = int(np.count_nonzero(below_bed_finite))
    max_below_bed_candidate_depth_m = (
        float(np.max(depths_m[below_bed_finite])) if below_bed_finite_candidate_count else None
    )

    if mask_at_depths is not None:
        mask_wet = np.isfinite(mask_at_depths) & (np.asarray(mask_at_depths, dtype=float) > 0.5)
    else:
        mask_wet = np.ones_like(depths_m, dtype=bool)

    eligible = finite & within_water_column & mask_wet
    if not np.any(eligible):
        return DeepestValidLevel(
            None,
            None,
            None,
            None,
            below_bed_finite_candidate_count=below_bed_finite_candidate_count,
            max_below_bed_candidate_depth_m=max_below_bed_candidate_depth_m,
        )

    eligible_indices = np.flatnonzero(eligible)
    deepest_index = int(eligible_indices[np.argmax(depths_m[eligible_indices])])
    return DeepestValidLevel(
        depth_m=float(depths_m[deepest_index]),
        uo_m_s=float(uo_at_depths[deepest_index]),
        vo_m_s=float(vo_at_depths[deepest_index]),
        depth_index=deepest_index,
        below_bed_finite_candidate_count=below_bed_finite_candidate_count,
        max_below_bed_candidate_depth_m=max_below_bed_candidate_depth_m,
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
    is materially deeper than the model's own seafloor. Given the shared
    tolerance with `select_deepest_valid_standard_level`'s own water-column
    condition, a depth selected as eligible will always report a
    non-negative (valid) height here -- the two must never disagree.
    """

    if model_bathymetry_m is None or current_sample_depth_m is None:
        return None, True

    height = model_bathymetry_m - current_sample_depth_m
    is_valid = height >= -tolerance_m
    return height, is_valid
