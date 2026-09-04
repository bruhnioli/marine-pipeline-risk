"""Grain-size percentile (D10/D50/D90) derivation from BGS PSA phi bins (MAR-008).

Scope and interpretation (mandatory reading before touching this module)
--------------------------------------------------------------------------
The BGS "Offshore samples: particle size analysis" layer exposes phi-bin
fields under two overlapping naming families for the same value -- e.g.
`PHI_4` and `PHI_4_0` are the same phi position (4.0), and `PHI_MI_6_5` is
the negative-phi twin of a positive `PHI_6_5` (`MI_` = "minus"). Summing
every populated `PHI_*`/`PHI_MI_*` field blindly double-counts any phi
position that happens to be populated under both its aliases. This module
therefore first collapses every populated field into (at most) one value
per phi position, refusing to guess when two aliases for the same position
disagree.

A second, easy-to-miss correctness risk: a PSA record's populated phi bins
may cover only PART of the whole sample -- e.g. a detailed sieve breakdown
of just the sand fraction, with the gravel and/or mud fraction reported
only as a single bulk GRAV/MUD percentage and no phi-bin detail at all
(confirmed against a real live BGS record: its populated phi bins summed to
almost exactly its SAND-fraction mass, not its total WEIGHT, while GRAV was
29%). Computing a "whole-sample" D50 from bins that only describe the sand
fraction would silently misrepresent it. `derive_grain_percentiles` guards
against this by checking that the populated bins' phi range plausibly
covers every gravel/sand/mud fraction that is not negligible, and (when
possible) that the bins' own total is consistent with the sample's stated
`WEIGHT`.

A third, separate correctness risk (MAR-008A): when `PHI_UNITS` is
"percent", the populated bins are themselves supposed to already sum to
~100% of the whole sample. A materially incomplete percentage distribution
(e.g. bins summing to 80%) must never be silently renormalized to 100% by
the cumulative-fraction math -- that would fabricate a whole-sample
percentile out of a partial one. `derive_grain_percentiles` therefore
validates the populated-bin total against 100% (within
`PHI_PERCENT_TOTAL_TOLERANCE_PCT`, a data-QA heuristic, not a physical
threshold) before deriving anything from a percent-unit distribution. This
check is independent of, and additional to, the gravel/sand/mud coverage
guard above: a percent distribution can sum to ~100% of just its own
sub-fraction (e.g. the sand fraction alone) and still be rejected by the
coverage guard for the same reason a mass distribution would be.

D10/D50/D90 semantics
----------------------
Following the standard geotechnical/sedimentology "percent finer" (percent
passing) convention: D50 is the diameter such that 50% of the sample's mass
is finer (smaller) than it, D10 the diameter such that only 10% is finer
(so D10 is the SMALL end: D10 < D50 < D90 in millimetres). Since
`phi = -log2(d_mm)`, finer material has LARGER phi, so D10 corresponds to
the LARGEST phi crossing of the three, D90 to the smallest. Percentiles are
interpolated linearly in phi space against the cumulative-mass curve built
from the coarsest bin outward (the standard "graphic interpolation" method
for grouped grain-size data), then converted back to millimetres.

Never derive D50 from Folk class, a class-lookup table, GSM percentages, or
the BGS predictive product (Section 11 of the ticket) -- this module only
ever derives it from a record's own internally consistent phi bins.
"""

import math
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

# --- Statuses -----------------------------------------------------------

DERIVED_FROM_PERCENT_BINS = "DERIVED_FROM_PERCENT_BINS"
DERIVED_FROM_NORMALIZED_MASS_BINS = "DERIVED_FROM_NORMALIZED_MASS_BINS"
INSUFFICIENT_BINS = "INSUFFICIENT_BINS"
UNKNOWN_UNITS = "UNKNOWN_UNITS"
INVALID_TOTAL = "INVALID_TOTAL"
AMBIGUOUS_BIN_SCHEME = "AMBIGUOUS_BIN_SCHEME"

# --- Scientific constants (Wentworth/Udden scale -- not project heuristics) --

GRAVEL_SAND_PHI_BOUNDARY = -1.0  # 2 mm
SAND_MUD_PHI_BOUNDARY = 4.0  # 0.0625 mm

# Project heuristic for planning only -- never a physical/statistical
# threshold: a gravel/sand/mud fraction at or below this is treated as
# negligible enough that a missing phi-bin breakdown for it does not, by
# itself, invalidate an otherwise-valid whole-sample percentile derivation.
NEGLIGIBLE_FRACTION_PCT = 1.0

# Project heuristic for planning only -- how closely the populated phi
# bins' own total must match the sample's stated WEIGHT (when both are in
# grams) to be trusted as whole-sample coverage rather than e.g. a
# sand-fraction-only sieve breakdown.
BIN_WEIGHT_TOTAL_RELATIVE_TOLERANCE = 0.05

# Project heuristic for planning only, a data-QA check -- never a
# physical/statistical threshold: how many percentage points a percent-unit
# phi-bin total may deviate from 100% before the distribution is treated as
# materially incomplete/excessive rather than trustworthy whole-sample
# coverage (MAR-008A).
PHI_PERCENT_TOTAL_TOLERANCE_PCT = 2.0

_MIN_BINS_FOR_PERCENTILES = 3

_PHI_FIELD_PATTERN = re.compile(r"^PHI_(MI_)?(\d+)(?:_(\d+))?$")
_MASS_LIKE_UNIT_NAMES = ("gram", "grams", "g")
_PERCENT_UNIT_NAMES = ("percent", "%")


@dataclass(frozen=True)
class GrainPercentileResult:
    """Everything needed to reproduce (or distrust) a percentile derivation."""

    d10_mm: float | None
    d50_mm: float | None
    d90_mm: float | None
    status: str
    phi_bin_scheme: str | None
    phi_bin_count: int
    phi_total_before_normalization: float | None
    units: str | None
    normalized: bool
    note: str = ""


def phi_to_mm(phi: float) -> float:
    """`d_mm = 2 ** (-phi)`."""

    return 2.0 ** (-phi)


def mm_to_phi(d_mm: float) -> float:
    """`phi = -log2(d_mm)`."""

    return -math.log2(d_mm)


def _parse_phi_field_name(name: str) -> float | None:
    """Parse a `PHI_*`/`PHI_MI_*` field name into its phi value, or None.

    `PHI_4` and `PHI_4_0` both parse to `4.0` (the same phi position, by
    design -- BGS's schema carries both a legacy short name and a decimal
    name for whole-phi positions); `PHI_MI_6_5` parses to `-6.5` (`MI_` =
    "minus", used for phi more negative than the finest whole-number
    boundary the short/legacy names cover).
    """

    match = _PHI_FIELD_PATTERN.match(name)
    if match is None:
        return None
    is_negative, int_part, frac_part = match.groups()
    magnitude = float(int_part)
    if frac_part:
        magnitude += float(frac_part) / (10 ** len(frac_part))
    return -magnitude if is_negative else magnitude


def extract_phi_bins(raw_properties: dict[str, Any]) -> dict[float, float] | None:
    """Collapse every populated `PHI_*`/`PHI_MI_*` field into one phi->value map.

    Returns `None` (never a guess) if two field aliases for the same phi
    position (e.g. `PHI_4` and `PHI_4_0`) are both populated and materially
    disagree -- an unresolvable internal inconsistency in the source
    record. Returns `{}` if no phi field is populated at all.
    """

    by_phi: dict[float, float] = {}
    for name, value in raw_properties.items():
        if value is None or isinstance(value, bool) or not isinstance(value, int | float):
            continue
        phi = _parse_phi_field_name(name)
        if phi is None:
            continue
        value = float(value)
        if phi in by_phi and not math.isclose(by_phi[phi], value, rel_tol=1e-9, abs_tol=1e-9):
            return None
        by_phi[phi] = value
    return by_phi


def _uniform_step(sorted_phis: list[float], tolerance: float = 1e-6) -> float | None:
    """The constant spacing between consecutive phi positions, or None if not uniform."""

    if len(sorted_phis) < 2:
        return None
    diffs = [b - a for a, b in pairwise(sorted_phis)]
    step = diffs[0]
    if step <= 0 or any(abs(d - step) > tolerance for d in diffs):
        return None
    return step


def _interpolate_phi_for_cumulative_target(
    sorted_phis: list[float], cumulative_fractions: list[float], step: float, target: float
) -> float:
    """The phi at which the ascending cumulative-mass curve crosses `target`."""

    prev_phi = sorted_phis[0] - step
    prev_cum = 0.0
    for phi, cum in zip(sorted_phis, cumulative_fractions, strict=True):
        if cum >= target:
            if math.isclose(cum, prev_cum):
                return phi
            fraction = (target - prev_cum) / (cum - prev_cum)
            return prev_phi + fraction * (phi - prev_phi)
        prev_phi, prev_cum = phi, cum
    return sorted_phis[-1]


def derive_grain_percentiles(
    *,
    raw_properties: dict[str, Any],
    phi_units: str | None,
    gravel_pct: float | None,
    sand_pct: float | None,
    mud_pct: float | None,
    gsm_units: str | None,
    weight: float | None,
    weight_units: str | None,
) -> GrainPercentileResult:
    """Derive D10/D50/D90 from one PSA record's own phi bins, or explain why not.

    Every branch that refuses to compute a percentile sets a specific
    `status` (see the module-level status constants) and a human-readable
    `note` -- callers must never treat a null result as "no data", only as
    "not safely derivable this way, for this stated reason".
    """

    def _null(
        status: str, note: str, bin_count: int = 0, total: float | None = None
    ) -> GrainPercentileResult:
        return GrainPercentileResult(
            d10_mm=None,
            d50_mm=None,
            d90_mm=None,
            status=status,
            phi_bin_scheme=None,
            phi_bin_count=bin_count,
            phi_total_before_normalization=total,
            units=phi_units,
            normalized=False,
            note=note,
        )

    if not phi_units or not phi_units.strip():
        return _null(UNKNOWN_UNITS, "PHI_UNITS not reported for this record.")

    units_normalized = phi_units.strip().lower()
    is_percent = units_normalized in _PERCENT_UNIT_NAMES
    is_mass = units_normalized in _MASS_LIKE_UNIT_NAMES
    if not is_percent and not is_mass:
        return _null(UNKNOWN_UNITS, f"Unrecognized PHI_UNITS={phi_units!r}; refusing to guess.")

    by_phi = extract_phi_bins(raw_properties)
    if by_phi is None:
        return _null(
            AMBIGUOUS_BIN_SCHEME,
            "Two field aliases for the same phi position (e.g. PHI_4 vs PHI_4_0) report "
            "materially different values; refusing to guess which is correct.",
        )
    if len(by_phi) < _MIN_BINS_FOR_PERCENTILES:
        return _null(
            INSUFFICIENT_BINS,
            f"Only {len(by_phi)} populated phi bin(s); at least {_MIN_BINS_FOR_PERCENTILES} "
            "are needed to interpolate a percentile.",
            bin_count=len(by_phi),
        )

    sorted_phis = sorted(by_phi)
    step = _uniform_step(sorted_phis)
    if step is None:
        return _null(
            AMBIGUOUS_BIN_SCHEME,
            "Populated phi bins are not spaced at a single uniform phi step; this is the "
            "signature of overlapping/mixed bin-scheme resolutions being combined.",
            bin_count=len(sorted_phis),
        )

    total = sum(by_phi[phi] for phi in sorted_phis)
    if total <= 0:
        return _null(
            INVALID_TOTAL,
            "Populated phi bins sum to zero or less.",
            bin_count=len(sorted_phis),
            total=total,
        )

    # Whole-sample coverage guard (see module docstring): a materially
    # present gravel/sand/mud fraction with zero phi bins in its
    # Wentworth/Udden phi sub-range means the populated bins describe only
    # PART of the sample.
    if gsm_units and gsm_units.strip().lower() in _PERCENT_UNIT_NAMES:
        has_gravel_bins = any(phi < GRAVEL_SAND_PHI_BOUNDARY for phi in sorted_phis)
        has_sand_bins = any(
            GRAVEL_SAND_PHI_BOUNDARY <= phi < SAND_MUD_PHI_BOUNDARY for phi in sorted_phis
        )
        has_mud_bins = any(phi >= SAND_MUD_PHI_BOUNDARY for phi in sorted_phis)
        missing = [
            f"{label} ({pct:.1f}%)"
            for label, pct, has_bins in (
                ("gravel", gravel_pct, has_gravel_bins),
                ("sand", sand_pct, has_sand_bins),
                ("mud", mud_pct, has_mud_bins),
            )
            if pct is not None and pct > NEGLIGIBLE_FRACTION_PCT and not has_bins
        ]
        if missing:
            return _null(
                INSUFFICIENT_BINS,
                "Populated phi bins do not cover a materially present fraction of the "
                f"whole sample ({', '.join(missing)} has no phi-bin breakdown at all; "
                f"project heuristic: negligible threshold {NEGLIGIBLE_FRACTION_PCT}%); "
                "computing a percentile from a partial fraction only would misrepresent "
                "it as a whole-sample value.",
                bin_count=len(sorted_phis),
                total=total,
            )

    # Percent-unit bins must themselves sum to ~100% of the whole sample --
    # never silently renormalized when materially incomplete/excessive
    # (MAR-008A). Checked in addition to, not instead of, the coverage guard
    # above: a sand-only percent distribution that happens to sum to ~100%
    # of just the sand sub-fraction is still caught by that guard, not this
    # one, when gravel/mud are materially present.
    if is_percent:
        percent_total_diff = abs(total - 100.0)
        if percent_total_diff > PHI_PERCENT_TOTAL_TOLERANCE_PCT:
            return _null(
                INVALID_TOTAL,
                f"Populated phi bins sum to {total:.2f}% (PHI_UNITS=percent), which is "
                f"{percent_total_diff:.2f} percentage points from 100% -- exceeding the "
                f"{PHI_PERCENT_TOTAL_TOLERANCE_PCT:g}-percentage-point project data-QA "
                "tolerance for planning only (not a physical threshold); the bins likely "
                "do not represent the whole sample and must not be silently renormalized.",
                bin_count=len(sorted_phis),
                total=total,
            )

    if (
        is_mass
        and weight is not None
        and weight > 0
        and weight_units
        and weight_units.strip().lower() in _MASS_LIKE_UNIT_NAMES
    ):
        relative_diff = abs(total - weight) / weight
        if relative_diff > BIN_WEIGHT_TOTAL_RELATIVE_TOLERANCE:
            return _null(
                INVALID_TOTAL,
                f"Populated phi bins sum to {total:.2f} but WEIGHT={weight:.2f} "
                f"{weight_units} (relative difference {relative_diff:.1%} exceeds the "
                f"{BIN_WEIGHT_TOTAL_RELATIVE_TOLERANCE:.0%} project-heuristic tolerance "
                "for planning only); the bins likely do not represent the whole sample.",
                bin_count=len(sorted_phis),
                total=total,
            )

    cumulative_fractions = []
    running = 0.0
    for phi in sorted_phis:
        running += by_phi[phi]
        cumulative_fractions.append(running / total)

    # D10 is the SMALL end (10% of the sample is finer than it), which is
    # the LARGEST phi crossing of the three -- see module docstring.
    phi_d10 = _interpolate_phi_for_cumulative_target(sorted_phis, cumulative_fractions, step, 0.90)
    phi_d50 = _interpolate_phi_for_cumulative_target(sorted_phis, cumulative_fractions, step, 0.50)
    phi_d90 = _interpolate_phi_for_cumulative_target(sorted_phis, cumulative_fractions, step, 0.10)

    return GrainPercentileResult(
        d10_mm=phi_to_mm(phi_d10),
        d50_mm=phi_to_mm(phi_d50),
        d90_mm=phi_to_mm(phi_d90),
        status=DERIVED_FROM_PERCENT_BINS if is_percent else DERIVED_FROM_NORMALIZED_MASS_BINS,
        phi_bin_scheme=f"uniform {step:g}-phi steps, {sorted_phis[0]:g}..{sorted_phis[-1]:g}",
        phi_bin_count=len(sorted_phis),
        phi_total_before_normalization=total,
        units=phi_units,
        normalized=True,
        note="",
    )
