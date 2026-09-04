"""Offline unit tests for marine_engine.sediment.grain_size.

Uses small synthetic PSA (particle size analysis) field dictionaries built
by hand, with analytically-known percentile answers -- never real BGS
sample records -- and never touches the network (the module under test is
pure Python with no I/O of its own).
"""

import pytest

from marine_engine.sediment import grain_size


def _five_equal_mass_bins(bin_value: float = 20.0) -> dict[str, float]:
    """Five equal-mass phi bins at phi = -1, 0, 1, 2, 3 (uniform 1.0-phi step).

    Field names follow the real BGS PHI_*/PHI_MI_* naming convention
    described in the grain_size module docstring.
    """

    return {
        "PHI_MI_1_0": bin_value,
        "PHI_0_0": bin_value,
        "PHI_1_0": bin_value,
        "PHI_2_0": bin_value,
        "PHI_3_0": bin_value,
    }


# --- phi_to_mm / mm_to_phi ----------------------------------------------------


def test_phi_to_mm_and_mm_to_phi_known_values():
    assert grain_size.phi_to_mm(0) == pytest.approx(1.0)
    assert grain_size.phi_to_mm(1) == pytest.approx(0.5)
    assert grain_size.phi_to_mm(2) == pytest.approx(0.25)

    for phi in (0.0, 1.0, 2.0):
        d_mm = grain_size.phi_to_mm(phi)
        assert grain_size.mm_to_phi(d_mm) == pytest.approx(phi)


# --- extract_phi_bins ----------------------------------------------------------


def test_extract_phi_bins_collapses_duplicate_alias_with_same_value():
    raw_properties = {"PHI_4": 5.0, "PHI_4_0": 5.0}

    bins = grain_size.extract_phi_bins(raw_properties)

    assert bins == {4.0: 5.0}
    assert len(bins) == 1


def test_extract_phi_bins_returns_none_on_conflicting_alias_values():
    raw_properties = {"PHI_4": 5.0, "PHI_4_0": 9.0}

    assert grain_size.extract_phi_bins(raw_properties) is None


def test_extract_phi_bins_parses_negative_mi_prefix():
    raw_properties = {"PHI_MI_6_5": 3.0, "PHI_MI_0_25": 1.0}

    bins = grain_size.extract_phi_bins(raw_properties)

    assert bins == {-6.5: 3.0, -0.25: 1.0}


# --- derive_grain_percentiles: units guards -------------------------------------


def test_derive_grain_percentiles_unknown_units_when_phi_units_missing():
    result = grain_size.derive_grain_percentiles(
        raw_properties=_five_equal_mass_bins(),
        phi_units=None,
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.UNKNOWN_UNITS
    assert result.d10_mm is None
    assert result.d50_mm is None
    assert result.d90_mm is None


def test_derive_grain_percentiles_unknown_units_when_phi_units_unrecognized():
    result = grain_size.derive_grain_percentiles(
        raw_properties=_five_equal_mass_bins(),
        phi_units="furlongs",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.UNKNOWN_UNITS


# --- derive_grain_percentiles: bin-count / bin-scheme guards --------------------


def test_derive_grain_percentiles_insufficient_bins_below_three():
    raw_properties = {"PHI_1_0": 10.0, "PHI_2_0": 10.0}

    result = grain_size.derive_grain_percentiles(
        raw_properties=raw_properties,
        phi_units="grams",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.INSUFFICIENT_BINS
    assert result.phi_bin_count == 2


def test_derive_grain_percentiles_ambiguous_when_bins_not_uniformly_spaced():
    # phi = 0.0, 0.5, 2.0 -- a 0.5-phi step followed by a 1.5-phi step.
    raw_properties = {"PHI_0_0": 10.0, "PHI_0_5": 10.0, "PHI_2_0": 10.0}

    result = grain_size.derive_grain_percentiles(
        raw_properties=raw_properties,
        phi_units="grams",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.AMBIGUOUS_BIN_SCHEME


# --- derive_grain_percentiles: the known synthetic distribution -----------------


def test_derive_grain_percentiles_known_synthetic_distribution():
    # 5 equal-mass bins at phi = -1, 0, 1, 2, 3 (step 1.0), 20 g each, total
    # 100 g. Cumulative mass from the coarsest bin outward: 0.2, 0.4, 0.6,
    # 0.8, 1.0 at phi = -1, 0, 1, 2, 3 respectively.
    #   D50 (target 0.50): interpolates between phi=0 (0.4) and phi=1 (0.6)
    #     -> phi=0.5 -> d50_mm = 2**-0.5
    #   D10 (target 0.90, the fine end): between phi=2 (0.8) and phi=3 (1.0)
    #     -> phi=2.5 -> d10_mm = 2**-2.5
    #   D90 (target 0.10, the coarse end): between the pre-first-bin anchor
    #     phi=-2 (0.0) and phi=-1 (0.2) -> phi=-1.5 -> d90_mm = 2**1.5
    result = grain_size.derive_grain_percentiles(
        raw_properties=_five_equal_mass_bins(20.0),
        phi_units="grams",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.DERIVED_FROM_NORMALIZED_MASS_BINS
    assert result.d10_mm < result.d50_mm < result.d90_mm
    assert result.d50_mm == pytest.approx(2.0**-0.5, rel=1e-3)
    assert result.d10_mm == pytest.approx(2.0**-2.5, rel=1e-3)
    assert result.d90_mm == pytest.approx(2.0**1.5, rel=1e-3)


def test_derive_grain_percentiles_percent_units_status():
    result = grain_size.derive_grain_percentiles(
        raw_properties=_five_equal_mass_bins(20.0),
        phi_units="percent",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.DERIVED_FROM_PERCENT_BINS


# --- derive_grain_percentiles: whole-sample coverage guard ----------------------


def test_derive_grain_percentiles_insufficient_when_gravel_fraction_has_no_phi_bins():
    """MAR-008: confirmed against a real BGS record whose populated phi bins
    covered only the sand fraction while GRAV was 29% -- the coverage guard
    must catch this rather than silently deriving a "whole sample" D50 from
    a sand-only breakdown."""

    # phi bins only in the sand range (-0.5, 0.0, 0.5, 1.0), no gravel bins.
    raw_properties = {
        "PHI_MI_0_5": 15.0,
        "PHI_0_0": 25.0,
        "PHI_0_5": 30.0,
        "PHI_1_0": 30.0,
    }

    result = grain_size.derive_grain_percentiles(
        raw_properties=raw_properties,
        phi_units="grams",
        gravel_pct=29.0,
        sand_pct=70.0,
        mud_pct=1.0,
        gsm_units="percent",
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.INSUFFICIENT_BINS


def test_derive_grain_percentiles_invalid_total_when_bins_dont_match_weight():
    # Bins sum to 50 g, but the sample's stated WEIGHT is 200 g -- the bins
    # clearly don't represent the whole sample.
    raw_properties = {"PHI_0_0": 10.0, "PHI_1_0": 20.0, "PHI_2_0": 20.0}

    result = grain_size.derive_grain_percentiles(
        raw_properties=raw_properties,
        phi_units="grams",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=200.0,
        weight_units="grams",
    )

    assert result.status == grain_size.INVALID_TOTAL


# --- derive_grain_percentiles: percent-unit whole-sample total validation (MAR-008A) --


def test_derive_grain_percentiles_percent_total_exactly_100_is_valid():
    result = grain_size.derive_grain_percentiles(
        raw_properties=_five_equal_mass_bins(20.0),  # 5 x 20 = 100%
        phi_units="percent",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.DERIVED_FROM_PERCENT_BINS
    assert result.d50_mm is not None
    assert result.phi_total_before_normalization == pytest.approx(100.0)
    assert result.normalized is True


def test_derive_grain_percentiles_percent_total_within_tolerance_is_valid():
    # 5 x 19.96 = 99.8% -- within PHI_PERCENT_TOTAL_TOLERANCE_PCT (2.0) of 100%.
    result = grain_size.derive_grain_percentiles(
        raw_properties=_five_equal_mass_bins(19.96),
        phi_units="percent",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.DERIVED_FROM_PERCENT_BINS
    assert result.d50_mm is not None
    # The ORIGINAL (not renormalized-to-100) total must remain recorded.
    assert result.phi_total_before_normalization == pytest.approx(99.8)


def test_derive_grain_percentiles_percent_total_incomplete_is_invalid():
    # 5 x 16.0 = 80% -- materially incomplete, must never be silently
    # renormalized to 100% and treated as a valid whole-sample distribution.
    result = grain_size.derive_grain_percentiles(
        raw_properties=_five_equal_mass_bins(16.0),
        phi_units="percent",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.INVALID_TOTAL
    assert result.d10_mm is None
    assert result.d50_mm is None
    assert result.d90_mm is None
    assert result.phi_total_before_normalization == pytest.approx(80.0)


def test_derive_grain_percentiles_percent_total_excess_is_invalid():
    # 5 x 24.0 = 120% -- materially excessive, same tolerance check applies
    # symmetrically above 100%, not just below it.
    result = grain_size.derive_grain_percentiles(
        raw_properties=_five_equal_mass_bins(24.0),
        phi_units="percent",
        gravel_pct=None,
        sand_pct=None,
        mud_pct=None,
        gsm_units=None,
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.INVALID_TOTAL
    assert result.d50_mm is None
    assert result.phi_total_before_normalization == pytest.approx(120.0)


def test_derive_grain_percentiles_percent_partial_fraction_trap_still_rejected():
    """The coverage guard must still dominate even when the percent bins
    themselves sum to exactly 100% -- of just the sand sub-fraction, not
    the whole sample (gravel is materially present with zero gravel bins).

    Same bins as the mass-unit partial-fraction-trap test above (they sum
    to exactly 100), but reported as PHI_UNITS=percent this time -- proving
    a "valid-looking" 100% percent total does not bypass the coverage guard.
    """

    raw_properties = {
        "PHI_MI_0_5": 15.0,
        "PHI_0_0": 25.0,
        "PHI_0_5": 30.0,
        "PHI_1_0": 30.0,
    }

    result = grain_size.derive_grain_percentiles(
        raw_properties=raw_properties,
        phi_units="percent",
        gravel_pct=29.0,
        sand_pct=70.0,
        mud_pct=1.0,
        gsm_units="percent",
        weight=None,
        weight_units=None,
    )

    assert result.status == grain_size.INSUFFICIENT_BINS
    assert result.d50_mm is None
