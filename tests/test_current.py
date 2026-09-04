"""Offline unit tests for marine_engine.metocean.current.

Uses small, hand-picked synthetic uo/vo/depth arrays with analytically
known answers -- never real 1.5 km 3D current product data -- and never
touches the network (the module under test is pure numpy with no I/O).
"""

import numpy as np
import pytest

from marine_engine.metocean import current

FORBIDDEN_NAME_SUBSTRINGS = ("bottom_current", "seabed_current", "current_at_seabed")


# --- compute_current_speed_m_s / compute_current_direction_to_deg -------------


def test_speed_and_direction_pure_eastward():
    uo = np.array([1.0])
    vo = np.array([0.0])

    speed = current.compute_current_speed_m_s(uo, vo)
    direction = current.compute_current_direction_to_deg(uo, vo)

    assert speed[0] == pytest.approx(1.0)
    assert direction[0] == pytest.approx(90.0)


def test_speed_and_direction_pure_northward():
    uo = np.array([0.0])
    vo = np.array([1.0])

    speed = current.compute_current_speed_m_s(uo, vo)
    direction = current.compute_current_direction_to_deg(uo, vo)

    assert speed[0] == pytest.approx(1.0)
    assert direction[0] == pytest.approx(0.0)


def test_speed_and_direction_pure_westward():
    uo = np.array([-1.0])
    vo = np.array([0.0])

    direction = current.compute_current_direction_to_deg(uo, vo)

    # atan2(-1, 0) is -90 deg; the % 360 modulo must wrap it into [0, 360).
    assert direction[0] == pytest.approx(270.0)
    assert 0.0 <= direction[0] < 360.0


def test_speed_and_direction_pure_southward():
    uo = np.array([0.0])
    vo = np.array([-1.0])

    direction = current.compute_current_direction_to_deg(uo, vo)

    assert direction[0] == pytest.approx(180.0)


def test_speed_zero_when_both_components_zero():
    uo = np.array([0.0])
    vo = np.array([0.0])

    speed = current.compute_current_speed_m_s(uo, vo)

    assert speed[0] == pytest.approx(0.0)


def test_speed_and_direction_work_on_arrays_not_just_scalars():
    uo = np.array([1.0, 0.0, -1.0])
    vo = np.array([0.0, 1.0, 0.0])

    speed = current.compute_current_speed_m_s(uo, vo)
    direction = current.compute_current_direction_to_deg(uo, vo)

    assert speed == pytest.approx([1.0, 1.0, 1.0])
    assert direction == pytest.approx([90.0, 0.0, 270.0])


# --- select_deepest_valid_standard_level ---------------------------------------


def test_select_deepest_valid_picks_the_max_depth_among_finite_pairs():
    depths_m = np.array([0.0, 5.0, 10.0, 25.0, 50.0])
    uo_at_depths = np.array([0.10, 0.20, 0.30, 0.40, np.nan])
    vo_at_depths = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    result = current.select_deepest_valid_standard_level(depths_m, uo_at_depths, vo_at_depths)

    assert result.depth_m == pytest.approx(25.0)
    assert result.depth_index == 3
    assert result.uo_m_s == pytest.approx(0.40)
    assert result.vo_m_s == pytest.approx(0.04)


def test_select_deepest_valid_never_selects_an_invalid_deeper_cell_when_intermediate_valid():
    depths_m = np.array([0.0, 5.0, 10.0, 25.0, 30.0])
    uo_at_depths = np.array([0.10, 0.20, 0.30, 0.40, np.nan])
    vo_at_depths = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    result = current.select_deepest_valid_standard_level(depths_m, uo_at_depths, vo_at_depths)

    # depth=30 (index 4) is the deepest cell in the array but is invalid
    # (uo is NaN there) -- the deepest VALID cell, depth=25 at index 3,
    # must win instead of the invalid deeper one.
    assert result.depth_m == pytest.approx(25.0)
    assert result.depth_index == 3
    assert result.depth_index != 4


def test_select_deepest_valid_works_with_unsorted_depths():
    # Descending order, matching the real product's own [5000, ..., 0]
    # depth ordering. The deepest two cells (50, 30) are invalid; the
    # deepest VALID cell is 25, not the array's first or last entry.
    depths_m = np.array([50.0, 30.0, 25.0, 10.0, 5.0, 0.0])
    uo_at_depths = np.array([np.nan, np.nan, 0.50, 0.30, 0.20, 0.10])
    vo_at_depths = np.array([0.05, np.nan, 0.05, 0.03, 0.02, 0.01])

    result = current.select_deepest_valid_standard_level(depths_m, uo_at_depths, vo_at_depths)

    assert result.depth_m == pytest.approx(25.0)
    assert result.depth_index == 2
    assert result.uo_m_s == pytest.approx(0.50)
    assert result.vo_m_s == pytest.approx(0.05)


def test_select_deepest_valid_returns_all_none_when_nothing_finite():
    depths_m = np.array([0.0, 5.0, 10.0])
    uo_at_depths = np.full(3, np.nan)
    vo_at_depths = np.full(3, np.nan)

    result = current.select_deepest_valid_standard_level(depths_m, uo_at_depths, vo_at_depths)

    assert result.depth_m is None
    assert result.uo_m_s is None
    assert result.vo_m_s is None
    assert result.depth_index is None


def test_select_deepest_valid_requires_both_uo_and_vo_finite():
    depths_m = np.array([0.0, 5.0, 10.0])
    uo_at_depths = np.array([0.10, 0.20, 0.30])
    vo_at_depths = np.array([0.01, 0.02, np.nan])

    result = current.select_deepest_valid_standard_level(depths_m, uo_at_depths, vo_at_depths)

    # depth=10 has a finite uo but a NaN vo, so it is excluded even though
    # it is otherwise the deepest cell -- only depth=5 has BOTH finite.
    assert result.depth_m == pytest.approx(5.0)
    assert result.depth_index == 1
    assert result.uo_m_s == pytest.approx(0.20)
    assert result.vo_m_s == pytest.approx(0.02)


# --- compute_height_above_model_bed_m ------------------------------------------


def test_height_above_model_bed_worked_example():
    height, is_valid = current.compute_height_above_model_bed_m(27.0, 25.0)

    assert height == pytest.approx(2.0)
    assert is_valid is True


def test_height_above_model_bed_invalid_when_sample_deeper_than_model_bed():
    height, is_valid = current.compute_height_above_model_bed_m(20.0, 25.0)

    assert height == pytest.approx(-5.0)
    assert is_valid is False


def test_height_above_model_bed_none_when_either_input_missing():
    height, is_valid = current.compute_height_above_model_bed_m(None, 25.0)
    assert height is None
    assert is_valid is True

    height, is_valid = current.compute_height_above_model_bed_m(20.0, None)
    assert height is None
    assert is_valid is True


def test_height_above_model_bed_valid_within_tiny_floating_tolerance():
    # Exactly zero: model bathymetry and sample depth coincide.
    height, is_valid = current.compute_height_above_model_bed_m(15.0, 15.0)
    assert height == pytest.approx(0.0, abs=1e-12)
    assert is_valid is True

    # Exactly -tolerance_m (the module's own constant): still valid --
    # the boundary is inclusive, not strictly greater-than.
    tolerance_m = current.HEIGHT_ABOVE_BED_TOLERANCE_M
    height, is_valid = current.compute_height_above_model_bed_m(0.0, tolerance_m)
    assert height == pytest.approx(-tolerance_m, abs=1e-15)
    assert is_valid is True


# --- naming regression (MAR-009 Section 8/35) ----------------------------------


def test_naming_never_uses_forbidden_bottom_current_terms():
    """The deepest-valid-standard-level current is explicitly NOT the
    model's true bottom cell, and the module docstring is strict that it
    must never be exposed under a "bottom current"/"seabed current"/
    "near-bed current" (unqualified)/"current_at_seabed" name -- the only
    canonical name is `deepest_valid_standard_level_current`.

    This checks the module's actual public API (`dir(current)`), not the
    raw source text -- the docstring itself legitimately names these
    forbidden terms in prose to explain the rule, so grepping the whole
    file would false-fail on its own explanation. Inspecting `dir()`
    guards against the real regression: a future function/constant
    accidentally named one of the forbidden terms.
    """

    public_names = [name for name in dir(current) if not name.startswith("_")]

    for forbidden in FORBIDDEN_NAME_SUBSTRINGS:
        offenders = [name for name in public_names if forbidden in name.lower()]
        assert offenders == []
