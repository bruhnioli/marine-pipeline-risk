"""Offline unit tests for marine_engine.metocean.wave.

Uses small hand-picked synthetic direction/height/period values with simple
round numbers -- never any real project metocean data -- and never touches
the network. `wave.py` is pure numpy (no I/O), so these tests exercise it
directly with in-memory scalars and arrays only.
"""

import numpy as np
import pytest

from marine_engine.metocean import wave

# --- normalize_direction_deg ---------------------------------------------


def test_normalize_direction_deg_wraps_values_above_360():
    result = wave.normalize_direction_deg(370.0)
    assert result == pytest.approx(10.0)


def test_normalize_direction_deg_wraps_negative_values():
    result = wave.normalize_direction_deg(-10.0)
    assert result == pytest.approx(350.0)


def test_normalize_direction_deg_leaves_in_range_values_unchanged():
    result = wave.normalize_direction_deg(180.0)
    assert result == pytest.approx(180.0)


# --- derive_wave_direction_to_deg (FROM is authoritative, TO is derived) -----


def test_derive_wave_direction_to_deg_adds_180():
    assert wave.derive_wave_direction_to_deg(0.0) == pytest.approx(180.0)
    assert wave.derive_wave_direction_to_deg(90.0) == pytest.approx(270.0)


def test_derive_wave_direction_to_deg_wraps_around_360():
    # 270 + 180 = 450, which wraps to 90.
    result = wave.derive_wave_direction_to_deg(270.0)
    assert result == pytest.approx(90.0)


def test_derive_wave_direction_to_deg_works_on_arrays():
    from_deg = np.array([0.0, 45.0, 90.0, 180.0, 270.0])
    expected_to_deg = [180.0, 225.0, 270.0, 0.0, 90.0]

    result = wave.derive_wave_direction_to_deg(from_deg)

    assert result.tolist() == pytest.approx(expected_to_deg)


# --- circular_mean_deg -------------------------------------------------------


def test_circular_mean_of_359_and_1_degrees_is_near_zero_not_180():
    """The single most important test in this file (Section 13).

    A plain arithmetic mean of 359 and 1 would wrongly give 180 -- proper
    circular statistics must instead put the mean near 0/360.
    """

    result = wave.circular_mean_deg(np.array([359.0, 1.0]))

    assert result is not None
    assert result < 2 or result > 358
    assert abs(result - 180) > 10


def test_circular_mean_of_identical_directions_returns_that_direction():
    result = wave.circular_mean_deg(np.array([45.0, 45.0, 45.0]))
    assert result == pytest.approx(45.0)


def test_circular_mean_ignores_nan_values():
    result = wave.circular_mean_deg(np.array([90.0, np.nan, 90.0]))
    assert result == pytest.approx(90.0)


def test_circular_mean_returns_none_when_all_nan():
    result = wave.circular_mean_deg(np.array([np.nan, np.nan]))
    assert result is None


def test_circular_mean_returns_none_for_perfectly_opposing_directions():
    # Exactly opposite directions in equal proportion cancel out -- no
    # well-defined mean exists, so None is returned rather than a fabricated
    # direction.
    result = wave.circular_mean_deg(np.array([0.0, 180.0]))
    assert result is None


# --- validate_significant_wave_height (Section 28) ---------------------------


def test_validate_significant_wave_height_accepts_zero_and_positive():
    result = wave.validate_significant_wave_height(np.array([0.0, 1.5, 3.0]))
    assert result.tolist() == [True, True, True]


def test_validate_significant_wave_height_rejects_negative():
    # A negative Hs is never valid -- it must be explicitly flagged False,
    # not silently clipped to 0 or accepted.
    result = wave.validate_significant_wave_height(np.array([-0.5, 1.0]))
    assert result.tolist() == [False, True]


def test_validate_significant_wave_height_rejects_nan_and_inf():
    result = wave.validate_significant_wave_height(np.array([np.nan, np.inf, 1.0]))
    assert result.tolist() == [False, False, True]


# --- validate_wave_period_s ---------------------------------------------------


def test_validate_wave_period_rejects_zero_and_negative():
    result = wave.validate_wave_period_s(np.array([0.0, -5.0, 8.5]))
    assert result.tolist() == [False, False, True]
