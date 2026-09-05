"""Offline unit tests for marine_engine.metocean.current_normalization (MAR-010).

Small hand-built synthetic DataFrames only -- never the real PL854 route or
real Copernicus data, never network access.
"""

import numpy as np
import pandas as pd
import pytest

from marine_engine.metocean import current_normalization as norm

# A small, physically arbitrary but strictly positive roughness for scale-factor
# tests that don't care about the specific fixed scenario values.
Z0_GENERIC = 1e-4


def _make_primary_current_row(
    *,
    current_node_id: str = "current_0000_0000",
    time_utc: pd.Timestamp | None = None,
    uo_m_s: float = 0.3,
    vo_m_s: float = 0.4,
    current_sample_depth_m: float | None = 20.0,
    model_bathymetry_m: float = 25.0,
    height_above_model_bed_m: float | None = 5.0,
    height_above_model_bed_valid: bool = True,
    source_dataset: str = "TEST_DATASET",
) -> dict:
    return {
        "current_node_id": current_node_id,
        "time_utc": time_utc or pd.Timestamp("2025-01-01T00:00", tz="UTC"),
        "uo_m_s": uo_m_s,
        "vo_m_s": vo_m_s,
        "current_speed_m_s": float(np.hypot(uo_m_s, vo_m_s)) if uo_m_s is not None else None,
        "current_sample_depth_m": current_sample_depth_m,
        "model_bathymetry_m": model_bathymetry_m,
        "height_above_model_bed_m": height_above_model_bed_m,
        "height_above_model_bed_valid": height_above_model_bed_valid,
        "source_dataset": source_dataset,
    }


# --- compute_log_profile_scale_factor (Section 2, 17-A/B/C) -------------------------


def test_scale_factor_is_exactly_one_when_target_equals_reference():
    """17-A: z_target == z_reference -> scale factor exactly 1."""

    z = 3.0
    scale = norm.compute_log_profile_scale_factor(z, z, Z0_GENERIC)
    assert scale == pytest.approx(1.0, abs=0.0)


def test_scale_factor_below_one_when_target_below_reference():
    """17-B: z_target < z_reference -> scale factor < 1."""

    scale = norm.compute_log_profile_scale_factor(0.5, 5.0, Z0_GENERIC)
    assert scale < 1.0


def test_scale_factor_above_one_when_target_above_reference():
    """17-C: z_target > z_reference -> scale factor > 1."""

    scale = norm.compute_log_profile_scale_factor(5.0, 0.5, Z0_GENERIC)
    assert scale > 1.0


def test_scale_factor_varies_with_roughness():
    """A rougher z0 changes the ratio -- roughness is a real sensitivity dimension."""

    smooth = norm.compute_log_profile_scale_factor(1.0, 3.0, 5e-6)
    rough = norm.compute_log_profile_scale_factor(1.0, 3.0, 3e-4)
    assert smooth != pytest.approx(rough)


# --- is_log_profile_input_valid / classify_vertical_domain_status (17-F/G) ----------


def test_is_log_profile_input_valid_true_for_well_formed_inputs():
    result = norm.is_log_profile_input_valid(1.0, np.array([2.0, 5.0]), Z0_GENERIC)
    assert result.tolist() == [True, True]


def test_is_log_profile_input_valid_rejects_non_positive_z_r():
    """17-F: z_r <= 0 (or NaN) is rejected, never silently accepted."""

    result = norm.is_log_profile_input_valid(1.0, np.array([0.0, -1.0, np.nan, 2.0]), Z0_GENERIC)
    assert result.tolist() == [False, False, False, True]


def test_is_log_profile_input_valid_rejects_non_positive_z0_and_z_t():
    """17-F: a contrived non-positive z0 or z_t must also be rejected."""

    assert not norm.is_log_profile_input_valid(1.0, np.array([2.0]), 0.0)[0]
    assert not norm.is_log_profile_input_valid(1.0, np.array([2.0]), -1e-4)[0]
    assert not norm.is_log_profile_input_valid(0.0, np.array([2.0]), Z0_GENERIC)[0]


def test_classify_vertical_domain_status_within_and_outside():
    status = norm.classify_vertical_domain_status(np.array([0.10, 0.30, 0.31, np.nan]))
    assert status.tolist() == [
        norm.WITHIN_VERTICAL_SCREEN,
        norm.WITHIN_VERTICAL_SCREEN,
        norm.OUTSIDE_VERTICAL_SCREEN,
        norm.OUTSIDE_VERTICAL_SCREEN,
    ]


# --- build_current_only_1m_sensitivity_hourly ---------------------------------------


def test_hourly_sensitivity_row_count_is_reference_rows_times_five_scenarios():
    """17-J: exactly (reference rows) x 5 -- never duplicated across chainage stations
    (this function never even sees chainage stations, only real support nodes)."""

    rows = [
        _make_primary_current_row(
            current_node_id="A", time_utc=pd.Timestamp(f"2025-01-01T0{h}:00", tz="UTC")
        )
        for h in range(3)
    ] + [
        _make_primary_current_row(
            current_node_id="B", time_utc=pd.Timestamp(f"2025-01-01T0{h}:00", tz="UTC")
        )
        for h in range(3)
    ]
    df = pd.DataFrame(rows)

    result = norm.build_current_only_1m_sensitivity_hourly(df)

    assert len(result) == 6 * 5
    assert set(result["current_node_id"].unique()) == {"A", "B"}


def test_hourly_sensitivity_emits_exactly_five_scenarios_no_sixth():
    """17-H: exactly the five fixed scenarios, no implicit sixth "canonical" one."""

    df = pd.DataFrame([_make_primary_current_row()])

    result = norm.build_current_only_1m_sensitivity_hourly(df)

    assert set(result["roughness_scenario"]) == {name for name, _ in norm.ROUGHNESS_SCENARIOS_M}
    assert len(result) == 5
    assert result["roughness_scenario"].nunique() == 5


def test_hourly_sensitivity_skips_rows_with_no_reference_sample():
    """A timestamp with no MAR-009A/B canonical reference sample contributes nothing."""

    df = pd.DataFrame(
        [
            _make_primary_current_row(current_sample_depth_m=None, height_above_model_bed_m=None),
        ]
    )

    result = norm.build_current_only_1m_sensitivity_hourly(df)

    assert result.empty


def test_hourly_sensitivity_direction_preserved():
    """17-D: scaling both components by the same positive scalar preserves direction."""

    df = pd.DataFrame([_make_primary_current_row(uo_m_s=0.6, vo_m_s=-0.8)])

    result = norm.build_current_only_1m_sensitivity_hourly(df)

    reference_angle = np.degrees(np.arctan2(0.6, -0.8)) % 360.0
    for _, row in result.iterrows():
        if pd.isna(row["current_only_1m_uo_m_s"]):
            continue
        scaled_angle = (
            np.degrees(np.arctan2(row["current_only_1m_uo_m_s"], row["current_only_1m_vo_m_s"]))
            % 360.0
        )
        assert scaled_angle == pytest.approx(reference_angle, abs=1e-9)


def test_hourly_sensitivity_zero_vector_remains_zero():
    """17-E: a zero reference vector normalizes to a zero vector regardless of scale factor."""

    df = pd.DataFrame([_make_primary_current_row(uo_m_s=0.0, vo_m_s=0.0)])

    result = norm.build_current_only_1m_sensitivity_hourly(df)

    assert (result["current_only_1m_uo_m_s"] == 0.0).all()
    assert (result["current_only_1m_vo_m_s"] == 0.0).all()
    assert (result["current_only_1m_speed_m_s"] == 0.0).all()


def test_hourly_sensitivity_rejects_non_positive_reference_height():
    """17-F: height_above_model_bed_m <= 0 -> normalized values null, never a division by zero."""

    df = pd.DataFrame(
        [_make_primary_current_row(height_above_model_bed_m=0.0, model_bathymetry_m=25.0)]
    )

    result = norm.build_current_only_1m_sensitivity_hourly(df)

    assert result["current_only_1m_speed_m_s"].isna().all()
    assert result["log_profile_scale_factor"].isna().all()


def test_hourly_sensitivity_outside_vertical_screen_nulls_normalized_values():
    """17-G: z_r/h_model > 0.30 -> no normalized value, explicit OUTSIDE status,
    row is NOT silently dropped."""

    df = pd.DataFrame(
        [
            _make_primary_current_row(
                current_node_id="OUTSIDE",
                height_above_model_bed_m=10.0,
                model_bathymetry_m=20.0,  # ratio 0.50 > 0.30
            ),
            _make_primary_current_row(
                current_node_id="WITHIN",
                height_above_model_bed_m=2.0,
                model_bathymetry_m=20.0,  # ratio 0.10 <= 0.30
            ),
        ]
    )

    result = norm.build_current_only_1m_sensitivity_hourly(df)

    outside_rows = result[result["current_node_id"] == "OUTSIDE"]
    within_rows = result[result["current_node_id"] == "WITHIN"]
    assert len(outside_rows) == 5  # still present, never dropped
    assert (
        outside_rows["log_profile_vertical_domain_status"] == norm.OUTSIDE_VERTICAL_SCREEN
    ).all()
    assert outside_rows["current_only_1m_speed_m_s"].isna().all()
    assert (within_rows["log_profile_vertical_domain_status"] == norm.WITHIN_VERTICAL_SCREEN).all()
    assert within_rows["current_only_1m_speed_m_s"].notna().all()


def test_hourly_sensitivity_column_schema_matches_constant():
    df = pd.DataFrame([_make_primary_current_row()])
    result = norm.build_current_only_1m_sensitivity_hourly(df)
    assert list(result.columns) == list(norm.CURRENT_ONLY_1M_SENSITIVITY_COLUMNS)


def test_hourly_sensitivity_scientific_role_is_current_only():
    df = pd.DataFrame([_make_primary_current_row()])
    result = norm.build_current_only_1m_sensitivity_hourly(df)
    assert (result["scientific_role"] == "CURRENT_ONLY_LOG_PROFILE_SENSITIVITY").all()


def test_hourly_sensitivity_empty_input():
    result = norm.build_current_only_1m_sensitivity_hourly(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == list(norm.CURRENT_ONLY_1M_SENSITIVITY_COLUMNS)


# --- compute_vertical_domain_summary -------------------------------------------------


def test_compute_vertical_domain_summary_dedupes_across_scenarios():
    df = pd.DataFrame(
        [
            _make_primary_current_row(
                current_node_id="A", height_above_model_bed_m=2.0, model_bathymetry_m=20.0
            ),
            _make_primary_current_row(
                current_node_id="B", height_above_model_bed_m=8.0, model_bathymetry_m=20.0
            ),
        ]
    )
    hourly = norm.build_current_only_1m_sensitivity_hourly(df)

    summary = norm.compute_vertical_domain_summary(hourly)

    # 2 real reference rows, not 2*5 -- the dedup by (node,time) must collapse scenarios.
    assert summary["total_reference_rows"] == 2
    assert summary["z_r_min"] == pytest.approx(2.0)
    assert summary["z_r_max"] == pytest.approx(8.0)
    assert summary["rows_outside_screen"] == 1  # node B: 8/20 = 0.40 > 0.30


def test_compute_vertical_domain_summary_empty_input():
    summary = norm.compute_vertical_domain_summary(pd.DataFrame())
    assert summary["total_reference_rows"] == 0
    assert summary["rows_outside_screen"] == 0
    assert summary["z_r_min"] is None


# --- compute_current_only_1m_sensitivity_stats + completeness -----------------------


def test_sensitivity_stats_completeness_and_percentiles():
    times = [pd.Timestamp(f"2025-01-01T{h:02d}:00", tz="UTC") for h in range(4)]
    rows = [
        _make_primary_current_row(
            current_node_id="A",
            time_utc=t,
            uo_m_s=0.1 * (i + 1),
            vo_m_s=0.0,
            height_above_model_bed_m=2.0,
            model_bathymetry_m=20.0,
        )
        for i, t in enumerate(times)
    ]
    df = pd.DataFrame(rows)
    hourly = norm.build_current_only_1m_sensitivity_hourly(df)

    stats = norm.compute_current_only_1m_sensitivity_stats(hourly)

    assert set(stats["roughness_scenario"]) == {name for name, _ in norm.ROUGHNESS_SCENARIOS_M}
    silt_row = stats[
        (stats["current_node_id"] == "A") & (stats["roughness_scenario"] == "SILT")
    ].iloc[0]
    assert silt_row["valid_hour_count"] == 4
    assert silt_row["completeness_pct"] == pytest.approx(100.0)


def test_sensitivity_stats_raises_when_duplicate_rows_exceed_100_pct():
    """Mirrors MAR-009B: completeness must never exceed 100%, a hard failure."""

    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    times = [base_time + pd.Timedelta(hours=h) for h in [0, 1, 1, 2]]  # hour 1 duplicated
    rows = [_make_primary_current_row(current_node_id="A", time_utc=t) for t in times]
    df = pd.DataFrame(rows)
    hourly = norm.build_current_only_1m_sensitivity_hourly(df)

    with pytest.raises(norm.NormalizationCompletenessError):
        norm.compute_current_only_1m_sensitivity_stats(hourly)


def test_sensitivity_stats_column_schema_matches_constant():
    df = pd.DataFrame([_make_primary_current_row()])
    hourly = norm.build_current_only_1m_sensitivity_hourly(df)
    stats = norm.compute_current_only_1m_sensitivity_stats(hourly)
    assert list(stats.columns) == list(norm.CURRENT_ONLY_1M_SENSITIVITY_STATS_COLUMNS)


def test_sensitivity_stats_empty_input():
    stats = norm.compute_current_only_1m_sensitivity_stats(pd.DataFrame())
    assert stats.empty


# --- compute_current_only_1m_sensitivity_envelope (Section 7, 17-I) ----------------


def test_sensitivity_envelope_is_min_max_not_average():
    """17-I: envelope is the actual min/max across the five scenarios, never an average."""

    stats_df = pd.DataFrame(
        {
            "current_node_id": ["A"] * 5,
            "roughness_scenario": [name for name, _ in norm.ROUGHNESS_SCENARIOS_M],
            "speed_1m_p95_m_s": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )

    envelope = norm.compute_current_only_1m_sensitivity_envelope(stats_df)

    row = envelope.iloc[0]
    assert row["speed_1m_p95_sensitivity_min_m_s"] == pytest.approx(0.10)
    assert row["speed_1m_p95_sensitivity_max_m_s"] == pytest.approx(0.50)
    assert row["speed_1m_p95_sensitivity_width_m_s"] == pytest.approx(0.40)
    # Never the average of the five values (0.30) mistaken for the min or max.
    average = sum([0.10, 0.20, 0.30, 0.40, 0.50]) / 5
    assert row["speed_1m_p95_sensitivity_min_m_s"] != pytest.approx(average)
    assert row["speed_1m_p95_sensitivity_max_m_s"] != pytest.approx(average)


def test_sensitivity_envelope_empty_input():
    envelope = norm.compute_current_only_1m_sensitivity_envelope(pd.DataFrame())
    assert envelope.empty


# --- No forbidden downstream-physics column names (Section 17-O) -------------------


def test_no_output_schema_contains_forbidden_downstream_terms():
    forbidden = ("bed_shear", "shields", "mobility", "risk")
    all_columns = [
        *norm.CURRENT_ONLY_1M_SENSITIVITY_COLUMNS,
        *norm.CURRENT_ONLY_1M_SENSITIVITY_STATS_COLUMNS,
        *norm.CURRENT_ONLY_1M_SENSITIVITY_ENVELOPE_COLUMNS,
    ]
    columns_lower = [c.lower() for c in all_columns]
    for term in forbidden:
        assert not any(term in column for column in columns_lower)
