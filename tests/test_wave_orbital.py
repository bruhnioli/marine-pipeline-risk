"""Offline unit tests for marine_engine.metocean.wave_orbital (MAR-011/MAR-011A).

Small hand-built synthetic DataFrames only -- never the real PL854 route or
real Copernicus data, never network access.
"""

import numpy as np
import pandas as pd
import pytest

from marine_engine.metocean import evidence as metocean_evidence
from marine_engine.metocean import wave_orbital as orb

# A representative, physically arbitrary case within TR155's own reported
# better-than-1%-accuracy range: h=25 m, Tz=6 s -> Tn=sqrt(25/9.80665)=1.597s,
# t=1.597/6=0.266 (<= 0.54).
H_M = 25.0
TZ_S = 6.0


def _make_wave_row(
    *,
    wave_node_id: str = "wave_0000_0000",
    time_utc: pd.Timestamp | None = None,
    hs_m: float = 1.5,
    tm02_s: float = TZ_S,
    tp_s: float = 8.0,
    tm10_s: float = 7.0,
    model_bathymetry_m: float = H_M,
    wave_mean_direction_from_deg: float = 90.0,
    wave_mean_direction_to_deg: float = 270.0,
    source_dataset: str = "TEST_WAVE_DATASET",
) -> dict:
    return {
        "wave_node_id": wave_node_id,
        "time_utc": time_utc or pd.Timestamp("2025-01-01T00:00", tz="UTC"),
        "hs_m": hs_m,
        "tm02_s": tm02_s,
        "tp_s": tp_s,
        "tm10_s": tm10_s,
        "model_bathymetry_m": model_bathymetry_m,
        "wave_mean_direction_from_deg": wave_mean_direction_from_deg,
        "wave_mean_direction_to_deg": wave_mean_direction_to_deg,
        "source_dataset": source_dataset,
    }


# --- Core formula unit tests ---------------------------------------------------------


def test_natural_scaling_period_matches_formula():
    tn = orb.compute_natural_scaling_period_s(H_M)
    assert tn == pytest.approx(np.sqrt(H_M / 9.80665))


def test_t_parameter_and_a_and_urms_hand_computed():
    tn = orb.compute_natural_scaling_period_s(H_M)
    t = orb.compute_soulsby_smallman_t_parameter(tn, TZ_S)
    a = orb.compute_soulsby_smallman_a(t)
    urms = orb.compute_orbital_velocity_rms_m_s(1.5, tn, a, t)

    expected_tn = np.sqrt(H_M / 9.80665)
    expected_t = expected_tn / TZ_S
    expected_a = (6500.0 + (0.56 + 15.54 * expected_t) ** 6) ** (1.0 / 6.0)
    expected_urms = 0.25 * 1.5 / (expected_tn * (1.0 + expected_a * expected_t**2) ** 3)

    assert t == pytest.approx(expected_t)
    assert a == pytest.approx(expected_a)
    assert urms == pytest.approx(expected_urms)


def test_t_parameter_within_reported_accuracy_range_for_representative_case():
    tn = orb.compute_natural_scaling_period_s(H_M)
    t = orb.compute_soulsby_smallman_t_parameter(tn, TZ_S)
    assert 0.0 <= t <= orb.REPORTED_1PCT_ACCURACY_MAX_T


# --- build_wave_orbital_velocity_3hourly: core behaviour ----------------------------


def test_hs_zero_gives_urms_zero():
    """Hs = 0, valid h/Tz -> Urms = 0."""

    df = pd.DataFrame([_make_wave_row(hs_m=0.0)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    assert result.iloc[0]["wave_orbital_velocity_rms_near_bed_m_s"] == pytest.approx(0.0)


def test_urms_scales_linearly_with_hs():
    """Larger Hs at the same h/Tz -> Urms scales linearly with Hs."""

    df = pd.DataFrame(
        [_make_wave_row(hs_m=1.0), _make_wave_row(hs_m=2.0), _make_wave_row(hs_m=3.0)]
    )
    result = orb.build_wave_orbital_velocity_3hourly(df)
    urms = result["wave_orbital_velocity_rms_near_bed_m_s"].to_numpy()

    assert urms[1] == pytest.approx(2.0 * urms[0])
    assert urms[2] == pytest.approx(3.0 * urms[0])


def test_larger_tz_at_fixed_hs_and_depth_increases_orbital_response():
    """Longer waves penetrate deeper -> larger Tz (smaller t) increases Urms
    (fixed Hs/h), consistent with the formula's own shallow/deep asymptotics."""

    df = pd.DataFrame(
        [_make_wave_row(tm02_s=4.0), _make_wave_row(tm02_s=8.0), _make_wave_row(tm02_s=12.0)]
    )
    result = orb.build_wave_orbital_velocity_3hourly(df)
    urms = result["wave_orbital_velocity_rms_near_bed_m_s"].to_numpy()

    assert np.all(np.isfinite(urms))
    assert urms[0] < urms[1] < urms[2]


def test_changing_vtpk_alone_does_not_change_canonical_urms():
    """19-D: Tz = VTM02 is used -- changing VTPK (tp_s) alone must not change Urms."""

    df = pd.DataFrame([_make_wave_row(tp_s=8.0), _make_wave_row(tp_s=99.0)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    urms = result["wave_orbital_velocity_rms_near_bed_m_s"].to_numpy()
    assert urms[0] == pytest.approx(urms[1])


def test_changing_vtm10_alone_does_not_change_canonical_urms():
    """19-E: changing VTM10 (tm10_s) alone must not change Urms."""

    df = pd.DataFrame([_make_wave_row(tm10_s=5.0), _make_wave_row(tm10_s=50.0)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    urms = result["wave_orbital_velocity_rms_near_bed_m_s"].to_numpy()
    assert urms[0] == pytest.approx(urms[1])


def test_equivalent_amplitude_is_exactly_sqrt2_times_urms():
    """19-F."""

    df = pd.DataFrame([_make_wave_row()])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    row = result.iloc[0]
    assert row["wave_orbital_velocity_equivalent_amplitude_m_s"] == pytest.approx(
        np.sqrt(2.0) * row["wave_orbital_velocity_rms_near_bed_m_s"]
    )


def test_equivalent_peak_period_is_exactly_128_times_tz():
    """19-G."""

    df = pd.DataFrame([_make_wave_row(tm02_s=6.5)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    assert result.iloc[0]["equivalent_peak_period_from_tz_s"] == pytest.approx(1.28 * 6.5)


def test_observed_tp_remains_unchanged_and_separate():
    """19-H: the real Copernicus VTPK value is preserved, never overwritten by the
    derived equivalent-peak-period diagnostic."""

    df = pd.DataFrame([_make_wave_row(tp_s=9.3, tm02_s=6.0)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    row = result.iloc[0]
    assert row["tp_s"] == pytest.approx(9.3)
    assert row["equivalent_peak_period_from_tz_s"] == pytest.approx(1.28 * 6.0)
    assert row["tp_s"] != pytest.approx(row["equivalent_peak_period_from_tz_s"])
    assert row["observed_to_equivalent_peak_period_ratio"] == pytest.approx(9.3 / (1.28 * 6.0))


# --- MAR-011A Section 10 A/B: t > 0.54 no longer nulls Urms -------------------------


def test_t_above_threshold_with_valid_inputs_still_produces_finite_urms():
    """10-A: t > 0.54 with valid Hs/Tz/depth must still produce a finite Urms --
    the previous (incorrect) MAR-011 behaviour nulled ~14.5% of the real PL854
    record solely because of this; TR155 never says the method is invalid here."""

    # A very short Tz relative to Tn pushes t well above 0.54.
    df = pd.DataFrame([_make_wave_row(tm02_s=0.5, model_bathymetry_m=25.0)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    row = result.iloc[0]

    assert row["soulsby_smallman_t_parameter"] > orb.REPORTED_1PCT_ACCURACY_MAX_T
    assert pd.notna(row["wave_orbital_velocity_rms_near_bed_m_s"])
    assert np.isfinite(row["wave_orbital_velocity_rms_near_bed_m_s"])
    assert pd.notna(row["wave_orbital_velocity_equivalent_amplitude_m_s"])
    # Raw inputs are of course still preserved.
    assert row["hs_m"] == pytest.approx(1.5)
    assert row["model_bathymetry_m"] == pytest.approx(25.0)


def test_t_above_threshold_is_marked_outside_reported_accuracy_range():
    """10-B: the same row is marked OUTSIDE_REPORTED_BETTER_THAN_1PCT_ACCURACY_RANGE."""

    df = pd.DataFrame([_make_wave_row(tm02_s=0.5, model_bathymetry_m=25.0)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    row = result.iloc[0]

    assert row["soulsby_smallman_accuracy_status"] == orb.OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE


def test_within_range_row_is_marked_within_reported_accuracy_range():
    df = pd.DataFrame([_make_wave_row(tm02_s=TZ_S)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    row = result.iloc[0]

    assert row["soulsby_smallman_accuracy_status"] == orb.WITHIN_REPORTED_1PCT_ACCURACY_RANGE
    assert pd.notna(row["wave_orbital_velocity_rms_near_bed_m_s"])


def test_hs_over_depth_is_diagnostic_only_and_never_rejects_data():
    """10-L (context)/19-L: even Hs > h must still produce a computed Urms --
    no invented breaking-wave cutoff anywhere in this module."""

    df = pd.DataFrame([_make_wave_row(hs_m=30.0, model_bathymetry_m=25.0, tm02_s=6.0)])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    row = result.iloc[0]
    assert row["hs_over_model_depth"] == pytest.approx(30.0 / 25.0)
    assert row["hs_over_model_depth"] > 1.0
    assert pd.notna(row["wave_orbital_velocity_rms_near_bed_m_s"])


def test_invalid_depth_produces_no_canonical_orbital_estimate():
    """10-H: non-positive/invalid depth -> still no canonical Urms, no crash --
    genuine invalidity is unaffected by the MAR-011A accuracy-semantics fix."""

    df = pd.DataFrame(
        [
            _make_wave_row(wave_node_id="ZERO_DEPTH", model_bathymetry_m=0.0),
            _make_wave_row(wave_node_id="NEG_DEPTH", model_bathymetry_m=-5.0),
            _make_wave_row(wave_node_id="NAN_DEPTH", model_bathymetry_m=float("nan")),
        ]
    )
    result = orb.build_wave_orbital_velocity_3hourly(df)
    assert (
        result["soulsby_smallman_accuracy_status"] == orb.OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE
    ).all()
    assert result["wave_orbital_velocity_rms_near_bed_m_s"].isna().all()


def test_invalid_tz_produces_no_canonical_orbital_estimate():
    """10-H: non-positive Tz -> still no canonical Urms, no crash."""

    df = pd.DataFrame(
        [
            _make_wave_row(wave_node_id="ZERO_TZ", tm02_s=0.0),
            _make_wave_row(wave_node_id="NEG_TZ", tm02_s=-3.0),
        ]
    )
    result = orb.build_wave_orbital_velocity_3hourly(df)
    assert (
        result["soulsby_smallman_accuracy_status"] == orb.OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE
    ).all()
    assert result["wave_orbital_velocity_rms_near_bed_m_s"].isna().all()


def test_no_duplicate_node_time_rows():
    df = pd.DataFrame([_make_wave_row(), _make_wave_row()])  # identical (node, time)
    result = orb.build_wave_orbital_velocity_3hourly(df)

    integrity = metocean_evidence.validate_temporal_integrity(
        result, time_column="time_utc", node_column="wave_node_id"
    )
    assert integrity["duplicate_node_time_row_count"] == 1


def test_builder_never_requires_any_current_column():
    """10-current-only: no current data are required by the MAR-011 calculation."""

    df = pd.DataFrame([_make_wave_row()])
    assert not any("current" in c.lower() for c in df.columns)
    result = orb.build_wave_orbital_velocity_3hourly(df)
    assert not any("current" in c.lower() for c in result.columns)


def test_hourly_column_schema_matches_constant():
    df = pd.DataFrame([_make_wave_row()])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    assert list(result.columns) == list(orb.WAVE_ORBITAL_VELOCITY_COLUMNS)


def test_hourly_scientific_role_and_tz_source():
    df = pd.DataFrame([_make_wave_row()])
    result = orb.build_wave_orbital_velocity_3hourly(df)
    assert (result["scientific_role"] == orb.SCIENTIFIC_ROLE).all()
    assert (result["tz_source"] == "VTM02").all()


def test_hourly_row_count_matches_input_one_to_one():
    """No fan-out, no dropping -- every (wave_node_id, time_utc) input row survives."""

    rows = [
        _make_wave_row(wave_node_id=node, time_utc=pd.Timestamp(f"2025-01-01T0{h}:00", tz="UTC"))
        for node in ("A", "B")
        for h in range(3)
    ]
    df = pd.DataFrame(rows)
    result = orb.build_wave_orbital_velocity_3hourly(df)
    assert len(result) == len(df) == 6


def test_hourly_empty_input():
    result = orb.build_wave_orbital_velocity_3hourly(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == list(orb.WAVE_ORBITAL_VELOCITY_COLUMNS)


# --- classify_soulsby_smallman_accuracy_status / is_orbital_velocity_input_valid ----


def test_classify_accuracy_status_boundaries():
    status = orb.classify_soulsby_smallman_accuracy_status(
        np.array([0.0, 0.54, 0.5401, -0.1, np.nan])
    )
    assert status.tolist() == [
        orb.WITHIN_REPORTED_1PCT_ACCURACY_RANGE,
        orb.WITHIN_REPORTED_1PCT_ACCURACY_RANGE,
        orb.OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE,
        orb.OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE,
        orb.OUTSIDE_REPORTED_1PCT_ACCURACY_RANGE,
    ]


def test_is_orbital_velocity_input_valid_true_for_well_formed_inputs():
    assert orb.is_orbital_velocity_input_valid(1.5, 6.0, 25.0)


def test_is_orbital_velocity_input_valid_rejects_negative_hs():
    assert not orb.is_orbital_velocity_input_valid(-0.1, 6.0, 25.0)


def test_is_orbital_velocity_input_valid_accepts_zero_hs():
    assert orb.is_orbital_velocity_input_valid(0.0, 6.0, 25.0)


# --- compute_wave_orbital_domain_summary (MAR-011A Section 4) ----------------------


def test_domain_summary_separates_accuracy_range_from_validity():
    """10-C: t > 0.54 does NOT reduce input-data validity/completeness -- both the
    within- and outside-range row are counted as valid; only genuinely invalid
    rows would be excluded from `input_valid_count`."""

    df = pd.DataFrame(
        [
            _make_wave_row(wave_node_id="A", tm02_s=6.0),  # within reported accuracy range
            _make_wave_row(wave_node_id="B", tm02_s=0.3),  # outside reported accuracy range
            _make_wave_row(wave_node_id="C", model_bathymetry_m=-1.0),  # genuinely invalid
        ]
    )
    hourly = orb.build_wave_orbital_velocity_3hourly(df)

    summary = orb.compute_wave_orbital_domain_summary(hourly)

    assert summary["total_rows"] == 3
    assert summary["input_valid_count"] == 2  # A and B, not the invalid-depth C
    assert summary["within_reported_1pct_accuracy_count"] == 1
    assert summary["outside_reported_1pct_accuracy_count"] == 1
    # C's raw (invalid) depth is still preserved, not dropped -- it is the min;
    # A/B's valid H_M depth is the max.
    assert summary["model_bathymetry_m_max"] == pytest.approx(H_M)


def test_domain_summary_empty_input():
    summary = orb.compute_wave_orbital_domain_summary(pd.DataFrame())
    assert summary["total_rows"] == 0
    assert summary["input_valid_count"] == 0
    assert summary["within_reported_1pct_accuracy_count"] == 0
    assert summary["outside_reported_1pct_accuracy_count"] == 0


# --- compute_wave_orbital_velocity_stats + completeness (MAR-011A Sections 4-5) -----


def test_stats_completeness_and_percentiles():
    times = [pd.Timestamp(f"2025-01-01T{h:02d}:00", tz="UTC") for h in (0, 3, 6, 9)]
    rows = [
        _make_wave_row(wave_node_id="A", time_utc=t, hs_m=1.0 + 0.1 * i)
        for i, t in enumerate(times)
    ]
    df = pd.DataFrame(rows)
    hourly = orb.build_wave_orbital_velocity_3hourly(df)

    stats = orb.compute_wave_orbital_velocity_stats(hourly)

    assert len(stats) == 1
    row = stats.iloc[0]
    assert row["input_valid_count"] == 4
    assert row["input_data_completeness_pct"] == pytest.approx(100.0)
    assert row["hs_max_m"] == pytest.approx(1.3)


def test_stats_completeness_not_reduced_by_rows_outside_accuracy_range():
    """10-C at the stats level: a node entirely outside the reported accuracy
    range must still show 100% input-data completeness."""

    times = [pd.Timestamp(f"2025-01-01T{h:02d}:00", tz="UTC") for h in (0, 3, 6)]
    rows = [_make_wave_row(wave_node_id="A", time_utc=t, tm02_s=0.3) for t in times]
    df = pd.DataFrame(rows)
    hourly = orb.build_wave_orbital_velocity_3hourly(df)

    stats = orb.compute_wave_orbital_velocity_stats(hourly)
    row = stats.iloc[0]

    assert row["outside_reported_1pct_accuracy_count"] == 3
    assert row["input_valid_count"] == 3
    assert row["input_data_completeness_pct"] == pytest.approx(100.0)
    assert row["within_reported_1pct_accuracy_pct"] == pytest.approx(0.0)
    assert row["outside_reported_1pct_accuracy_pct"] == pytest.approx(100.0)


def test_stats_canonical_mean_includes_rows_outside_accuracy_range():
    """10-D: the canonical orbital_rms_mean_m_s (and by construction every other
    canonical statistic) must reflect ALL physically valid rows, not only the
    <= 0.54 subset. (Per TR155's own physics, a large `t` gives a materially
    SMALL Urms, not a large one -- so this checks inclusion via the mean,
    which is exactly and unambiguously computable, rather than assuming
    outside-range rows push extremes upward.)
    """

    times = [pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=3 * i) for i in range(4)]
    rows = [
        _make_wave_row(wave_node_id="A", time_utc=times[0], hs_m=1.0, tm02_s=TZ_S),
        _make_wave_row(wave_node_id="A", time_utc=times[1], hs_m=1.0, tm02_s=TZ_S),
        _make_wave_row(wave_node_id="A", time_utc=times[2], hs_m=1.0, tm02_s=0.3),
        _make_wave_row(wave_node_id="A", time_utc=times[3], hs_m=1.0, tm02_s=0.3),
    ]
    df = pd.DataFrame(rows)
    hourly = orb.build_wave_orbital_velocity_3hourly(df)
    stats = orb.compute_wave_orbital_velocity_stats(hourly)
    row = stats.iloc[0]

    full_record_urms = hourly["wave_orbital_velocity_rms_near_bed_m_s"]
    within_mask = (
        hourly["soulsby_smallman_accuracy_status"] == orb.WITHIN_REPORTED_1PCT_ACCURACY_RANGE
    )
    within_only_mean = hourly.loc[within_mask, "wave_orbital_velocity_rms_near_bed_m_s"].mean()

    assert row["input_valid_count"] == 4  # all four rows counted, none nulled for t > 0.54
    assert full_record_urms.notna().sum() == 4
    assert row["orbital_rms_mean_m_s"] == pytest.approx(full_record_urms.mean())
    # The two outside-range rows have a materially smaller Urms (TR155's own
    # "orbital velocities are very small" statement) -- their mere presence in
    # the canonical mean must pull it below the within-range-only mean.
    assert row["orbital_rms_mean_m_s"] < within_only_mean


def test_conditional_accuracy_range_p95_is_explicitly_named_and_scoped_to_subset():
    """10-E: the conditional <=0.54 p95 is present under its own explicit name,
    computed strictly from the within-range subset -- never silently swapped
    in for the canonical full-record statistic."""

    times = [pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=3 * i) for i in range(4)]
    rows = [
        _make_wave_row(wave_node_id="A", time_utc=times[0], hs_m=1.0, tm02_s=TZ_S),
        _make_wave_row(wave_node_id="A", time_utc=times[1], hs_m=3.0, tm02_s=TZ_S),
        _make_wave_row(wave_node_id="A", time_utc=times[2], hs_m=1.0, tm02_s=0.3),
        _make_wave_row(wave_node_id="A", time_utc=times[3], hs_m=1.0, tm02_s=0.3),
    ]
    df = pd.DataFrame(rows)
    hourly = orb.build_wave_orbital_velocity_3hourly(df)
    stats = orb.compute_wave_orbital_velocity_stats(hourly)
    row = stats.iloc[0]

    within_mask = (
        hourly["soulsby_smallman_accuracy_status"] == orb.WITHIN_REPORTED_1PCT_ACCURACY_RANGE
    )
    expected_within_p95 = hourly.loc[
        within_mask, "wave_orbital_velocity_rms_near_bed_m_s"
    ].quantile(0.95)

    assert "orbital_rms_p95_within_reported_1pct_accuracy_range_m_s" in stats.columns
    assert row["orbital_rms_p95_within_reported_1pct_accuracy_range_m_s"] == pytest.approx(
        expected_within_p95
    )
    # It is computed from strictly fewer (the within-only) rows than the
    # canonical statistic, so it is never a silent stand-in for it.
    assert within_mask.sum() < len(hourly)


def test_stats_raises_when_duplicate_rows_exceed_100_pct():
    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    times = [base_time + pd.Timedelta(hours=3 * h) for h in [0, 1, 1, 2]]  # step 1 duplicated
    rows = [_make_wave_row(wave_node_id="A", time_utc=t) for t in times]
    df = pd.DataFrame(rows)
    hourly = orb.build_wave_orbital_velocity_3hourly(df)

    with pytest.raises(orb.OrbitalVelocityCompletenessError):
        orb.compute_wave_orbital_velocity_stats(hourly)


def test_stats_column_schema_matches_constant():
    df = pd.DataFrame([_make_wave_row()])
    hourly = orb.build_wave_orbital_velocity_3hourly(df)
    stats = orb.compute_wave_orbital_velocity_stats(hourly)
    assert list(stats.columns) == list(orb.WAVE_ORBITAL_VELOCITY_STATS_COLUMNS)


def test_stats_empty_input():
    stats = orb.compute_wave_orbital_velocity_stats(pd.DataFrame())
    assert stats.empty


# --- No forbidden downstream-physics column names -----------------------------------


def test_no_output_schema_contains_forbidden_downstream_terms():
    forbidden = ("bed_shear", "shields", "mobility", "risk")
    all_columns = [
        *orb.WAVE_ORBITAL_VELOCITY_COLUMNS,
        *orb.WAVE_ORBITAL_VELOCITY_STATS_COLUMNS,
    ]
    columns_lower = [c.lower() for c in all_columns]
    for term in forbidden:
        assert not any(term in column for column in columns_lower)
