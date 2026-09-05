"""Offline unit tests for marine_engine.metocean.combined_bed_shear (MAR-012).

Small hand-built synthetic DataFrames only -- never the real PL854 route or
real Copernicus data, never network access. Lettered comments (A, B, C, ...)
map directly to MAR-012 Section 30's required test list.
"""

import numpy as np
import pandas as pd
import pytest

from marine_engine.metocean import combined_bed_shear as cbs

WORKING_CRS = "EPSG:32631"


# --- A/B/C/D: Soulsby mean/max combined stress special cases (Sections 15-16) -------


def test_A_zero_current_finite_wave_gives_tau_max_equal_tau_wave():
    tau_c = np.array([0.0])
    tau_w = np.array([1.234])
    phi = np.array([np.nan])  # null, as it would genuinely be for zero current
    tau_m = cbs.compute_soulsby_mean_combined_stress_pa(tau_c, tau_w)
    tau_max = cbs.compute_soulsby_max_combined_stress_pa(tau_c, tau_w, tau_m, phi)

    assert tau_m[0] == pytest.approx(0.0)
    assert tau_max[0] == pytest.approx(tau_w[0])


def test_B_finite_current_zero_wave_gives_tau_max_equal_tau_current():
    tau_c = np.array([2.5])
    tau_w = np.array([0.0])
    phi = np.array([45.0])  # phi is irrelevant here but still finite
    tau_m = cbs.compute_soulsby_mean_combined_stress_pa(tau_c, tau_w)
    tau_max = cbs.compute_soulsby_max_combined_stress_pa(tau_c, tau_w, tau_m, phi)

    assert tau_m[0] == pytest.approx(tau_c[0])
    assert tau_max[0] == pytest.approx(tau_c[0])


def test_C_both_zero_gives_all_stresses_zero():
    tau_c = np.array([0.0])
    tau_w = np.array([0.0])
    phi = np.array([np.nan])
    tau_m = cbs.compute_soulsby_mean_combined_stress_pa(tau_c, tau_w)
    tau_max = cbs.compute_soulsby_max_combined_stress_pa(tau_c, tau_w, tau_m, phi)

    assert tau_m[0] == pytest.approx(0.0)
    assert tau_max[0] == pytest.approx(0.0)


def test_D_phi_zero_gives_larger_or_equal_tau_max_than_phi_90():
    tau_c = np.array([1.0, 1.0])
    tau_w = np.array([0.8, 0.8])
    tau_m = cbs.compute_soulsby_mean_combined_stress_pa(tau_c, tau_w)

    tau_max_aligned = cbs.compute_soulsby_max_combined_stress_pa(
        tau_c, tau_w, tau_m, np.array([0.0, 0.0])
    )
    tau_max_orthogonal = cbs.compute_soulsby_max_combined_stress_pa(
        tau_c, tau_w, tau_m, np.array([90.0, 90.0])
    )

    assert (tau_max_aligned >= tau_max_orthogonal).all()
    assert tau_max_aligned[0] > tau_max_orthogonal[0]
    # Hand-computed: aligned = tau_m + tau_w; orthogonal = sqrt(tau_m^2 + tau_w^2)
    assert tau_max_aligned[0] == pytest.approx(tau_m[0] + tau_w[0])
    assert tau_max_orthogonal[0] == pytest.approx(np.sqrt(tau_m[0] ** 2 + tau_w[0] ** 2))


# --- E/F: wave-current axis angle folding (Section 14) ------------------------------


def test_E_from_to_direction_conversion_preserves_wave_axis_after_fold():
    current_dir = np.array([0.0])
    wave_from = np.array([90.0])
    wave_to = (wave_from + 180.0) % 360.0  # project's from->to convention

    phi_using_from = cbs.fold_wave_current_axis_angle_deg(current_dir, wave_from)
    phi_using_to = cbs.fold_wave_current_axis_angle_deg(current_dir, wave_to)

    assert phi_using_from[0] == pytest.approx(phi_using_to[0])


def test_F_direction_differences_0_and_180_give_same_axis():
    current_dir = np.array([10.0, 10.0])
    wave_dir = np.array([10.0, 190.0])  # diffs of 0 and 180

    phi = cbs.fold_wave_current_axis_angle_deg(current_dir, wave_dir)

    assert phi[0] == pytest.approx(0.0)
    assert phi[1] == pytest.approx(0.0)


def test_fold_angle_orthogonal_case():
    phi = cbs.fold_wave_current_axis_angle_deg(np.array([0.0]), np.array([90.0]))
    assert phi[0] == pytest.approx(90.0)


def test_fold_angle_folds_obtuse_difference_around_90():
    # 135 degrees apart folds to 45 (180 - 135), exploiting the wave axis's
    # own 180-degree symmetry.
    phi = cbs.fold_wave_current_axis_angle_deg(np.array([0.0]), np.array([135.0]))
    assert phi[0] == pytest.approx(45.0)


def test_fold_angle_null_when_current_direction_is_null():
    phi = cbs.fold_wave_current_axis_angle_deg(np.array([np.nan]), np.array([90.0]))
    assert np.isnan(phi[0])


# --- G/H: current friction velocity inversion (Section 6) ---------------------------


def test_G_current_friction_velocity_matches_hand_calculation():
    z0 = 1e-4  # COARSE_SAND
    u1m = 0.5
    u_star = cbs.compute_current_friction_velocity_m_s(np.array([u1m]), z0)
    expected = 0.40 * u1m / np.log((1.0 + z0) / z0)
    assert u_star[0] == pytest.approx(expected)


def test_G_zero_current_gives_zero_friction_velocity_and_zero_stress():
    u_star = cbs.compute_current_friction_velocity_m_s(np.array([0.0]), 1e-4)
    tau_c = cbs.compute_current_bed_shear_stress_pa(u_star)
    assert u_star[0] == pytest.approx(0.0)
    assert tau_c[0] == pytest.approx(0.0)


def test_G_missing_current_propagates_to_nan_not_a_fabricated_zero():
    u_star = cbs.compute_current_friction_velocity_m_s(np.array([np.nan]), 1e-4)
    tau_c = cbs.compute_current_bed_shear_stress_pa(u_star)
    assert np.isnan(u_star[0])
    assert np.isnan(tau_c[0])


def test_H_larger_u1m_produces_larger_tau_current():
    z0 = 1e-4
    u_star = cbs.compute_current_friction_velocity_m_s(np.array([0.1, 0.5, 1.0]), z0)
    tau_c = cbs.compute_current_bed_shear_stress_pa(u_star)
    assert tau_c[0] < tau_c[1] < tau_c[2]


# --- I/J: wave friction factor branches (Sections 10-12) -----------------------------


def test_I_wave_friction_factor_uses_max_of_two_branches_never_average():
    f_ws = np.array([0.01, 0.05])
    f_wr = np.array([0.03, 0.02])
    uw = np.array([0.5, 0.5])

    f_w, branch = cbs.compute_wave_friction_factor(f_ws, f_wr, uw)

    assert f_w[0] == pytest.approx(0.03)
    assert branch[0] == cbs.ROUGH
    assert f_w[1] == pytest.approx(0.05)
    assert branch[1] == cbs.SMOOTH_OR_LAMINAR
    # Never the average of the two branches.
    assert f_w[0] != pytest.approx((f_ws[0] + f_wr[0]) / 2.0)


def test_wave_friction_factor_null_for_calm_sea():
    f_ws = np.array([0.02])
    f_wr = np.array([0.03])
    uw = np.array([0.0])
    f_w, branch = cbs.compute_wave_friction_factor(f_ws, f_wr, uw)
    assert np.isnan(f_w[0])
    assert branch[0] is None


def test_J_reynolds_regime_switches_correctly_at_5e5():
    rw = np.array([1e5, cbs.WAVE_REYNOLDS_TRANSITION, cbs.WAVE_REYNOLDS_TRANSITION + 1.0, 1e6])
    regime = cbs.classify_wave_reynolds_regime(rw)

    assert regime.tolist() == [
        cbs.LAMINAR_BRANCH,
        cbs.LAMINAR_BRANCH,
        cbs.SMOOTH_TURBULENT_BRANCH,
        cbs.SMOOTH_TURBULENT_BRANCH,
    ]

    f_ws = cbs.compute_wave_friction_smooth_branch(rw)
    expected_laminar = 2.0 * np.power(rw[:2], -0.5)
    expected_turbulent = 0.0521 * np.power(rw[2:], -0.187)
    assert f_ws[:2] == pytest.approx(expected_laminar)
    assert f_ws[2:] == pytest.approx(expected_turbulent)


def test_wave_reynolds_number_undefined_for_zero_uw_or_zero_excursion():
    rw = cbs.compute_wave_reynolds_number(np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0]))
    assert np.isnan(rw[0])
    assert np.isnan(rw[1])
    assert np.isfinite(rw[2])


def test_rough_branch_never_derives_z0_from_d50():
    """Sanity: `compute_wave_friction_rough_branch` takes z0 directly, matching the
    fixed roughness scenario -- there is no D50-to-z0 derivation path anywhere here."""

    a_wave = np.array([0.2])
    f_wr = cbs.compute_wave_friction_rough_branch(a_wave, 1e-4)
    expected = 1.39 * (0.2 / 1e-4) ** (-0.52)
    assert f_wr[0] == pytest.approx(expected)


# --- Wave semi-orbital excursion / calm-sea guards (Section 8) ----------------------


def test_calm_sea_gives_zero_excursion_and_zero_wave_stress():
    a_wave = cbs.compute_wave_semi_orbital_excursion_m(np.array([0.0]), np.array([np.nan]))
    tau_w = cbs.compute_wave_bed_shear_stress_pa(np.array([0.0]), np.array([np.nan]))
    assert a_wave[0] == pytest.approx(0.0)
    assert tau_w[0] == pytest.approx(0.0)


def test_missing_uw_propagates_to_nan_excursion_not_a_fabricated_zero():
    a_wave = cbs.compute_wave_semi_orbital_excursion_m(np.array([np.nan]), np.array([8.0]))
    assert np.isnan(a_wave[0])


# --- ks_m diagnostic (Section 4) -----------------------------------------------------


def test_ks_m_is_exactly_30_times_z0():
    ks = cbs.compute_ks_m(np.array([5e-6, 1e-4]))
    assert ks[0] == pytest.approx(30.0 * 5e-6)
    assert ks[1] == pytest.approx(30.0 * 1e-4)


# --- Spatial node pairing (Section 19, P/Q) ------------------------------------------


def test_P_nodes_paired_by_coordinate_not_by_node_id_string():
    """Deliberately mismatched naming, identical coordinates -- pairing must succeed
    via coordinate identity alone."""

    current_nodes = pd.DataFrame(
        {"node_id": ["current_AAA"], "longitude": [1.666667], "latitude": [53.364861]}
    )
    wave_nodes = pd.DataFrame(
        {"node_id": ["wave_ZZZ"], "longitude": [1.666667], "latitude": [53.364861]}
    )

    pairs = cbs.build_hydro_pairs(current_nodes, wave_nodes, working_crs=WORKING_CRS)

    assert len(pairs) == 1
    assert pairs.iloc[0]["current_node_id"] == "current_AAA"
    assert pairs.iloc[0]["wave_node_id"] == "wave_ZZZ"
    assert pairs.iloc[0]["coordinate_separation_m"] == pytest.approx(0.0, abs=1e-3)
    assert pairs.iloc[0]["hydro_pair_id"] == "current_AAA__wave_ZZZ"


def test_hydro_pairs_column_schema_matches_constant():
    current_nodes = pd.DataFrame(
        {"node_id": ["current_A"], "longitude": [1.7], "latitude": [53.37]}
    )
    wave_nodes = pd.DataFrame({"node_id": ["wave_A"], "longitude": [1.7], "latitude": [53.37]})
    pairs = cbs.build_hydro_pairs(current_nodes, wave_nodes, working_crs=WORKING_CRS)
    assert list(pairs.columns) == list(cbs.HYDRO_PAIRS_COLUMNS)


def test_Q_unreconcilable_coordinates_hard_fail():
    """A current node with no wave node anywhere near it must hard fail, never
    silently nearest-neighbour to a distant, unrelated cell."""

    current_nodes = pd.DataFrame(
        {
            "node_id": ["current_A", "current_B"],
            "longitude": [1.666667, 1.696970],  # real ~2 km grid spacing
            "latitude": [53.364861, 53.364861],
        }
    )
    wave_nodes = pd.DataFrame(
        {"node_id": ["wave_A"], "longitude": [1.666667], "latitude": [53.364861]}
    )

    with pytest.raises(cbs.UnreconciledHydroNodeError):
        cbs.build_hydro_pairs(current_nodes, wave_nodes, working_crs=WORKING_CRS)


def test_hydro_pairs_empty_input():
    result = cbs.build_hydro_pairs(pd.DataFrame(), pd.DataFrame(), working_crs=WORKING_CRS)
    assert result.empty
    assert list(result.columns) == list(cbs.HYDRO_PAIRS_COLUMNS)


# --- Full-row builder: temporal join + physics (Sections 18, 20) --------------------

ROUGHNESS_SCENARIO_NAMES = [name for name, _ in cbs.ROUGHNESS_SCENARIOS_M]


def _current_hourly_df(
    *,
    node_id: str = "current_A",
    times: list[pd.Timestamp] | None = None,
    speed_m_s: float = 0.5,
) -> pd.DataFrame:
    times = times or [pd.Timestamp("2025-01-01T00:00", tz="UTC")]
    rows = []
    for t in times:
        for scenario, z0 in cbs.ROUGHNESS_SCENARIOS_M:
            rows.append(
                {
                    "current_node_id": node_id,
                    "time_utc": t,
                    "roughness_scenario": scenario,
                    "z0_m": z0,
                    "current_only_1m_uo_m_s": speed_m_s,
                    "current_only_1m_vo_m_s": 0.0,
                    "current_only_1m_speed_m_s": speed_m_s,
                }
            )
    return pd.DataFrame(rows)


def _wave_3hourly_df(
    *,
    node_id: str = "wave_A",
    times: list[pd.Timestamp] | None = None,
    urms: float = 0.3,
    tz_s: float = 6.0,
    tp_s: float = 8.0,
    wave_direction_to_deg: float = 90.0,
) -> pd.DataFrame:
    times = times or [pd.Timestamp("2025-01-01T00:00", tz="UTC")]
    rows = []
    for t in times:
        rows.append(
            {
                "wave_node_id": node_id,
                "time_utc": t,
                "wave_orbital_velocity_rms_near_bed_m_s": urms,
                "wave_orbital_velocity_equivalent_amplitude_m_s": np.sqrt(2.0) * urms,
                "equivalent_peak_period_from_tz_s": 1.28 * tz_s,
                "tp_s": tp_s,
                "wave_mean_direction_to_deg": wave_direction_to_deg,
            }
        )
    return pd.DataFrame(rows)


def _hydro_pairs_df(*, current_node_id="current_A", wave_node_id="wave_A") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hydro_pair_id": f"{current_node_id}__{wave_node_id}",
                "current_node_id": current_node_id,
                "wave_node_id": wave_node_id,
                "current_longitude": 1.7,
                "current_latitude": 53.37,
                "wave_longitude": 1.7,
                "wave_latitude": 53.37,
                "coordinate_separation_m": 0.0,
            }
        ]
    )


def test_builder_column_schema_matches_constant():
    result = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df()
    )
    assert list(result.columns) == list(cbs.COMBINED_BED_SHEAR_3HOURLY_COLUMNS)


def test_builder_empty_inputs():
    result = cbs.build_combined_bed_shear_3hourly(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert result.empty
    assert list(result.columns) == list(cbs.COMBINED_BED_SHEAR_3HOURLY_COLUMNS)


# --- N/O: exact-timestamp join only (Section 18) -------------------------------------


def test_N_only_the_shared_exact_timestamp_survives_the_join():
    t0 = pd.Timestamp("2025-01-01T00:00", tz="UTC")
    t1 = pd.Timestamp("2025-01-01T01:00", tz="UTC")
    t2 = pd.Timestamp("2025-01-01T02:00", tz="UTC")
    current_df = _current_hourly_df(times=[t0, t1, t2])
    wave_df = _wave_3hourly_df(times=[t1])  # only the middle hour is a wave timestamp

    result = cbs.build_combined_bed_shear_3hourly(current_df, wave_df, _hydro_pairs_df())

    assert set(result["time_utc"].unique()) == {t1}
    assert len(result) == len(ROUGHNESS_SCENARIO_NAMES)  # one row per scenario, one timestamp


def test_O_unmatched_timestamps_produce_no_fabricated_rows():
    t0 = pd.Timestamp("2025-01-01T00:00", tz="UTC")
    t5 = pd.Timestamp("2025-01-01T05:00", tz="UTC")
    current_df = _current_hourly_df(times=[t0])
    wave_df = _wave_3hourly_df(times=[t5])  # no overlap at all

    result = cbs.build_combined_bed_shear_3hourly(current_df, wave_df, _hydro_pairs_df())

    assert result.empty


# --- K/L/M: wave amplitude/period source discipline (Sections 7, 13) ---------------


def test_K_tau_wave_uses_equivalent_amplitude_column_not_rms_directly():
    result = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(), _wave_3hourly_df(urms=0.3), _hydro_pairs_df()
    )
    row = result.iloc[0]

    assert row["wave_orbital_velocity_equivalent_amplitude_m_s"] == pytest.approx(
        np.sqrt(2.0) * row["wave_orbital_velocity_rms_m_s"]
    )
    manual_tau_w_using_amplitude = (
        0.5
        * cbs.RHO_WATER_KG_M3
        * row["wave_friction_factor"]
        * row["wave_orbital_velocity_equivalent_amplitude_m_s"] ** 2
    )
    manual_tau_w_using_rms = (
        0.5
        * cbs.RHO_WATER_KG_M3
        * row["wave_friction_factor"]
        * row["wave_orbital_velocity_rms_m_s"] ** 2
    )
    assert row["tau_wave_pa"] == pytest.approx(manual_tau_w_using_amplitude)
    assert row["tau_wave_pa"] != pytest.approx(manual_tau_w_using_rms)


def test_L_representative_period_is_passthrough_of_128_times_tz():
    result = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(), _wave_3hourly_df(tz_s=6.5), _hydro_pairs_df()
    )
    assert result.iloc[0]["representative_wave_period_s"] == pytest.approx(1.28 * 6.5)


def test_M_changing_observed_vtpk_alone_does_not_change_tau_wave():
    result_a = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(), _wave_3hourly_df(tp_s=8.0), _hydro_pairs_df()
    )
    result_b = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(), _wave_3hourly_df(tp_s=99.0), _hydro_pairs_df()
    )
    assert result_a.iloc[0]["tau_wave_pa"] == pytest.approx(result_b.iloc[0]["tau_wave_pa"])
    assert result_a.iloc[0]["observed_tp_s"] != result_b.iloc[0]["observed_tp_s"]


# --- R: exactly five scenarios, no best/average scenario (Section 4, 23) -----------


def test_R_exactly_five_scenarios_present_no_extra_no_average():
    result = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df()
    )
    assert len(cbs.ROUGHNESS_SCENARIOS_M) == 5
    assert sorted(result["roughness_scenario"].unique()) == sorted(ROUGHNESS_SCENARIO_NAMES)
    assert "BEST_ESTIMATE" not in result["roughness_scenario"].unique()
    assert "AVERAGE" not in result["roughness_scenario"].unique()


def test_same_z0_used_consistently_for_current_and_wave_within_a_scenario_row():
    result = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df()
    )
    for _, row in result.iterrows():
        expected_z0 = dict(cbs.ROUGHNESS_SCENARIOS_M)[row["roughness_scenario"]]
        assert row["z0_m"] == pytest.approx(expected_z0)
        assert row["ks_m"] == pytest.approx(30.0 * expected_z0)


# --- S: sensitivity envelope from real five-scenario outputs (Section 23) ----------


def test_S_envelope_min_max_width_come_from_real_scenario_values():
    stats_df = pd.DataFrame(
        {
            "hydro_pair_id": ["P1"] * 5,
            "roughness_scenario": ROUGHNESS_SCENARIO_NAMES,
            "tau_max_combined_p95_pa": [0.10, 0.20, 0.15, 0.25, 0.05],
            "tau_max_combined_p99_pa": [0.20, 0.30, 0.25, 0.35, 0.10],
        }
    )
    envelope = cbs.compute_sensitivity_envelope(stats_df)
    row = envelope.iloc[0]

    assert row["tau_max_p95_sensitivity_min_pa"] == pytest.approx(0.05)
    assert row["tau_max_p95_sensitivity_max_pa"] == pytest.approx(0.25)
    assert row["tau_max_p95_sensitivity_width_pa"] == pytest.approx(0.20)
    assert row["tau_max_p99_sensitivity_min_pa"] == pytest.approx(0.10)
    assert row["tau_max_p99_sensitivity_max_pa"] == pytest.approx(0.35)
    assert row["tau_max_p99_sensitivity_width_pa"] == pytest.approx(0.25)


def test_envelope_never_averages_scenarios():
    stats_df = pd.DataFrame(
        {
            "hydro_pair_id": ["P1"] * 5,
            "roughness_scenario": ROUGHNESS_SCENARIO_NAMES,
            "tau_max_combined_p95_pa": [1.0, 2.0, 3.0, 4.0, 5.0],
            "tau_max_combined_p99_pa": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    envelope = cbs.compute_sensitivity_envelope(stats_df)
    average = np.mean([1.0, 2.0, 3.0, 4.0, 5.0])
    assert envelope.iloc[0]["tau_max_p95_sensitivity_min_pa"] != pytest.approx(average)
    assert envelope.iloc[0]["tau_max_p95_sensitivity_max_pa"] != pytest.approx(average)


def test_envelope_empty_input():
    result = cbs.compute_sensitivity_envelope(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == list(cbs.SENSITIVITY_ENVELOPE_COLUMNS)


# --- Combined node statistics (Section 22) -------------------------------------------


def test_combined_stats_column_schema_matches_constant():
    result = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(), _wave_3hourly_df(), _hydro_pairs_df()
    )
    stats = cbs.compute_combined_bed_shear_stats(result)
    assert list(stats.columns) == list(cbs.COMBINED_BED_SHEAR_STATS_COLUMNS)
    assert len(stats) == 5  # one row per (hydro_pair_id, roughness_scenario)


def test_combined_stats_completeness_never_exceeds_100_pct():
    times = [
        pd.Timestamp("2025-01-01T00:00", tz="UTC") + pd.Timedelta(hours=3 * i) for i in range(4)
    ]
    result = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(times=times), _wave_3hourly_df(times=times), _hydro_pairs_df()
    )
    stats = cbs.compute_combined_bed_shear_stats(result)
    assert (stats["completeness_pct"] <= 100.0001).all()


def test_temporal_alignment_summary_reports_distinct_timestamps():
    times = [
        pd.Timestamp("2025-01-01T00:00", tz="UTC") + pd.Timedelta(hours=3 * i) for i in range(4)
    ]
    result = cbs.build_combined_bed_shear_3hourly(
        _current_hourly_df(times=times), _wave_3hourly_df(times=times), _hydro_pairs_df()
    )
    summary = cbs.compute_temporal_alignment_summary(result)
    assert summary["matched_timestamp_count"] == 4
    assert summary["expected_3hour_count"] == 4
    assert summary["completeness_pct"] == pytest.approx(100.0)
    assert summary["overlap_start_time_utc"] == times[0]
    assert summary["overlap_end_time_utc"] == times[-1]


def test_temporal_alignment_summary_empty_input():
    summary = cbs.compute_temporal_alignment_summary(pd.DataFrame())
    assert summary["matched_timestamp_count"] == 0
    assert summary["completeness_pct"] is None


# --- Long-term wave-only shear context (Section 21) ----------------------------------


def test_long_term_wave_only_stats_column_schema_matches_constant():
    wave_df = _wave_3hourly_df(
        times=[
            pd.Timestamp("2025-01-01T00:00", tz="UTC") + pd.Timedelta(hours=3 * i) for i in range(4)
        ]
    )
    overlap_keys = wave_df[["wave_node_id", "time_utc"]].iloc[[0]]
    result = cbs.compute_wave_only_bed_shear_long_term_stats(wave_df, overlap_keys_df=overlap_keys)
    assert list(result.columns) == list(cbs.WAVE_ONLY_BED_SHEAR_LONG_TERM_STATS_COLUMNS)
    assert len(result) == 5  # one node x five scenarios


def test_long_term_wave_only_stats_overlap_subset_is_smaller_than_full_record():
    times = [
        pd.Timestamp("2025-01-01T00:00", tz="UTC") + pd.Timedelta(hours=3 * i) for i in range(10)
    ]
    wave_df = _wave_3hourly_df(times=times)
    overlap_keys = wave_df[["wave_node_id", "time_utc"]].iloc[:3]

    result = cbs.compute_wave_only_bed_shear_long_term_stats(wave_df, overlap_keys_df=overlap_keys)
    row = result.iloc[0]

    assert row["long_term_valid_count"] == 10
    assert row["overlap_valid_count"] == 3
    assert row["overlap_valid_count"] < row["long_term_valid_count"]


def test_long_term_wave_only_stats_ratio_is_descriptive_not_fabricated_when_empty():
    wave_df = _wave_3hourly_df(times=[pd.Timestamp("2025-01-01T00:00", tz="UTC")])
    empty_overlap = wave_df[["wave_node_id", "time_utc"]].iloc[0:0]
    result = cbs.compute_wave_only_bed_shear_long_term_stats(wave_df, overlap_keys_df=empty_overlap)
    row = result.iloc[0]
    assert row["overlap_valid_count"] == 0
    assert row["overlap_to_long_term_tau_wave_p95_ratio"] is None


# --- X: forbidden downstream-physics terms never appear in any output schema -------


def test_X_no_output_schema_contains_forbidden_downstream_terms():
    forbidden = ("shields", "theta", "mobility", "risk")
    all_columns = [
        *cbs.COMBINED_BED_SHEAR_3HOURLY_COLUMNS,
        *cbs.WAVE_ONLY_BED_SHEAR_LONG_TERM_STATS_COLUMNS,
        *cbs.COMBINED_BED_SHEAR_STATS_COLUMNS,
        *cbs.SENSITIVITY_ENVELOPE_COLUMNS,
        *cbs.HYDRO_PAIRS_COLUMNS,
    ]
    columns_lower = [c.lower() for c in all_columns]
    for term in forbidden:
        assert not any(term in column for column in columns_lower), term


def test_representativeness_warning_matches_required_verbatim_text():
    assert cbs.REPRESENTATIVENESS_WARNING == (
        "COMBINED BED-SHEAR STATISTICS ARE BASED ON THE CONTEMPORANEOUS PRIMARY-CURRENT / "
        "WAVE OVERLAP, NOT THE FULL 1980–2026 WAVE RECORD AND NOT A 25-YEAR RETURN-PERIOD "
        "ANALYSIS."
    )
