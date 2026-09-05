"""Offline unit tests for marine_engine.providers.metocean.acquisition.

No network access -- these operate on locally-written synthetic files and
small in-memory xarray Datasets only (never the real PL854 Copernicus
Marine data).
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marine_engine.providers.metocean import acquisition as acq


def _write_file(path: Path, content: bytes = b"synthetic-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# --- historical cutoff -----------------------------------------------------------


def test_compute_historical_cutoff_is_midnight_of_the_day_48h_before():
    now_utc = datetime(2026, 9, 4, 20, 0, 0, tzinfo=UTC)

    cutoff = acq.compute_historical_cutoff(now_utc)

    # Sanity-check the 48h arithmetic against the module's own buffer constant.
    safe_moment = now_utc - timedelta(hours=acq.CUTOFF_BUFFER_HOURS)
    assert safe_moment == datetime(2026, 9, 2, 20, 0, 0, tzinfo=UTC)
    assert cutoff == datetime(2026, 9, 2, 0, 0, 0, tzinfo=UTC)


def test_compute_historical_cutoff_handles_a_month_boundary():
    # 48h before this moment crosses from March into February.
    now_utc = datetime(2025, 3, 1, 5, 0, 0, tzinfo=UTC)

    cutoff = acq.compute_historical_cutoff(now_utc)

    safe_moment = now_utc - timedelta(hours=acq.CUTOFF_BUFFER_HOURS)
    assert safe_moment == datetime(2025, 2, 27, 5, 0, 0, tzinfo=UTC)
    assert cutoff == datetime(2025, 2, 27, 0, 0, 0, tzinfo=UTC)


# --- monthly chunks ----------------------------------------------------------------


def test_generate_monthly_chunks_within_one_month():
    start = datetime(2024, 5, 10, tzinfo=UTC)
    end = datetime(2024, 5, 20, tzinfo=UTC)

    chunks = acq.generate_monthly_chunks(start, end)

    assert chunks == [(start, end)]


def test_generate_monthly_chunks_spans_multiple_months():
    start = datetime(2024, 1, 15, tzinfo=UTC)
    end = datetime(2024, 4, 10, tzinfo=UTC)

    chunks = acq.generate_monthly_chunks(start, end)

    assert chunks == [
        (datetime(2024, 1, 15, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)),
        (datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC)),
        (datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC)),
        (datetime(2024, 4, 1, tzinfo=UTC), datetime(2024, 4, 10, tzinfo=UTC)),
    ]
    # No gaps or overlaps: each chunk's end is exactly the next chunk's start.
    for (_, chunk_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
        assert chunk_end == next_start


def test_generate_monthly_chunks_handles_december_to_january_year_rollover():
    start = datetime(2023, 12, 10, tzinfo=UTC)
    end = datetime(2024, 2, 5, tzinfo=UTC)

    chunks = acq.generate_monthly_chunks(start, end)

    assert chunks == [
        (datetime(2023, 12, 10, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)),
        (datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 5, tzinfo=UTC)),
    ]
    # The year increments correctly at the December -> January boundary.
    assert chunks[0][1] == datetime(2024, 1, 1, tzinfo=UTC)
    assert chunks[1][0] == datetime(2024, 1, 1, tzinfo=UTC)


def test_generate_monthly_chunks_empty_when_start_not_before_end():
    same_instant = datetime(2024, 6, 1, tzinfo=UTC)

    after = datetime(2024, 6, 2, tzinfo=UTC)
    before = datetime(2024, 6, 1, tzinfo=UTC)

    assert acq.generate_monthly_chunks(same_instant, same_instant) == []
    assert acq.generate_monthly_chunks(after, before) == []


# --- yearly chunks -----------------------------------------------------------------


def test_generate_yearly_chunks_within_one_year():
    start = datetime(2024, 3, 1, tzinfo=UTC)
    end = datetime(2024, 9, 1, tzinfo=UTC)

    chunks = acq.generate_yearly_chunks(start, end)

    assert chunks == [(start, end)]


def test_generate_yearly_chunks_spans_multiple_years_including_leap_year():
    start = datetime(2019, 6, 1, tzinfo=UTC)
    end = datetime(2021, 3, 1, tzinfo=UTC)

    chunks = acq.generate_yearly_chunks(start, end)

    assert chunks == [
        (datetime(2019, 6, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)),
        (datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)),
        (datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 3, 1, tzinfo=UTC)),
    ]
    # Boundaries land on Jan 1 of both intervening years.
    assert chunks[0][1] == datetime(2020, 1, 1, tzinfo=UTC)
    assert chunks[1][0] == datetime(2020, 1, 1, tzinfo=UTC)
    assert chunks[1][1] == datetime(2021, 1, 1, tzinfo=UTC)
    assert chunks[2][0] == datetime(2021, 1, 1, tzinfo=UTC)

    # The chunk covering all of 2020 fully brackets the leap day, Feb 29 2020.
    leap_day = datetime(2020, 2, 29, tzinfo=UTC)
    year_2020_start, year_2020_end = chunks[1]
    assert year_2020_start <= leap_day < year_2020_end


def test_generate_yearly_chunks_empty_when_start_not_before_end():
    same_instant = datetime(2024, 1, 1, tzinfo=UTC)

    after = datetime(2024, 6, 1, tzinfo=UTC)
    before = datetime(2024, 1, 1, tzinfo=UTC)

    assert acq.generate_yearly_chunks(same_instant, same_instant) == []
    assert acq.generate_yearly_chunks(after, before) == []


# --- manifest schema / idempotency --------------------------------------------------


def _default_record_kwargs(file_path: Path) -> dict:
    """Baseline kwargs for `record_acquisition`; tests override individual fields."""

    return {
        "provider": "CopernicusMarine",
        "product_id": "GLOBAL_ANALYSISFORECAST_PHY_001_024",
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "evidence_role": "metocean_hindcast",
        "variables": ["thetao", "so"],
        "requested_bbox": [-10.0, 50.0, 2.0, 60.0],
        "requested_depths": [0.0, 10.0],
        "requested_start": datetime(2024, 1, 1, tzinfo=UTC),
        "requested_end": datetime(2024, 2, 1, tzinfo=UTC),
        "actual_start": "2024-01-01T00:00:00",
        "actual_end": "2024-01-31T23:00:00",
        "temporal_resolution": "P1D",
        "local_path": file_path,
        "toolbox_version": "1.6.2",
        "licence": "Copernicus Marine Licence",
        "downloaded_at": datetime(2024, 2, 2, tzinfo=UTC),
    }


def test_record_acquisition_writes_all_manifest_fields(tmp_path: Path):
    file_path = _write_file(tmp_path / "data.nc", b"fake-netcdf-bytes")
    manifest_path = tmp_path / "manifest.json"
    kwargs = _default_record_kwargs(file_path)

    entry = acq.record_acquisition(manifest_path, **kwargs)

    for field_name in acq.MANIFEST_ENTRY_FIELDS:
        assert field_name in entry
    assert entry["sha256"] == acq.compute_sha256(file_path)
    assert entry["file_size_bytes"] == file_path.stat().st_size
    assert entry["requested_start"] == kwargs["requested_start"].isoformat()
    assert entry["requested_end"] == kwargs["requested_end"].isoformat()


def test_record_acquisition_static_dataset_uses_sentinel_not_crash(tmp_path: Path):
    file_path = _write_file(tmp_path / "static.nc", b"time-invariant-bytes")
    manifest_path = tmp_path / "manifest.json"
    kwargs = {**_default_record_kwargs(file_path), "requested_start": None, "requested_end": None}

    entry = acq.record_acquisition(manifest_path, **kwargs)

    assert entry["requested_start"] == acq.STATIC_TIME_SENTINEL
    assert entry["requested_end"] == acq.STATIC_TIME_SENTINEL


def test_already_acquired_returns_entry_when_identity_and_file_match(tmp_path: Path):
    file_path = _write_file(tmp_path / "data.nc", b"fake-netcdf-bytes")
    manifest_path = tmp_path / "manifest.json"
    kwargs = _default_record_kwargs(file_path)

    acq.record_acquisition(manifest_path, **kwargs)

    found = acq.already_acquired(
        manifest_path,
        product_id=kwargs["product_id"],
        dataset_id=kwargs["dataset_id"],
        variables=kwargs["variables"],
        requested_bbox=kwargs["requested_bbox"],
        requested_depths=kwargs["requested_depths"],
        requested_start=kwargs["requested_start"].isoformat(),
        requested_end=kwargs["requested_end"].isoformat(),
    )

    # Same request, chunk idempotency: an identical re-run must not re-download.
    assert found is not None
    assert found["local_path"] == str(file_path)


def test_already_acquired_none_when_variables_differ(tmp_path: Path):
    file_path = _write_file(tmp_path / "data.nc", b"fake-netcdf-bytes")
    manifest_path = tmp_path / "manifest.json"
    kwargs = _default_record_kwargs(file_path)

    acq.record_acquisition(manifest_path, **kwargs)

    found = acq.already_acquired(
        manifest_path,
        product_id=kwargs["product_id"],
        dataset_id=kwargs["dataset_id"],
        variables=["thetao"],  # differs from the recorded ["thetao", "so"]
        requested_bbox=kwargs["requested_bbox"],
        requested_depths=kwargs["requested_depths"],
        requested_start=kwargs["requested_start"].isoformat(),
        requested_end=kwargs["requested_end"].isoformat(),
    )

    # A request that actually differs must never be silently skipped.
    assert found is None


def test_already_acquired_none_when_file_deleted(tmp_path: Path):
    file_path = _write_file(tmp_path / "data.nc", b"fake-netcdf-bytes")
    manifest_path = tmp_path / "manifest.json"
    kwargs = _default_record_kwargs(file_path)

    acq.record_acquisition(manifest_path, **kwargs)
    file_path.unlink()

    found = acq.already_acquired(
        manifest_path,
        product_id=kwargs["product_id"],
        dataset_id=kwargs["dataset_id"],
        variables=kwargs["variables"],
        requested_bbox=kwargs["requested_bbox"],
        requested_depths=kwargs["requested_depths"],
        requested_start=kwargs["requested_start"].isoformat(),
        requested_end=kwargs["requested_end"].isoformat(),
    )

    # The manifest entry alone is not enough -- the file must still exist on disk.
    assert found is None


def test_record_acquisition_upserts_same_identity_not_duplicates(tmp_path: Path):
    file_path = _write_file(tmp_path / "data.nc", b"v1-bytes")
    manifest_path = tmp_path / "manifest.json"
    kwargs = _default_record_kwargs(file_path)

    acq.record_acquisition(manifest_path, **kwargs)

    # Simulate a re-run after interruption, for the exact same identity.
    file_path.write_bytes(b"v2-bytes-longer-than-v1")
    acq.record_acquisition(manifest_path, **kwargs)

    entries = acq.load_manifest(manifest_path)

    assert len(entries) == 1  # no duplicate row
    assert entries[0]["sha256"] == acq.compute_sha256(file_path)  # reflects the re-run, not v1


def test_isoformat_or_static_sentinel():
    moment = datetime(2024, 5, 1, 12, 30, 0, tzinfo=UTC)

    assert acq.isoformat_or_static_sentinel(moment) == moment.isoformat()
    assert acq.isoformat_or_static_sentinel(None) == acq.STATIC_TIME_SENTINEL


# --- deduplicate_time_coordinate (MAR-009B) -----------------------------------------
#
# Synthetic datasets shaped like the three real products -- never the real
# PL854 data, never network access. Each simulates the real bug: Copernicus
# subset() is inclusive of `end_datetime`, so two adjacent monthly/yearly
# acquisition chunks each carry their shared boundary instant, and
# concatenating them (`xr.open_mfdataset(..., combine="by_coords")`) does
# not itself deduplicate.


def _make_4d_current_dataset(
    times, uo_values, vo_values, *, depths=(0.0,), latitudes=(53.0,), longitudes=(1.0,)
) -> xr.Dataset:
    """Primary-current-shaped (time, depth, latitude, longitude)."""

    return xr.Dataset(
        {
            "uo": (("time", "depth", "latitude", "longitude"), uo_values),
            "vo": (("time", "depth", "latitude", "longitude"), vo_values),
        },
        coords={
            "time": times,
            "depth": list(depths),
            "latitude": list(latitudes),
            "longitude": list(longitudes),
        },
    )


def _make_surface_current_dataset(
    times, uo_values, vo_values, *, latitudes=(53.0,), longitudes=(1.0,)
) -> xr.Dataset:
    """Long-term-surface-current-shaped (time, latitude, longitude)."""

    return xr.Dataset(
        {
            "uo": (("time", "latitude", "longitude"), uo_values),
            "vo": (("time", "latitude", "longitude"), vo_values),
        },
        coords={"time": times, "latitude": list(latitudes), "longitude": list(longitudes)},
    )


def _make_wave_dataset(times, hs_values, *, latitudes=(53.0,), longitudes=(1.0,)) -> xr.Dataset:
    """Wave-shaped (time, latitude, longitude), real variable name VHM0."""

    return xr.Dataset(
        {"VHM0": (("time", "latitude", "longitude"), hs_values)},
        coords={"time": times, "latitude": list(latitudes), "longitude": list(longitudes)},
    )


def test_deduplicate_time_coordinate_noop_when_no_duplicates():
    times = pd.date_range("2024-07-01", periods=4, freq="h", tz="UTC")
    shape = (4, 1, 1, 1)
    ds = _make_4d_current_dataset(times, np.arange(4.0).reshape(shape), np.zeros(shape))

    deduped, result = acq.deduplicate_time_coordinate(ds)

    assert result == acq.TemporalDeduplicationResult(4, 4, 0)
    assert len(deduped["time"]) == 4


def test_deduplicate_time_coordinate_collapses_consistent_monthly_boundary():
    """Two monthly current chunks sharing exactly one boundary hour, consistent data."""

    times_a = pd.date_range("2024-07-01", periods=4, freq="h", tz="UTC")  # hours 0,1,2,3
    times_b = pd.date_range("2024-07-01 03:00", periods=4, freq="h", tz="UTC")  # hours 3,4,5,6
    shape = (4, 1, 1, 1)
    uo_a = np.array([0.0, 1.0, 2.0, 3.0]).reshape(shape)
    vo_a = np.array([0.0, 2.0, 4.0, 6.0]).reshape(shape)
    # chunk_b's hour-3 (its own index 0) must match chunk_a's hour-3 (index 3).
    uo_b = np.array([3.0, 4.0, 5.0, 6.0]).reshape(shape)
    vo_b = np.array([6.0, 8.0, 10.0, 12.0]).reshape(shape)
    chunk_a = _make_4d_current_dataset(times_a, uo_a, vo_a)
    chunk_b = _make_4d_current_dataset(times_b, uo_b, vo_b)
    combined = xr.concat([chunk_a, chunk_b], dim="time")
    assert len(combined["time"]) == 8  # raw, still overlapping

    deduped, result = acq.deduplicate_time_coordinate(combined)

    assert result == acq.TemporalDeduplicationResult(
        raw_time_count=8, unique_time_count=7, duplicate_boundary_timestamp_count=1
    )
    assert len(deduped["time"]) == 7
    deduped_time_index = pd.Index(deduped["time"].to_numpy())
    assert deduped_time_index.is_unique
    assert deduped_time_index.is_monotonic_increasing
    # The single canonical hour-3 row keeps its (consistent) value, not a duplicate.
    assert float(deduped["uo"].isel(time=3).to_numpy().item()) == pytest.approx(3.0)


def test_deduplicate_time_coordinate_raises_on_inconsistent_monthly_boundary():
    """Same shape as above, but the two chunks disagree at the shared boundary hour."""

    times_a = pd.date_range("2024-07-01", periods=4, freq="h", tz="UTC")
    times_b = pd.date_range("2024-07-01 03:00", periods=4, freq="h", tz="UTC")
    shape = (4, 1, 1, 1)
    uo_a = np.array([0.0, 1.0, 2.0, 3.0]).reshape(shape)
    vo_a = np.zeros(shape)
    uo_b = np.array([999.0, 4.0, 5.0, 6.0]).reshape(shape)  # disagrees with chunk_a's 3.0
    vo_b = np.zeros(shape)
    combined = xr.concat(
        [
            _make_4d_current_dataset(times_a, uo_a, vo_a),
            _make_4d_current_dataset(times_b, uo_b, vo_b),
        ],
        dim="time",
    )

    with pytest.raises(acq.DuplicateTimestampConflictError):
        acq.deduplicate_time_coordinate(combined)


def test_deduplicate_time_coordinate_treats_nan_as_equal_to_nan():
    """A genuinely absent sample at the boundary must not be flagged as a conflict."""

    times_a = pd.date_range("2024-07-01", periods=4, freq="h", tz="UTC")
    times_b = pd.date_range("2024-07-01 03:00", periods=4, freq="h", tz="UTC")
    shape = (4, 1, 1, 1)
    uo_a = np.array([0.0, 1.0, 2.0, np.nan]).reshape(shape)
    uo_b = np.array([np.nan, 4.0, 5.0, 6.0]).reshape(shape)  # shared hour: NaN in both
    zeros = np.zeros(shape)
    combined = xr.concat(
        [
            _make_4d_current_dataset(times_a, uo_a, zeros),
            _make_4d_current_dataset(times_b, uo_b, zeros),
        ],
        dim="time",
    )

    deduped, result = acq.deduplicate_time_coordinate(combined)

    assert result.duplicate_boundary_timestamp_count == 1
    assert np.isnan(deduped["uo"].isel(time=3).to_numpy().item())


def test_deduplicate_time_coordinate_yearly_long_term_current_overlap():
    """Two yearly long-term-current chunks sharing the Dec31/Jan1 boundary hour."""

    times_2024 = pd.date_range("2024-12-31 22:00", periods=3, freq="h", tz="UTC")
    times_2025 = pd.date_range("2025-01-01 00:00", periods=3, freq="h", tz="UTC")
    shape = (3, 1, 1)
    uo_2024 = np.array([0.1, 0.2, 0.3]).reshape(shape)
    uo_2025 = np.array([0.3, 0.4, 0.5]).reshape(shape)  # index 0 (Jan 1 00:00) matches 0.3
    zeros = np.zeros(shape)
    combined = xr.concat(
        [
            _make_surface_current_dataset(times_2024, uo_2024, zeros),
            _make_surface_current_dataset(times_2025, uo_2025, zeros),
        ],
        dim="time",
    )
    assert len(combined["time"]) == 6

    deduped, result = acq.deduplicate_time_coordinate(combined)

    assert result == acq.TemporalDeduplicationResult(6, 5, 1)
    assert len(deduped["time"]) == 5
    assert pd.Index(deduped["time"].to_numpy()).is_unique


def test_deduplicate_time_coordinate_yearly_wave_overlap():
    """Two yearly wave chunks sharing the Dec31/Jan1 3-hourly boundary step."""

    times_2024 = pd.date_range("2024-12-31 18:00", periods=3, freq="3h", tz="UTC")
    times_2025 = pd.date_range("2025-01-01 00:00", periods=3, freq="3h", tz="UTC")
    shape = (3, 1, 1)
    hs_2024 = np.array([1.0, 1.2, 1.4]).reshape(shape)
    hs_2025 = np.array([1.4, 1.6, 1.8]).reshape(shape)  # index 0 (Jan 1 00:00) matches 1.4
    combined = xr.concat(
        [_make_wave_dataset(times_2024, hs_2024), _make_wave_dataset(times_2025, hs_2025)],
        dim="time",
    )
    assert len(combined["time"]) == 6

    deduped, result = acq.deduplicate_time_coordinate(combined)

    assert result == acq.TemporalDeduplicationResult(6, 5, 1)
    assert len(deduped["time"]) == 5


def test_deduplicate_time_coordinate_sorts_regardless_of_concatenation_order():
    """Even if chunks were concatenated out of chronological order, output is sorted."""

    times_a = pd.date_range("2024-07-01", periods=4, freq="h", tz="UTC")
    times_b = pd.date_range("2024-07-01 03:00", periods=4, freq="h", tz="UTC")
    shape = (4, 1, 1, 1)
    uo_a = np.array([0.0, 1.0, 2.0, 3.0]).reshape(shape)
    uo_b = np.array([3.0, 4.0, 5.0, 6.0]).reshape(shape)
    zeros = np.zeros(shape)
    # Deliberately reversed concatenation order (chunk_b BEFORE chunk_a).
    combined = xr.concat(
        [
            _make_4d_current_dataset(times_b, uo_b, zeros),
            _make_4d_current_dataset(times_a, uo_a, zeros),
        ],
        dim="time",
    )

    deduped, result = acq.deduplicate_time_coordinate(combined)

    assert result.duplicate_boundary_timestamp_count == 1
    deduped_times = pd.Index(deduped["time"].to_numpy())
    assert deduped_times.is_monotonic_increasing
    assert list(deduped_times) == sorted(deduped_times)
