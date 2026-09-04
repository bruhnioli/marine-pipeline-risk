"""Offline unit tests for marine_engine.providers.metocean.acquisition.

No network access, no xarray -- these operate on locally-written synthetic
files and in-memory data only.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

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
