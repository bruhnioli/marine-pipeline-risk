"""Offline unit tests for marine_engine.providers.bathymetry.acquisition.

No network access -- these operate on locally-written files only.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from marine_engine.providers.bathymetry import acquisition as acq


def _write_file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# --- raw-path policy -----------------------------------------------------------


def test_raw_dataset_dir_layout():
    result = acq.raw_dataset_dir(Path("data/raw"), "BGS", "GB02SS0001")
    assert result == Path("data/raw/bathymetry/bgs/GB02SS0001")


def test_raw_dataset_dir_lowercases_source_only():
    result = acq.raw_dataset_dir(Path("data/raw"), "EMODnet", "emodnet__mean")
    assert result == Path("data/raw/bathymetry/emodnet/emodnet__mean")


# --- SHA-256 --------------------------------------------------------------------


def test_compute_sha256_matches_hashlib_reference(tmp_path: Path):
    content = b"some raw bathymetry bytes, unmodified"
    file_path = _write_file(tmp_path / "sample.tif", content)

    assert acq.compute_sha256(file_path) == hashlib.sha256(content).hexdigest()


def test_compute_sha256_differs_for_different_content(tmp_path: Path):
    a = _write_file(tmp_path / "a.tif", b"aaaa")
    b = _write_file(tmp_path / "b.tif", b"bbbb")

    assert acq.compute_sha256(a) != acq.compute_sha256(b)


# --- manifest schema -------------------------------------------------------------


def test_record_acquisition_writes_all_required_fields(tmp_path: Path):
    file_path = _write_file(tmp_path / "emodnet__mean.tif", b"fake-geotiff-bytes")
    manifest_path = tmp_path / "manifest.json"

    entry = acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://ows.emodnet-bathymetry.eu/wcs",
        request_parameters={"coverageId": "emodnet__mean"},
        local_path=file_path,
        licence="CC BY 4.0",
        acquisition_year=2024,
        horizontal_crs="EPSG:4326",
        vertical_datum="LAT",
        nominal_resolution_m=115.0,
        acquired_at=datetime.now(UTC),
    )

    for field_name in acq.MANIFEST_ENTRY_FIELDS:
        assert field_name in entry
    assert entry["raw_unmodified"] is True
    assert entry["file_size_bytes"] == file_path.stat().st_size
    assert entry["sha256"] == acq.compute_sha256(file_path)


def test_manifest_persists_across_loads(tmp_path: Path):
    file_path = _write_file(tmp_path / "a.tif", b"content")
    manifest_path = tmp_path / "manifest.json"

    acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://example.invalid",
        request_parameters={},
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    reloaded = acq.load_manifest(manifest_path)
    assert len(reloaded) == 1
    assert reloaded[0]["dataset_id"] == "emodnet__mean"


# --- idempotency -----------------------------------------------------------------


def test_record_acquisition_is_idempotent_by_source_and_dataset_id(tmp_path: Path):
    file_path = _write_file(tmp_path / "a.tif", b"v1")
    manifest_path = tmp_path / "manifest.json"

    acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://example.invalid",
        request_parameters={"v": 1},
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    # Re-run for the SAME source+dataset_id (e.g. a refreshed pull of the
    # same current release) -- must replace, not duplicate.
    file_path.write_bytes(b"v2")
    acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://example.invalid",
        request_parameters={"v": 2},
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    entries = acq.load_manifest(manifest_path)
    assert len(entries) == 1  # no duplicate row
    assert entries[0]["request_parameters"] == {"v": 2}


def test_different_dataset_id_never_overwrites_another_epoch(tmp_path: Path):
    file_2022 = _write_file(tmp_path / "2022.tif", b"2022-data")
    file_2024 = _write_file(tmp_path / "2024.tif", b"2024-data")
    manifest_path = tmp_path / "manifest.json"

    for dataset_id, path in (("emodnet__mean_2022", file_2022), ("emodnet__mean", file_2024)):
        acq.record_acquisition(
            manifest_path,
            source="EMODnet",
            dataset_id=dataset_id,
            source_url_or_service="https://example.invalid",
            request_parameters={},
            local_path=path,
            licence=None,
            acquisition_year=None,
            horizontal_crs=None,
            vertical_datum=None,
            nominal_resolution_m=None,
            acquired_at=datetime.now(UTC),
        )

    entries = acq.load_manifest(manifest_path)
    assert len(entries) == 2  # both epochs preserved as distinct entries
    assert file_2022.read_bytes() == b"2022-data"  # untouched
    assert file_2024.read_bytes() == b"2024-data"  # untouched


def test_already_acquired_skips_when_file_and_params_match(tmp_path: Path):
    file_path = _write_file(tmp_path / "a.tif", b"content")
    manifest_path = tmp_path / "manifest.json"
    params = {"coverageId": "emodnet__mean", "bbox_wgs84": [1, 2, 3, 4]}

    acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://example.invalid",
        request_parameters=params,
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    existing = acq.already_acquired(manifest_path, "EMODnet", "emodnet__mean", params)
    assert existing is not None
    assert existing["local_path"] == str(file_path)


def test_already_acquired_none_when_params_differ(tmp_path: Path):
    file_path = _write_file(tmp_path / "a.tif", b"content")
    manifest_path = tmp_path / "manifest.json"

    acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://example.invalid",
        request_parameters={"bbox_wgs84": [1, 2, 3, 4]},
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    # A different (e.g. regenerated) AOI bbox must NOT be treated as already acquired.
    existing = acq.already_acquired(
        manifest_path, "EMODnet", "emodnet__mean", {"bbox_wgs84": [9, 9, 9, 9]}
    )
    assert existing is None


def test_already_acquired_none_when_file_deleted(tmp_path: Path):
    file_path = _write_file(tmp_path / "a.tif", b"content")
    manifest_path = tmp_path / "manifest.json"
    params = {"v": 1}

    acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://example.invalid",
        request_parameters=params,
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    file_path.unlink()

    assert acq.already_acquired(manifest_path, "EMODnet", "emodnet__mean", params) is None
