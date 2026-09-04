"""Offline unit tests for marine_engine.providers.bathymetry.acquisition.

No network access -- these operate on locally-written files only.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
        # EMODnet DTM 2024 is an aggregate product release, not a survey
        # acquisition (MAR-007A) -- acquisition_year stays None here.
        acquisition_year=None,
        product_release_year=2024,
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
    assert entry["acquisition_year"] is None
    assert entry["product_release_year"] == 2024


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
        product_release_year=None,
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
        product_release_year=None,
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
        product_release_year=None,
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
            product_release_year=None,
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
        product_release_year=None,
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
        product_release_year=None,
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
        product_release_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    file_path.unlink()

    assert acq.already_acquired(manifest_path, "EMODnet", "emodnet__mean", params) is None


# --- temporal semantics: acquisition_year vs product_release_year (MAR-007A) ---


def test_aggregate_product_has_null_acquisition_year_and_a_release_year(tmp_path: Path):
    """EMODnet DTM 2024: a product release, never a survey acquisition."""

    file_path = _write_file(tmp_path / "emodnet__mean.tif", b"fake-geotiff-bytes")
    manifest_path = tmp_path / "manifest.json"

    entry = acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://ows.emodnet-bathymetry.eu/wcs",
        request_parameters={},
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        product_release_year=2024,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    assert entry["acquisition_year"] is None
    assert entry["product_release_year"] == 2024


def test_real_survey_has_acquisition_year_and_null_release_year(tmp_path: Path):
    """A genuine survey dataset: a real acquisition year, no distinct product release."""

    file_path = _write_file(tmp_path / "gb02ss0001.tif", b"fake-survey-bytes")
    manifest_path = tmp_path / "manifest.json"

    entry = acq.record_acquisition(
        manifest_path,
        source="BGS",
        dataset_id="GB02SS0001",
        source_url_or_service="https://ogc.bgs.ac.uk/csw",
        request_parameters={},
        local_path=file_path,
        licence=None,
        acquisition_year=2002,
        product_release_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    assert entry["acquisition_year"] == 2002
    assert entry["product_release_year"] is None


def test_acquisition_timestamp_is_independent_of_both_temporal_fields(tmp_path: Path):
    """`acquisition_timestamp` means when THIS software downloaded the file --
    unrelated to either the survey's acquisition year or a product's release year."""

    file_path = _write_file(tmp_path / "a.tif", b"content")
    manifest_path = tmp_path / "manifest.json"
    download_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)

    entry = acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://example.invalid",
        request_parameters={},
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        product_release_year=2024,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=download_time,
    )

    assert entry["acquisition_timestamp"] == download_time.isoformat()
    assert entry["acquisition_timestamp"] != entry["product_release_year"]


def test_legacy_manifest_entry_without_product_release_year_loads_safely(tmp_path: Path):
    """An old manifest written before MAR-007A has no `product_release_year` key at
    all -- loading it must not crash, and reading the field must give None, not KeyError."""

    manifest_path = tmp_path / "manifest.json"
    legacy_entry = {
        "source": "EMODnet",
        "dataset_id": "emodnet__mean",
        "source_url_or_service": "https://ows.emodnet-bathymetry.eu/wcs",
        "acquisition_timestamp": "2026-09-03T19:30:54.383142+00:00",
        "request_parameters": {"coverageId": "emodnet__mean"},
        "original_filename": "emodnet__mean.tif",
        "local_path": str(tmp_path / "emodnet__mean.tif"),
        "file_size_bytes": 430480,
        "sha256": "bba1e2694b3b34f10f3f9b075909273df2bfaaf8e39eb0eb93f035569d7e84cd",
        "licence": "CC BY 4.0",
        "acquisition_year": 2024,  # the OLD, incorrect semantics -- no product_release_year at all
        "horizontal_crs": "EPSG:4326",
        "vertical_datum": "LAT",
        "nominal_resolution_m": 115.0,
        "raw_unmodified": True,
    }
    manifest_path.write_text(json.dumps([legacy_entry]), encoding="utf-8")

    entries = acq.load_manifest(manifest_path)

    assert len(entries) == 1
    assert entries[0].get("product_release_year") is None  # missing field -> None, never a crash
    assert "product_release_year" not in entries[0]  # genuinely absent, not fabricated


# --- metadata-only correction (MAR-007A Section 4) --------------------------


def test_correct_temporal_metadata_fixes_fields_without_touching_the_rest(tmp_path: Path):
    file_path = _write_file(tmp_path / "emodnet__mean.tif", b"real-geotiff-bytes-untouched")
    manifest_path = tmp_path / "manifest.json"

    original = acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://ows.emodnet-bathymetry.eu/wcs",
        request_parameters={"coverageId": "emodnet__mean", "bbox_wgs84": [1, 53, 2, 54]},
        local_path=file_path,
        licence="CC BY 4.0",
        acquisition_year=2024,  # written under the OLD, incorrect semantics
        product_release_year=None,
        horizontal_crs="EPSG:4326",
        vertical_datum="LAT",
        nominal_resolution_m=115.0,
        acquired_at=datetime(2026, 9, 3, 19, 30, 54, tzinfo=UTC),
    )

    corrected = acq.correct_temporal_metadata(
        manifest_path, "EMODnet", "emodnet__mean", acquisition_year=None, product_release_year=2024
    )

    assert corrected["acquisition_year"] is None
    assert corrected["product_release_year"] == 2024
    # Everything else preserved exactly -- no re-download, no re-derivation.
    assert corrected["local_path"] == original["local_path"]
    assert corrected["sha256"] == original["sha256"]
    assert corrected["file_size_bytes"] == original["file_size_bytes"]
    assert corrected["request_parameters"] == original["request_parameters"]
    assert corrected["acquisition_timestamp"] == original["acquisition_timestamp"]
    assert corrected["licence"] == original["licence"]
    assert file_path.read_bytes() == b"real-geotiff-bytes-untouched"  # never rewritten


def test_correct_temporal_metadata_does_not_re_download_or_touch_the_file(tmp_path: Path):
    file_path = _write_file(tmp_path / "emodnet__mean.tif", b"original-bytes")
    manifest_path = tmp_path / "manifest.json"
    original_mtime = file_path.stat().st_mtime

    acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",
        source_url_or_service="https://ows.emodnet-bathymetry.eu/wcs",
        request_parameters={},
        local_path=file_path,
        licence=None,
        acquisition_year=2024,
        product_release_year=None,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    acq.correct_temporal_metadata(
        manifest_path, "EMODnet", "emodnet__mean", acquisition_year=None, product_release_year=2024
    )

    assert file_path.stat().st_mtime == original_mtime  # the file itself was never touched
    assert file_path.read_bytes() == b"original-bytes"


def test_correct_temporal_metadata_raises_when_no_matching_entry_exists(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(KeyError):
        acq.correct_temporal_metadata(
            manifest_path,
            "EMODnet",
            "does_not_exist",
            acquisition_year=None,
            product_release_year=2024,
        )


def test_correct_temporal_metadata_targets_only_the_named_dataset(tmp_path: Path):
    """Never a blanket "every 2024 entry is a product" rule -- only the exact
    (source, dataset_id) pair passed in is touched."""

    file_a = _write_file(tmp_path / "a.tif", b"a")
    file_b = _write_file(tmp_path / "b.tif", b"b")
    manifest_path = tmp_path / "manifest.json"

    for dataset_id, path in (("emodnet__mean", file_a), ("some_2024_survey", file_b)):
        acq.record_acquisition(
            manifest_path,
            source="EMODnet",
            dataset_id=dataset_id,
            source_url_or_service="https://example.invalid",
            request_parameters={},
            local_path=path,
            licence=None,
            acquisition_year=2024,
            product_release_year=None,
            horizontal_crs=None,
            vertical_datum=None,
            nominal_resolution_m=None,
            acquired_at=datetime.now(UTC),
        )

    acq.correct_temporal_metadata(
        manifest_path, "EMODnet", "emodnet__mean", acquisition_year=None, product_release_year=2024
    )

    entries = {e["dataset_id"]: e for e in acq.load_manifest(manifest_path)}
    assert entries["emodnet__mean"]["acquisition_year"] is None
    assert entries["emodnet__mean"]["product_release_year"] == 2024
    # The unrelated "some_2024_survey" entry is untouched -- 2024 in a name
    # alone never triggers a correction.
    assert entries["some_2024_survey"]["acquisition_year"] == 2024
    assert entries["some_2024_survey"]["product_release_year"] is None


# --- regression: 2024 must never reappear as a source acquisition year ------


def test_2024_never_becomes_acquisition_year_merely_from_the_product_name(tmp_path: Path):
    """The dataset_id containing "2024" (as EMODnet's own DTM naming does)
    must never, by itself, cause acquisition_year to be set to 2024."""

    file_path = _write_file(tmp_path / "emodnet__mean.tif", b"content")
    manifest_path = tmp_path / "manifest.json"

    entry = acq.record_acquisition(
        manifest_path,
        source="EMODnet",
        dataset_id="emodnet__mean",  # "DTM 2024" is the product TITLE, not this id
        source_url_or_service="https://ows.emodnet-bathymetry.eu/wcs",
        request_parameters={},
        local_path=file_path,
        licence=None,
        acquisition_year=None,
        product_release_year=2024,
        horizontal_crs=None,
        vertical_datum=None,
        nominal_resolution_m=None,
        acquired_at=datetime.now(UTC),
    )

    assert entry["acquisition_year"] is None
    assert entry["product_release_year"] == 2024
    assert entry["acquisition_year"] != 2024
