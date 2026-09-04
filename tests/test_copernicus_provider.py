"""Offline unit tests for marine_engine.providers.metocean.copernicus.

Fully offline: never calls the real `copernicusmarine` package, never
touches the network, and never requires any Copernicus Marine credentials.
Every test monkeypatches the exact attributes the module under test calls
(`copernicus.copernicusmarine.login` / `.describe` / `.subset`) with small
fakes built from `types.SimpleNamespace`, mirroring the real Toolbox's
pydantic-model attribute access (`.products`, `.datasets`, ... rather than
dict indexing) that `copernicus.py` relies on throughout.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from marine_engine.providers.metocean import copernicus

# --- ensure_authenticated ------------------------------------------------------


def test_ensure_authenticated_raises_when_login_check_returns_false(monkeypatch):
    def fake_login(*, check_credentials_valid):
        assert check_credentials_valid is True
        return False

    monkeypatch.setattr(copernicus.copernicusmarine, "login", fake_login)

    with pytest.raises(copernicus.CopernicusAuthenticationRequiredError):
        copernicus.ensure_authenticated()


def test_ensure_authenticated_does_not_raise_when_login_check_returns_true(monkeypatch):
    def fake_login(*, check_credentials_valid):
        return True

    monkeypatch.setattr(copernicus.copernicusmarine, "login", fake_login)

    copernicus.ensure_authenticated()


# --- confirm_live_dataset_id ---------------------------------------------------


def _catalogue_with_dataset_ids(dataset_ids: list[str]) -> SimpleNamespace:
    datasets = [SimpleNamespace(dataset_id=dataset_id) for dataset_id in dataset_ids]
    return SimpleNamespace(products=[SimpleNamespace(datasets=datasets)])


def test_confirm_live_dataset_id_returns_id_when_found(monkeypatch):
    expected_id = "cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i"
    catalogue = _catalogue_with_dataset_ids(["other_dataset_a", expected_id, "other_dataset_b"])

    def fake_describe(*, product_id, disable_progress_bar):
        assert product_id == "NWSHELF_ANALYSISFORECAST_PHY_004_013"
        assert disable_progress_bar is True
        return catalogue

    monkeypatch.setattr(copernicus.copernicusmarine, "describe", fake_describe)

    result = copernicus.confirm_live_dataset_id("NWSHELF_ANALYSISFORECAST_PHY_004_013", expected_id)

    assert result == expected_id


def test_confirm_live_dataset_id_raises_when_not_found_and_lists_available(monkeypatch):
    catalogue = _catalogue_with_dataset_ids(["available_dataset_one", "available_dataset_two"])

    def fake_describe(*, product_id, disable_progress_bar):
        return catalogue

    monkeypatch.setattr(copernicus.copernicusmarine, "describe", fake_describe)

    with pytest.raises(copernicus.CopernicusDatasetNotFoundError) as exc_info:
        copernicus.confirm_live_dataset_id("SOME_PRODUCT_ID", "missing_dataset_id")

    message = str(exc_info.value)
    assert "available_dataset_one" in message
    assert "available_dataset_two" in message


def test_confirm_live_dataset_id_raises_when_no_products(monkeypatch):
    empty_catalogue = SimpleNamespace(products=[])

    def fake_describe(*, product_id, disable_progress_bar):
        return empty_catalogue

    monkeypatch.setattr(copernicus.copernicusmarine, "describe", fake_describe)

    with pytest.raises(copernicus.CopernicusDatasetNotFoundError):
        copernicus.confirm_live_dataset_id("SOME_PRODUCT_ID", "some_dataset_id")


# --- get_dataset_time_range_ms -------------------------------------------------


def _catalogue_with_coordinates(coordinates: list[SimpleNamespace]) -> SimpleNamespace:
    """A minimal but fully nested fake catalogue matching `_find_time_range_ms`'s walk.

    products -> datasets -> versions -> parts -> services -> variables -> coordinates
    """

    variable = SimpleNamespace(coordinates=coordinates)
    service = SimpleNamespace(variables=[variable])
    part = SimpleNamespace(services=[service])
    version = SimpleNamespace(parts=[part])
    dataset = SimpleNamespace(versions=[version])
    product = SimpleNamespace(datasets=[dataset])
    return SimpleNamespace(products=[product])


def test_get_dataset_time_range_ms_extracts_min_max(monkeypatch):
    catalogue = _catalogue_with_coordinates(
        [SimpleNamespace(coordinate_id="time", minimum_value=1000, maximum_value=9000)]
    )

    def fake_describe(*, dataset_id, disable_progress_bar):
        assert dataset_id == "some_dataset_id"
        assert disable_progress_bar is True
        return catalogue

    monkeypatch.setattr(copernicus.copernicusmarine, "describe", fake_describe)

    result = copernicus.get_dataset_time_range_ms("some_dataset_id")

    assert result == (1000, 9000)


def test_get_dataset_time_range_ms_returns_none_when_no_time_coordinate(monkeypatch):
    catalogue = _catalogue_with_coordinates(
        [
            SimpleNamespace(coordinate_id="latitude", minimum_value=-10, maximum_value=10),
            SimpleNamespace(coordinate_id="longitude", minimum_value=-20, maximum_value=20),
        ]
    )

    def fake_describe(*, dataset_id, disable_progress_bar):
        return catalogue

    monkeypatch.setattr(copernicus.copernicusmarine, "describe", fake_describe)

    result = copernicus.get_dataset_time_range_ms("some_dataset_id")

    assert result is None


# --- toolbox_version ------------------------------------------------------------


def test_toolbox_version_returns_copernicusmarine_version(monkeypatch):
    monkeypatch.setattr(copernicus.copernicusmarine, "__version__", "1.2.3")

    assert copernicus.toolbox_version() == "1.2.3"


# --- subset_dataset --------------------------------------------------------------


def _subset_dataset_kwargs(*, output_directory: Path, output_filename: str = "out.nc") -> dict:
    return {
        "dataset_id": "x",
        "variables": ["uo"],
        "minimum_longitude": 0,
        "maximum_longitude": 1,
        "minimum_latitude": 0,
        "maximum_latitude": 1,
        "start_datetime": None,
        "end_datetime": None,
        "minimum_depth": None,
        "maximum_depth": None,
        "output_directory": output_directory,
        "output_filename": output_filename,
    }


def test_subset_dataset_raises_before_calling_subset_when_unauthenticated(monkeypatch, tmp_path):
    subset_calls = []

    def fake_login(*, check_credentials_valid):
        return False

    def fake_subset(**kwargs):
        subset_calls.append(kwargs)
        return SimpleNamespace(file_path="unused", file_size=0, status="unused")

    monkeypatch.setattr(copernicus.copernicusmarine, "login", fake_login)
    monkeypatch.setattr(copernicus.copernicusmarine, "subset", fake_subset)

    with pytest.raises(copernicus.CopernicusAuthenticationRequiredError):
        copernicus.subset_dataset(**_subset_dataset_kwargs(output_directory=tmp_path))

    assert subset_calls == []


def test_subset_dataset_calls_subset_and_builds_result_when_authenticated(monkeypatch, tmp_path):
    output_path = tmp_path / "out.nc"
    output_path.write_bytes(b"x")

    def fake_login(*, check_credentials_valid):
        return True

    def fake_subset(**kwargs):
        return SimpleNamespace(
            file_path=str(output_path), file_size=1234, status="success", message="ok"
        )

    monkeypatch.setattr(copernicus.copernicusmarine, "login", fake_login)
    monkeypatch.setattr(copernicus.copernicusmarine, "subset", fake_subset)

    result = copernicus.subset_dataset(**_subset_dataset_kwargs(output_directory=tmp_path))

    assert result.local_path == output_path
    assert result.file_size_bytes == 1234
    assert result.status == "success"
    assert result.message == "ok"


def test_subset_dataset_result_message_defaults_to_none_when_absent(monkeypatch, tmp_path):
    output_path = tmp_path / "out.nc"
    output_path.write_bytes(b"x")

    def fake_login(*, check_credentials_valid):
        return True

    def fake_subset(**kwargs):
        return SimpleNamespace(file_path=str(output_path), file_size=1, status="ok")

    monkeypatch.setattr(copernicus.copernicusmarine, "login", fake_login)
    monkeypatch.setattr(copernicus.copernicusmarine, "subset", fake_subset)

    result = copernicus.subset_dataset(**_subset_dataset_kwargs(output_directory=tmp_path))

    assert result.message is None


def test_subset_dataset_creates_output_directory_when_missing(monkeypatch, tmp_path):
    nested_output_directory = tmp_path / "nested" / "dir"
    output_path = nested_output_directory / "out.nc"

    def fake_login(*, check_credentials_valid):
        return True

    def fake_subset(**kwargs):
        # Real `copernicusmarine.subset` writes into `output_directory`, so it
        # must already exist by the time this fake is called.
        assert nested_output_directory.is_dir()
        return SimpleNamespace(file_path=str(output_path), file_size=1, status="ok", message=None)

    monkeypatch.setattr(copernicus.copernicusmarine, "login", fake_login)
    monkeypatch.setattr(copernicus.copernicusmarine, "subset", fake_subset)

    assert not nested_output_directory.exists()

    copernicus.subset_dataset(**_subset_dataset_kwargs(output_directory=nested_output_directory))

    assert nested_output_directory.is_dir()
