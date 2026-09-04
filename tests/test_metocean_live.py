"""Live smoke tests against the real Copernicus Marine Toolbox catalogue.

Excluded from the default `pytest` run (see `-m "not live"` in
pyproject.toml) so the normal suite never depends on Copernicus Marine
being reachable. Run explicitly with:

    uv run pytest -m live

Every test here only reads public catalogue metadata
(`copernicusmarine.describe`), which needs network access but no
credentials -- except `test_small_one_day_subset_can_be_acquired`, the one
test that attempts a real data download. That one skips itself (rather
than failing) when `copernicus.subset_dataset` reports no credentials are
configured, which is the expected state on a machine that has never run
`copernicusmarine login`.
"""

from datetime import UTC, datetime, timedelta

import copernicusmarine
import pytest

from marine_engine.providers.metocean import copernicus

pytestmark = pytest.mark.live


def _described_dataset(dataset_id: str):
    """The one dataset object for `dataset_id` from a scoped `describe()` call."""

    result = copernicusmarine.describe(dataset_id=dataset_id, disable_progress_bar=True)
    return result.products[0].datasets[0]


def _variable_short_names(dataset) -> set[str]:
    """Every variable short name across all of `dataset`'s versions/parts/services.

    Walked defensively across every version and part (not just the first),
    mirroring the same defensive nested walk `copernicus.py` itself uses in
    `_find_time_range_ms` -- a dataset's variables can be split across more
    than one part (e.g. a static dataset's "default" and "bathy" parts), so
    trusting only the first part risks a false negative here.
    """

    return {
        variable.short_name
        for version in dataset.versions
        for part in version.parts
        for service in part.services
        for variable in service.variables
    }


def _variable_coordinate_ids(
    dataset, short_name: str, service_name_substring: str = "arco"
) -> set[str] | None:
    """Coordinate ids for the first `short_name` variable found on an "*arco*" service.

    Returns `None` if no such service/variable combination is found, so
    callers can treat "no coordinates exposed here" as acceptable rather
    than asserting a specific catalogue layout that may not hold.
    """

    for version in dataset.versions:
        for part in version.parts:
            for service in part.services:
                if service_name_substring not in service.service_name:
                    continue
                for variable in service.variables:
                    if variable.short_name == short_name:
                        return {coordinate.coordinate_id for coordinate in variable.coordinates}
    return None


def test_primary_current_product_metadata_reachable():
    result = copernicusmarine.describe(
        product_id=copernicus.PRIMARY_CURRENT_PRODUCT_ID, disable_progress_bar=True
    )

    assert result.products


def test_primary_current_dataset_id_confirmed_live():
    confirmed_id = copernicus.confirm_live_dataset_id(
        copernicus.PRIMARY_CURRENT_PRODUCT_ID, copernicus.PRIMARY_CURRENT_DATASET_ID
    )

    assert confirmed_id == copernicus.PRIMARY_CURRENT_DATASET_ID


def test_primary_current_dataset_has_expected_uo_vo_depth_dimensions():
    dataset = _described_dataset(copernicus.PRIMARY_CURRENT_DATASET_ID)

    short_names = _variable_short_names(dataset)
    assert "uo" in short_names
    assert "vo" in short_names

    # A 3D product should carry a depth coordinate on "uo" wherever an ARCO
    # service exposes coordinates for it -- if none do, that's still fine
    # for a reachability/shape smoke test.
    uo_coordinate_ids = _variable_coordinate_ids(dataset, "uo")
    if uo_coordinate_ids is not None:
        assert "depth" in uo_coordinate_ids


def test_primary_current_static_dataset_has_deptho_and_deptho_lev():
    dataset = _described_dataset(copernicus.PRIMARY_CURRENT_STATIC_DATASET_ID)

    short_names = _variable_short_names(dataset)

    # The live catalogue names this variable "deptho_lev" (not
    # "deptho_lev_interp" -- confirmed against the real describe() output,
    # never assumed from the ticket's own guessed naming).
    assert "deptho" in short_names
    assert "deptho_lev" in short_names


def test_wave_product_reachable_with_expected_variables():
    product_result = copernicusmarine.describe(
        product_id=copernicus.WAVE_PRODUCT_ID, disable_progress_bar=True
    )
    assert product_result.products

    dataset = _described_dataset(copernicus.WAVE_DATASET_ID)
    short_names = _variable_short_names(dataset)

    for expected_variable in ("VHM0", "VTPK", "VTM02", "VTM10", "VMDR"):
        assert expected_variable in short_names


def test_long_term_surface_current_dataset_id_confirmed_live():
    confirmed_id = copernicus.confirm_live_dataset_id(
        copernicus.LONG_TERM_CURRENT_PRODUCT_ID, copernicus.LONG_TERM_CURRENT_DATASET_ID
    )

    assert confirmed_id == copernicus.LONG_TERM_CURRENT_DATASET_ID


def test_long_term_surface_current_time_range_starts_1993_or_earlier():
    time_range_ms = copernicus.get_dataset_time_range_ms(copernicus.LONG_TERM_CURRENT_DATASET_ID)

    assert time_range_ms is not None
    minimum_ms, _maximum_ms = time_range_ms
    start_year = datetime.fromtimestamp(minimum_ms / 1000, tz=UTC).year

    # Documented to start 1993-01-01 -- "<=" (not "==") to tolerate minor
    # metadata drift rather than hard-coding an exact date as a permanent
    # constant.
    assert start_year <= 1993


def test_small_one_day_subset_can_be_acquired(tmp_path):
    """The only test in this module that needs a real, configured account.

    Requests a single historical day over a tiny bounding box and a 5 m
    depth slice -- a deliberately small, fast smoke test, never a real
    acquisition. Skips (rather than fails) when Copernicus Marine
    credentials are not configured, which is the expected state here.
    """

    start_of_day = (datetime.now(UTC) - timedelta(days=5)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_of_day = start_of_day + timedelta(days=1)

    try:
        result = copernicus.subset_dataset(
            dataset_id=copernicus.PRIMARY_CURRENT_DATASET_ID,
            variables=["uo", "vo"],
            minimum_longitude=1.7,
            maximum_longitude=1.9,
            minimum_latitude=53.3,
            maximum_latitude=53.5,
            start_datetime=start_of_day,
            end_datetime=end_of_day,
            minimum_depth=0,
            maximum_depth=5,
            output_directory=tmp_path,
            output_filename="smoke.nc",
        )
    except copernicus.CopernicusAuthenticationRequiredError:
        pytest.skip(
            "Copernicus Marine credentials not configured -- skipping the one live test "
            "that needs real data download"
        )

    assert result.local_path.exists()
    assert result.file_size_bytes > 0
