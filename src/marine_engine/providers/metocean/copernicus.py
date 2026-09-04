"""Copernicus Marine Toolbox wrapper for PL854 metocean forcing evidence (MAR-009).

Source provenance
------------------
Three official Copernicus Marine products, confirmed live against the
Toolbox's own catalogue metadata (not assumed from documentation) on
2026-09-04:

- `NWSHELF_ANALYSISFORECAST_PHY_004_013` -- Met Office North-West Shelf
  coupled hydrodynamic-wave analysis/forecast. The 3D hourly current
  dataset is `cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i`
  (33 standard depth levels, `uo`/`vo`/`wo`, m/s); its static companion
  `cmems_mod_nws_phy_anfc_1.5km_static` carries `deptho`
  (sea_floor_depth_below_geoid), `deptho_lev` (model_level_number_at_sea_
  floor -- the ticket refers to this as `deptho_lev_interp`; the live
  dataset's actual variable name is `deptho_lev`, used here under that
  real name), and `mask` (sea_binary_mask, has a depth dimension). This
  is a ROLLING catalogue: live inspection on 2026-09-04 showed data from
  2024-07-20 through roughly a week past the inspection date (forecast) --
  never a fixed multi-decade record.
- `NWSHELF_MULTIYEAR_PHY_004_009` -- Met Office North-West Shelf physical
  reanalysis. The HOURLY 2D current dataset is
  `cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i` (confirmed live: 1993-01-01
  onward). A DAILY 3D current dataset
  (`cmems_mod_nws_phy-uv_my_7km-3D_P1D-m`) also exists on this product --
  it is a 25-hour tide-removing mean and must NEVER be substituted for the
  hourly instantaneous current (Section 7 of the ticket).
- `NWSHELF_REANALYSIS_WAV_004_015` -- WAVEWATCH III North-West European
  Shelf wave reanalysis. Dataset id `MetO-NWS-WAV-RAN` (confirmed live:
  1980-01-01 onward, 3-hourly); static companion
  `cmems_mod_nws_wav_my_1.5km-3D_static` carries `deptho`/`mask`.

These dataset ids are re-confirmed live at the start of every real
acquisition run (`confirm_live_dataset_id`) rather than trusted blindly --
if the live catalogue no longer lists the expected id, this raises rather
than guessing (Section 2 of the ticket: "Do not guess if the identifier
changes").

Authentication
---------------
`copernicusmarine.open_dataset`/`subset` fall back to an INTERACTIVE
username/password prompt when no credentials are configured -- unsafe in
an automated context (it would block on stdin). `ensure_authenticated`
uses the Toolbox's own non-interactive
`copernicusmarine.login(check_credentials_valid=True)` check instead,
confirmed live to return a plain `False` (no prompt, no exception) when no
credentials exist. Every acquisition function calls this FIRST and raises
`CopernicusAuthenticationRequiredError` before ever reaching `subset`, so a
missing-credentials run always fails cleanly and non-interactively.
Credentials are never accepted as a config/YAML value, logged, or written
to the manifest -- only the Toolbox's own credential store
(`$HOME/.copernicusmarine`) or its environment variables are used, exactly
as the normal `copernicusmarine login` process already works.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import copernicusmarine

# --- Product / dataset identifiers (confirmed live 2026-09-04) --------------

PRIMARY_CURRENT_PRODUCT_ID = "NWSHELF_ANALYSISFORECAST_PHY_004_013"
PRIMARY_CURRENT_DATASET_ID = "cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i"
PRIMARY_CURRENT_STATIC_DATASET_ID = "cmems_mod_nws_phy_anfc_1.5km_static"

LONG_TERM_CURRENT_PRODUCT_ID = "NWSHELF_MULTIYEAR_PHY_004_009"
LONG_TERM_CURRENT_DATASET_ID = "cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i"
LONG_TERM_CURRENT_STATIC_DATASET_ID = "cmems_mod_nws_phy_my_7km-3D_static"
# Explicitly forbidden (Section 7): a 25-hour tide-removing mean, never a
# substitute for hourly instantaneous current.
LONG_TERM_CURRENT_FORBIDDEN_DAILY_DATASET_ID = "cmems_mod_nws_phy-uv_my_7km-3D_P1D-m"

WAVE_PRODUCT_ID = "NWSHELF_REANALYSIS_WAV_004_015"
WAVE_DATASET_ID = "MetO-NWS-WAV-RAN"
WAVE_STATIC_DATASET_ID = "cmems_mod_nws_wav_my_1.5km-3D_static"

# --- Variables ---------------------------------------------------------------

PRIMARY_CURRENT_VARIABLES = ("uo", "vo")  # `wo` preserved if present, never used physically here
PRIMARY_CURRENT_STATIC_VARIABLES = ("deptho", "deptho_lev", "mask")
LONG_TERM_CURRENT_VARIABLES = ("uo", "vo")
WAVE_VARIABLES = ("VHM0", "VTPK", "VTM02", "VTM10", "VMDR", "VSDX", "VSDY")
WAVE_STATIC_VARIABLES = ("deptho", "mask")

# --- Evidence roles (canonical naming, never substituted for one another) ---

PRIMARY_CURRENT_EVIDENCE_ROLE = "PRIMARY_CURRENT_EVIDENCE"
LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE = "LONG_TERM_SURFACE_CURRENT_CONTEXT"
PRIMARY_WAVE_CLIMATE_ROLE = "PRIMARY_WAVE_CLIMATE"


class CopernicusAuthenticationRequiredError(RuntimeError):
    """Copernicus Marine credentials are not configured.

    Raised before any real data request is attempted -- never after an
    interactive prompt has already been risked.
    """


class CopernicusDatasetNotFoundError(RuntimeError):
    """An expected dataset id is not present in the live Copernicus Marine catalogue.

    Never guessed at or substituted -- the caller must re-confirm the
    correct id from the live catalogue this error lists.
    """


def toolbox_version() -> str:
    """The installed Copernicus Marine Toolbox version, for manifest provenance."""

    return copernicusmarine.__version__


def ensure_authenticated() -> None:
    """Raise `CopernicusAuthenticationRequiredError` if not authenticated, non-interactively.

    Never triggers the Toolbox's own interactive username/password prompt
    (confirmed live: `login(check_credentials_valid=True)` itself never
    prompts, unlike `open_dataset`/`subset` with no credentials configured).
    """

    if not copernicusmarine.login(check_credentials_valid=True):
        raise CopernicusAuthenticationRequiredError(
            "Copernicus Marine credentials are not configured (checked the environment "
            "variables and the Toolbox's own credentials file). Authenticate using the "
            "normal Copernicus Marine Toolbox login process, then re-run this command:\n"
            "  uv run copernicusmarine login\n"
            "(or set COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD). "
            "Register for a free account at https://data.marine.copernicus.eu/register if "
            "needed. Credentials must never be entered into Claude or pasted into chat."
        )


def confirm_live_dataset_id(product_id: str, expected_dataset_id: str) -> str:
    """Confirm `expected_dataset_id` is still listed for `product_id` in the live catalogue.

    Never guesses: raises `CopernicusDatasetNotFoundError` (listing what IS
    actually available) rather than silently proceeding with a stale id if
    the rolling catalogue has changed (Section 2 of the ticket).
    """

    catalogue = copernicusmarine.describe(product_id=product_id, disable_progress_bar=True)
    if not catalogue.products:
        raise CopernicusDatasetNotFoundError(
            f"No product found for product_id={product_id!r} in the live Copernicus Marine "
            "catalogue."
        )
    dataset_ids = [dataset.dataset_id for dataset in catalogue.products[0].datasets]
    if expected_dataset_id not in dataset_ids:
        raise CopernicusDatasetNotFoundError(
            f"Expected dataset_id={expected_dataset_id!r} not found in the live catalogue for "
            f"product_id={product_id!r}. Available dataset ids: {sorted(dataset_ids)}. "
            "Refusing to guess -- update the expected dataset id only after confirming the "
            "correct one from this live list."
        )
    return expected_dataset_id


def _find_time_range_ms(catalogue: Any) -> tuple[int, int] | None:
    """The dataset's own overall (min, max) time coordinate, in epoch milliseconds.

    Walks the catalogue's nested service/variable/coordinate structure
    defensively -- any variable's time coordinate reflects the same
    dataset-wide extent, so the first one found is used.
    """

    for product in catalogue.products:
        for dataset in product.datasets:
            for version in dataset.versions:
                for part in version.parts:
                    for service in part.services:
                        for variable in service.variables:
                            for coordinate in variable.coordinates:
                                if coordinate.coordinate_id == "time":
                                    return (
                                        int(coordinate.minimum_value),
                                        int(coordinate.maximum_value),
                                    )
    return None


def get_dataset_time_range_ms(dataset_id: str) -> tuple[int, int] | None:
    """The live dataset's own available (min, max) time coordinate (epoch ms), or None."""

    catalogue = copernicusmarine.describe(dataset_id=dataset_id, disable_progress_bar=True)
    return _find_time_range_ms(catalogue)


@dataclass(frozen=True)
class SubsetResult:
    """What actually happened when a chunk was subset from Copernicus Marine."""

    local_path: Path
    file_size_bytes: int
    status: str
    message: str | None


def subset_dataset(
    *,
    dataset_id: str,
    variables: list[str],
    minimum_longitude: float,
    maximum_longitude: float,
    minimum_latitude: float,
    maximum_latitude: float,
    start_datetime: datetime | None,
    end_datetime: datetime | None,
    minimum_depth: float | None,
    maximum_depth: float | None,
    output_directory: Path,
    output_filename: str,
) -> SubsetResult:
    """Download one chunk from Copernicus Marine to disk, authenticating first.

    Never falls back to an interactive prompt -- `ensure_authenticated`
    raises before this ever calls the Toolbox's own `subset`.
    """

    ensure_authenticated()

    output_directory.mkdir(parents=True, exist_ok=True)
    response = copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        output_directory=output_directory,
        output_filename=output_filename,
        file_format="netcdf",
        overwrite=True,
        disable_progress_bar=True,
    )
    return SubsetResult(
        local_path=Path(response.file_path),
        file_size_bytes=int(response.file_size),
        status=str(response.status),
        message=getattr(response, "message", None),
    )
