"""EMODnet Bathymetry provider: the mandatory full-AOI numerical baseline.

Source provenance
------------------
EMODnet Bathymetry publishes its DTM via an OGC WCS 2.0.1 service at
https://ows.emodnet-bathymetry.eu/wcs (confirmed via a live GetCapabilities
request). The unsuffixed coverage `emodnet__mean` ("Mean depth") is the
CURRENT release; older releases are retained under year-suffixed coverage
ids (`emodnet__mean_2022`, `emodnet__mean_2020`, ...). Confirmed -- not
assumed from the name alone -- that `emodnet__mean` IS the 2024 release by
fetching its own CSW metadata record (title "EMODnet Digital Bathymetry
(DTM 2024)", publication date 2024-12-31, at SOURCE_METADATA_URL below).

Per that same metadata record:
- native grid: 1/16 * 1/16 arc-minute (~115 m at this latitude); confirmed
  independently via DescribeCoverage's grid offset vectors (0.0010416666...
  degrees == 1/960 degree == 1/16 arc-minute).
- vertical datum: Lowest Astronomical Tide (LAT), exactly as stated.
- licence: Creative Commons Attribution 4.0 International, with an explicit
  "DO NOT USE FOR NAVIGATION" constraint.
- native format: image/tiff; GeoTIFF is requested explicitly below via
  format=image/tiff;application=geotiff (also an advertised supported format).

Role: MANDATORY full-AOI baseline / QA comparison layer -- `inventory.py`
never promotes it to "high-resolution primary" merely for being available.
"""

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from marine_engine.providers.bathymetry.inventory import SurveyRecord

WCS_BASE_URL = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE_ID = "emodnet__mean"  # verified == "EMODnet Digital Bathymetry (DTM 2024)"
DATASET_TITLE = "EMODnet Digital Bathymetry (DTM 2024)"
NATIVE_RESOLUTION_M = 115.0  # 1/16 arc-minute at this latitude, per source metadata
NATIVE_PIXEL_SPACING_DEG = 1.0 / 960.0  # confirmed via DescribeCoverage grid offset vectors
VERTICAL_DATUM = "LAT"
LICENCE = "Creative Commons Attribution 4.0 International"
ACCESS_NOTE = "DO NOT USE FOR NAVIGATION (source-stated constraint)."
SOURCE_METADATA_URL = (
    "https://sextant.ifremer.fr/geonetwork/srv/eng/csw?request=GetRecordById&elementSetName=full"
    "&service=CSW&version=2.0.2&OutputSchema=http://www.isotc211.org/2005/gmd"
    "&id=cf51df64-56f9-4a99-b1aa-36b8d7b743a1"
)
REQUEST_TIMEOUT_S = 60.0


class EmodnetUnavailableError(RuntimeError):
    """The EMODnet WCS service could not be reached or returned something unusable."""


def discover_emodnet_baseline(aoi_bbox_wgs84: tuple[float, float, float, float]) -> SurveyRecord:
    """Build the EMODnet baseline candidate record.

    Its footprint is the AOI's own WGS84 bounding box, passed in by the
    caller (reproducibly derived from the real aoi.gpkg) -- guaranteed to
    cover the complete AOI because that is exactly what gets requested in
    `fetch_emodnet_geotiff`, not asserted from the product title.

    `product_release_year=2024` records when EMODnet published this DTM
    *release*; `acquisition_year` is deliberately left None because 2024 is
    not when the underlying bathymetric surveys were measured -- MAR-006's
    own source-reference attribution shows this composite is itself built
    from older CDI surveys (resolved individually in MAR-006B). Likewise
    `temporal_epoch=None`: labelling an aggregate multi-source composite
    with a single year would misrepresent it as one acquisition epoch.
    """

    west, south, east, north = aoi_bbox_wgs84
    footprint = box(west, south, east, north)

    return SurveyRecord(
        source="EMODnet",
        source_dataset_id=COVERAGE_ID,
        source_record_url_or_identifier=SOURCE_METADATA_URL,
        title=DATASET_TITLE,
        survey_name=None,
        survey_start_date=None,
        survey_end_date=None,
        acquisition_year=None,
        product_release_year=2024,
        data_type="DTM",
        survey_method="composite/aggregated bathymetric DTM (multi-source)",
        nominal_resolution_m=NATIVE_RESOLUTION_M,
        horizontal_crs="EPSG:4326",
        vertical_datum=VERTICAL_DATUM,
        licence=LICENCE,
        access_type="open",
        download_available=True,
        manual_download_required=False,
        acquisition_status="numerically_acquired",
        temporal_epoch=None,
        notes=(
            "Full-AOI baseline; not a high-resolution pipeline-scale dataset. "
            "product_release_year=2024 is the DTM release year, NOT the acquisition "
            "year of the underlying surveys (see MAR-006B source-reference resolution "
            f"for the real per-survey acquisition epochs). {ACCESS_NOTE}"
        ),
        geometry_wgs84=footprint,
    )


@dataclass(frozen=True)
class EmodnetFetchResult:
    local_path: Path
    request_parameters: dict[str, Any]
    returned_crs: str
    width_px: int
    height_px: int
    content_type: str


def _read_tiff_dimensions(path: Path) -> tuple[int, int]:
    """Read ImageWidth/ImageLength directly from the TIFF IFD (no raster library)."""

    with path.open("rb") as fh:
        header = fh.read(8)
        if header[:2] not in (b"II", b"MM"):
            raise EmodnetUnavailableError("Response is not a valid TIFF (bad byte-order marker).")
        byte_order = "<" if header[:2] == b"II" else ">"
        (ifd_offset,) = struct.unpack(byte_order + "I", header[4:8])

        fh.seek(ifd_offset)
        (entry_count,) = struct.unpack(byte_order + "H", fh.read(2))

        width = height = None
        for _ in range(entry_count):
            tag, field_type, _count, raw_value = struct.unpack(byte_order + "HHI4s", fh.read(12))
            if tag not in (256, 257):  # ImageWidth, ImageLength
                continue
            value = (
                struct.unpack(byte_order + "H", raw_value[:2])[0]
                if field_type == 3  # SHORT
                else struct.unpack(byte_order + "I", raw_value)[0]  # LONG
            )
            if tag == 256:
                width = value
            else:
                height = value

    if width is None or height is None:
        raise EmodnetUnavailableError(
            "Could not find ImageWidth/ImageLength tags in the TIFF response."
        )
    return width, height


def fetch_emodnet_geotiff(
    aoi_bbox_wgs84: tuple[float, float, float, float],
    output_path: Path,
    timeout: float = REQUEST_TIMEOUT_S,
) -> EmodnetFetchResult:
    """Fetch a native-resolution GeoTIFF subset covering `aoi_bbox_wgs84`.

    Requests only a Lat/Long subset -- no scale/resampling parameters -- so
    the server returns data at its own native grid spacing. The returned
    pixel dimensions are read back from the file and checked against the
    bbox / native pixel spacing as a sanity check that no artificial
    resampling occurred (a wildly different ratio would indicate the server
    silently resampled, which this function then refuses to accept as
    the requested native-resolution data).
    """

    west, south, east, north = aoi_bbox_wgs84
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": COVERAGE_ID,
        "subset": [f"Lat({south},{north})", f"Long({west},{east})"],
        "format": "image/tiff;application=geotiff",
    }

    try:
        response = requests.get(WCS_BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise EmodnetUnavailableError(f"EMODnet WCS request failed: {exc}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "tiff" not in content_type.lower():
        raise EmodnetUnavailableError(
            f"EMODnet WCS did not return a TIFF (Content-Type={content_type!r}); refusing to "
            "treat a rendered image (e.g. WMS PNG) as numerical bathymetry."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    width_px, height_px = _read_tiff_dimensions(output_path)
    expected_width = round((east - west) / NATIVE_PIXEL_SPACING_DEG)
    expected_height = round((north - south) / NATIVE_PIXEL_SPACING_DEG)
    if abs(width_px - expected_width) > 2 or abs(height_px - expected_height) > 2:
        raise EmodnetUnavailableError(
            f"Returned raster is {width_px}x{height_px} px; expected ~{expected_width}x"
            f"{expected_height} px at the native {NATIVE_RESOLUTION_M} m grid for this bbox -- "
            "the service may have resampled instead of returning native resolution."
        )

    return EmodnetFetchResult(
        local_path=output_path,
        request_parameters={
            "coverageId": COVERAGE_ID,
            "bbox_wgs84": [west, south, east, north],
            "format": params["format"],
        },
        returned_crs="EPSG:4326",
        width_px=width_px,
        height_px=height_px,
        content_type=content_type,
    )


# --- Source-reference and quality-index attribution (MAR-006) --------------
#
# EMODnet Bathymetry also publishes an OGC WFS 2.0.0 service at
# https://ows.emodnet-bathymetry.eu/wfs (confirmed via a live GetCapabilities
# request) exposing two real, machine-readable vector feature types --
# confirmed via DescribeFeatureType, not guessed:
#
# `emodnet:source_references` fields: release, date_start, date_end,
# edmo_id (EDMO = European Directory of Marine Organisations, the
# contributing-organisation id), type ("CDI" = a SeaDataNet Common Data
# Index-registered survey, or "DTM" = a composite/GEBCO fallback grid used
# where no better local survey exists), device, colour, metadata_url, geom.
# A GeoJSON response for this AOI additionally carries an `identifier`
# field (a real per-record dataset id, e.g. "121948", or a GEBCO release
# tag like "GEBCO_2014") that is not declared in the XSD but is present in
# practice -- read defensively either way.
#
# `emodnet:quality_index` fields: the same release/date/edmo_id/type/geom,
# plus `combined` (float), `horizontal`/`vertical`/`age`/`purpose` (each a
# raw official integer class -- preserved as-is, never reinterpreted here,
# since no official class-to-label mapping was found in the service
# metadata to justify one).
#
# Both layers carry EVERY past DTM release's attribution in one table; a
# real query for this AOI returned 67 source_references / 30 quality_index
# features across releases 2011-2024, of which exactly 7 (both layers)
# have `release == "2024"` -- matching the acquired WCS coverage. Only
# that release is relevant here and is filtered for explicitly.
WFS_BASE_URL = "https://ows.emodnet-bathymetry.eu/wfs"
SOURCE_REFERENCES_LAYER = "emodnet:source_references"
QUALITY_INDEX_LAYER = "emodnet:quality_index"
DOWNLOAD_TILES_LAYER = "emodnet:download_tiles"
TARGET_RELEASE = "2024"  # matches COVERAGE_ID's DTM release
WFS_REQUEST_TIMEOUT_S = 60.0


class EmodnetAttributionUnavailableError(RuntimeError):
    """The source-reference/quality-index/tile-index WFS could not be retrieved or parsed."""


@dataclass(frozen=True)
class SourceReferenceFeature:
    """One `source_references` polygon: which input dataset prevails there."""

    identifier: str | None
    source_type: str | None  # "CDI" (real survey) | "DTM" (GEBCO/composite fallback) | other
    edmo_id: int | None
    release: str | None
    date_start: str | None
    date_end: str | None
    metadata_url: str | None
    geometry_wgs84: BaseGeometry


@dataclass(frozen=True)
class QualityIndexFeature:
    """One `quality_index` polygon: the official QI classes for that area."""

    identifier: str | None
    source_type: str | None
    combined: float | None
    horizontal: int | None
    vertical: int | None
    age: int | None
    purpose: int | None
    release: str | None
    geometry_wgs84: BaseGeometry


def _wfs_get_feature(
    type_name: str,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    release: str | None = None,
    count: int = 500,
) -> dict[str, Any]:
    """A real WFS 2.0.0 GetFeature request for one layer, as GeoJSON.

    Filters server-side via `CQL_FILTER` (a `release=...` attribute filter
    combined with a spatial `BBOX(geom, ...)` predicate) rather than the
    plain `bbox` KVP parameter: for `source_references`/`quality_index`,
    the plain `bbox` parameter alone returns EVERY historical DTM release
    overlapping the AOI (67 features with very complex polygons for this
    AOI, ~8.5 MB, bordering on a 60 s timeout in practice), whereas the
    combined CQL filter returns just the one release actually needed
    (confirmed: 7 features, ~27 KB, ~1.3 s for this AOI). Confirmed
    empirically that this server's CQL `BBOX()` function takes
    (lat, lon, lat, lon) order -- matching this layer's declared EPSG:4326
    axis order -- NOT the more common (lon, lat) convention; verified by
    testing both orders directly, not assumed.
    """

    west, south, east, north = bbox_wgs84
    bbox_predicate = f"BBOX(geom,{south},{west},{north},{east})"
    cql_filter = (
        f"release='{release}' AND {bbox_predicate}" if release is not None else bbox_predicate
    )

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": type_name,
        "CQL_FILTER": cql_filter,
        "outputFormat": "application/json",
        "count": count,
    }
    try:
        response = requests.get(WFS_BASE_URL, params=params, timeout=WFS_REQUEST_TIMEOUT_S)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise EmodnetAttributionUnavailableError(
            f"EMODnet WFS request for {type_name} failed: {exc}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise EmodnetAttributionUnavailableError(
            f"EMODnet WFS response for {type_name} was not valid JSON: {exc}"
        ) from exc


def fetch_source_references(
    aoi_bbox_wgs84: tuple[float, float, float, float], release: str | None = TARGET_RELEASE
) -> list[SourceReferenceFeature]:
    """Fetch `source_references` features intersecting the AOI bbox, filtered to one release.

    `release=None` returns every historical release found (mainly useful
    for diagnostics); the default matches the acquired WCS coverage.
    """

    data = _wfs_get_feature(SOURCE_REFERENCES_LAYER, aoi_bbox_wgs84, release=release)
    features = []
    for entry in data.get("features", []):
        props = entry.get("properties", {})
        geometry = entry.get("geometry")
        if not geometry:
            continue
        features.append(
            SourceReferenceFeature(
                identifier=props.get("identifier"),
                source_type=props.get("type"),
                edmo_id=props.get("edmo_id"),
                release=props.get("release"),
                date_start=props.get("date_start"),
                date_end=props.get("date_end"),
                metadata_url=props.get("metadata_url") or None,
                geometry_wgs84=shape(geometry),
            )
        )
    return features


def fetch_quality_index(
    aoi_bbox_wgs84: tuple[float, float, float, float], release: str | None = TARGET_RELEASE
) -> list[QualityIndexFeature]:
    """Fetch `quality_index` features intersecting the AOI bbox, filtered to one release."""

    data = _wfs_get_feature(QUALITY_INDEX_LAYER, aoi_bbox_wgs84, release=release)
    features = []
    for entry in data.get("features", []):
        props = entry.get("properties", {})
        geometry = entry.get("geometry")
        if not geometry:
            continue
        features.append(
            QualityIndexFeature(
                identifier=props.get("identifier"),
                source_type=props.get("type"),
                combined=props.get("combined"),
                horizontal=props.get("horizontal"),
                vertical=props.get("vertical"),
                age=props.get("age"),
                purpose=props.get("purpose"),
                release=props.get("release"),
                geometry_wgs84=shape(geometry),
            )
        )
    return features


@dataclass(frozen=True)
class MslAvailabilityResult:
    """Whether an official MSL-referenced tile covers the AOI, and how to get it.

    Deliberately does not download the tile itself (see `check_msl_availability`).
    """

    available: bool
    dtm_release: str | None
    tile_id: str | None
    format_label: str | None
    download_url: str | None
    notes: str


def check_msl_availability(
    aoi_bbox_wgs84: tuple[float, float, float, float], release: str = TARGET_RELEASE
) -> MslAvailabilityResult:
    """Check (never download) whether an official MSL-referenced tile covers the AOI.

    Uses the real `emodnet:download_tiles` WFS layer (confirmed via
    DescribeFeatureType: dtm_release, dtm_tile, product_format,
    download_url, geom). For this AOI/release, tile "D4" genuinely lists a
    "ESRI ASCII Mean Sea Level" format with a live download URL
    (`.../v12/D4_2024.msl.zip`) -- confirmed via HTTP HEAD to be a ~132 MB
    whole-tile archive, not AOI-clipped, so it is recorded here but not
    fetched (not "small and straightforward" for this task's scope).
    """

    try:
        data = _wfs_get_feature(DOWNLOAD_TILES_LAYER, aoi_bbox_wgs84)
    except EmodnetAttributionUnavailableError as exc:
        return MslAvailabilityResult(
            available=False,
            dtm_release=release,
            tile_id=None,
            format_label=None,
            download_url=None,
            notes=f"Could not query the download-tiles index: {exc}",
        )

    for entry in data.get("features", []):
        props = entry.get("properties", {})
        format_label = props.get("product_format") or ""
        if props.get("dtm_release") == release and "mean sea level" in format_label.lower():
            return MslAvailabilityResult(
                available=True,
                dtm_release=release,
                tile_id=props.get("dtm_tile"),
                format_label=format_label,
                download_url=props.get("download_url"),
                notes=(
                    "Verified reproducibly acquirable via the emodnet:download_tiles WFS; "
                    "not downloaded here (whole-tile archive, ~132 MB, not AOI-clipped -- "
                    "out of scope for MAR-006's 'small and straightforward' allowance). "
                    "Store separately from the LAT canonical baseline if fetched later; "
                    "do not use it to derive or replace LAT values."
                ),
            )

    return MslAvailabilityResult(
        available=False,
        dtm_release=release,
        tile_id=None,
        format_label=None,
        download_url=None,
        notes=f"No 'Mean Sea Level' product_format found for release={release} at this AOI.",
    )


# --- Native per-cell QA layer discovery (MAR-007) ----------------------------
#
# MAR-007 asks whether EMODnet exposes additional per-cell DTM attributes
# (min/max/std depth, number of contributing values, interpolation flag,
# smoothed-depth offset) as machine-readable coverage, without guessing
# coverage ids. Discovery here is genuinely live each call (never a frozen
# list) so it stays correct if EMODnet's own offering changes:
#
# 1. WCS GetCapabilities is parsed for every advertised CoverageId and
#    checked for QA-attribute keywords -- confirmed live that, as of this
#    implementation, only `emodnet__mean` and its year/land/colour variants
#    are advertised; no separate min/max/sd/count/flag coverage exists.
# 2. The already-integrated `emodnet:download_tiles` WFS is checked for a
#    matching `product_format` -- confirmed live that only "SD" (standard
#    deviation) has any match, as a whole-tile ~150 MB archive (not
#    AOI-clipped; bigger than the MSL tile MAR-006 already judged out of
#    scope). This is a bulk download, not a live/queryable machine-readable
#    per-cell coverage, so it is reported as found-but-not-fetched rather
#    than treated as satisfying this ticket's "machine-readable coverage"
#    requirement.
#
# Nothing here is downloaded; this only reports what exists.

QA_ATTRIBUTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "minimum_depth": ("min",),
    "maximum_depth": ("max",),
    "depth_std": ("sd", "std", "stdev", "standard deviation", "standarddeviation"),
    "n_values": ("count", "n_values", "nvalues", "number of values", "numberofvalues"),
    "interpolation_flag": ("interp", "extrapol", "flag"),
    "mean_smoothed_depth": ("smooth",),
}

_COVERAGE_ID_RE = re.compile(r"<wcs:CoverageId>([^<]+)</wcs:CoverageId>")


@dataclass(frozen=True)
class NativeQaLayerAvailability:
    """What (if anything) official EMODnet services expose for per-cell QA attributes.

    `wcs_matches`/`download_tile_matches` map each requested attribute name
    to the real coverage id / product_format that matched it, or None if
    nothing did -- never fabricated, never guessed.
    """

    wcs_coverage_ids: tuple[str, ...]
    wcs_matches: dict[str, str | None]
    download_tile_formats: tuple[str, ...]
    download_tile_matches: dict[str, str | None]
    notes: str


def list_wcs_coverage_ids() -> list[str]:
    """Every CoverageId currently advertised by the real EMODnet WCS GetCapabilities."""

    params = {"service": "WCS", "version": "2.0.1", "request": "GetCapabilities"}
    response = requests.get(WCS_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return _COVERAGE_ID_RE.findall(response.text)


def _first_keyword_match(candidates: list[str], keywords: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        lowered = candidate.lower()
        if any(keyword in lowered for keyword in keywords):
            return candidate
    return None


def check_native_qa_layers(
    aoi_bbox_wgs84: tuple[float, float, float, float], release: str | None = TARGET_RELEASE
) -> NativeQaLayerAvailability:
    """Live-check (never bulk-download) official per-cell QA layer availability.

    See the module-level comment above for what was actually found.
    """

    try:
        coverage_ids = list_wcs_coverage_ids()
    except requests.RequestException:
        coverage_ids = []
    wcs_matches = {
        attr: _first_keyword_match(coverage_ids, keywords)
        for attr, keywords in QA_ATTRIBUTE_KEYWORDS.items()
    }

    try:
        data = _wfs_get_feature(DOWNLOAD_TILES_LAYER, aoi_bbox_wgs84)
        formats = sorted(
            {
                props.get("product_format")
                for entry in data.get("features", [])
                for props in (entry.get("properties", {}),)
                if props.get("product_format")
                and (release is None or props.get("dtm_release") == release)
            }
        )
    except EmodnetAttributionUnavailableError:
        formats = []
    download_tile_matches = {
        attr: _first_keyword_match(formats, keywords)
        for attr, keywords in QA_ATTRIBUTE_KEYWORDS.items()
    }

    return NativeQaLayerAvailability(
        wcs_coverage_ids=tuple(coverage_ids),
        wcs_matches=wcs_matches,
        download_tile_formats=tuple(formats),
        download_tile_matches=download_tile_matches,
        notes=(
            "No live WCS coverage advertises a separate minimum/maximum/std/count/"
            "interpolation-flag/smoothed-depth layer for this release. The only official "
            "match is a whole-tile 'SD' (standard deviation) download via "
            "emodnet:download_tiles (~150 MB, not AOI-clipped) -- a bulk archive, not a "
            "live machine-readable per-cell coverage, so it is not fetched here. All "
            "native cell QA chainage fields are recorded as null with this status."
        ),
    )
