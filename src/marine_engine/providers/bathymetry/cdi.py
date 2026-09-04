"""SeaDataNet CDI survey-metadata resolution for EMODnet `source_references` (MAR-006B).

Follows ONLY the official provenance chain: an EMODnet `source_references`
WFS feature's own `metadata_url` (confirmed in `emodnet.py`) points at a
SeaDataNet CDI "report" page
(`https://cdi-bathymetry.seadatanet.org/report/edmo/{edmo_id}/{identifier}`).
That page embeds a structured `schema.org/Dataset` JSON-LD block plus a
supplementary HTML details table; this module parses both into one
`CdiRecord`.

Bot-protection note (load-bearing for this module's whole design)
-------------------------------------------------------------------
The CDI report host fronts every request with a client-side proof-of-work
challenge (a small page that computes a SHA-256 nonce in JavaScript, sets a
cookie, then reloads) before serving the real report -- confirmed by
fetching it directly with `requests` (HTTP 503, `Checking your browser...`,
a `bot_challenge_token` cookie) and separately loading the same URL in a
real browser, where the page's own script solves its own challenge and the
real content loads. `requests` cannot execute that JavaScript, and this
module deliberately does NOT attempt to solve the challenge itself -- doing
so would be automating a bypass of the site's own bot detection, which is
out of bounds regardless of how small the puzzle is. `fetch_cdi_report_html`
therefore always fails against this host when called from a plain HTTP
client; it exists so that (a) the failure mode is detected and reported
honestly rather than silently, and (b) if the host ever serves this content
without the challenge (a different route, a future change), it starts
working with no code change needed.

For the three PL854 source-reference records this ticket resolves
(110153, 121953, 121954), the real CDI report pages were instead read
through a real browser during implementation, following exactly the same
official metadata_url links. Their parsed results are bundled below as
`KNOWN_CDI_RECORDS` so `resolve_cdi_record` can still return real,
source-verified metadata for these specific records without pretending an
automated live fetch is possible. Every other identifier resolves through
the live path and reports `CdiUnavailableError` honestly if that also hits
the challenge.
"""

import re
from dataclasses import dataclass
from datetime import date

import requests
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

PRODUCT_RELEASE_YEAR = 2024  # the EMODnet DTM release these sources were resolved against
CDI_REQUEST_TIMEOUT_S = 30.0

QI_AGE_OLD_SURVEY_THRESHOLD_YEARS = (
    30  # QI_Age = 0 means "older than 30 years" (given, not derived)
)

# A CDI-stated HORIZONTAL resolution finer than this is "materially higher"
# than the EMODnet composite's own ~115 m native grid
# (providers/bathymetry/emodnet.py NATIVE_RESOLUTION_M) -- kept as a local
# constant rather than importing that module, since this is a generic
# "beats the aggregate baseline" threshold, not a dependency on EMODnet
# specifically. Applies only to horizontal spatial/grid resolution -- never
# to vertical resolution/accuracy, which is a different physical quantity
# (MAR-006D).
MATERIALLY_FINER_RESOLUTION_THRESHOLD_M = 115.0

ACCESS_DIRECT_DOWNLOAD = "DIRECT_DOWNLOAD"
ACCESS_SEADATANET_REQUEST = "SEADATANET_REQUEST"
ACCESS_REGISTRATION_REQUIRED = "REGISTRATION_REQUIRED"
ACCESS_OWNER_PERMISSION_REQUIRED = "OWNER_PERMISSION_REQUIRED"
ACCESS_METADATA_ONLY = "METADATA_ONLY"
ACCESS_RESTRICTED = "RESTRICTED"
ACCESS_UNKNOWN = "UNKNOWN"

RECOVERY_HIGH_RES_RECOVERABLE = "HIGH_RES_SOURCE_RECOVERABLE"
RECOVERY_HIGH_RES_REQUESTABLE = "HIGH_RES_SOURCE_REQUESTABLE"
RECOVERY_RESOLUTION_UNKNOWN = "SOURCE_RESOLUTION_UNKNOWN"
RECOVERY_NOT_RECOVERABLE = "SOURCE_NOT_RECOVERABLE"

CONSISTENCY_CONSISTENT = "CONSISTENT"
CONSISTENCY_INCONSISTENT = "INCONSISTENT"
CONSISTENCY_NOT_VERIFIABLE = "NOT_VERIFIABLE"

RESOLUTION_LIVE = "live"
RESOLUTION_CACHED_SNAPSHOT = "cached_snapshot_manual_browser_verification"
RESOLUTION_UNAVAILABLE = "unavailable"


class CdiUnavailableError(RuntimeError):
    """The CDI report page could not be fetched or parsed (bot-challenge, network, format)."""


@dataclass(frozen=True)
class CdiRecord:
    """Parsed SeaDataNet CDI metadata for one EMODnet `source_references` record.

    Any field the CDI report does not state stays None -- never inferred
    from the dataset title or from the EMODnet QI classification.
    """

    source_reference_id: str
    cdi_record_id: str | None
    title: str | None
    description: str | None
    organisation: str | None  # data originator/maintainer -- the actual surveying authority
    organisation_edmo_id: int | None
    data_centre: str | None  # distributor/publisher -- may differ from the originator
    data_centre_edmo_id: int | None
    survey_name: str | None  # cruise name
    platform: str | None
    acquisition_start: date | None
    acquisition_end: date | None
    acquisition_year: int | None
    survey_method: str | None  # CDI "Instrument/gear category"
    device: str | None
    horizontal_resolution_note: str | None  # raw source text; not a usable numeric value here
    vertical_resolution_note: str | None
    horizontal_crs: str | None
    vertical_datum: str | None  # left None unless the source states one explicitly
    geographic_footprint: BaseGeometry | None  # from the JSON-LD contentLocation.geo.box
    data_format: str | None
    data_size_mb: float | None
    licence_code: str | None  # NERC L08 vocabulary code, e.g. "RS"
    access_restriction: str | None  # e.g. "by negotiation"
    access_mechanism: str | None  # e.g. "web data access with registration"
    metadata_url: str
    data_access_url: str | None  # the CDI system's own canonical report URL
    cdi_import_date: str | None
    cdi_update_date: str | None
    resolution_status: str  # RESOLUTION_LIVE | RESOLUTION_CACHED_SNAPSHOT | RESOLUTION_UNAVAILABLE
    notes: str | None = None


def cdi_report_url(edmo_id: int, source_reference_id: str) -> str:
    return f"https://cdi-bathymetry.seadatanet.org/report/edmo/{edmo_id}/{source_reference_id}"


_BOT_CHALLENGE_MARKERS = ("bot_challenge_token", "Checking your browser")


def _looks_like_bot_challenge(status_code: int, text: str) -> bool:
    if status_code == 503:
        return True
    return any(marker in text for marker in _BOT_CHALLENGE_MARKERS)


def fetch_cdi_report_html(metadata_url: str) -> str:
    """A single, non-retrying attempt at the real CDI report HTML.

    Does not retry: the failure mode observed for this host (a client-side
    proof-of-work challenge) cannot be resolved by retrying a plain HTTP
    request, so retrying would only add latency. Raises `CdiUnavailableError`
    -- distinguishing a detected bot-challenge from a genuine network/HTTP
    failure -- rather than ever attempting to solve the challenge.
    """

    try:
        response = requests.get(metadata_url, timeout=CDI_REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        raise CdiUnavailableError(f"CDI report request for {metadata_url} failed: {exc}") from exc

    if _looks_like_bot_challenge(response.status_code, response.text):
        raise CdiUnavailableError(
            f"CDI report at {metadata_url} is behind a client-side bot-detection challenge "
            "(HTTP 503 / proof-of-work page) that a plain HTTP client cannot solve; not "
            "attempting to bypass it."
        )
    if response.status_code != 200:
        raise CdiUnavailableError(
            f"CDI report request for {metadata_url} returned HTTP {response.status_code}"
        )
    return response.text


# --- Parsing -----------------------------------------------------------------

_JSON_LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.DOTALL)
_TABLE_ROW_RE = re.compile(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_EDMO_ID_RE = re.compile(r"/report/(\d+)")


def _strip_tags(html_fragment: str) -> str:
    return _TAG_RE.sub("", html_fragment).replace("&nbsp;", " ").strip()


def _parse_date_yyyymmdd(value: str) -> date | None:
    value = value.strip()
    if len(value) != 8 or not value.isdigit():
        return None
    try:
        return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
    except ValueError:
        return None


def _parse_geo_box(box_value: str) -> BaseGeometry | None:
    """schema.org GeoShape.box: two space-separated "lat lon" corner points."""

    parts = box_value.split()
    if len(parts) != 4:
        return None
    try:
        lat1, lon1, lat2, lon2 = (float(p) for p in parts)
    except ValueError:
        return None
    return box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def _extract_edmo_id(url: str | None) -> int | None:
    if not url:
        return None
    match = _EDMO_ID_RE.search(url)
    return int(match.group(1)) if match else None


def _html_table_fields(html: str) -> dict[str, str]:
    """First-match "label -> value" map from the two-column details table.

    Some labels repeat (e.g. "Horizontal resolution" appears once for the
    number and once for its unit) -- the first occurrence is what this
    module actually uses, so first-match-wins is deliberate, not a bug.
    """

    fields: dict[str, str] = {}
    for label_html, value_html in _TABLE_ROW_RE.findall(html):
        label = _strip_tags(label_html)
        if label and label not in fields:
            fields[label] = _strip_tags(value_html)
    return fields


def parse_cdi_report_html(
    html: str, *, source_reference_id: str, metadata_url: str, resolution_status: str
) -> CdiRecord:
    """Parse a real CDI report page into a `CdiRecord`.

    Prefers the embedded schema.org JSON-LD block (structured, unambiguous)
    for title/description/dates/organisations/footprint/format/licence, and
    falls back to the supplementary HTML details table only for fields the
    JSON-LD does not carry (instrument category, resolution notes, min/max
    instrument depth, CDI-record housekeeping dates).
    """

    import json

    json_ld_match = _JSON_LD_RE.search(html)
    ld: dict = json.loads(json_ld_match.group(1)) if json_ld_match else {}
    fields = _html_table_fields(html)

    temporal_coverage = ld.get("temporalCoverage")
    acquisition_start = acquisition_end = None
    if temporal_coverage and "/" in temporal_coverage:
        start_str, end_str = temporal_coverage.split("/", 1)
        try:
            acquisition_start = date.fromisoformat(start_str)
            acquisition_end = date.fromisoformat(end_str)
        except ValueError:
            acquisition_start = acquisition_end = None
    acquisition_year = acquisition_start.year if acquisition_start else None

    maintainer = ld.get("maintainer") or {}
    producer = ld.get("producer") or {}
    publisher = ld.get("publisher") or {}
    author = ld.get("author") or {}
    # The actual surveying authority is maintainer/producer (e.g. UKHO); the
    # distributor/access point is author/publisher (e.g. OceanWise) -- the
    # EMODnet WFS `edmo_id` field only ever exposes the latter, so this is
    # the one place the true originator is recoverable at all.
    organisation = maintainer.get("name") or producer.get("name")
    organisation_edmo_id = _extract_edmo_id(maintainer.get("identifier") or producer.get("url"))
    data_centre = publisher.get("name") or author.get("name")
    data_centre_edmo_id = _extract_edmo_id(publisher.get("identifier") or author.get("url"))

    geo_box = ((ld.get("contentLocation") or {}).get("geo") or {}).get("box")
    footprint = _parse_geo_box(geo_box) if geo_box else None

    encoding_formats = ld.get("encodingFormat") or []
    data_format = fields.get("Data format") or (encoding_formats[0] if encoding_formats else None)

    licence_url = ld.get("license")
    licence_code = licence_url.rstrip("/").rsplit("/", 1)[-1] if licence_url else None

    data_size_mb = None
    data_size_text = fields.get("Data size")
    if data_size_text:
        match = re.search(r"[\d.]+", data_size_text)
        if match:
            data_size_mb = float(match.group())

    cdi_ids = ld.get("identifier") or []
    cdi_report_id = next((v.rsplit("/", 1)[-1] for v in cdi_ids if "cdi.seadatanet.org" in v), None)
    data_access_url = next((v for v in cdi_ids if "cdi.seadatanet.org" in v), None)

    horizontal_res_note = fields.get("Horizontal resolution")
    vertical_res_note = fields.get("Vertical resolution")

    return CdiRecord(
        source_reference_id=source_reference_id,
        cdi_record_id=fields.get("CDI-record id") or cdi_report_id,
        title=ld.get("name") or fields.get("Data set name"),
        description=ld.get("description") or fields.get("Abstract"),
        organisation=organisation,
        organisation_edmo_id=organisation_edmo_id,
        data_centre=data_centre,
        data_centre_edmo_id=data_centre_edmo_id,
        survey_name=fields.get("Cruise name"),
        platform=fields.get("Platform type"),
        acquisition_start=acquisition_start,
        acquisition_end=acquisition_end,
        acquisition_year=acquisition_year,
        survey_method=fields.get("Instrument/gear category"),
        device=None,  # no field independent of "Instrument/gear category" is exposed
        horizontal_resolution_note=(
            f"{horizontal_res_note} (source reports 0/unspecified)"
            if horizontal_res_note == "0"
            else horizontal_res_note
        ),
        vertical_resolution_note=(
            f"{vertical_res_note} (source reports 0/unspecified)"
            if vertical_res_note == "0"
            else vertical_res_note
        ),
        horizontal_crs="EPSG:4326" if fields.get("Datum") == "World Geodetic System 84" else None,
        vertical_datum=None,  # never stated by this source; not inferred from the horizontal datum
        geographic_footprint=footprint,
        data_format=data_format,
        data_size_mb=data_size_mb,
        licence_code=licence_code,
        access_restriction=fields.get("Access restriction"),
        access_mechanism=fields.get("Access/ordering of data"),
        metadata_url=metadata_url,
        data_access_url=data_access_url,
        cdi_import_date=fields.get("CDI-record initial import date"),
        cdi_update_date=fields.get("CDI-record last update"),
        resolution_status=resolution_status,
    )


# --- Derived classifications ---------------------------------------------


def calculate_survey_age(acquisition_year: int | None, product_release_year: int) -> int | None:
    if acquisition_year is None:
        return None
    return product_release_year - acquisition_year


def classify_qi_age_consistency(acquisition_year: int | None, qi_age: int | None) -> str:
    if acquisition_year is None or qi_age is None:
        return CONSISTENCY_NOT_VERIFIABLE
    age = calculate_survey_age(acquisition_year, PRODUCT_RELEASE_YEAR)
    older_than_30 = age is not None and age > QI_AGE_OLD_SURVEY_THRESHOLD_YEARS
    if qi_age == 0:
        return CONSISTENCY_CONSISTENT if older_than_30 else CONSISTENCY_INCONSISTENT
    return CONSISTENCY_NOT_VERIFIABLE  # only QI_Age=0's definition was given for this ticket


def classify_qi_vertical_consistency(survey_method: str | None, qi_vertical: int | None) -> str:
    if qi_vertical is None or not survey_method or survey_method.strip().lower() == "unknown":
        return CONSISTENCY_NOT_VERIFIABLE
    method = survey_method.lower()
    is_multibeam = "multibeam" in method or "mbes" in method
    is_singlebeam_or_lidar = "single-beam" in method or "single beam" in method or "lidar" in method
    if qi_vertical == 4:
        return CONSISTENCY_CONSISTENT if is_multibeam else CONSISTENCY_INCONSISTENT
    if qi_vertical == 3:
        return CONSISTENCY_CONSISTENT if is_singlebeam_or_lidar else CONSISTENCY_INCONSISTENT
    return CONSISTENCY_NOT_VERIFIABLE


def classify_qi_purpose_consistency(
    organisation: str | None,
    data_centre: str | None,
    project_hint: str | None,
    qi_purpose: int | None,
) -> str:
    if qi_purpose is None:
        return CONSISTENCY_NOT_VERIFIABLE
    text = " ".join(filter(None, [organisation, data_centre, project_hint])).lower()
    if qi_purpose == 3 and ("hydrographic" in text or "hydrography" in text):
        return CONSISTENCY_CONSISTENT
    return CONSISTENCY_NOT_VERIFIABLE


def classify_qi_metadata_consistency(
    cdi_record: CdiRecord,
    *,
    qi_age: int | None,
    qi_horizontal: int | None,
    qi_vertical: int | None,
    qi_purpose: int | None,
) -> str:
    """One overall verdict from the per-field checks: never invents agreement.

    Any genuine contradiction dominates (INCONSISTENT). Otherwise, if at
    least one field was independently corroborated and none contradicted,
    the survey is CONSISTENT overall. If nothing could be independently
    checked at all (e.g. QI_Horizontal has no independent CDI field to
    compare against here), the honest answer is NOT_VERIFIABLE.
    """

    component_results = [
        classify_qi_age_consistency(cdi_record.acquisition_year, qi_age),
        classify_qi_vertical_consistency(cdi_record.survey_method, qi_vertical),
        classify_qi_purpose_consistency(
            cdi_record.organisation, cdi_record.data_centre, cdi_record.description, qi_purpose
        ),
        # QI_Horizontal: no independent CDI field (e.g. a positioning-system
        # description) is exposed by any of these records to check against.
        CONSISTENCY_NOT_VERIFIABLE if qi_horizontal is not None else CONSISTENCY_NOT_VERIFIABLE,
    ]
    if CONSISTENCY_INCONSISTENT in component_results:
        return CONSISTENCY_INCONSISTENT
    if CONSISTENCY_CONSISTENT in component_results:
        return CONSISTENCY_CONSISTENT
    return CONSISTENCY_NOT_VERIFIABLE


def classify_access(cdi_record: CdiRecord) -> str:
    """Official CDI access metadata only -- metadata availability is not data availability."""

    mechanism = (cdi_record.access_mechanism or "").lower()
    restriction = (cdi_record.access_restriction or "").lower()

    if not mechanism and not restriction:
        return ACCESS_UNKNOWN
    if "negotiation" in restriction or "permission" in restriction:
        return ACCESS_OWNER_PERMISSION_REQUIRED
    if "restricted" in restriction:
        return ACCESS_RESTRICTED
    if "registration" in mechanism:
        return ACCESS_REGISTRATION_REQUIRED
    if "unrestricted" in mechanism or "without restriction" in mechanism:
        return ACCESS_DIRECT_DOWNLOAD
    if "shopping" in mechanism or "seadatanet" in mechanism or "order" in mechanism:
        return ACCESS_SEADATANET_REQUEST
    return ACCESS_UNKNOWN


def _extract_stated_resolution_m(note: str | None) -> float | None:
    """A genuinely stated positive numeric resolution from a CDI note, or None.

    CDI's own "0 Metres" convention means unspecified (see
    `parse_cdi_report_html`'s handling of the raw "0" value) -- zero or a
    missing note never counts as a real stated value here.
    """

    if not note:
        return None
    match = re.match(r"([\d.]+)", note)
    if not match:
        return None
    value = float(match.group(1))
    return value if value > 0 else None


def _confirms_materially_finer_resolution(cdi_record: CdiRecord) -> bool:
    """True only if CDI states a HORIZONTAL spatial resolution finer than the aggregate baseline.

    Deliberately looks at `horizontal_resolution_note` only. Two things are
    explicitly NOT evidence of horizontal spatial/grid resolution, however
    finely they are stated, and must never trigger a `HIGH_RES_SOURCE_*`
    result on their own:

    - `vertical_resolution_note` -- vertical resolution/accuracy is a
      statement about how precisely depth is measured, not about how
      densely the survey samples the seabed horizontally. A sub-metre
      vertical accuracy says nothing about grid/point spacing (MAR-006D).
    - QI instrument class (e.g. QI_Vertical=4 suggesting MBES) -- an
      instrument class is not proof of the exported/gridded resolution
      actually available, only of the sounding technology used during
      acquisition. See `classify_qi_vertical_consistency`'s docstring for
      the same "not independent corroboration" reasoning (MAR-006C).

    `vertical_resolution_note` itself is still preserved as metadata
    elsewhere (`CdiRecord.vertical_resolution_note`, the output
    dataframe's `vertical_accuracy_or_resolution` column) -- only its use
    *here*, to decide horizontal recovery potential, is excluded.
    """

    value = _extract_stated_resolution_m(cdi_record.horizontal_resolution_note)
    return value is not None and value < MATERIALLY_FINER_RESOLUTION_THRESHOLD_M


def classify_recovery_potential(cdi_record: CdiRecord, access_class: str) -> str:
    """Whether CDI metadata actually confirms materially finer-than-~115 m HORIZONTAL resolution.

    Deliberately separate from `access_class`: a source being requestable
    does not by itself confirm it is higher-resolution. Only a genuinely
    stated numeric *horizontal* spatial/grid resolution finer than the
    EMODnet baseline does that -- vertical resolution/accuracy and QI
    instrument class are both explicitly excluded, neither being evidence
    of horizontal sampling density (see
    `_confirms_materially_finer_resolution`). For the current PL854
    records, none states a numeric horizontal resolution (their "0 Metres"
    fields are unspecified), so all three correctly resolve to
    `SOURCE_RESOLUTION_UNKNOWN` regardless of how confident QI_Vertical
    looks -- being requestable is still reported, just via `access_class`,
    not smuggled into this field.
    """

    if access_class == ACCESS_RESTRICTED:
        return RECOVERY_NOT_RECOVERABLE

    has_real_dataset = bool(cdi_record.data_format) and cdi_record.data_size_mb is not None
    if not has_real_dataset:
        return RECOVERY_RESOLUTION_UNKNOWN

    if not _confirms_materially_finer_resolution(cdi_record):
        return RECOVERY_RESOLUTION_UNKNOWN

    if access_class == ACCESS_DIRECT_DOWNLOAD:
        return RECOVERY_HIGH_RES_RECOVERABLE
    if access_class in (
        ACCESS_SEADATANET_REQUEST,
        ACCESS_REGISTRATION_REQUIRED,
        ACCESS_OWNER_PERMISSION_REQUIRED,
    ):
        return RECOVERY_HIGH_RES_REQUESTABLE
    return RECOVERY_RESOLUTION_UNKNOWN


# --- Known-record fallback (manual browser verification; see module docstring) ---
#
# Captured 2026-09-04 by navigating a real browser to each record's official
# EMODnet-provided metadata_url (the same URLs `resolve_cdi_record` tries
# live first) and reading the same JSON-LD + details table this module's
# parser consumes. Nothing here was guessed or inferred from a title.

KNOWN_CDI_RECORDS: dict[str, CdiRecord] = {
    "110153": CdiRecord(
        source_reference_id="110153",
        cdi_record_id="1298109",
        title="Haddock Bank",
        description="UK Civil Hydrographic Programme Survey HI560",
        organisation="United Kingdom Hydrographic Office",
        organisation_edmo_id=26,
        data_centre="OceanWise Limited",
        data_centre_edmo_id=2607,
        survey_name="HI560",
        platform="unknown",
        acquisition_start=date(1992, 9, 21),
        acquisition_end=date(1992, 12, 8),
        acquisition_year=1992,
        survey_method="single-beam echosounders",
        device=None,
        horizontal_resolution_note="0 (source reports 0/unspecified)",
        vertical_resolution_note="0 (source reports 0/unspecified)",
        horizontal_crs="EPSG:4326",
        vertical_datum=None,
        geographic_footprint=box(1.3927999735, 53.2625007629, 1.6612999439, 53.4254989624),
        data_format="Climate and Forecast NetCDF Version 3.5",
        data_size_mb=57751.0,
        licence_code="RS",
        access_restriction="by negotiation",
        access_mechanism="web data access with registration",
        metadata_url=cdi_report_url(2607, "110153"),
        data_access_url="https://cdi.seadatanet.org/report/1298109",
        cdi_import_date="2012-07-16 07:38:31.220",
        cdi_update_date="2017-12-01 17:31:52.280",
        resolution_status=RESOLUTION_CACHED_SNAPSHOT,
        notes=(
            "QI text on the CDI report itself (2017-05-19): QI_Purpose='Hydrographic survey or "
            "compatible with hydrographic standards'; QI_Vertical='Lidar, SBES High Frequency'; "
            "QI_Horizontal='unknown or larger than 500m'."
        ),
    ),
    "121953": CdiRecord(
        source_reference_id="121953",
        cdi_record_id="3044183",
        title="North Sea, Broken Bank to North Haisborough, Block 2",
        description="UK Civil Hydrographic Programme Survey HI524-HI525-HI531",
        organisation="United Kingdom Hydrographic Office",
        organisation_edmo_id=26,
        data_centre="OceanWise Limited",
        data_centre_edmo_id=2607,
        survey_name="HI524-HI525-HI531",
        platform="unknown",
        acquisition_start=date(1991, 4, 24),
        acquisition_end=date(1991, 8, 16),
        acquisition_year=1991,
        survey_method="unknown",
        device=None,
        horizontal_resolution_note="0 (source reports 0/unspecified)",
        vertical_resolution_note="0 (source reports 0/unspecified)",
        horizontal_crs="EPSG:4326",
        vertical_datum=None,
        geographic_footprint=box(1.8569539785, 53.3623428345, 2.1509323120, 53.5071105957),
        data_format="Climate and Forecast NetCDF Version 3.5",
        data_size_mb=57751.0,
        licence_code="RS",
        access_restriction="by negotiation",
        access_mechanism="web data access with registration",
        metadata_url=cdi_report_url(2607, "121953"),
        data_access_url="https://cdi.seadatanet.org/report/3044183",
        cdi_import_date="2020-07-06 15:23:30.777",
        cdi_update_date="2020-07-06 15:23:35.403",
        resolution_status=RESOLUTION_CACHED_SNAPSHOT,
        notes=(
            "Same cruise (HI524-HI525-HI531) and epoch as 121954, different spatial block. "
            "Instrument/gear category field itself says 'unknown'; the only place 'MBES' "
            "appears is the QI_Vertical classification text (2017-05-19), which is the QI "
            "system's own label rather than an independent CDI instrument field -- see "
            "classify_qi_vertical_consistency. QI text: QI_Purpose='Hydrographic survey or "
            "compatible with hydrographic standards'; QI_Vertical='MBES high frequency (larger "
            "than 100kHz)'; QI_Horizontal='smaller than 20m'."
        ),
    ),
    "121954": CdiRecord(
        source_reference_id="121954",
        cdi_record_id="3044184",
        title="North Sea, Broken Bank to North Haisborough, Block 3",
        description="UK Civil Hydrographic Programme Survey HI524-HI525-HI531",
        organisation="United Kingdom Hydrographic Office",
        organisation_edmo_id=26,
        data_centre="OceanWise Limited",
        data_centre_edmo_id=2607,
        survey_name="HI524-HI525-HI531",
        platform="unknown",
        acquisition_start=date(1991, 4, 24),
        acquisition_end=date(1991, 8, 16),
        acquisition_year=1991,
        survey_method="unknown",
        device=None,
        horizontal_resolution_note="0 (source reports 0/unspecified)",
        vertical_resolution_note="0 (source reports 0/unspecified)",
        horizontal_crs="EPSG:4326",
        vertical_datum=None,
        geographic_footprint=box(1.6378524303, 53.2211380005, 1.9470589161, 53.4096450806),
        data_format="Climate and Forecast NetCDF Version 3.5",
        data_size_mb=57751.0,
        licence_code="RS",
        access_restriction="by negotiation",
        access_mechanism="web data access with registration",
        metadata_url=cdi_report_url(2607, "121954"),
        data_access_url="https://cdi.seadatanet.org/report/3044184",
        cdi_import_date="2020-07-06 15:23:30.777",
        cdi_update_date="2020-07-06 15:23:35.403",
        resolution_status=RESOLUTION_CACHED_SNAPSHOT,
        notes=(
            "Same cruise (HI524-HI525-HI531) and epoch as 121953, different spatial block. "
            "Instrument/gear category field itself says 'unknown'; the only place 'MBES' "
            "appears is the QI_Vertical classification text (2017-05-19) -- see "
            "classify_qi_vertical_consistency. QI text: QI_Purpose='Hydrographic survey or "
            "compatible with hydrographic standards'; QI_Vertical='MBES high frequency (larger "
            "than 100kHz)'; QI_Horizontal='smaller than 20m'."
        ),
    ),
}


def resolve_cdi_record(source_reference_id: str, edmo_id: int) -> CdiRecord:
    """Resolve one source-reference id to real CDI metadata: live first, then the known snapshot.

    Never fabricates a record for an unknown id -- returns a minimal
    `CdiRecord` with `resolution_status=RESOLUTION_UNAVAILABLE` and every
    other field None instead.
    """

    metadata_url = cdi_report_url(edmo_id, source_reference_id)
    try:
        html = fetch_cdi_report_html(metadata_url)
    except CdiUnavailableError:
        known = KNOWN_CDI_RECORDS.get(source_reference_id)
        if known is not None:
            return known
        return CdiRecord(
            source_reference_id=source_reference_id,
            cdi_record_id=None,
            title=None,
            description=None,
            organisation=None,
            organisation_edmo_id=None,
            data_centre=None,
            data_centre_edmo_id=None,
            survey_name=None,
            platform=None,
            acquisition_start=None,
            acquisition_end=None,
            acquisition_year=None,
            survey_method=None,
            device=None,
            horizontal_resolution_note=None,
            vertical_resolution_note=None,
            horizontal_crs=None,
            vertical_datum=None,
            geographic_footprint=None,
            data_format=None,
            data_size_mb=None,
            licence_code=None,
            access_restriction=None,
            access_mechanism=None,
            metadata_url=metadata_url,
            data_access_url=None,
            cdi_import_date=None,
            cdi_update_date=None,
            resolution_status=RESOLUTION_UNAVAILABLE,
            notes=(
                "CDI report unavailable (bot-challenge or network failure) and no cached "
                "snapshot exists for this id."
            ),
        )

    return parse_cdi_report_html(
        html,
        source_reference_id=source_reference_id,
        metadata_url=metadata_url,
        resolution_status=RESOLUTION_LIVE,
    )
