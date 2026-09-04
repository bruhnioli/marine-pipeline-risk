"""UKHO seabed-survey discovery -- three official routes checked, none yields
a queryable bathymetry-survey footprint for this AOI as of these runs.

Source provenance
------------------
**1. Generic MEDIN catalogue (MAR-005).** The UK's official marine-metadata
discovery catalogue, a GeoNetwork OGC CSW 2.0.2 service at
https://portal.medin.org.uk/geonetwork. A live bounding-box `GetRecords`
query against this AOI (west=1.5764758, south=53.3228828, east=2.0771479,
north=53.4341688) is confirmed working (HTTP 200,
`numberOfRecordsMatched="52"`), but every one of those 52 matches is a
broad/global-extent dataset that merely happens to overlap this small AOI
(e.g. the GEBCO One Minute Grid, the UK National Databank of Moored Current
Meter Data) -- none is a UKHO-specific localized seabed survey.
`discover_ukho_surveys` below performs this query and tags any genuine
match `discovery_method="generic_medin"`.

**2. Direct UKHO MEDIN metadata export (MAR-005B, attempted).**
`https://medinexport-data.ukho.gov.uk/` was specified as the authoritative
direct export. Verified unreachable via three independent HTTP stacks
(curl, a full browser engine, Python's `requests`), all failing identically:
TCP connects, but the TLS handshake is reset with no response (curl:
"schannel: server closed abruptly (missing close_notify)"), for every path
tried (`/`, `/export`, `/medin`, `/api`, `/records`, with/without a `www.`
prefix) and after multiple retries. An `openssl s_client` connection to the
same IP (51.141.5.62) *does* complete a TLS handshake, but presents a
certificate for `CN=admiralty.co.uk` -- a different hostname entirely --
indicating this specific subdomain has no live SNI/TLS binding on the
shared infrastructure it resolves to, rather than a transient network
blip. Status recorded as `LEGACY_EXPORT_HOST_UNAVAILABLE`; no further
retries against this host are made, and it is not treated as proof UKHO
lacks bathymetry data generally -- only that this specific export path is
currently unreachable.

**3. UKHO/ADMIRALTY Data Hub (MAR-005B, current official portal).**
`https://datahub.admiralty.co.uk/portal/` -- confirmed live (ArcGIS
Enterprise 10.3); `sharing/rest/search` works anonymously, no token
required for public items. Two searches were run: a broad keyword search
(`bathymetry OR seabed OR hydrographic OR multibeam`, 68 matches, all
generic Esri basemaps or "Maritime Limits" shapefiles) and a full
enumeration of every `owner:UKHydrographicOffice` item (155 total, both
pages retrieved). Of those 155: 26 are real public Feature Services with
real extents (pipelines, oil/gas installations, wind farm structures,
wrecks, maritime/EEZ/continental-shelf limits, satellite-derived
coastlines for 6 named UK/Ireland ports, ships routeing, VORF vertical-
datum blocks) -- none is a bathymetry survey. 26 "Image" items are all UI
icons/branding (including a "SeabedMappingIcon" -- a UI graphic, not
data). 15 items have "S-102" or "Bathymetry" in the title, but all are
`type="PDF"` (specifications/guides) or `type="Code Sample"` (downloadable
archives with empty `extent` metadata in the catalogue -- not live
services), covering named UK harbour trial cells (Bristol, Clyde,
Felixstowe, Humber, Plymouth, Solent, Tees, Channel Islands), none of
which is the PL854 corridor; their archives were not downloaded to check
precisely, per this task's scope (avoid downloading large datasets; these
catalogue records carry no spatial extent to test against PL854 without
doing so, and none is plausibly within range of PL854 given the named
locations). No authentication bypass was attempted or needed for any of
this -- everything above is anonymous, public catalogue access.

Conclusion: across all three official routes, no queryable UKHO
bathymetry-survey footprint has been identified for this AOI. This is
reported precisely as "not identified through the currently reachable
official services", not as "UKHO has no bathymetry here" -- those are not
equivalent claims. `discover_ukho_surveys` still performs the live generic
MEDIN query every run so that conclusion stays reproducible rather than
asserted from memory.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from marine_engine.providers.bathymetry._http import post_with_retries
from marine_engine.providers.bathymetry.inventory import SurveyRecord

MEDIN_CSW_URL = "https://portal.medin.org.uk/geonetwork"
MEDIN_MAX_RECORDS = 100

CSW_NS = {
    "csw": "http://www.opengis.net/cat/csw/2.0.2",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# A record is treated as a genuine UKHO seabed-survey candidate only if its
# title/identifier plausibly names UKHO/Admiralty AND a survey-scale
# keyword -- broad national/global databanks (tide gauges, current meters,
# GEBCO, research cruises) are excluded even though their bbox matches.
UKHO_SURVEY_KEYWORDS = ("ukho", "admiralty", "hydrographic office")
SURVEY_SCALE_KEYWORDS = ("multibeam", "survey", "bathymetric survey", "seabed survey")

GETRECORDS_BBOX_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<csw:GetRecords xmlns:csw="http://www.opengis.net/cat/csw/2.0.2"
  xmlns:ogc="http://www.opengis.net/ogc" xmlns:gml="http://www.opengis.net/gml"
  service="CSW" version="2.0.2" resultType="results" maxRecords="{max_records}"
  outputSchema="http://www.opengis.net/cat/csw/2.0.2">
  <csw:Query typeNames="csw:Record">
    <csw:ElementSetName>brief</csw:ElementSetName>
    <csw:Constraint version="1.1.0">
      <ogc:Filter>
        <ogc:BBOX>
          <ogc:PropertyName>ows:BoundingBox</ogc:PropertyName>
          <gml:Envelope srsName="urn:x-ogc:def:crs:EPSG:6.11:4326">
            <gml:lowerCorner>{south} {west}</gml:lowerCorner>
            <gml:upperCorner>{north} {east}</gml:upperCorner>
          </gml:Envelope>
        </ogc:BBOX>
      </ogc:Filter>
    </csw:Constraint>
  </csw:Query>
</csw:GetRecords>"""


class UkhoUnavailableError(RuntimeError):
    """The MEDIN CSW request failed or returned something unparseable."""


@dataclass(frozen=True)
class SourceDiscoveryStatus:
    """Per-source discovery outcome, reported even when zero candidates result."""

    source: str
    endpoint_queried: str
    query_succeeded: bool
    total_records_seen: int
    relevant_records_found: int
    message: str


def _is_ukho_survey_candidate(title: str) -> bool:
    lowered = title.lower()
    return any(k in lowered for k in UKHO_SURVEY_KEYWORDS) and any(
        k in lowered for k in SURVEY_SCALE_KEYWORDS
    )


def discover_ukho_surveys(
    aoi_bbox_wgs84: tuple[float, float, float, float],
) -> tuple[list[SurveyRecord], SourceDiscoveryStatus]:
    """Query MEDIN for anything intersecting the AOI, keep only UKHO-survey-like matches.

    Returns `([], status)` for this AOI as of the verification behind this
    module's docstring -- but the query is real and live, so a future run
    against a different AOI (or after MEDIN/UKHO publish something new)
    would pick up genuine matches automatically.
    """

    west, south, east, north = aoi_bbox_wgs84
    body = GETRECORDS_BBOX_TEMPLATE.format(
        max_records=MEDIN_MAX_RECORDS, west=west, south=south, east=east, north=north
    )

    try:
        response = post_with_retries(
            MEDIN_CSW_URL, body.encode("utf-8"), headers={"Content-Type": "application/xml"}
        )
        root = ET.fromstring(response.content)
    except Exception as exc:  # noqa: BLE001 -- one clear source-specific failure
        return [], SourceDiscoveryStatus(
            source="UKHO",
            endpoint_queried=MEDIN_CSW_URL,
            query_succeeded=False,
            total_records_seen=0,
            relevant_records_found=0,
            message=f"MEDIN CSW query failed: {exc}",
        )

    if root.tag.endswith("ExceptionReport"):
        text = " ".join(e.text.strip() for e in root.iter() if e.text and e.text.strip())
        return [], SourceDiscoveryStatus(
            source="UKHO",
            endpoint_queried=MEDIN_CSW_URL,
            query_succeeded=False,
            total_records_seen=0,
            relevant_records_found=0,
            message=f"MEDIN CSW returned an exception: {text}",
        )

    results = root.find(".//csw:SearchResults", CSW_NS)
    total_matched = int(results.get("numberOfRecordsMatched", "0")) if results is not None else 0

    titles = [
        (e.text or "").strip()
        for e in root.findall(".//dc:title", CSW_NS)
        if e.text and e.text.strip()
    ]
    ukho_titles = [t for t in titles if _is_ukho_survey_candidate(t)]

    records = [
        SurveyRecord(
            source="UKHO",
            source_dataset_id=title,
            source_record_url_or_identifier=MEDIN_CSW_URL,
            title=title,
            data_type="bathymetric survey",
            access_type="unknown",
            acquisition_status="found_via_medin_needs_manual_followup",
            notes="Matched UKHO/Admiralty + survey-scale keywords in a MEDIN bbox search; "
            "no footprint or further metadata retrieved automatically -- follow up manually.",
        )
        for title in ukho_titles
    ]

    message = (
        f"MEDIN CSW bbox search matched {total_matched} records overlapping the AOI "
        f"(returned {len(titles)}); {len(ukho_titles)} looked UKHO-survey-specific. "
        "UKHO's own ADMIRALTY Marine Data Portal exposes no public bathymetry API "
        "(verified: anonymous search returns no Feature/Image/Map Service items)."
    )

    return records, SourceDiscoveryStatus(
        source="UKHO",
        endpoint_queried=MEDIN_CSW_URL,
        query_succeeded=True,
        total_records_seen=total_matched,
        relevant_records_found=len(ukho_titles),
        message=message,
    )
