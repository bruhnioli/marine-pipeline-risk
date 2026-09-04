"""BGS (British Geological Survey) historical survey metadata provider.

Source provenance
------------------
BGS publishes marine survey metadata through its own GeoNetwork OGC CSW
2.0.2 catalogue at https://metadata.bgs.ac.uk/geonetwork/srv/eng/csw
(confirmed via live GetRecords/GetRecordById requests). Two request styles
are used here:

- `GetRecords` with a `CQL_TEXT` `AnyText LIKE` constraint to find a record
  by its BGS site-survey reference number. Returns a compact Dublin Core
  summary: title, abstract, access rights, a free-text source/method
  field, and a bounding box.
- `GetRecordById` with `outputSchema=.../gmd` to fetch the full ISO19139
  record for fields the summary omits: horizontal CRS, nominal resolution,
  exact survey start/end dates, and an unambiguous west/east/south/north
  bounding box (`EX_GeographicBoundingBox`) -- used in preference to the
  summary's `ows:BoundingBox`, whose axis order (lat, lon for EPSG:4326)
  is easy to get backwards.

Known candidates for this study (from the ticket, used only as search
terms -- every fact below is read back from the live record, never
assumed): GB02SS0001 ("Anglia Field Development / Anglia North West") and
CS03SS0003 ("Saturn Proposed Platform -> 49/16 LOGGS Tie-In pipeline/cable
route"). A third candidate, the 2003 DTI SEA5 survey covering Ower Bank,
has a data.gov.uk metadata mirror but its live BGS/NERC catalogue record
returns an access-denied exception, and the only footprint discoverable
via that mirror describes a location off north Scotland -- not the
Southern North Sea -- almost certainly a copy-paste error against a
sibling "SEA5" record; that footprint is deliberately not used (see
`discover_sea5_ower_bank`).
"""

import xml.etree.ElementTree as ET
from typing import Any

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from marine_engine.providers.bathymetry._http import get_with_retries
from marine_engine.providers.bathymetry.inventory import SurveyRecord

CSW_BASE_URL = "https://metadata.bgs.ac.uk/geonetwork/srv/eng/csw"
GEONETWORK_RECORD_URL = "https://metadata.bgs.ac.uk/geonetwork/srv/api/records/{uuid}"

CSW_NS = {
    "csw": "http://www.opengis.net/cat/csw/2.0.2",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dct": "http://purl.org/dc/terms/",
    "ows": "http://www.opengis.net/ows",
}
ISO_NS = {
    "gmd": "http://www.isotc211.org/2005/gmd",
    "gco": "http://www.isotc211.org/2005/gco",
    "gml": "http://www.opengis.net/gml/3.2",
}

SEA5_BGS_UUID = "aba64100-c16b-4de3-e044-0003ba6f30bd"
SEA5_DATA_GOV_UK_URL = (
    "https://www.data.gov.uk/dataset/4a6eb7cf-fda3-4430-acd1-f968b86cbd67/"
    "2003-strategic-environmental-assessment-sea5-wessex-explorer-sea5-survey-seabed-sampling-video-"
)


class BgsUnavailableError(RuntimeError):
    """A BGS CSW request failed, matched nothing, or returned an exception report."""


def _request_xml(params: dict[str, Any]) -> ET.Element:
    try:
        response = get_with_retries(CSW_BASE_URL, params=params)
    except Exception as exc:  # noqa: BLE001 -- surfaced as one clear source-specific error
        raise BgsUnavailableError(f"BGS CSW request failed: {exc}") from exc

    root = ET.fromstring(response.content)
    if root.tag.endswith("ExceptionReport"):
        text = " ".join(e.text.strip() for e in root.iter() if e.text and e.text.strip())
        raise BgsUnavailableError(f"BGS CSW returned an exception: {text}")
    return root


def _search_by_reference(reference: str) -> ET.Element | None:
    root = _request_xml(
        {
            "SERVICE": "CSW",
            "VERSION": "2.0.2",
            "REQUEST": "GetRecords",
            "resultType": "results",
            "outputSchema": "http://www.opengis.net/cat/csw/2.0.2",
            "elementSetName": "full",
            "constraintLanguage": "CQL_TEXT",
            "constraint_language_version": "1.1.0",
            "constraint": f"AnyText like '%{reference}%'",
        }
    )
    results = root.find(".//csw:SearchResults", CSW_NS)
    if results is None or results.get("numberOfRecordsMatched") in (None, "0"):
        return None
    return root.find(".//csw:Record", CSW_NS)


def _get_record_by_id(uuid: str) -> ET.Element:
    root = _request_xml(
        {
            "SERVICE": "CSW",
            "VERSION": "2.0.2",
            "REQUEST": "GetRecordById",
            "elementSetName": "full",
            "outputSchema": "http://www.isotc211.org/2005/gmd",
            "id": uuid,
        }
    )
    metadata = root.find(".//gmd:MD_Metadata", ISO_NS)
    if metadata is None:
        raise BgsUnavailableError(f"No MD_Metadata found in the GetRecordById response for {uuid}.")
    return metadata


def _text(elem: ET.Element | None) -> str | None:
    return elem.text.strip() if elem is not None and elem.text else None


def _parse_brief_fields(record: ET.Element) -> dict[str, Any]:
    return {
        "uuid": _text(record.find("dc:identifier", CSW_NS)),
        "title": _text(record.find("dc:title", CSW_NS)),
        "abstract": _text(record.find("dct:abstract", CSW_NS)),
        "rights": _text(record.find("dc:rights", CSW_NS)),
        "method_text": _text(record.find("dc:source", CSW_NS)),
        "geoindex_url": _text(record.find("dc:URI", CSW_NS)),
    }


def _parse_iso_fields(metadata: ET.Element) -> dict[str, Any]:
    crs_code = _text(
        metadata.find(
            "gmd:referenceSystemInfo/gmd:MD_ReferenceSystem/gmd:referenceSystemIdentifier"
            "/gmd:RS_Identifier/gmd:code/gco:CharacterString",
            ISO_NS,
        )
    )

    distance_elem = metadata.find(
        "gmd:identificationInfo/gmd:MD_DataIdentification/gmd:spatialResolution"
        "/gmd:MD_Resolution/gmd:distance/gco:Distance",
        ISO_NS,
    )
    resolution_m = None
    if distance_elem is not None and distance_elem.text:
        uom = distance_elem.get("uom", "")
        if "9001" in uom or uom.lower().endswith("m") or uom.lower() == "metre":
            resolution_m = float(distance_elem.text)

    temporal_path = (
        "gmd:identificationInfo/gmd:MD_DataIdentification/gmd:extent/gmd:EX_Extent"
        "/gmd:temporalElement/gmd:EX_TemporalExtent/gmd:extent/gml:TimePeriod/"
    )
    start_date = _text(metadata.find(temporal_path + "gml:beginPosition", ISO_NS))
    end_date = _text(metadata.find(temporal_path + "gml:endPosition", ISO_NS))

    bbox_path = (
        "gmd:identificationInfo/gmd:MD_DataIdentification/gmd:extent/gmd:EX_Extent"
        "/gmd:geographicElement/gmd:EX_GeographicBoundingBox/"
    )
    west = _text(metadata.find(bbox_path + "gmd:westBoundLongitude/gco:Decimal", ISO_NS))
    east = _text(metadata.find(bbox_path + "gmd:eastBoundLongitude/gco:Decimal", ISO_NS))
    south = _text(metadata.find(bbox_path + "gmd:southBoundLatitude/gco:Decimal", ISO_NS))
    north = _text(metadata.find(bbox_path + "gmd:northBoundLatitude/gco:Decimal", ISO_NS))
    footprint: BaseGeometry | None = None
    if None not in (west, east, south, north):
        footprint = box(float(west), float(south), float(east), float(north))

    access_code = _text(
        metadata.find(
            "gmd:identificationInfo/gmd:MD_DataIdentification/gmd:resourceConstraints"
            "/gmd:MD_LegalConstraints/gmd:accessConstraints/gmd:MD_RestrictionCode",
            ISO_NS,
        )
    )

    return {
        "horizontal_crs": f"EPSG:{crs_code.rsplit(':', 1)[-1]}" if crs_code else None,
        "nominal_resolution_m": resolution_m,
        "survey_start_date": start_date,
        "survey_end_date": end_date,
        "footprint_wgs84": footprint,
        "access_restriction_code": access_code,
    }


def _stub_record(
    reference: str, known_title: str, known_year: int, failure_reason: str
) -> SurveyRecord:
    """A record for a known candidate whose live BGS metadata could not be confirmed this run.

    Carries only what the ticket itself already stated -- never a guessed
    footprint, resolution, or access status.
    """

    return SurveyRecord(
        source="BGS",
        source_dataset_id=reference,
        title=known_title,
        acquisition_year=known_year,
        temporal_epoch=str(known_year),
        acquisition_status="live_verification_failed",
        notes=f"Live BGS catalogue query failed this run: {failure_reason}",
    )


def _build_record(
    reference: str, fields: dict[str, Any], iso_fields: dict[str, Any]
) -> SurveyRecord:
    start = iso_fields.get("survey_start_date")
    year = int(start[:4]) if start else None
    restricted = iso_fields.get("access_restriction_code") == "otherRestrictions"

    return SurveyRecord(
        source="BGS",
        source_dataset_id=reference,
        source_record_url_or_identifier=GEONETWORK_RECORD_URL.format(uuid=fields["uuid"]),
        title=fields.get("title"),
        survey_name=fields.get("title"),
        survey_start_date=start,
        survey_end_date=iso_fields.get("survey_end_date"),
        acquisition_year=year,
        data_type="geophysical site survey",
        survey_method=fields.get("method_text"),
        nominal_resolution_m=iso_fields.get("nominal_resolution_m"),
        horizontal_crs=iso_fields.get("horizontal_crs"),
        vertical_datum=None,  # not stated by the source for this record
        licence=fields.get("rights"),
        access_type="restricted" if restricted else fields.get("rights"),
        download_available=False if restricted else None,
        manual_download_required=True if restricted else None,
        acquisition_status="metadata_verified_data_restricted"
        if restricted
        else "metadata_verified",
        temporal_epoch=str(year) if year else None,
        notes=(
            f"{fields.get('abstract') or ''} Access: {fields.get('rights')}. "
            f"Interactive viewer only (no download/API): {fields.get('geoindex_url')}"
        ).strip(),
        geometry_wgs84=iso_fields.get("footprint_wgs84"),
    )


def _discover_by_reference(reference: str, known_title: str, known_year: int) -> SurveyRecord:
    try:
        brief = _search_by_reference(reference)
        if brief is None:
            return _stub_record(reference, known_title, known_year, "0 records matched.")
        fields = _parse_brief_fields(brief)
        iso_fields = _parse_iso_fields(_get_record_by_id(fields["uuid"]))
    except BgsUnavailableError as exc:
        return _stub_record(reference, known_title, known_year, str(exc))

    return _build_record(reference, fields, iso_fields)


def discover_gb02ss0001() -> SurveyRecord:
    return _discover_by_reference(
        "GB02SS0001", "Anglia Field Development / Anglia North West Site Survey", 2002
    )


def discover_cs03ss0003() -> SurveyRecord:
    return _discover_by_reference(
        "CS03SS0003", "48/10 Saturn -> 49/16 LOGGS Tie-In Pipeline/Cable Route Survey", 2003
    )


def discover_sea5_ower_bank() -> SurveyRecord:
    """The 2003 DTI SEA5 Southern North Sea survey (Race Bank, Docking Shoal, Ower Bank).

    See module docstring: the live record 403s and its only mirrored
    footprint is almost certainly a copy-paste error from an unrelated
    Moray Firth survey, so no footprint is used here -- spatial overlap
    with PL854 cannot be authoritatively verified this run.
    """

    try:
        _get_record_by_id(SEA5_BGS_UUID)
        notes_prefix = ""
    except BgsUnavailableError as exc:
        notes_prefix = f"Live BGS/NERC record inaccessible this run ({exc}). "

    return SurveyRecord(
        source="BGS",
        source_dataset_id="SEA5-OWER-BANK",
        source_record_url_or_identifier=SEA5_DATA_GOV_UK_URL,
        title="2003 DTI SEA5 (RV Wessex Explorer) seabed sampling/video/geophysical survey",
        survey_name="SEA5 -- Race Bank, Docking Shoal and Ower Bank",
        survey_start_date="2003-08-29",
        survey_end_date="2003-09-26",
        acquisition_year=2003,
        data_type="multibeam + sidescan geophysical survey",
        survey_method="multibeam echo sounder, sidescan sonar, seabed sampling, video",
        nominal_resolution_m=None,  # not stated; full ISO record inaccessible
        # EPSG:32631 as stated in the data.gov.uk mirror's spatial-reference-system field
        horizontal_crs="EPSG:32631",
        vertical_datum=None,
        licence="Open Government Licence (Crown Copyright; BEIS/DECC SEA data)",
        access_type="manual_only",
        download_available=False,
        manual_download_required=True,
        acquisition_status="footprint_unverifiable",
        temporal_epoch="2003",
        notes=(
            notes_prefix
            + "Abstract states '3 processed gridded multibeam files are available' via an "
            "interactive search portal only (webapps.bgs.ac.uk/data/sea/app/search); no direct "
            "download/API link found. Footprint could not be authoritatively verified: the only "
            "published footprint for this record is off north Scotland (Moray Firth), not the "
            "Southern North Sea -- almost certainly a metadata copy-paste error -- so it is "
            "deliberately not used rather than reported as a false negative."
        ),
        geometry_wgs84=None,
    )


def discover_bgs_surveys() -> list[SurveyRecord]:
    """Discover all three approved BGS/DTI historical survey candidates."""

    return [discover_gb02ss0001(), discover_cs03ss0003(), discover_sea5_ower_bank()]
