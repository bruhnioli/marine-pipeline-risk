"""Offline unit tests for the bathymetry source modules (ukho, bgs, emodnet).

All network calls are monkeypatched with canned responses shaped like the
real services (trimmed from the actual verified responses used to build
these modules) or with simulated failures. No network access.
"""

import struct

import pytest
import requests

from marine_engine.providers.bathymetry import bgs, emodnet, ukho


class FakeResponse:
    def __init__(self, content: bytes, headers: dict | None = None, status_code: int = 200):
        self.content = content
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeJsonResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def _build_minimal_tiff(width: int, height: int) -> bytes:
    """A tiny valid little-endian TIFF with only ImageWidth/ImageLength tags."""

    header = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
    entry_count = struct.pack("<H", 2)
    entry_width = struct.pack("<HHI", 256, 3, 1) + struct.pack("<H", width) + b"\x00\x00"
    entry_height = struct.pack("<HHI", 257, 3, 1) + struct.pack("<H", height) + b"\x00\x00"
    next_ifd = struct.pack("<I", 0)
    return header + entry_count + entry_width + entry_height + next_ifd


# --- EMODnet: numeric request construction -----------------------------------


def test_emodnet_fetch_requests_lat_long_subset_no_resampling(monkeypatch, tmp_path):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        width = round((2.0 - 1.0) / emodnet.NATIVE_PIXEL_SPACING_DEG)
        height = round((54.0 - 53.0) / emodnet.NATIVE_PIXEL_SPACING_DEG)
        return FakeResponse(
            _build_minimal_tiff(width, height), headers={"Content-Type": "image/tiff"}
        )

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    result = emodnet.fetch_emodnet_geotiff((1.0, 53.0, 2.0, 54.0), tmp_path / "out.tif")

    assert captured["url"] == emodnet.WCS_BASE_URL
    params = captured["params"]
    assert params["coverageId"] == emodnet.COVERAGE_ID
    assert params["subset"] == ["Lat(53.0,54.0)", "Long(1.0,2.0)"]
    # No scale/resampling parameters requested -- native resolution only.
    assert "scaleSize" not in params
    assert "resampling" not in params
    assert result.returned_crs == "EPSG:4326"


def test_emodnet_fetch_rejects_non_tiff_response(monkeypatch, tmp_path):
    def fake_get(url, params=None, timeout=None):
        return FakeResponse(b"<html>not a tiff</html>", headers={"Content-Type": "image/png"})

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    with pytest.raises(emodnet.EmodnetUnavailableError):
        emodnet.fetch_emodnet_geotiff((1.0, 53.0, 2.0, 54.0), tmp_path / "out.tif")


def test_emodnet_fetch_rejects_wrong_dimensions_as_possible_resampling(monkeypatch, tmp_path):
    def fake_get(url, params=None, timeout=None):
        # Return a raster far smaller than the native-resolution expectation
        # for this bbox -- as if the server silently resampled/downsampled.
        return FakeResponse(_build_minimal_tiff(4, 4), headers={"Content-Type": "image/tiff"})

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    with pytest.raises(emodnet.EmodnetUnavailableError):
        emodnet.fetch_emodnet_geotiff((1.0, 53.0, 2.0, 54.0), tmp_path / "out.tif")


def test_emodnet_fetch_network_failure_raises_clear_error(monkeypatch, tmp_path):
    def fake_get(url, params=None, timeout=None):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    with pytest.raises(emodnet.EmodnetUnavailableError):
        emodnet.fetch_emodnet_geotiff((1.0, 53.0, 2.0, 54.0), tmp_path / "out.tif")


def test_discover_emodnet_baseline_footprint_matches_requested_bbox():
    bbox = (1.5764758, 53.3228828, 2.0771479, 53.4341688)
    record = emodnet.discover_emodnet_baseline(bbox)

    assert record.source == "EMODnet"
    assert record.geometry_wgs84 is not None
    assert record.geometry_wgs84.bounds == pytest.approx(bbox, rel=1e-6)
    assert record.download_available is True
    assert record.vertical_datum == "LAT"


# --- BGS: real-shaped Dublin Core + ISO19139 parsing --------------------------

BGS_BRIEF_RECORD_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<csw:GetRecordsResponse xmlns:csw="http://www.opengis.net/cat/csw/2.0.2">
  <csw:SearchStatus timestamp="2026-09-03T19:17:22Z" />
  <csw:SearchResults numberOfRecordsMatched="1" numberOfRecordsReturned="1" elementSet="full" nextRecord="0">
    <csw:Record xmlns:ows="http://www.opengis.net/ows" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dct="http://purl.org/dc/terms/">
      <dc:identifier>a6d7eee1-3ca8-0370-e044-0003ba9b0d98</dc:identifier>
      <dc:title>2002, GDF Suez, Anglia Field Development and Anglia North West, Site Survey, BGS Reference Number GB02SS0001</dc:title>
      <dct:abstract>An oil and gas industry site survey acquired between August and September 2002.</dct:abstract>
      <dc:rights>otherRestrictions</dc:rights>
      <dc:source>Data was collected using multibeam echo sounder, side scan sonar.</dc:source>
      <ows:BoundingBox crs="urn:ogc:def:crs:EPSG:6.6:4326">
        <ows:LowerCorner>53.3333 1.4000</ows:LowerCorner>
        <ows:UpperCorner>53.5000 1.6000</ows:UpperCorner>
      </ows:BoundingBox>
      <dc:URI protocol="WWW:LINK-1.0-http--link">http://mapapps2.bgs.ac.uk/geoindex_offshore/home.html?OGI_siteSvyId=2203375</dc:URI>
    </csw:Record>
  </csw:SearchResults>
</csw:GetRecordsResponse>"""

BGS_ZERO_MATCH_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<csw:GetRecordsResponse xmlns:csw="http://www.opengis.net/cat/csw/2.0.2">
  <csw:SearchStatus timestamp="2026-09-03T19:17:22Z" />
  <csw:SearchResults numberOfRecordsMatched="0" numberOfRecordsReturned="0" elementSet="full" nextRecord="0" />
</csw:GetRecordsResponse>"""

BGS_ISO_RECORD_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<csw:GetRecordByIdResponse xmlns:csw="http://www.opengis.net/cat/csw/2.0.2">
  <gmd:MD_Metadata xmlns:gmd="http://www.isotc211.org/2005/gmd" xmlns:gco="http://www.isotc211.org/2005/gco" xmlns:gml="http://www.opengis.net/gml/3.2">
    <gmd:referenceSystemInfo>
      <gmd:MD_ReferenceSystem>
        <gmd:referenceSystemIdentifier>
          <gmd:RS_Identifier>
            <gmd:code><gco:CharacterString>urn:ogc:def:crs:EPSG::4230</gco:CharacterString></gmd:code>
          </gmd:RS_Identifier>
        </gmd:referenceSystemIdentifier>
      </gmd:MD_ReferenceSystem>
    </gmd:referenceSystemInfo>
    <gmd:identificationInfo>
      <gmd:MD_DataIdentification>
        <gmd:resourceConstraints>
          <gmd:MD_LegalConstraints>
            <gmd:accessConstraints>
              <gmd:MD_RestrictionCode codeListValue="otherRestrictions">otherRestrictions</gmd:MD_RestrictionCode>
            </gmd:accessConstraints>
          </gmd:MD_LegalConstraints>
        </gmd:resourceConstraints>
        <gmd:spatialResolution>
          <gmd:MD_Resolution>
            <gmd:distance><gco:Distance uom="urn:ogc:def:uom:EPSG::9001">5</gco:Distance></gmd:distance>
          </gmd:MD_Resolution>
        </gmd:spatialResolution>
        <gmd:extent>
          <gmd:EX_Extent>
            <gmd:geographicElement>
              <gmd:EX_GeographicBoundingBox>
                <gmd:westBoundLongitude><gco:Decimal>1.4000</gco:Decimal></gmd:westBoundLongitude>
                <gmd:eastBoundLongitude><gco:Decimal>1.6000</gco:Decimal></gmd:eastBoundLongitude>
                <gmd:southBoundLatitude><gco:Decimal>53.3333</gco:Decimal></gmd:southBoundLatitude>
                <gmd:northBoundLatitude><gco:Decimal>53.5000</gco:Decimal></gmd:northBoundLatitude>
              </gmd:EX_GeographicBoundingBox>
            </gmd:geographicElement>
            <gmd:temporalElement>
              <gmd:EX_TemporalExtent>
                <gmd:extent>
                  <gml:TimePeriod gml:id="_x">
                    <gml:beginPosition>2002-08-29</gml:beginPosition>
                    <gml:endPosition>2002-09-03</gml:endPosition>
                  </gml:TimePeriod>
                </gmd:extent>
              </gmd:EX_TemporalExtent>
            </gmd:temporalElement>
          </gmd:EX_Extent>
        </gmd:extent>
      </gmd:MD_DataIdentification>
    </gmd:identificationInfo>
  </gmd:MD_Metadata>
</csw:GetRecordByIdResponse>"""

BGS_EXCEPTION_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows" version="1.2.0">
  <ows:Exception exceptionCode="NoApplicableCode">
    <ows:ExceptionText>OperationNotAllowedEx : Operation not allowed</ows:ExceptionText>
  </ows:Exception>
</ows:ExceptionReport>"""


def test_bgs_discover_gb02ss0001_parses_real_shaped_response(monkeypatch):
    responses = {"GetRecords": BGS_BRIEF_RECORD_XML, "GetRecordById": BGS_ISO_RECORD_XML}

    def fake_get(url, params=None):
        return FakeResponse(responses[params["REQUEST"]])

    monkeypatch.setattr(bgs, "get_with_retries", fake_get)

    record = bgs.discover_gb02ss0001()

    assert record.source_dataset_id == "GB02SS0001"
    assert record.survey_start_date == "2002-08-29"
    assert record.survey_end_date == "2002-09-03"
    assert record.acquisition_year == 2002
    assert record.nominal_resolution_m == 5.0
    assert record.horizontal_crs == "EPSG:4230"
    assert record.vertical_datum is None  # never stated, never guessed
    assert record.download_available is False
    assert record.manual_download_required is True
    assert record.geometry_wgs84 is not None
    assert record.geometry_wgs84.bounds == pytest.approx((1.4, 53.3333, 1.6, 53.5), rel=1e-3)


def test_bgs_discover_falls_back_to_stub_on_zero_matches(monkeypatch):
    def fake_get(url, params=None):
        return FakeResponse(BGS_ZERO_MATCH_XML)

    monkeypatch.setattr(bgs, "get_with_retries", fake_get)

    record = bgs.discover_gb02ss0001()

    assert record.source_dataset_id == "GB02SS0001"
    assert record.acquisition_status == "live_verification_failed"
    assert record.geometry_wgs84 is None
    # Known facts from the ticket are still preserved, never fabricated further.
    assert record.acquisition_year == 2002


def test_bgs_discover_falls_back_to_stub_on_network_failure(monkeypatch):
    def fake_get(url, params=None):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(bgs, "get_with_retries", fake_get)

    record = bgs.discover_cs03ss0003()

    assert record.acquisition_status == "live_verification_failed"
    assert "simulated network failure" in record.notes


def test_bgs_sea5_handles_inaccessible_live_record_without_crashing(monkeypatch):
    def fake_get(url, params=None):
        return FakeResponse(BGS_EXCEPTION_XML)

    monkeypatch.setattr(bgs, "get_with_retries", fake_get)

    record = bgs.discover_sea5_ower_bank()

    assert record.source_dataset_id == "SEA5-OWER-BANK"
    assert record.geometry_wgs84 is None  # deliberately not using the mismatched footprint
    assert "inaccessible" in record.notes.lower()
    # Known ticket-provided facts are still preserved.
    assert record.acquisition_year == 2003


def test_bgs_discover_bgs_surveys_returns_all_three_even_if_one_source_fails(monkeypatch):
    call_count = {"n": 0}

    def flaky_get(url, params=None):
        call_count["n"] += 1
        if params.get("REQUEST") == "GetRecordById" and "aba64100" not in params.get("id", ""):
            raise requests.ConnectionError("simulated failure")
        return FakeResponse(
            BGS_ZERO_MATCH_XML if params.get("REQUEST") == "GetRecords" else BGS_EXCEPTION_XML
        )

    monkeypatch.setattr(bgs, "get_with_retries", flaky_get)

    records = bgs.discover_bgs_surveys()

    assert len(records) == 3
    assert {r.source_dataset_id for r in records} == {"GB02SS0001", "CS03SS0003", "SEA5-OWER-BANK"}


# --- UKHO: MEDIN bbox search filtering + failure behaviour --------------------

MEDIN_MIXED_RESPONSE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<csw:GetRecordsResponse xmlns:csw="http://www.opengis.net/cat/csw/2.0.2" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <csw:SearchStatus timestamp="2026-09-03T19:23:14Z" />
  <csw:SearchResults numberOfRecordsMatched="3" numberOfRecordsReturned="3" elementSet="brief" nextRecord="0">
    <csw:Record><dc:identifier>1</dc:identifier><dc:title>General Bathymetric Chart of the Oceans (GEBCO) One Minute Grid</dc:title></csw:Record>
    <csw:Record><dc:identifier>2</dc:identifier><dc:title>UK National Databank of Moored Current Meter Data (1967-)</dc:title></csw:Record>
    <csw:Record><dc:identifier>3</dc:identifier><dc:title>UKHO Admiralty Multibeam Survey of the Southern North Sea</dc:title></csw:Record>
  </csw:SearchResults>
</csw:GetRecordsResponse>"""


def test_ukho_discover_filters_broad_datasets_keeps_ukho_specific(monkeypatch):
    def fake_post(url, data, headers=None):
        return FakeResponse(MEDIN_MIXED_RESPONSE_XML)

    monkeypatch.setattr(ukho, "post_with_retries", fake_post)

    records, status = ukho.discover_ukho_surveys((1.0, 53.0, 2.0, 54.0))

    assert status.query_succeeded is True
    assert status.total_records_seen == 3
    assert len(records) == 1
    assert "UKHO Admiralty" in records[0].title


def test_ukho_discover_reports_zero_when_only_broad_datasets_match(monkeypatch):
    only_broad_xml = MEDIN_MIXED_RESPONSE_XML.replace(
        b"<dc:title>UKHO Admiralty Multibeam Survey of the Southern North Sea</dc:title>",
        b"<dc:title>GESLA sea level dataset</dc:title>",
    )

    def fake_post(url, data, headers=None):
        return FakeResponse(only_broad_xml)

    monkeypatch.setattr(ukho, "post_with_retries", fake_post)

    records, status = ukho.discover_ukho_surveys((1.0, 53.0, 2.0, 54.0))

    assert records == []
    assert status.query_succeeded is True
    assert status.relevant_records_found == 0


def test_ukho_discover_network_failure_returns_status_not_exception(monkeypatch):
    def fake_post(url, data, headers=None):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(ukho, "post_with_retries", fake_post)

    records, status = ukho.discover_ukho_surveys((1.0, 53.0, 2.0, 54.0))

    assert records == []
    assert status.query_succeeded is False
    assert "simulated network failure" in status.message


def test_ukho_discover_handles_exception_report_body(monkeypatch):
    def fake_post(url, data, headers=None):
        return FakeResponse(BGS_EXCEPTION_XML)  # any ows:ExceptionReport body works here

    monkeypatch.setattr(ukho, "post_with_retries", fake_post)

    records, status = ukho.discover_ukho_surveys((1.0, 53.0, 2.0, 54.0))

    assert records == []
    assert status.query_succeeded is False


# --- EMODnet: source-reference / quality-index / MSL attribution (MAR-006) ---

SOURCE_REFERENCES_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "identifier": "121954",
                "type": "CDI",
                "edmo_id": 2607,
                "release": "2024",
                "date_start": "2020-01-01",
                "date_end": "2020-06-01",
                "metadata_url": "https://cdi-bathymetry.seadatanet.org/report/edmo/2607/121954",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1.6, 53.3], [1.9, 53.3], [1.9, 53.4], [1.6, 53.4], [1.6, 53.3]]],
            },
        }
    ],
}

QUALITY_INDEX_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "identifier": "121954",
                "type": "CDI",
                "combined": 76.92,
                "horizontal": 2,
                "vertical": 3,
                "age": 3,
                "purpose": 3,
                "release": "2024",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1.6, 53.3], [1.9, 53.3], [1.9, 53.4], [1.6, 53.4], [1.6, 53.3]]],
            },
        }
    ],
}

DOWNLOAD_TILES_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "dtm_release": "2024",
                "dtm_tile": "D4",
                "product_format": "ESRI ASCII Mean Sea Level",
                "download_url": "https://downloads.emodnet-bathymetry.eu/v12/D4_2024.msl.zip",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1.0, 53.0], [3.0, 53.0], [3.0, 54.0], [1.0, 54.0], [1.0, 53.0]]],
            },
        }
    ],
}


def test_wfs_get_feature_builds_lat_lon_order_bbox_and_release_filter(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeJsonResponse({"type": "FeatureCollection", "features": []})

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    emodnet._wfs_get_feature(
        emodnet.SOURCE_REFERENCES_LAYER, (1.0, 53.0, 2.0, 54.0), release="2024"
    )

    assert captured["url"] == emodnet.WFS_BASE_URL
    params = captured["params"]
    assert params["typeNames"] == emodnet.SOURCE_REFERENCES_LAYER
    # (lat, lon, lat, lon) order -- matching this server's declared EPSG:4326
    # axis order, confirmed empirically (see _wfs_get_feature's docstring).
    assert params["CQL_FILTER"] == "release='2024' AND BBOX(geom,53.0,1.0,54.0,2.0)"


def test_wfs_get_feature_without_release_omits_attribute_filter(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return FakeJsonResponse({"type": "FeatureCollection", "features": []})

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    emodnet._wfs_get_feature(emodnet.DOWNLOAD_TILES_LAYER, (1.0, 53.0, 2.0, 54.0))

    assert captured["params"]["CQL_FILTER"] == "BBOX(geom,53.0,1.0,54.0,2.0)"


def test_wfs_get_feature_network_failure_raises_attribution_unavailable(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    with pytest.raises(emodnet.EmodnetAttributionUnavailableError):
        emodnet._wfs_get_feature(emodnet.SOURCE_REFERENCES_LAYER, (1.0, 53.0, 2.0, 54.0))


def test_wfs_get_feature_non_json_response_raises_attribution_unavailable(monkeypatch):
    class BadJsonResponse(FakeJsonResponse):
        def json(self):
            raise ValueError("not json")

    def fake_get(url, params=None, timeout=None):
        return BadJsonResponse({})

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    with pytest.raises(emodnet.EmodnetAttributionUnavailableError):
        emodnet._wfs_get_feature(emodnet.SOURCE_REFERENCES_LAYER, (1.0, 53.0, 2.0, 54.0))


def test_fetch_source_references_parses_real_shaped_geojson(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return FakeJsonResponse(SOURCE_REFERENCES_GEOJSON)

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    features = emodnet.fetch_source_references((1.0, 53.0, 2.0, 54.0))

    assert len(features) == 1
    feature = features[0]
    assert feature.identifier == "121954"
    assert feature.source_type == "CDI"
    assert feature.edmo_id == 2607
    assert feature.metadata_url.startswith("https://cdi-bathymetry.seadatanet.org")
    assert feature.geometry_wgs84.is_valid


def test_fetch_source_references_unavailable_raises_not_silently_empty(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    with pytest.raises(emodnet.EmodnetAttributionUnavailableError):
        emodnet.fetch_source_references((1.0, 53.0, 2.0, 54.0))


def test_fetch_quality_index_parses_real_shaped_geojson_preserving_raw_classes(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return FakeJsonResponse(QUALITY_INDEX_GEOJSON)

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    features = emodnet.fetch_quality_index((1.0, 53.0, 2.0, 54.0))

    assert len(features) == 1
    feature = features[0]
    assert feature.horizontal == 2
    assert feature.vertical == 3
    assert feature.age == 3
    assert feature.purpose == 3
    assert feature.combined == pytest.approx(76.92)


def test_fetch_quality_index_unavailable_raises(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    with pytest.raises(emodnet.EmodnetAttributionUnavailableError):
        emodnet.fetch_quality_index((1.0, 53.0, 2.0, 54.0))


def test_check_msl_availability_finds_matching_tile(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return FakeJsonResponse(DOWNLOAD_TILES_GEOJSON)

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    result = emodnet.check_msl_availability((1.0, 53.0, 2.0, 54.0))

    assert result.available is True
    assert result.tile_id == "D4"
    assert result.dtm_release == "2024"
    assert "mean sea level" in result.format_label.lower()
    assert result.download_url.endswith(".msl.zip")
    assert "not downloaded" in result.notes.lower()


def test_check_msl_availability_no_matching_format_returns_unavailable(monkeypatch):
    no_msl = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "dtm_release": "2024",
                    "dtm_tile": "D4",
                    "product_format": "ESRI ASCII LAT",
                    "download_url": "https://example.invalid/D4_2024.lat.zip",
                },
                "geometry": DOWNLOAD_TILES_GEOJSON["features"][0]["geometry"],
            }
        ],
    }

    def fake_get(url, params=None, timeout=None):
        return FakeJsonResponse(no_msl)

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    result = emodnet.check_msl_availability((1.0, 53.0, 2.0, 54.0))

    assert result.available is False
    assert result.download_url is None


def test_check_msl_availability_network_failure_returns_result_not_exception(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(emodnet.requests, "get", fake_get)

    result = emodnet.check_msl_availability((1.0, 53.0, 2.0, 54.0))

    assert result.available is False
    assert "simulated network failure" in result.notes
