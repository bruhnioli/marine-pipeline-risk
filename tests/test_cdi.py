"""Offline unit tests for marine_engine.providers.bathymetry.cdi.

The HTML fixtures below are trimmed-but-real captures of the actual
SeaDataNet CDI report pages for PL854 source-reference ids 110153/121953
(read through a real browser during MAR-006B implementation, since a
client-side proof-of-work challenge blocks plain HTTP clients -- see
cdi.py's module docstring). No network access.
"""

import pytest
import requests

from marine_engine.providers.bathymetry import cdi

# Real (trimmed) CDI report content for source-reference 110153 -- single-beam,
# 1992, UKHO/OceanWise. Trimmed of unrelated map/vocabulary markup but the
# JSON-LD block and the fields the parser reads are verbatim.
HTML_110153 = """<table class="browse-elastic-results"><tbody>
<tr><td>Data set name</td><td>Haddock Bank</td></tr>
<tr><td>Abstract</td><td>UK Civil Hydrographic Programme Survey HI560</td></tr>
<tr><td>Data format</td><td><ul><li><a href="http://vocab.nerc.ac.uk/collection/L24/current/CF">Climate and Forecast NetCDF</a> <span><b>Version</b> 3.5</span></li></ul></td></tr>
<tr><td>Data size</td><td> 57751.000 MB<br></td></tr>
<tr><td>Datum</td><td>World Geodetic System 84</td></tr>
<tr><td>Minimum instrument depth (m)</td><td>37.6</td></tr>
<tr><td>Maximum instrument depth (m)</td><td>6.3</td></tr>
<tr><td>Instrument/gear category</td><td><ul><li><a>single-beam echosounders</a></li></ul></td></tr>
<tr><td>Horizontal resolution</td><td>0</td></tr>
<tr><td>Horizontal resolution</td><td>Metres</td></tr>
<tr><td>Vertical resolution</td><td>0</td></tr>
<tr><td>Vertical resolution</td><td>Metres</td></tr>
<tr><td>Platform type</td><td><ul><li><a>unknown</a></li></ul></td></tr>
<tr><td>Cruise name</td><td>HI560</td></tr>
<tr><td>Alternative cruise name</td><td>110153</td></tr>
<tr><td>Data originator</td><td><a href="https://edmo.seadatanet.org/report/26">United Kingdom Hydrographic Office </a></td></tr>
<tr><td>Data custodian</td><td><a href="https://edmo.seadatanet.org/report/26">United Kingdom Hydrographic Office </a></td></tr>
<tr><td>Project name</td><td><a href="https://edmerp.seadatanet.org/report/11812">UKCHP - UK Civil Hydrography Programme (monitoring project)</a></td></tr>
<tr><td>Data Distributor</td><td><a href="https://edmo.seadatanet.org/report/2607">OceanWise Limited </a></td></tr>
<tr><td>Access/ordering of data</td><td>web data access with registration</td></tr>
<tr><td>Access restriction</td><td><ul><li><a title="CDI:L08::RS">by negotiation</a></li></ul></td></tr>
<tr><td>Quality info</td><td><table class="table-dq-objects"><tbody><tr><th>Name</th><th>Date</th><th>Comment</th></tr>
<tr><td>QI_Purpose</td><td>2017-05-19</td><td>Hydrographic survey or compatible with hydrographic standards</td></tr>
<tr><td>QI_Vertical</td><td>2017-05-19</td><td>Lidar, SBES High Frequency</td></tr>
<tr><td>QI_Horizontal</td><td>2017-05-19</td><td>unknown or larger than 500m</td></tr>
</tbody></table></td></tr>
<tr><td>CDI-record id</td><td>1298109</td></tr>
<tr><td>CDI-record initial import date</td><td>2012-07-16 07:38:31.220</td></tr>
<tr><td>CDI-record last update</td><td>2017-12-01 17:31:52.280</td></tr>
<tr><td>Point of contact</td><td><a href="https://edmo.seadatanet.org/report/2607">OceanWise Limited </a></td></tr>
<script type="application/ld+json">{
    "@context": {"@vocab": "https://schema.org/"},
    "@type": "Dataset",
    "name": "Haddock Bank",
    "description": "UK Civil Hydrographic Programme Survey HI560",
    "temporalCoverage": "1992-09-21/1992-12-08",
    "url": "https://cdi.seadatanet.org/report/1298109",
    "contentLocation": {"@type": "Place", "geo": {"@type": "GeoShape", "box": "53.2625007629 1.3927999735 53.4254989624 1.6612999439"}},
    "usageInfo": "by negotiation(RS)",
    "identifier": ["110153", "https://cdi.seadatanet.org/report/1298109"],
    "author": {"@type": "Organization", "name": "OceanWise Limited", "url": "https://edmo.seadatanet.org/report/2607", "identifier": "https://edmo.seadatanet.org/report/2607"},
    "maintainer": {"@type": "Organization", "name": "United Kingdom Hydrographic Office", "url": "https://edmo.seadatanet.org/report/26", "identifier": "https://edmo.seadatanet.org/report/26"},
    "producer": {"@type": "Organization", "name": "United Kingdom Hydrographic Office", "url": "https://edmo.seadatanet.org/report/26", "identifier": "https://edmo.seadatanet.org/report/26"},
    "publisher": {"@type": "Organization", "name": "OceanWise Limited", "url": "https://edmo.seadatanet.org/report/2607", "identifier": "https://edmo.seadatanet.org/report/2607"},
    "encodingFormat": ["application/netcdf"],
    "license": "http://vocab.nerc.ac.uk/collection/L08/current/RS/"
}</script></tbody></table>"""

# Real content for 121953 -- same cruise/epoch as 121954, "unknown" instrument
# field despite QI_Vertical=4's own label mentioning MBES (see the module
# docstring's discussion of why that is NOT independent corroboration).
HTML_121953 = """<table class="browse-elastic-results"><tbody>
<tr><td>Data set name</td><td>North Sea, Broken Bank to North Haisborough, Block 2</td></tr>
<tr><td>Abstract</td><td>UK Civil Hydrographic Programme Survey HI524-HI525-HI531</td></tr>
<tr><td>Data format</td><td><ul><li><a>Climate and Forecast NetCDF</a> <span><b>Version</b> 3.5</span></li></ul></td></tr>
<tr><td>Data size</td><td> 57751.000 MB<br></td></tr>
<tr><td>Datum</td><td>World Geodetic System 84</td></tr>
<tr><td>Instrument/gear category</td><td><ul><li><a>unknown</a></li></ul></td></tr>
<tr><td>Horizontal resolution</td><td>0</td></tr>
<tr><td>Horizontal resolution</td><td>Metres</td></tr>
<tr><td>Vertical resolution</td><td>0</td></tr>
<tr><td>Vertical resolution</td><td>Metres</td></tr>
<tr><td>Cruise name</td><td>HI524-HI525-HI531</td></tr>
<tr><td>Alternative cruise name</td><td>121953</td></tr>
<tr><td>Data originator</td><td><a href="https://edmo.seadatanet.org/report/26">United Kingdom Hydrographic Office </a></td></tr>
<tr><td>Data Distributor</td><td><a href="https://edmo.seadatanet.org/report/2607">OceanWise Limited </a></td></tr>
<tr><td>Access/ordering of data</td><td>web data access with registration</td></tr>
<tr><td>Access restriction</td><td><ul><li><a>by negotiation</a></li></ul></td></tr>
<tr><td>Quality info</td><td><table class="table-dq-objects"><tbody><tr><th>Name</th><th>Date</th><th>Comment</th></tr>
<tr><td>QI_Purpose</td><td>2017-05-19</td><td>Hydrographic survey or compatible with hydrographic standards</td></tr>
<tr><td>QI_Vertical</td><td>2017-05-19</td><td>MBES high frequency (larger than 100kHz)</td></tr>
<tr><td>QI_Horizontal</td><td>2017-05-19</td><td>smaller than 20m</td></tr>
</tbody></table></td></tr>
<tr><td>CDI-record id</td><td>3044183</td></tr>
<tr><td>CDI-record initial import date</td><td>2020-07-06 15:23:30.777</td></tr>
<tr><td>CDI-record last update</td><td>2020-07-06 15:23:35.403</td></tr>
<script type="application/ld+json">{
    "@context": {"@vocab": "https://schema.org/"},
    "@type": "Dataset",
    "name": "North Sea, Broken Bank to North Haisborough, Block 2",
    "description": "UK Civil Hydrographic Programme Survey HI524-HI525-HI531",
    "temporalCoverage": "1991-04-24/1991-08-16",
    "url": "https://cdi.seadatanet.org/report/3044183",
    "contentLocation": {"@type": "Place", "geo": {"@type": "GeoShape", "box": "53.3623428345 1.8569539785 53.5071105957 2.1509323120"}},
    "usageInfo": "by negotiation(RS)",
    "identifier": ["121953", "https://cdi.seadatanet.org/report/3044183"],
    "author": {"@type": "Organization", "name": "OceanWise Limited", "url": "https://edmo.seadatanet.org/report/2607", "identifier": "https://edmo.seadatanet.org/report/2607"},
    "maintainer": {"@type": "Organization", "name": "United Kingdom Hydrographic Office", "url": "https://edmo.seadatanet.org/report/26", "identifier": "https://edmo.seadatanet.org/report/26"},
    "producer": {"@type": "Organization", "name": "United Kingdom Hydrographic Office", "url": "https://edmo.seadatanet.org/report/26", "identifier": "https://edmo.seadatanet.org/report/26"},
    "publisher": {"@type": "Organization", "name": "OceanWise Limited", "url": "https://edmo.seadatanet.org/report/2607", "identifier": "https://edmo.seadatanet.org/report/2607"},
    "encodingFormat": ["application/netcdf"],
    "license": "http://vocab.nerc.ac.uk/collection/L08/current/RS/"
}</script></tbody></table>"""

BOT_CHALLENGE_HTML = """<!DOCTYPE html><html><head><title>Marine Data Access - SeaDataNet CDI</title></head>
<body><div class="checking-browser"><h1>Checking your browser&hellip;</h1></div>
<script>var COOKIE = "bot_challenge_token";</script></body></html>"""


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


# --- parse_cdi_report_html: JSON-LD + HTML table fields ----------------------


def test_parse_cdi_report_html_extracts_json_ld_fields():
    record = cdi.parse_cdi_report_html(
        HTML_110153,
        source_reference_id="110153",
        metadata_url="https://cdi-bathymetry.seadatanet.org/report/edmo/2607/110153",
        resolution_status=cdi.RESOLUTION_LIVE,
    )

    assert record.title == "Haddock Bank"
    assert record.description == "UK Civil Hydrographic Programme Survey HI560"
    assert record.acquisition_start.isoformat() == "1992-09-21"
    assert record.acquisition_end.isoformat() == "1992-12-08"
    assert record.acquisition_year == 1992
    assert record.data_format == "Climate and Forecast NetCDF Version 3.5"
    assert record.data_size_mb == pytest.approx(57751.0)
    assert record.licence_code == "RS"
    assert record.cdi_record_id == "1298109"
    assert record.data_access_url == "https://cdi.seadatanet.org/report/1298109"
    assert record.geographic_footprint is not None
    assert record.geographic_footprint.bounds == pytest.approx(
        (1.3927999735, 53.2625007629, 1.6612999439, 53.4254989624)
    )


def test_parse_cdi_report_html_separates_originator_from_distributor():
    """The actual surveying authority (maintainer/producer) must never be
    conflated with the distributor/access-point (author/publisher) -- the
    EMODnet WFS `edmo_id` field only ever exposes the latter."""

    record = cdi.parse_cdi_report_html(
        HTML_110153,
        source_reference_id="110153",
        metadata_url="https://x",
        resolution_status="live",
    )

    assert record.organisation == "United Kingdom Hydrographic Office"
    assert record.organisation_edmo_id == 26
    assert record.data_centre == "OceanWise Limited"
    assert record.data_centre_edmo_id == 2607
    assert record.organisation_edmo_id != record.data_centre_edmo_id


def test_parse_cdi_report_html_extracts_html_table_supplementary_fields():
    record = cdi.parse_cdi_report_html(
        HTML_110153,
        source_reference_id="110153",
        metadata_url="https://x",
        resolution_status="live",
    )

    assert record.survey_method == "single-beam echosounders"
    assert record.survey_name == "HI560"
    assert record.cdi_import_date == "2012-07-16 07:38:31.220"
    assert record.cdi_update_date == "2017-12-01 17:31:52.280"
    assert record.horizontal_crs == "EPSG:4326"


def test_parse_cdi_report_html_zero_resolution_marked_unspecified_not_fabricated():
    record = cdi.parse_cdi_report_html(
        HTML_110153,
        source_reference_id="110153",
        metadata_url="https://x",
        resolution_status="live",
    )

    # The source literally reports "0" -- this must never be surfaced as a
    # real numeric resolution value; it is preserved as an explanatory note.
    assert record.horizontal_resolution_note is not None
    assert "unspecified" in record.horizontal_resolution_note
    assert record.vertical_resolution_note is not None
    assert "unspecified" in record.vertical_resolution_note


def test_parse_cdi_report_html_missing_json_ld_falls_back_to_table_fields():
    html_without_json_ld = HTML_110153.split("<script")[0] + "</tbody></table>"

    record = cdi.parse_cdi_report_html(
        html_without_json_ld,
        source_reference_id="110153",
        metadata_url="https://x",
        resolution_status="live",
    )

    assert record.title == "Haddock Bank"  # from the HTML table's "Data set name"
    assert record.acquisition_year is None  # only the JSON-LD carries temporalCoverage
    assert record.geographic_footprint is None


def test_parse_cdi_report_html_121953_confirms_mbes_qi_but_unknown_instrument_field():
    record = cdi.parse_cdi_report_html(
        HTML_121953,
        source_reference_id="121953",
        metadata_url="https://x",
        resolution_status="live",
    )

    assert record.survey_method == "unknown"
    assert record.acquisition_year == 1991


# --- fetch_cdi_report_html: bot-challenge / network failure handling --------


def test_fetch_cdi_report_html_detects_503_bot_challenge(monkeypatch):
    def fake_get(url, timeout=None):
        return FakeResponse(BOT_CHALLENGE_HTML, status_code=503)

    monkeypatch.setattr(cdi.requests, "get", fake_get)

    with pytest.raises(cdi.CdiUnavailableError, match="bot-detection"):
        cdi.fetch_cdi_report_html("https://cdi-bathymetry.seadatanet.org/report/edmo/2607/110153")


def test_fetch_cdi_report_html_detects_challenge_marker_even_without_503(monkeypatch):
    def fake_get(url, timeout=None):
        return FakeResponse(BOT_CHALLENGE_HTML, status_code=200)

    monkeypatch.setattr(cdi.requests, "get", fake_get)

    with pytest.raises(cdi.CdiUnavailableError):
        cdi.fetch_cdi_report_html("https://x")


def test_fetch_cdi_report_html_network_failure_raises(monkeypatch):
    def fake_get(url, timeout=None):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(cdi.requests, "get", fake_get)

    with pytest.raises(cdi.CdiUnavailableError, match="simulated network failure"):
        cdi.fetch_cdi_report_html("https://x")


def test_fetch_cdi_report_html_non_200_non_challenge_raises(monkeypatch):
    def fake_get(url, timeout=None):
        return FakeResponse("<html>not found</html>", status_code=404)

    monkeypatch.setattr(cdi.requests, "get", fake_get)

    with pytest.raises(cdi.CdiUnavailableError, match="404"):
        cdi.fetch_cdi_report_html("https://x")


def test_fetch_cdi_report_html_success_returns_real_text(monkeypatch):
    def fake_get(url, timeout=None):
        return FakeResponse(HTML_110153, status_code=200)

    monkeypatch.setattr(cdi.requests, "get", fake_get)

    html = cdi.fetch_cdi_report_html("https://x")
    assert "Haddock Bank" in html


# --- resolve_cdi_record: live-first, cached-snapshot fallback ---------------


def test_resolve_cdi_record_uses_live_fetch_when_available(monkeypatch):
    def fake_get(url, timeout=None):
        return FakeResponse(HTML_121953, status_code=200)

    monkeypatch.setattr(cdi.requests, "get", fake_get)

    record = cdi.resolve_cdi_record("121953", edmo_id=2607)

    assert record.resolution_status == cdi.RESOLUTION_LIVE
    assert record.title == "North Sea, Broken Bank to North Haisborough, Block 2"


def test_resolve_cdi_record_falls_back_to_known_snapshot_on_bot_challenge(monkeypatch):
    def fake_get(url, timeout=None):
        return FakeResponse(BOT_CHALLENGE_HTML, status_code=503)

    monkeypatch.setattr(cdi.requests, "get", fake_get)

    record = cdi.resolve_cdi_record("110153", edmo_id=2607)

    assert record.resolution_status == cdi.RESOLUTION_CACHED_SNAPSHOT
    assert record.title == "Haddock Bank"
    assert record.acquisition_year == 1992


def test_resolve_cdi_record_unknown_id_with_fetch_failure_is_unavailable_not_fabricated(
    monkeypatch,
):
    def fake_get(url, timeout=None):
        return FakeResponse(BOT_CHALLENGE_HTML, status_code=503)

    monkeypatch.setattr(cdi.requests, "get", fake_get)

    record = cdi.resolve_cdi_record("999999", edmo_id=2607)

    assert record.resolution_status == cdi.RESOLUTION_UNAVAILABLE
    assert record.title is None
    assert record.acquisition_year is None
    assert record.organisation is None


# --- survey age -----------------------------------------------------------


def test_calculate_survey_age_known_year():
    assert cdi.calculate_survey_age(1990, 2024) == 34


def test_calculate_survey_age_none_when_year_missing():
    assert cdi.calculate_survey_age(None, 2024) is None


# --- QI consistency classification -----------------------------------------


def test_classify_qi_age_consistency_old_survey_matches_qi_age_zero():
    assert cdi.classify_qi_age_consistency(1990, qi_age=0) == cdi.CONSISTENCY_CONSISTENT


def test_classify_qi_age_consistency_recent_survey_contradicts_qi_age_zero():
    assert cdi.classify_qi_age_consistency(2020, qi_age=0) == cdi.CONSISTENCY_INCONSISTENT


def test_classify_qi_age_consistency_not_verifiable_when_year_missing():
    assert cdi.classify_qi_age_consistency(None, qi_age=0) == cdi.CONSISTENCY_NOT_VERIFIABLE


def test_classify_qi_vertical_consistency_mbes_matches_class_4():
    assert (
        cdi.classify_qi_vertical_consistency("multibeam echosounder", qi_vertical=4)
        == cdi.CONSISTENCY_CONSISTENT
    )


def test_classify_qi_vertical_consistency_singlebeam_matches_class_3():
    assert (
        cdi.classify_qi_vertical_consistency("single-beam echosounders", qi_vertical=3)
        == cdi.CONSISTENCY_CONSISTENT
    )


def test_classify_qi_vertical_consistency_not_verifiable_when_instrument_unknown():
    assert (
        cdi.classify_qi_vertical_consistency("unknown", qi_vertical=4)
        == cdi.CONSISTENCY_NOT_VERIFIABLE
    )


def test_classify_qi_vertical_consistency_inconsistent_when_mismatched():
    assert (
        cdi.classify_qi_vertical_consistency("single-beam echosounders", qi_vertical=4)
        == cdi.CONSISTENCY_INCONSISTENT
    )


def test_classify_qi_purpose_consistency_matches_hydrographic_context():
    result = cdi.classify_qi_purpose_consistency(
        "United Kingdom Hydrographic Office", "OceanWise Limited", None, qi_purpose=3
    )
    assert result == cdi.CONSISTENCY_CONSISTENT


def test_classify_qi_purpose_consistency_not_verifiable_without_hydrographic_context():
    result = cdi.classify_qi_purpose_consistency(
        "Acme Corp", "Acme Distributor", None, qi_purpose=3
    )
    assert result == cdi.CONSISTENCY_NOT_VERIFIABLE


def test_classify_qi_metadata_consistency_overall_consistent_for_110153():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    result = cdi.classify_qi_metadata_consistency(
        record, qi_age=0, qi_horizontal=0, qi_vertical=3, qi_purpose=3
    )
    assert result == cdi.CONSISTENCY_CONSISTENT


def test_classify_qi_metadata_consistency_inconsistency_dominates():
    record = cdi.KNOWN_CDI_RECORDS["110153"]  # single-beam, 1992
    # A deliberately contradictory QI_Vertical=4 (MBES) for a known single-beam survey.
    result = cdi.classify_qi_metadata_consistency(
        record, qi_age=0, qi_horizontal=0, qi_vertical=4, qi_purpose=3
    )
    assert result == cdi.CONSISTENCY_INCONSISTENT


def test_classify_qi_metadata_consistency_not_verifiable_when_nothing_checkable():
    record = cdi.KNOWN_CDI_RECORDS["121953"]  # instrument field is "unknown"
    result = cdi.classify_qi_metadata_consistency(
        record, qi_age=None, qi_horizontal=3, qi_vertical=4, qi_purpose=None
    )
    assert result == cdi.CONSISTENCY_NOT_VERIFIABLE


# --- access / recovery-potential classification -----------------------------


def test_classify_access_owner_permission_required_for_negotiation():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    assert cdi.classify_access(record) == cdi.ACCESS_OWNER_PERMISSION_REQUIRED


def test_classify_access_registration_required():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    modified = record.__class__(**{**record.__dict__, "access_restriction": None})
    assert cdi.classify_access(modified) == cdi.ACCESS_REGISTRATION_REQUIRED


def test_classify_access_direct_download():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    modified = record.__class__(
        **{
            **record.__dict__,
            "access_mechanism": "web data access without restriction",
            "access_restriction": None,
        }
    )
    assert cdi.classify_access(modified) == cdi.ACCESS_DIRECT_DOWNLOAD


def test_classify_access_seadatanet_request():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    modified = record.__class__(
        **{
            **record.__dict__,
            "access_mechanism": "SeaDataNet shopping basket order",
            "access_restriction": None,
        }
    )
    assert cdi.classify_access(modified) == cdi.ACCESS_SEADATANET_REQUEST


def test_classify_access_unknown_when_no_information():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    modified = record.__class__(
        **{**record.__dict__, "access_mechanism": None, "access_restriction": None}
    )
    assert cdi.classify_access(modified) == cdi.ACCESS_UNKNOWN


def test_classify_recovery_potential_resolution_unknown_for_requestable_but_unstated_resolution():
    """MAR-006C: this is the real state for all three current PL854 records --
    a genuine request path exists (registration + owner negotiation), but CDI
    states no numeric resolution, so recovery_potential must NOT claim
    HIGH_RES_*. Access is still reported correctly via `classify_access`,
    kept entirely separate from this field."""

    record = cdi.KNOWN_CDI_RECORDS["110153"]
    access_class = cdi.classify_access(record)

    assert access_class == cdi.ACCESS_OWNER_PERMISSION_REQUIRED  # requestable IS still true
    assert cdi.classify_recovery_potential(record, access_class) == cdi.RECOVERY_RESOLUTION_UNKNOWN


def test_classify_recovery_potential_requestable_when_resolution_confirmed():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    confirmed = record.__class__(**{**record.__dict__, "horizontal_resolution_note": "2 Metres"})
    assert (
        cdi.classify_recovery_potential(confirmed, cdi.ACCESS_OWNER_PERMISSION_REQUIRED)
        == cdi.RECOVERY_HIGH_RES_REQUESTABLE
    )


def test_classify_recovery_potential_recoverable_when_resolution_confirmed_and_direct_download():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    confirmed = record.__class__(**{**record.__dict__, "vertical_resolution_note": "1 Metres"})
    assert cdi.classify_recovery_potential(confirmed, cdi.ACCESS_DIRECT_DOWNLOAD) == (
        cdi.RECOVERY_HIGH_RES_RECOVERABLE
    )


def test_classify_recovery_potential_not_recoverable_when_restricted_even_with_confirmed_resolution():
    """Access establishing the source cannot be obtained dominates, regardless
    of whether resolution is confirmed."""

    record = cdi.KNOWN_CDI_RECORDS["110153"]
    confirmed = record.__class__(**{**record.__dict__, "horizontal_resolution_note": "2 Metres"})
    assert (
        cdi.classify_recovery_potential(confirmed, cdi.ACCESS_RESTRICTED)
        == cdi.RECOVERY_NOT_RECOVERABLE
    )


def test_classify_recovery_potential_resolution_unknown_without_a_real_dataset():
    record = cdi.KNOWN_CDI_RECORDS["110153"]
    modified = record.__class__(**{**record.__dict__, "data_format": None, "data_size_mb": None})
    assert (
        cdi.classify_recovery_potential(modified, cdi.ACCESS_UNKNOWN)
        == cdi.RECOVERY_RESOLUTION_UNKNOWN
    )


def test_classify_recovery_potential_zero_resolution_note_does_not_count_as_confirmed():
    """CDI's "0 Metres" convention means unspecified, not a real sub-metre claim."""

    record = cdi.KNOWN_CDI_RECORDS["110153"]
    assert record.horizontal_resolution_note is not None
    assert "unspecified" in record.horizontal_resolution_note
    assert (
        cdi.classify_recovery_potential(record, cdi.ACCESS_DIRECT_DOWNLOAD)
        == cdi.RECOVERY_RESOLUTION_UNKNOWN
    )


def test_classify_recovery_potential_qi_vertical_mbes_alone_is_not_proof_of_resolution():
    """QI_Vertical=4 (MBES) must never substitute for a stated numeric resolution."""

    record = cdi.KNOWN_CDI_RECORDS["121953"]  # QI_Vertical=4 in MAR-006, no stated resolution
    access_class = cdi.classify_access(record)
    assert cdi.classify_recovery_potential(record, access_class) == cdi.RECOVERY_RESOLUTION_UNKNOWN


def test_classify_recovery_potential_never_claims_a_specific_resolution():
    """No CDI record here states a numeric resolution -- recovery_potential must
    never imply one (e.g. must not say "1 m data available")."""

    record = cdi.KNOWN_CDI_RECORDS["121953"]
    assert record.horizontal_resolution_note is not None
    assert "1 m" not in record.horizontal_resolution_note
    assert "1m" not in record.horizontal_resolution_note


def test_access_classification_independent_of_recovery_classification():
    """Changing only the resolution note must not change access_class, and
    vice versa -- the two functions must never share hidden state."""

    record = cdi.KNOWN_CDI_RECORDS["110153"]
    access_before = cdi.classify_access(record)

    confirmed = record.__class__(**{**record.__dict__, "horizontal_resolution_note": "2 Metres"})
    access_after = cdi.classify_access(confirmed)

    assert access_before == access_after == cdi.ACCESS_OWNER_PERMISSION_REQUIRED
    assert cdi.classify_recovery_potential(
        record, access_before
    ) != cdi.classify_recovery_potential(confirmed, access_after)


def test_extract_stated_resolution_m_treats_zero_as_unspecified():
    assert cdi._extract_stated_resolution_m("0") is None
    assert cdi._extract_stated_resolution_m("0 (source reports 0/unspecified)") is None
    assert cdi._extract_stated_resolution_m(None) is None


def test_extract_stated_resolution_m_parses_real_value():
    assert cdi._extract_stated_resolution_m("2 Metres") == pytest.approx(2.0)
    assert cdi._extract_stated_resolution_m("0.5") == pytest.approx(0.5)


# --- known-snapshot integrity ------------------------------------------------


def test_known_cdi_records_are_marked_as_cached_snapshots():
    for record in cdi.KNOWN_CDI_RECORDS.values():
        assert record.resolution_status == cdi.RESOLUTION_CACHED_SNAPSHOT


def test_known_cdi_records_have_the_resolved_real_epochs():
    assert cdi.KNOWN_CDI_RECORDS["110153"].acquisition_year == 1992
    assert cdi.KNOWN_CDI_RECORDS["121953"].acquisition_year == 1991
    assert cdi.KNOWN_CDI_RECORDS["121954"].acquisition_year == 1991


def test_known_cdi_records_all_report_owner_permission_required_access():
    for record in cdi.KNOWN_CDI_RECORDS.values():
        assert cdi.classify_access(record) == cdi.ACCESS_OWNER_PERMISSION_REQUIRED


def test_known_cdi_records_all_report_resolution_unknown_not_high_res():
    """MAR-006C's expected real-world result (Section 7): a real request path
    exists for all three current PL854 records, but none states a numeric
    resolution, so none may be reported as a confirmed high-resolution
    source. This is the exact regression this ticket exists to lock in."""

    for source_reference_id, record in cdi.KNOWN_CDI_RECORDS.items():
        access_class = cdi.classify_access(record)
        recovery = cdi.classify_recovery_potential(record, access_class)
        assert recovery == cdi.RECOVERY_RESOLUTION_UNKNOWN, (
            f"{source_reference_id} unexpectedly resolved to {recovery!r}"
        )


def test_cdi_report_url_matches_the_real_pattern():
    assert cdi.cdi_report_url(2607, "110153") == (
        "https://cdi-bathymetry.seadatanet.org/report/edmo/2607/110153"
    )
