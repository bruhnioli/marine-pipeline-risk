"""Live smoke tests against the real UKHO/MEDIN, BGS, and EMODnet services.

Excluded from the default `pytest` run (see `-m "not live"` in
pyproject.toml) so the normal suite never depends on any of these being
reachable. Run explicitly with:

    uv run pytest -m live
"""

import pytest

from marine_engine.providers.bathymetry import bgs, cdi, emodnet, ukho

pytestmark = pytest.mark.live

PL854_AOI_BBOX_WGS84 = (1.5764758, 53.3228828, 2.0771479, 53.4341688)


def test_ukho_medin_query_is_reachable():
    records, status = ukho.discover_ukho_surveys(PL854_AOI_BBOX_WGS84)

    assert status.query_succeeded is True
    assert isinstance(records, list)


def test_bgs_gb02ss0001_is_reachable():
    record = bgs.discover_gb02ss0001()

    assert record.source_dataset_id == "GB02SS0001"
    # Either verified live, or a clearly-marked stub -- never silent failure.
    assert record.acquisition_status is not None


def test_emodnet_wcs_is_reachable(tmp_path):
    result = emodnet.fetch_emodnet_geotiff(PL854_AOI_BBOX_WGS84, tmp_path / "smoke.tif")

    assert result.local_path.exists()
    assert result.width_px > 0
    assert result.height_px > 0


def test_emodnet_source_references_wfs_is_reachable():
    features = emodnet.fetch_source_references(PL854_AOI_BBOX_WGS84)

    assert isinstance(features, list)
    assert len(features) > 0
    assert all(f.release == emodnet.TARGET_RELEASE for f in features)


def test_emodnet_quality_index_wfs_is_reachable():
    features = emodnet.fetch_quality_index(PL854_AOI_BBOX_WGS84)

    assert isinstance(features, list)
    assert len(features) > 0
    assert all(f.release == emodnet.TARGET_RELEASE for f in features)


def test_emodnet_msl_availability_check_is_reachable():
    result = emodnet.check_msl_availability(PL854_AOI_BBOX_WGS84)

    # Either a real acquisition mechanism was found, or a clear reason it
    # wasn't -- never a silent gap.
    assert result.notes


def test_emodnet_native_qa_layer_discovery_is_reachable():
    result = emodnet.check_native_qa_layers(PL854_AOI_BBOX_WGS84)

    assert len(result.wcs_coverage_ids) > 0
    assert "emodnet__mean" in result.wcs_coverage_ids
    assert result.notes


def test_cdi_report_host_bot_challenge_still_blocks_plain_http(tmp_path):
    """Documents current real-world behaviour, not a hard requirement.

    As of MAR-006B, the CDI report host fronts every request with a
    client-side proof-of-work challenge that a plain HTTP client cannot
    solve (see cdi.py's module docstring) -- `resolve_cdi_record` falls
    back to a manually-verified snapshot for PL854's three known ids. This
    test just confirms that finding is still accurate; if the host ever
    stops challenging plain clients, `fetch_cdi_report_html` should then
    return real HTML instead of raising, which this test also accepts.
    """

    url = cdi.cdi_report_url(2607, "110153")
    try:
        html = cdi.fetch_cdi_report_html(url)
    except cdi.CdiUnavailableError:
        return  # still challenged -- the documented, expected state
    assert "Haddock Bank" in html  # host stopped challenging; real content came through
