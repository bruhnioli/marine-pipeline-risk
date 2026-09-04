"""Live smoke tests against the real BGS sediment ArcGIS REST services.

Excluded from the default `pytest` run (see `-m "not live"` in
pyproject.toml) so the normal suite never depends on any of these being
reachable. Run explicitly with:

    uv run pytest -m live
"""

import pytest
from shapely.geometry import box

from marine_engine.providers.sediment import bgs

pytestmark = pytest.mark.live

# A small bbox-derived polygon over part of the real PL854 AOI -- exact
# precision doesn't matter for a reachability/schema smoke test, and no row
# count here is asserted as a permanent constant (live data changes).
PL854_AOI_BBOX_WGS84 = (1.5764758, 53.3228828, 2.0771479, 53.3800000)
PL854_AOI_POLYGON_WGS84 = box(*PL854_AOI_BBOX_WGS84)

EXPECTED_PSA_FIELDS = (
    "PSA_DATA_ID",
    "ACTIVITY_ID",
    "SAMPLE_NAME",
    "EQUIPMENT_TYPE",
    "DEPTH_TOP",
    "DEPTH_BASE",
    "FOLK_CLASS",
    "GRAV",
    "SAND",
    "MUD",
    "GSM_UNITS",
    "PHI_UNITS",
    "EQUIPMENT_START_DATE",
)


def test_bgs_psa_service_is_reachable_and_has_expected_fields():
    features = bgs.fetch_psa_observations(PL854_AOI_POLYGON_WGS84)

    assert isinstance(features, list)
    if features:
        properties = features[0]["properties"]
        for field in EXPECTED_PSA_FIELDS:
            assert field in properties


def test_bgs_psa_pagination_with_small_page_size_matches_default():
    """A small page_size must page transparently to the same total as one big page."""

    paged = bgs.fetch_psa_observations(PL854_AOI_POLYGON_WGS84, page_size=5)
    single_page = bgs.fetch_psa_observations(PL854_AOI_POLYGON_WGS84, page_size=2000)

    assert len(paged) == len(single_page)


def test_bgs_seabed_sediments_250k_is_reachable():
    features = bgs.fetch_seabed_sediments_250k(PL854_AOI_POLYGON_WGS84)

    assert isinstance(features, list)
    if features:
        properties = features[0]["properties"]
        assert "BGS_ID" in properties
        assert "FOLK_S" in properties
        assert "FOLK_D50" in properties


def test_bgs_predictive_folk_polygons_is_reachable():
    features = bgs.fetch_predictive_folk_polygons(PL854_AOI_POLYGON_WGS84)

    assert isinstance(features, list)
    if features:
        properties = features[0]["properties"]
        assert "FOLK_S" in properties
        assert "FOLK_CLASS" in properties


def test_bgs_predictive_percentage_identify_is_reachable():
    lon, lat = PL854_AOI_POLYGON_WGS84.centroid.x, PL854_AOI_POLYGON_WGS84.centroid.y

    value = bgs.fetch_predictive_percentage_at_point(lon, lat, bgs.PREDICTIVE_MUD_LAYER_ID)

    # Either a real pixel value or None (no data at this point) -- never an
    # exception for reachability itself.
    assert value is None or isinstance(value, float)
