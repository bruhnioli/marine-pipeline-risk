"""BGS (British Geological Survey) seabed sediment/substrate provider (MAR-008).

Source provenance
------------------
Three official BGS ArcGIS REST `MapServer` layers, confirmed live and their
field schemas inspected directly (not assumed from documentation) on
2026-09-04:

- `SDDS/Offshore_Sample_Data/MapServer/7` -- "Offshore samples: particle
  size analysis" (Tier 1, primary observational evidence). A genuine point
  feature layer; `maxRecordCount=2000`, pagination supported.
- `SDDS/Test/MapServer/3` -- "Seabed Sediments 250k" (Tier 2, regional
  interpreted substrate). A genuine polygon feature layer, 1:250,000 scale.
- `SDDS/Test/MapServer/7` -- "Predictive Seabed Sediments" (Tier 3,
  secondary model comparison only). A genuine polygon feature layer for the
  classified Folk class; the percentage sand/gravel/mud values live on
  SEPARATE raster layers of the same service (9=mud, 10=gravel, 11=sand)
  that do NOT support attribute queries at all (`/query` returns a hard
  `400`) -- the only confirmed working mechanism for those is the
  `MapServer/identify` operation, sampled one point at a time.

All three feature layers are queried spatially against the real AOI
polygon (as Esri JSON rings, `spatialRel=esriSpatialRelIntersects`), never
by title/location text matching, with `resultOffset`/`resultRecordCount`
pagination. The service's own `exceededTransferLimit` flag was observed set
even for tiny result sets in live testing, so it is not trusted alone as a
signal that more pages exist; a page is instead treated as the last one
once it returns fewer features than requested.
"""

import json
from typing import Any
from urllib.parse import urlencode

from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from marine_engine.providers.bathymetry._http import get_with_retries, post_with_retries

PSA_SERVICE_URL = "https://map.bgs.ac.uk/arcgis/rest/services/SDDS/Offshore_Sample_Data/MapServer/7"
SEABED_SEDIMENTS_250K_SERVICE_URL = (
    "https://map.bgs.ac.uk/arcgis/rest/services/SDDS/Test/MapServer/3"
)
PREDICTIVE_FOLK_SERVICE_URL = "https://map.bgs.ac.uk/arcgis/rest/services/SDDS/Test/MapServer/7"
PREDICTIVE_SERVICE_ROOT_URL = "https://map.bgs.ac.uk/arcgis/rest/services/SDDS/Test/MapServer"

PREDICTIVE_MUD_LAYER_ID = 9
PREDICTIVE_GRAVEL_LAYER_ID = 10
PREDICTIVE_SAND_LAYER_ID = 11

DEFAULT_PAGE_SIZE = 1000  # comfortably under the service's maxRecordCount=2000

PSA_DATASET_TITLE = "BGS Offshore samples: particle size analysis"
SEABED_SEDIMENTS_250K_DATASET_TITLE = "BGS Seabed Sediments 250k"
PREDICTIVE_DATASET_TITLE = "BGS Predictive Seabed Sediments UK"


class BgsSedimentUnavailableError(RuntimeError):
    """A BGS sediment ArcGIS REST request failed or returned an in-body error."""


def _esri_polygon_geometry_json(polygon_wgs84: BaseGeometry) -> dict[str, Any]:
    """Encode a Shapely (Multi)Polygon as Esri JSON `rings` (WGS84).

    Esri JSON requires exterior rings clockwise and interior rings
    (holes) counter-clockwise -- the opposite winding to GeoJSON/Shapely's
    own convention -- hence the `orient(..., sign=-1.0)`.
    """

    polygons = (
        list(polygon_wgs84.geoms) if polygon_wgs84.geom_type == "MultiPolygon" else [polygon_wgs84]
    )
    rings: list[list[list[float]]] = []
    for polygon in polygons:
        oriented = orient(polygon, sign=-1.0)
        rings.append([[float(x), float(y)] for x, y in oriented.exterior.coords])
        for interior in oriented.interiors:
            rings.append([[float(x), float(y)] for x, y in interior.coords])

    return {"rings": rings, "spatialReference": {"wkid": 4326}}


def _query_layer_paginated(
    layer_url: str, aoi_polygon_wgs84: BaseGeometry, *, page_size: int = DEFAULT_PAGE_SIZE
) -> list[dict[str, Any]]:
    """Fetch every GeoJSON feature intersecting `aoi_polygon_wgs84`, paginated.

    Stops once a page returns fewer features than requested -- the live
    service's `exceededTransferLimit` flag is not a reliable "more pages
    exist" signal (observed `true` even for a 5-of-5 result), so relying on
    it alone risks either an infinite loop or under-fetching.

    Uses POST (form-encoded body), never GET: the real PL854 AOI polygon has
    over 1500 vertices, and encoding that many coordinates as a GET query
    string produces a URL long enough that the live service resets the
    connection outright (confirmed against the real AOI, not a hypothetical)
    -- POST is the standard ArcGIS REST fix for a geometry payload this size.
    """

    geometry_json = _esri_polygon_geometry_json(aoi_polygon_wgs84)
    features: list[dict[str, Any]] = []
    offset = 0

    while True:
        params = {
            "geometry": json.dumps(geometry_json),
            "geometryType": "esriGeometryPolygon",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": page_size,
            "resultOffset": offset,
        }
        try:
            response = post_with_retries(
                f"{layer_url}/query",
                urlencode(params).encode("utf-8"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except Exception as exc:  # noqa: BLE001 -- one clear source-specific failure
            raise BgsSedimentUnavailableError(
                f"BGS ArcGIS REST query failed for {layer_url}: {exc}"
            ) from exc

        payload = response.json()
        if "error" in payload:
            raise BgsSedimentUnavailableError(
                f"BGS ArcGIS REST error querying {layer_url}: {payload['error']}"
            )

        page_features = payload.get("features", [])
        features.extend(page_features)
        if len(page_features) < page_size:
            break
        offset += page_size

    return features


def fetch_psa_observations(
    aoi_polygon_wgs84: BaseGeometry, *, page_size: int = DEFAULT_PAGE_SIZE
) -> list[dict[str, Any]]:
    """All PSA (particle size analysis) point records intersecting the AOI, paginated."""

    return _query_layer_paginated(PSA_SERVICE_URL, aoi_polygon_wgs84, page_size=page_size)


def fetch_seabed_sediments_250k(
    aoi_polygon_wgs84: BaseGeometry, *, page_size: int = DEFAULT_PAGE_SIZE
) -> list[dict[str, Any]]:
    """All Seabed Sediments 250k polygon records intersecting the AOI, paginated."""

    return _query_layer_paginated(
        SEABED_SEDIMENTS_250K_SERVICE_URL, aoi_polygon_wgs84, page_size=page_size
    )


def fetch_predictive_folk_polygons(
    aoi_polygon_wgs84: BaseGeometry, *, page_size: int = DEFAULT_PAGE_SIZE
) -> list[dict[str, Any]]:
    """All Predictive Seabed Sediments Folk-class grid-cell polygons intersecting the AOI."""

    return _query_layer_paginated(
        PREDICTIVE_FOLK_SERVICE_URL, aoi_polygon_wgs84, page_size=page_size
    )


def fetch_predictive_percentage_at_point(lon: float, lat: float, layer_id: int) -> float | None:
    """Sample a predictive sand/gravel/mud percentage raster at one point.

    Layers 9 (mud), 10 (gravel), 11 (sand) of the Predictive Seabed
    Sediments service are raster layers with no attribute-query support at
    all (`/query` returns a hard `400`, confirmed live) -- `identify` at a
    single point is the only confirmed-working mechanism, hence this is a
    per-point operation with no bulk/vectorized equivalent. Returns `None`
    (never a fabricated value) if the layer has no data at this point.
    """

    # `mapExtent`/`imageDisplay` are required by the `identify` operation's
    # API contract but are irrelevant to a single-point pixel lookup beyond
    # satisfying it -- an arbitrary small window centred on the point.
    half_extent_deg = 0.01
    map_extent = (
        f"{lon - half_extent_deg},{lat - half_extent_deg},"
        f"{lon + half_extent_deg},{lat + half_extent_deg}"
    )
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": f"all:{layer_id}",
        "tolerance": "1",
        "mapExtent": map_extent,
        "imageDisplay": "400,400,96",
        "returnGeometry": "false",
        "f": "json",
    }
    try:
        response = get_with_retries(f"{PREDICTIVE_SERVICE_ROOT_URL}/identify", params=params)
    except Exception as exc:  # noqa: BLE001 -- one clear source-specific failure
        raise BgsSedimentUnavailableError(f"BGS predictive-layer identify failed: {exc}") from exc

    payload = response.json()
    if "error" in payload:
        raise BgsSedimentUnavailableError(
            f"BGS predictive-layer identify error: {payload['error']}"
        )

    results = payload.get("results", [])
    if not results:
        return None
    value = results[0].get("attributes", {}).get("Stretch.Pixel Value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
