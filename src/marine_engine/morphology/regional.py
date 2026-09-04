"""Regional (kilometre-scale) seabed morphology context from the EMODnet baseline (MAR-007).

Scope and interpretation (mandatory reading before touching this module)
--------------------------------------------------------------------------
The EMODnet 2024 DTM covering PL854 is a ~115 m-class composite built from
CDI surveys resolved in MAR-006B/C as 1991/1992 -- i.e. every terrain
feature computed here describes "regional morphology represented by the
legacy bathymetric sources incorporated in EMODnet DTM 2024", never
"current seabed morphology". These broad, kilometre-scale features are
appropriate for regional gradient / bank-flank-channel context; they are
NOT appropriate for sand-wave crests/troughs, local pipeline scour, metre-
scale roughness, embedment, free-span, or any present-day fine seabed
geometry claim. No such claim is made anywhere in this module.

Sign convention (Section 3)
----------------------------
Canonical bathymetry is `depth_lat_m`, positive downward. For terrain
mathematics ONLY, this module works in `seabed_elevation_lat_m = -depth_lat_m`
so that positive TPI/relief means locally higher/shallower, negative means
locally lower/deeper -- ordinary elevation semantics. This is a sign
representation for internal maths only: it never modifies the canonical
MAR-006 DTM file and introduces no new vertical datum.

Analysis scales (Section 4)
-----------------------------
All neighbourhoods are physical map-unit radii (500/1000/2000 m), rounded
to the nearest whole 100 m-grid cell count for the actual computation.
These are broad, kilometre-scale windows -- never presented as resolving
metre-scale terrain.

Halo (Section 5)
------------------
The canonical MAR-006 AOI is polygon-clipped, so evaluating a 1-2 km moving
window directly against it would be biased at the clipped edge. This
module instead acquires a wider "halo" EMODnet raster (AOI buffered by
`HALO_BUFFER_M`), computes every morphology feature on that unclipped halo
grid, and only clips the FINAL results back to the canonical AOI. The halo
raster is an analysis-support artifact, not a replacement for the MAR-006
canonical DTM.
"""

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

from marine_engine.preprocessing import bathymetry
from marine_engine.providers.bathymetry.emodnet import NativeQaLayerAvailability

# --- Scales, thresholds, sign convention ------------------------------------

ANALYSIS_RADII_M: tuple[float, ...] = (500.0, 1000.0, 2000.0)
SLOPE_RADII_M: tuple[float, ...] = (500.0, 1000.0)
TPI_RADII_M: tuple[float, ...] = (1000.0, 2000.0)
RELIEF_RADII_M: tuple[float, ...] = (1000.0, 2000.0)
TERRAIN_STD_RADII_M: tuple[float, ...] = (1000.0, 2000.0)

MAX_ANALYSIS_RADIUS_M = max(ANALYSIS_RADII_M)
NATIVE_SOURCE_CELL_GUARD_M = 200.0  # >= 1 native ~115 m source cell, rounded up generously
HALO_BUFFER_M = 2200.0  # >= MAX_ANALYSIS_RADIUS_M + NATIVE_SOURCE_CELL_GUARD_M, per Section 5

MIN_VALID_NEIGHBORHOOD_FRACTION = (
    0.90  # Section 10: a data-support safeguard, not a physical threshold
)

DEFAULT_TARGET_RESOLUTION_M = bathymetry.DEFAULT_TARGET_RESOLUTION_M  # 100.0, reused from MAR-006
SOFTWARE_VERSION = "marine-engine 0.1.0 (MAR-007)"
PRODUCT_NAME = "EMODnet Digital Bathymetry (DTM 2024)"

SCIENTIFIC_LIMITATIONS: tuple[str, ...] = (
    "These features describe regional morphology represented by the legacy bathymetric "
    "sources incorporated in EMODnet DTM 2024 -- not current/present-day seabed morphology.",
    "The underlying PL854 source surveys were acquired in 1991-1992, ~32-33 years before "
    "the 2024 DTM release; 2024 is a product release year, never an acquisition year.",
    "EMODnet 2024 is ~115 m-class regional bathymetry projected onto a 100 m analysis grid; "
    "the grid spacing is an analysis choice, not measured resolution.",
    "Appropriate for broad bathymetric gradient, bank/flank/channel-scale morphology, and "
    "kilometre-scale relative topographic position and relief only.",
    "NOT appropriate for current sand-wave crests/troughs, megaripples, local pipeline "
    "scour, metre-scale roughness, embedment, free-span, or any present-day fine seabed "
    "geometry claim -- none of that is derived or implied here.",
    "No curvature, rugosity, TRI, or aspect is computed in this ticket (see module docs).",
    "Terrain standard deviation (if present) is broad terrain variability, never measurement "
    "uncertainty, vertical accuracy, or rugosity.",
    "No morphology/uncertainty/stability/hazard risk score is computed anywhere here.",
)


class RegionalMorphologyError(RuntimeError):
    """A morphology-scale precondition could not be met (e.g. no robust plane-fit method)."""


# --- Circular footprints and windowed moment sums (Section 6-9) -------------


def _radius_px(radius_m: float, cell_size_m: float) -> int:
    return max(1, round(radius_m / cell_size_m))


def _circular_footprint(radius_px: int) -> np.ndarray:
    coords = np.arange(-radius_px, radius_px + 1)
    xx, yy = np.meshgrid(coords, coords)
    return (xx**2 + yy**2) <= radius_px**2


def _offset_grids_m(radius_px: int, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(-radius_px, radius_px + 1) * cell_size_m
    dx, dy = np.meshgrid(coords, coords)
    return dx, dy


def _correlate(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Sum of `array[p + offset] * kernel[offset]` over the kernel's own offsets.

    Using `ndimage.correlate` (not `convolve`) deliberately: the kernel is
    indexed by *offset from the output pixel*, exactly matching how
    `_offset_grids_m`/`_circular_footprint` are built, with no 180-degree
    kernel flip to reason about.
    """

    return ndimage.correlate(array, kernel, mode="constant", cval=0.0)


@dataclass(frozen=True)
class NeighborhoodMoments:
    """Windowed sums needed by TPI / local relief / terrain-std / slope, for one radius."""

    n: np.ndarray  # count of valid cells in the neighborhood
    z: np.ndarray  # sum of z (elevation) over valid cells
    z2: np.ndarray  # sum of z^2 over valid cells
    valid_fraction: np.ndarray  # n / total footprint cell count
    footprint_cell_count: int
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    xx: np.ndarray | None = None
    yy: np.ndarray | None = None
    xy: np.ndarray | None = None
    xz: np.ndarray | None = None
    yz: np.ndarray | None = None


def compute_neighborhood_moments(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    radius_m: float,
    cell_size_m: float,
    *,
    need_plane: bool = False,
) -> NeighborhoodMoments:
    radius_px = _radius_px(radius_m, cell_size_m)
    footprint = _circular_footprint(radius_px).astype(float)
    footprint_cell_count = int(footprint.sum())

    valid_f = valid_mask.astype(float)
    z_filled = np.where(valid_mask, elevation, 0.0)

    n = _correlate(valid_f, footprint)
    z = _correlate(z_filled, footprint)
    z2 = _correlate(z_filled**2, footprint)
    valid_fraction = n / footprint_cell_count

    kwargs: dict[str, np.ndarray] = {}
    if need_plane:
        dx, dy = _offset_grids_m(radius_px, cell_size_m)
        kwargs = {
            "x": _correlate(valid_f, dx * footprint),
            "y": _correlate(valid_f, dy * footprint),
            "xx": _correlate(valid_f, (dx**2) * footprint),
            "yy": _correlate(valid_f, (dy**2) * footprint),
            "xy": _correlate(valid_f, (dx * dy) * footprint),
            "xz": _correlate(z_filled, dx * footprint),
            "yz": _correlate(z_filled, dy * footprint),
        }

    return NeighborhoodMoments(
        n=n,
        z=z,
        z2=z2,
        valid_fraction=valid_fraction,
        footprint_cell_count=footprint_cell_count,
        **kwargs,
    )


def _apply_validity_threshold(
    value: np.ndarray,
    valid_fraction: np.ndarray,
    threshold: float = MIN_VALID_NEIGHBORHOOD_FRACTION,
) -> np.ndarray:
    result = value.copy()
    result[valid_fraction < threshold] = np.nan
    return result


# --- Section 6: broad slope via local planar fit ----------------------------


def compute_slope_deg(
    elevation: np.ndarray, valid_mask: np.ndarray, radius_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit z = ax + by + c inside a circular neighborhood; slope = atan(sqrt(a^2+b^2)).

    Returns (slope_deg, valid_fraction) at every pixel of the input grid,
    with `slope_deg` already NaN'd wherever the neighborhood-validity
    threshold isn't met. Solved via the moment-sum normal equations (see
    `compute_neighborhood_moments`) rather than a raw 3x3 Horn finite
    difference, matching the ticket's explicit preference for a broad
    local-plane fit over a fine-scale one-cell method.
    """

    moments = compute_neighborhood_moments(
        elevation, valid_mask, radius_m, cell_size_m, need_plane=True
    )
    shape = elevation.shape

    a_matrix = np.stack(
        [
            np.stack([moments.xx, moments.xy, moments.x], axis=-1),
            np.stack([moments.xy, moments.yy, moments.y], axis=-1),
            np.stack([moments.x, moments.y, moments.n], axis=-1),
        ],
        axis=-2,
    )
    b_vector = np.stack([moments.xz, moments.yz, moments.z], axis=-1)

    solvable = (moments.n >= 3) & (moments.valid_fraction >= MIN_VALID_NEIGHBORHOOD_FRACTION)
    slope_deg = np.full(shape, np.nan)

    if np.any(solvable):
        det = np.linalg.det(a_matrix[solvable])
        nonsingular = np.abs(det) > 1e-6
        idx = np.flatnonzero(solvable.ravel())
        solvable_flat_mask = np.zeros(shape, dtype=bool).ravel()
        solvable_flat_mask[idx[nonsingular]] = True
        solvable_final = solvable_flat_mask.reshape(shape)

        if np.any(solvable_final):
            # `b` needs an explicit trailing right-hand-side dimension for a
            # batched solve (shape (N,3,1), not (N,3)) -- squeezed back off below.
            solutions = np.linalg.solve(
                a_matrix[solvable_final], b_vector[solvable_final][..., np.newaxis]
            )[..., 0]
            a_coef = solutions[:, 0]
            b_coef = solutions[:, 1]
            slope_rad = np.arctan(np.sqrt(a_coef**2 + b_coef**2))
            slope_deg[solvable_final] = np.degrees(slope_rad)

    return slope_deg, moments.valid_fraction


# --- Section 7: TPI (elevation semantics) -----------------------------------


def compute_tpi_m(
    elevation: np.ndarray, valid_mask: np.ndarray, radius_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """TPI = central elevation - mean neighborhood elevation.

    Positive => locally higher/shallower than the surrounding kilometre-
    scale terrain; negative => locally lower/deeper; near zero =>
    approximately flat/constant-slope context at that scale (Section 7).
    """

    moments = compute_neighborhood_moments(elevation, valid_mask, radius_m, cell_size_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_elevation = moments.z / moments.n
    tpi = elevation - mean_elevation
    tpi = np.where(valid_mask, tpi, np.nan)
    tpi = _apply_validity_threshold(tpi, moments.valid_fraction)
    return tpi, moments.valid_fraction


# --- Section 8: broad local relief ------------------------------------------


def compute_local_relief_m(
    elevation: np.ndarray, valid_mask: np.ndarray, radius_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """local_relief = max(elevation) - min(elevation) inside the neighborhood. Always >= 0."""

    radius_px = _radius_px(radius_m, cell_size_m)
    footprint = _circular_footprint(radius_px)

    filled_for_max = np.where(valid_mask, elevation, -np.inf)
    filled_for_min = np.where(valid_mask, elevation, np.inf)
    local_max = ndimage.maximum_filter(
        filled_for_max, footprint=footprint, mode="constant", cval=-np.inf
    )
    local_min = ndimage.minimum_filter(
        filled_for_min, footprint=footprint, mode="constant", cval=np.inf
    )

    relief = local_max - local_min
    relief[~np.isfinite(relief)] = np.nan

    moments = compute_neighborhood_moments(elevation, valid_mask, radius_m, cell_size_m)
    relief = _apply_validity_threshold(relief, moments.valid_fraction)
    return relief, moments.valid_fraction


# --- Section 9: optional broad terrain standard deviation -------------------


def compute_terrain_std_m(
    elevation: np.ndarray, valid_mask: np.ndarray, radius_m: float, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation of elevation within the neighborhood.

    Broad terrain variability ONLY -- never measurement uncertainty,
    vertical accuracy, or rugosity (Section 9).
    """

    moments = compute_neighborhood_moments(elevation, valid_mask, radius_m, cell_size_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = moments.z / moments.n
        variance = moments.z2 / moments.n - mean**2
    variance = np.clip(variance, 0.0, None)
    std = np.sqrt(variance)
    std = _apply_validity_threshold(std, moments.valid_fraction)
    return std, moments.valid_fraction


# --- Halo acquisition and preparation (Section 5) ---------------------------


def build_halo_bbox_wgs84(
    aoi_geometry_working: BaseGeometry, working_crs: str, halo_m: float = HALO_BUFFER_M
) -> tuple[float, float, float, float]:
    """The AOI buffered by `halo_m`, as a WGS84 bbox for the EMODnet WCS halo request."""

    buffered = aoi_geometry_working.buffer(halo_m)
    buffered_wgs84 = gpd.GeoSeries([buffered], crs=working_crs).to_crs("EPSG:4326").iloc[0]
    return tuple(float(v) for v in buffered_wgs84.bounds)


@dataclass(frozen=True)
class HaloElevationGrid:
    """The unclipped halo elevation grid ready for windowed morphology maths."""

    elevation: np.ndarray  # seabed_elevation_lat_m = -depth_lat_m (Section 3)
    valid_mask: np.ndarray
    transform: rasterio.Affine
    crs: str
    raw_stats: bathymetry.RawRasterStats
    sign_convention_observed: str
    source_sha256: str | None


def prepare_halo_elevation_grid(
    *,
    raw_halo_path: Path,
    raw_halo_manifest_entry: dict[str, Any] | None,
    working_crs: str,
    target_resolution_m: float = DEFAULT_TARGET_RESOLUTION_M,
    reference_depth_range_m: tuple[float, float] = bathymetry.DEFAULT_REFERENCE_DEPTH_RANGE_M,
) -> HaloElevationGrid:
    """Inspect -> verify sign -> convert -> reproject the halo raster, reusing MAR-006 utilities.

    Deliberately does NOT clip to the AOI here -- that would defeat the
    entire purpose of a halo. Only the FINAL morphology outputs are clipped
    (`build_all_morphology_layers`, via `bathymetry.mask_to_aoi_polygon`).
    """

    raw_stats = bathymetry.inspect_raw_raster(raw_halo_path)
    sign_convention = bathymetry.determine_sign_convention(raw_stats, reference_depth_range_m)

    with rasterio.open(raw_halo_path) as src:
        raw_band = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_bounds = tuple(src.bounds)

    canonical_depth = bathymetry.to_canonical_depth(raw_band, sign_convention)
    depth_working, transform = bathymetry.reproject_and_resample(
        canonical_depth, src_transform, src_crs, working_crs, src_bounds, target_resolution_m
    )
    elevation = -depth_working  # Section 3: seabed_elevation_lat_m
    valid_mask = ~np.isnan(elevation)

    return HaloElevationGrid(
        elevation=elevation,
        valid_mask=valid_mask,
        transform=transform,
        crs=working_crs,
        raw_stats=raw_stats,
        sign_convention_observed=sign_convention,
        source_sha256=(raw_halo_manifest_entry or {}).get("sha256"),
    )


# --- Assembling every morphology raster layer (Sections 6-9, clipped) -------


@dataclass(frozen=True)
class MorphologyLayer:
    name: str
    array: np.ndarray  # already clipped to the canonical AOI
    transform: rasterio.Affine
    radius_m: float
    unit: str
    description: str


_LAYER_SPECS: tuple[tuple[str, str, tuple[float, ...], str], ...] = (
    (
        "slope",
        "deg",
        SLOPE_RADII_M,
        "Broad local-plane-fit slope magnitude in degrees. NOT local pipe inclination or "
        "an individual sand-wave face slope.",
    ),
    (
        "tpi",
        "m",
        TPI_RADII_M,
        "Topographic Position Index (elevation semantics): positive=local high, "
        "negative=local depression, near zero=flat/constant-slope context.",
    ),
    (
        "local_relief",
        "m",
        RELIEF_RADII_M,
        "Broad local relief amplitude (max-min elevation) inside the neighborhood. "
        "NOT a sand-wave-height measurement.",
    ),
    (
        "terrain_std",
        "m",
        TERRAIN_STD_RADII_M,
        "Broad terrain variability (std of elevation) inside the neighborhood. NOT "
        "measurement uncertainty, vertical accuracy, or rugosity.",
    ),
)

_COMPUTE_FUNCS = {
    "slope": compute_slope_deg,
    "tpi": compute_tpi_m,
    "local_relief": compute_local_relief_m,
    "terrain_std": compute_terrain_std_m,
}


def build_all_morphology_layers(
    halo_grid: HaloElevationGrid, aoi_geometry_working: BaseGeometry
) -> list[MorphologyLayer]:
    """Compute every required/optional morphology layer on the halo, then clip each to the AOI."""

    cell_size_m = abs(halo_grid.transform.a)
    layers: list[MorphologyLayer] = []

    for feature_key, unit, radii, description in _LAYER_SPECS:
        compute = _COMPUTE_FUNCS[feature_key]
        for radius_m in radii:
            array, _valid_fraction = compute(
                halo_grid.elevation, halo_grid.valid_mask, radius_m, cell_size_m
            )
            clipped, clipped_transform = bathymetry.mask_to_aoi_polygon(
                array, halo_grid.transform, halo_grid.crs, aoi_geometry_working
            )
            layers.append(
                MorphologyLayer(
                    name=f"{feature_key}_{int(radius_m)}m_{unit}",
                    array=clipped,
                    transform=clipped_transform,
                    radius_m=radius_m,
                    unit=unit,
                    description=description,
                )
            )

    return layers


# --- Chainage sampling and provenance join (Sections 15-16) -----------------

CHAINAGE_MORPHOLOGY_COLUMNS = (
    "pipeline_id",
    "station_index",
    "chainage_m",
    "kp_label",
    "depth_lat_m",
    "slope_500m_deg",
    "slope_1000m_deg",
    "tpi_1000m_m",
    "tpi_2000m_m",
    "local_relief_1000m_m",
    "local_relief_2000m_m",
    "terrain_std_1000m_m",
    "terrain_std_2000m_m",
    "cell_shallowest_depth_lat_m",
    "cell_deepest_depth_lat_m",
    "cell_depth_range_m",
    "cell_depth_std_m",
    "cell_n_values",
    "cell_interpolation_flag",
    "mean_smoothed_depth_lat_m",
    "mean_smoothed_offset_pct",
    "source_reference_id",
    "source_acquisition_year",
    "source_acquisition_start",
    "source_acquisition_end",
    "source_age_at_2024_release_years",
    "qi_age",
    "qi_horizontal",
    "qi_vertical",
    "qi_purpose",
    "qi_combined",
)


def _sample_array_at_points(
    array: np.ndarray, transform: rasterio.Affine, points: list[tuple[float, float]]
) -> list[float | None]:
    height, width = array.shape
    values: list[float | None] = []
    for x, y in points:
        row, col = rasterio.transform.rowcol(transform, x, y)
        if 0 <= row < height and 0 <= col < width:
            value = array[row, col]
            values.append(float(value) if not np.isnan(value) else None)
        else:
            values.append(None)
    return values


def sample_morphology_layers_at_chainage(
    layers: list[MorphologyLayer], chainage_gdf_working: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Sample every clipped morphology raster at every chainage station, in gdf row order."""

    points = [(pt.x, pt.y) for pt in chainage_gdf_working.geometry]
    columns = {
        layer.name: _sample_array_at_points(layer.array, layer.transform, points)
        for layer in layers
    }
    return pd.DataFrame(columns)


def join_source_provenance(
    chainage_bathymetry_df: pd.DataFrame, cdi_sources_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach MAR-006B/C acquisition-epoch provenance to each chainage station.

    QI_* fields and `source_reference_id` already live in
    `chainage_bathymetry_df` per station (MAR-006); only the CDI-resolved
    acquisition-epoch fields are joined in here from
    `emodnet_cdi_sources.parquet`, keyed by `source_reference_id`. Never
    reduces any of this to one confidence score (Section 15). Every one of
    the 941 stations is retained regardless of match.
    """

    cdi_columns = cdi_sources_df.set_index("source_reference_id")[
        [
            "acquisition_year",
            "acquisition_start",
            "acquisition_end",
            "survey_age_at_product_release_year",
        ]
    ]
    joined = chainage_bathymetry_df.merge(
        cdi_columns, left_on="source_reference_id", right_index=True, how="left"
    )
    return joined.rename(
        columns={
            "acquisition_year": "source_acquisition_year",
            "acquisition_start": "source_acquisition_start",
            "acquisition_end": "source_acquisition_end",
            "survey_age_at_product_release_year": "source_age_at_2024_release_years",
        }
    )


@dataclass(frozen=True)
class RegionalMorphologyResult:
    chainage_df: pd.DataFrame
    layers: list[MorphologyLayer]
    halo_grid: HaloElevationGrid
    qa_layer_availability: NativeQaLayerAvailability
    aoi_identifier: str
    working_crs: str
    processing_timestamp: str


def build_regional_morphology(
    *,
    aoi_gdf: gpd.GeoDataFrame,
    chainage_gdf: gpd.GeoDataFrame,
    chainage_bathymetry_df: pd.DataFrame,
    cdi_sources_df: pd.DataFrame,
    raw_halo_path: Path,
    raw_halo_manifest_entry: dict[str, Any] | None,
    qa_layer_availability: NativeQaLayerAvailability,
    working_crs: str,
    aoi_identifier: str,
) -> RegionalMorphologyResult:
    """End-to-end MAR-007: halo elevation -> broad morphology -> AOI clip -> chainage join."""

    aoi_geometry_working = (
        gpd.GeoSeries(aoi_gdf.geometry, crs=aoi_gdf.crs).to_crs(working_crs).union_all()
    )

    halo_grid = prepare_halo_elevation_grid(
        raw_halo_path=raw_halo_path,
        raw_halo_manifest_entry=raw_halo_manifest_entry,
        working_crs=working_crs,
    )
    layers = build_all_morphology_layers(halo_grid, aoi_geometry_working)

    chainage_working = (
        chainage_gdf.to_crs(working_crs).sort_values("station_index").reset_index(drop=True)
    )
    chainage_bathymetry_sorted = chainage_bathymetry_df.sort_values("station_index").reset_index(
        drop=True
    )
    if len(chainage_working) != len(chainage_bathymetry_sorted):
        raise RegionalMorphologyError(
            f"chainage_gdf has {len(chainage_working)} stations but chainage_bathymetry "
            f"has {len(chainage_bathymetry_sorted)}; refusing to align mismatched station sets."
        )

    morphology_samples = sample_morphology_layers_at_chainage(layers, chainage_working)
    provenance = join_source_provenance(chainage_bathymetry_sorted, cdi_sources_df)

    # Native per-cell QA (Sections 11-14): confirmed unavailable as a live/
    # small machine-readable coverage for this release (see
    # `emodnet.check_native_qa_layers`) -- explicit null columns, not fabricated.
    station_count = len(chainage_working)
    qa_columns = pd.DataFrame(
        {
            "cell_shallowest_depth_lat_m": [None] * station_count,
            "cell_deepest_depth_lat_m": [None] * station_count,
            "cell_depth_range_m": [None] * station_count,
            "cell_depth_std_m": [None] * station_count,
            "cell_n_values": [None] * station_count,
            "cell_interpolation_flag": [None] * station_count,
            "mean_smoothed_depth_lat_m": [None] * station_count,
            "mean_smoothed_offset_pct": [None] * station_count,
        }
    )

    chainage_df = pd.concat(
        [
            provenance.reset_index(drop=True),
            morphology_samples.reset_index(drop=True),
            qa_columns.reset_index(drop=True),
        ],
        axis=1,
    )
    chainage_df = chainage_df[list(CHAINAGE_MORPHOLOGY_COLUMNS)]

    return RegionalMorphologyResult(
        chainage_df=chainage_df,
        layers=layers,
        halo_grid=halo_grid,
        qa_layer_availability=qa_layer_availability,
        aoi_identifier=aoi_identifier,
        working_crs=working_crs,
        processing_timestamp=datetime.now(UTC).isoformat(),
    )


# --- Output writing (Sections 16-18) ----------------------------------------


def write_morphology_raster(
    layer: MorphologyLayer, crs: str, output_path: Path, *, aoi_identifier: str
) -> Path:
    tags = {
        "product": PRODUCT_NAME,
        "product_release_year": "2024",
        "vertical_datum": "LAT",
        "morphology_sign_convention": (
            "seabed_elevation_lat_m = -depth_lat_m; positive value = locally higher/"
            "shallower terrain, negative value = locally lower/deeper terrain"
        ),
        "source_nominal_resolution_m": "115",
        "analysis_grid_spacing_m": str(DEFAULT_TARGET_RESOLUTION_M),
        "analysis_radius_m": str(layer.radius_m),
        "feature": layer.name,
        "unit": layer.unit,
        "description": layer.description,
        "source_age_warning": (
            "Underlying PL854 source surveys acquired 1991-1992 (~32-33 years before "
            "the 2024 EMODnet DTM release); this is legacy regional morphology "
            "represented by those sources, not current seabed morphology."
        ),
        "limitation": (
            "NOT appropriate for current sand-wave crests/troughs, local pipeline scour, "
            "metre-scale roughness, embedment, free-span, or present-day fine seabed geometry."
        ),
        "aoi_identifier": aoi_identifier,
        "software_version": SOFTWARE_VERSION,
    }
    bathymetry.write_canonical_raster(layer.array, layer.transform, crs, output_path, tags)
    return output_path


def write_chainage_regional_morphology(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def write_morphology_metadata(
    result: RegionalMorphologyResult,
    *,
    input_dtm_path: Path,
    raster_paths: dict[str, Path],
    output_path: Path,
) -> Path:
    metadata = {
        "input_canonical_dtm_path": str(input_dtm_path),
        "source_product": PRODUCT_NAME,
        "product_release_year": 2024,
        "source_nominal_resolution_m": 115.0,
        "analysis_grid_spacing_m": DEFAULT_TARGET_RESOLUTION_M,
        "underlying_acquisition_years": sorted(
            {
                int(y)
                for y in result.chainage_df["source_acquisition_year"].dropna().unique().tolist()
            }
        ),
        "maximum_analysis_radius_m": MAX_ANALYSIS_RADIUS_M,
        "halo_buffer_m": HALO_BUFFER_M,
        "slope_method": (
            "local planar fit z=ax+by+c over a circular neighborhood; slope=atan(sqrt(a^2+b^2))"
        ),
        "tpi_definition": (
            "central seabed elevation minus mean seabed elevation of the circular "
            "neighborhood (elevation semantics: seabed_elevation_lat_m = -depth_lat_m)"
        ),
        "neighborhood_validity_threshold": MIN_VALID_NEIGHBORHOOD_FRACTION,
        "neighborhood_validity_threshold_note": (
            "a numerical edge/data-support safeguard, not a physical threshold"
        ),
        "sign_convention_observed_on_halo": result.halo_grid.sign_convention_observed,
        "features": {
            "slope_500m_deg": {"radius_m": 500.0, "unit": "degrees"},
            "slope_1000m_deg": {"radius_m": 1000.0, "unit": "degrees"},
            "tpi_1000m_m": {"radius_m": 1000.0, "unit": "metres"},
            "tpi_2000m_m": {"radius_m": 2000.0, "unit": "metres"},
            "local_relief_1000m_m": {"radius_m": 1000.0, "unit": "metres"},
            "local_relief_2000m_m": {"radius_m": 2000.0, "unit": "metres"},
            "terrain_std_1000m_m": {"radius_m": 1000.0, "unit": "metres"},
            "terrain_std_2000m_m": {"radius_m": 2000.0, "unit": "metres"},
        },
        "native_qa_layer_availability": {
            "wcs_coverage_ids": list(result.qa_layer_availability.wcs_coverage_ids),
            "wcs_matches": result.qa_layer_availability.wcs_matches,
            "download_tile_formats": list(result.qa_layer_availability.download_tile_formats),
            "download_tile_matches": result.qa_layer_availability.download_tile_matches,
            "notes": result.qa_layer_availability.notes,
        },
        "raster_outputs": {name: str(path) for name, path in raster_paths.items()},
        "aoi_identifier": result.aoi_identifier,
        "processing_timestamp": result.processing_timestamp,
        "software_version": SOFTWARE_VERSION,
        "scientific_limitations": list(SCIENTIFIC_LIMITATIONS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_path


# --- Diagnostics report (Sections 19-20) ------------------------------------


def _describe(series: pd.Series) -> dict[str, float]:
    valid = series.dropna()
    if valid.empty:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
        }
    return {
        "count": int(valid.count()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
    }


def print_regional_morphology_report(result: RegionalMorphologyResult, *, file: Any = None) -> None:
    file = file or sys.stdout
    df = result.chainage_df
    lines = ["=== PL854 Regional Seabed Morphology (MAR-007) ===", ""]

    lines.append(
        f"Halo buffer: {HALO_BUFFER_M:.0f} m; "
        f"source sign observed: {result.halo_grid.sign_convention_observed}"
    )
    lines.append("")

    for col in ("slope_500m_deg", "slope_1000m_deg"):
        stats = _describe(df[col])
        p95 = float(df[col].dropna().quantile(0.95)) if stats["count"] else float("nan")
        p99 = float(df[col].dropna().quantile(0.99)) if stats["count"] else float("nan")
        lines.append(
            f"{col}: n={stats['count']} min={stats['min']:.3f} max={stats['max']:.3f} "
            f"mean={stats['mean']:.3f} median={stats['median']:.3f} p95={p95:.3f} p99={p99:.3f}"
        )

    lines.append("")
    for col in ("tpi_1000m_m", "tpi_2000m_m"):
        valid = df[col].dropna()
        stats = _describe(df[col])
        pos = int((valid > 0).sum())
        neg = int((valid < 0).sum())
        near_zero = int((valid.abs() <= 0.01).sum())
        lines.append(
            f"{col}: n={stats['count']} min={stats['min']:.3f} max={stats['max']:.3f} "
            f"mean={stats['mean']:.3f} median={stats['median']:.3f} "
            f"positive={pos} negative={neg} near_zero(<=0.01m)={near_zero}"
        )

    lines.append("")
    for col in ("local_relief_1000m_m", "local_relief_2000m_m"):
        stats = _describe(df[col])
        lines.append(
            f"{col}: n={stats['count']} min={stats['min']:.3f} max={stats['max']:.3f} "
            f"mean={stats['mean']:.3f} median={stats['median']:.3f}"
        )

    for col in ("terrain_std_1000m_m", "terrain_std_2000m_m"):
        if col in df.columns and df[col].notna().any():
            stats = _describe(df[col])
            lines.append(
                f"{col}: n={stats['count']} min={stats['min']:.3f} max={stats['max']:.3f} "
                f"mean={stats['mean']:.3f} median={stats['median']:.3f}"
            )

    lines.append("")
    lines.append(f"Native cell QA availability: {result.qa_layer_availability.notes}")

    lines.append("")
    lines.append("Chainage coverage by source_reference_id:")
    for source_id, count in df["source_reference_id"].value_counts(dropna=False).items():
        lines.append(f"  {source_id}: {count} stations")

    lines.append("")
    lines.append("Chainage coverage by acquisition year:")
    year_counts = df["source_acquisition_year"].value_counts(dropna=False)
    total = len(df)
    for year, count in year_counts.items():
        lines.append(f"  {year}: {count} stations ({100.0 * count / total:.1f}%)")

    lines.append("")
    lines.append("First 5 stations:")
    lines.append(df.head(5).to_string())
    lines.append("")
    lines.append("Last 5 stations:")
    lines.append(df.tail(5).to_string())

    lines.append("")
    lines.append("=" * 70)
    lines.append("REGIONAL MORPHOLOGY AGE WARNING")
    lines.append(
        "The terrain features describe morphology represented by source surveys "
        "acquired in 1991-1992 and incorporated into the EMODnet DTM 2024 product. "
        "They must not be interpreted as surveyed present-day seabed morphology."
    )
    lines.append("=" * 70)

    print("\n".join(lines), file=file)
