"""Canonical EMODnet baseline DTM: verified depth semantics, projected analysis grid.

Transforms the raw EMODnet numeric GeoTIFF (acquired in MAR-005) into a
clean, reproducible canonical baseline: empirically-verified positive-down
depth relative to LAT, reprojected to the working CRS at a 100 m analysis
grid using bilinear resampling, clipped to the real PL854 AOI polygon (not
just its bounding box), with source-reference/quality-index provenance and
depth sampled onto every chainage station.

Scope: this is a REGIONAL baseline -- EMODnet's own ~115 m-class product.
The 100 m analysis grid is a projected spacing choice, not a claim that
reprojection created new resolving power. No morphology (slope/curvature/
roughness/BPI), no erosion/deposition, no scour/free-span or risk
calculations -- those are later tickets.
"""

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import mapping

from marine_engine.providers.bathymetry.emodnet import (
    QualityIndexFeature,
    SourceReferenceFeature,
)

# Numerical tolerance band (not a physical one): the observed mean must be
# within 50%-150% of the known regional depth range, in EITHER sign, to be
# accepted as that convention -- wide enough for real variability, narrow
# enough to refuse a genuine anomaly (e.g. a mean near zero or in the
# thousands) rather than silently guessing.
SIGN_CHECK_TOLERANCE_LOW = 0.5
SIGN_CHECK_TOLERANCE_HIGH = 1.5
DEFAULT_REFERENCE_DEPTH_RANGE_M = (20.0, 28.0)  # PL854 regional context, from project documentation

SIGN_POSITIVE_DOWN_DEPTH = "positive_down_depth"
SIGN_NEGATIVE_ELEVATION = "negative_elevation"

DEFAULT_TARGET_RESOLUTION_M = 100.0
RESAMPLING_METHOD = "bilinear"
CANONICAL_VERTICAL_DATUM = "LAT"
SOFTWARE_VERSION = "marine-engine 0.1.0 (MAR-006)"


class InvalidRawRasterError(RuntimeError):
    """The raw raster is missing, unreadable, or has no finite pixels to inspect."""


class AmbiguousSignConventionError(RuntimeError):
    """The observed raster statistics do not clearly match either sign convention."""


@dataclass(frozen=True)
class RawRasterStats:
    """Empirical statistics read directly from the raw raster's pixel values."""

    width: int
    height: int
    source_crs: str
    nodata_value: float | None
    total_pixels: int
    nan_count: int
    finite_count: int
    min: float
    max: float
    mean: float
    median: float


def inspect_raw_raster(raw_path: Path) -> RawRasterStats:
    """Read the raw raster and compute real statistics -- no assumptions about sign."""

    if not raw_path.exists():
        raise InvalidRawRasterError(f"Raw raster not found: {raw_path}")

    with rasterio.open(raw_path) as src:
        band = src.read(1)
        width, height = src.width, src.height
        source_crs = src.crs.to_string() if src.crs else None
        nodata_value = src.nodata

    nan_mask = np.isnan(band)
    valid_mask = ~nan_mask
    if nodata_value is not None and not np.isnan(nodata_value):
        valid_mask &= band != nodata_value

    finite = band[valid_mask]
    if finite.size == 0:
        raise InvalidRawRasterError(f"Raw raster {raw_path} has no finite/valid pixels to inspect.")

    return RawRasterStats(
        width=width,
        height=height,
        source_crs=source_crs,
        nodata_value=float(nodata_value) if nodata_value is not None else None,
        total_pixels=band.size,
        nan_count=int(nan_mask.sum()),
        finite_count=int(finite.size),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
    )


def determine_sign_convention(
    stats: RawRasterStats,
    reference_depth_range_m: tuple[float, float] = DEFAULT_REFERENCE_DEPTH_RANGE_M,
) -> str:
    """Empirically classify the raster's vertical sign convention.

    Compares the SIGN and MAGNITUDE of the observed mean against a known
    regional depth range (from independent project documentation, used
    only as a sanity check) rather than trusting product documentation
    alone. Raises rather than guessing if neither convention fits.
    """

    low, high = reference_depth_range_m
    mean = stats.mean

    if mean < 0 and low * SIGN_CHECK_TOLERANCE_LOW <= abs(mean) <= high * SIGN_CHECK_TOLERANCE_HIGH:
        return SIGN_NEGATIVE_ELEVATION
    if mean > 0 and low * SIGN_CHECK_TOLERANCE_LOW <= mean <= high * SIGN_CHECK_TOLERANCE_HIGH:
        return SIGN_POSITIVE_DOWN_DEPTH

    raise AmbiguousSignConventionError(
        f"Observed raw mean {mean:.2f} does not clearly match the expected regional "
        f"depth magnitude ({low}-{high} m) in either sign convention; refusing to guess. "
        "Inspect the raster manually before proceeding."
    )


def to_canonical_depth(raw_band: np.ndarray, sign_convention: str) -> np.ndarray:
    """Convert a RAW pixel array to canonical positive-down depth_lat_m.

    Always takes the raw (unconverted) array plus the convention explicitly
    detected from it -- there is no "already converted" state to re-flip,
    so calling this cannot double-negate as long as callers always pass
    the original raw array (enforced by this module's own pipeline).
    """

    if sign_convention == SIGN_NEGATIVE_ELEVATION:
        return -raw_band.astype("float64")
    if sign_convention == SIGN_POSITIVE_DOWN_DEPTH:
        return raw_band.astype("float64")
    raise ValueError(f"Unknown sign convention: {sign_convention!r}")


def reproject_and_resample(
    canonical_source: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    working_crs: str,
    src_bounds: tuple[float, float, float, float],
    target_resolution_m: float = DEFAULT_TARGET_RESOLUTION_M,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Reproject an already sign-corrected depth array to `working_crs` at `target_resolution_m`.

    Uses bilinear resampling, appropriate for continuous depth data --
    nearest-neighbour would introduce blocky artefacts, and higher-order
    methods (cubic) could manufacture apparent detail the ~115 m-class
    source does not actually contain.
    """

    dst_transform, width, height = calculate_default_transform(
        src_crs,
        working_crs,
        canonical_source.shape[1],
        canonical_source.shape[0],
        *src_bounds,
        resolution=target_resolution_m,
    )

    destination = np.full((height, width), np.nan, dtype="float64")
    reproject(
        source=canonical_source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=working_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return destination, dst_transform


def mask_to_aoi_polygon(
    array: np.ndarray, transform: rasterio.Affine, crs: str, aoi_geometry: Any
) -> tuple[np.ndarray, rasterio.Affine]:
    """Mask+crop a projected array to an AOI polygon (not merely its bounding box).

    Pixels outside the real polygon become NoData (NaN); the output extent
    is cropped to the polygon's bounds, so the canonical raster represents
    the study AOI rather than a loose rectangle around it.
    """

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float64",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }
    with MemoryFile() as memfile, memfile.open(**profile) as dataset:
        dataset.write(array, 1)
        masked, masked_transform = rasterio_mask(
            dataset, [mapping(aoi_geometry)], nodata=np.nan, crop=True
        )
    return masked[0], masked_transform


@dataclass(frozen=True)
class CanonicalRasterStats:
    width: int
    height: int
    valid_pixel_count: int
    nodata_pixel_count: int
    valid_percent: float
    depth_min: float
    depth_max: float
    depth_mean: float
    depth_median: float


def _canonical_stats(array: np.ndarray) -> CanonicalRasterStats:
    valid_mask = ~np.isnan(array)
    valid = array[valid_mask]
    total = array.size
    valid_count = int(valid.size)

    if valid_count == 0:
        return CanonicalRasterStats(
            width=array.shape[1],
            height=array.shape[0],
            valid_pixel_count=0,
            nodata_pixel_count=total,
            valid_percent=0.0,
            depth_min=float("nan"),
            depth_max=float("nan"),
            depth_mean=float("nan"),
            depth_median=float("nan"),
        )

    return CanonicalRasterStats(
        width=array.shape[1],
        height=array.shape[0],
        valid_pixel_count=valid_count,
        nodata_pixel_count=total - valid_count,
        valid_percent=100.0 * valid_count / total,
        depth_min=float(np.min(valid)),
        depth_max=float(np.max(valid)),
        depth_mean=float(np.mean(valid)),
        depth_median=float(np.median(valid)),
    )


def write_canonical_raster(
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    output_path: Path,
    tags: dict[str, str],
) -> None:
    """Write the canonical depth raster as a tagged GeoTIFF with explicit NoData."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(array.astype("float32"), 1)
        dataset.update_tags(**tags)


@dataclass(frozen=True)
class CanonicalDtmReport:
    """Everything needed to summarize a completed canonical-DTM build."""

    source_path: Path
    source_sha256: str | None
    source_crs: str
    source_nominal_resolution_m: float
    raw_stats: RawRasterStats
    sign_convention_observed: str
    output_crs: str
    output_resolution_m: float
    resampling_method: str
    vertical_datum: str
    canonical_stats: CanonicalRasterStats
    output_raster_path: Path
    output_metadata_path: Path
    processing_timestamp: str


def build_canonical_dtm(
    *,
    raw_path: Path,
    raw_manifest_entry: dict[str, Any] | None,
    aoi_geometry_working: Any,
    working_crs: str,
    aoi_identifier: str,
    output_raster_path: Path,
    output_metadata_path: Path,
    target_resolution_m: float = DEFAULT_TARGET_RESOLUTION_M,
    reference_depth_range_m: tuple[float, float] = DEFAULT_REFERENCE_DEPTH_RANGE_M,
) -> CanonicalDtmReport:
    """End-to-end: inspect -> verify sign -> reproject/resample -> mask to AOI -> write."""

    raw_stats = inspect_raw_raster(raw_path)
    sign_convention = determine_sign_convention(raw_stats, reference_depth_range_m)

    with rasterio.open(raw_path) as src:
        raw_band = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_bounds = tuple(src.bounds)

    canonical_source = to_canonical_depth(raw_band, sign_convention)
    reprojected, reprojected_transform = reproject_and_resample(
        canonical_source, src_transform, src_crs, working_crs, src_bounds, target_resolution_m
    )
    masked, masked_transform = mask_to_aoi_polygon(
        reprojected, reprojected_transform, working_crs, aoi_geometry_working
    )

    canonical_stats = _canonical_stats(masked)
    processing_timestamp = datetime.now(UTC).isoformat()

    tags = {
        "product": "EMODnet Digital Bathymetry (DTM 2024)",
        "vertical_datum": CANONICAL_VERTICAL_DATUM,
        "value_semantics": "positive_down_depth_lat_m",
        "source_sign_convention_observed": sign_convention,
        "resampling_method": RESAMPLING_METHOD,
        "source_nominal_resolution_m": "115",
        "output_analysis_grid_spacing_m": str(target_resolution_m),
        "source_sha256": (raw_manifest_entry or {}).get("sha256", ""),
        "processing_timestamp": processing_timestamp,
        "software_version": SOFTWARE_VERSION,
        "aoi_identifier": aoi_identifier,
        "limitation": (
            "Regional ~115 m-class baseline; the 100 m grid is an analysis "
            "spacing, not measured resolution. Not sufficient alone for "
            "pipeline-scale scour, sand-wave geometry, or free-span detection."
        ),
    }
    write_canonical_raster(masked, masked_transform, working_crs, output_raster_path, tags)

    metadata = {
        "product": "EMODnet Digital Bathymetry (DTM 2024)",
        "source_version": "2024",
        "source_path": str(raw_path),
        "source_sha256": (raw_manifest_entry or {}).get("sha256"),
        "source_crs": raw_stats.source_crs,
        "output_crs": working_crs,
        "source_nominal_resolution_m": 115.0,
        "output_analysis_grid_spacing_m": target_resolution_m,
        "resampling_method": RESAMPLING_METHOD,
        "vertical_datum": CANONICAL_VERTICAL_DATUM,
        "source_sign_convention": sign_convention,
        "canonical_sign_convention": "positive_down_depth_lat_m",
        "aoi_identifier": aoi_identifier,
        "processing_timestamp": processing_timestamp,
        "software_version": SOFTWARE_VERSION,
        "raw_stats": asdict(raw_stats),
        "canonical_stats": asdict(canonical_stats),
        "scientific_limitations": [
            "EMODnet 2024 is approximately 115 m-class regional bathymetry.",
            "The 100 m projected grid is an analysis grid, not true 100 m measurement resolution.",
            "Appropriate for regional seabed context and broad morphology.",
            "NOT sufficient by itself for pipeline-scale local scour, "
            "metre-scale sand-wave geometry, or free-span detection.",
            "High-resolution MBES can replace/augment this baseline later "
            "without changing the canonical pipeline/chainage architecture.",
        ],
    }
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return CanonicalDtmReport(
        source_path=raw_path,
        source_sha256=(raw_manifest_entry or {}).get("sha256"),
        source_crs=raw_stats.source_crs,
        source_nominal_resolution_m=115.0,
        raw_stats=raw_stats,
        sign_convention_observed=sign_convention,
        output_crs=working_crs,
        output_resolution_m=target_resolution_m,
        resampling_method=RESAMPLING_METHOD,
        vertical_datum=CANONICAL_VERTICAL_DATUM,
        canonical_stats=canonical_stats,
        output_raster_path=output_raster_path,
        output_metadata_path=output_metadata_path,
        processing_timestamp=processing_timestamp,
    )


# --- Chainage depth + source/quality attribution ----------------------------

CHAINAGE_OUTPUT_COLUMNS = (
    "pipeline_id",
    "station_index",
    "chainage_m",
    "kp_label",
    "depth_lat_m",
    "bathymetry_source_product",
    "source_reference_id",
    "source_reference_type",
    "qi_age",
    "qi_horizontal",
    "qi_vertical",
    "qi_purpose",
    "qi_combined",
)


def _reproject_features_to_working_crs(features: list, working_crs: str) -> list[tuple[Any, Any]]:
    """Pair each feature with its geometry reprojected once to `working_crs`."""

    if not features:
        return []
    geometries_wgs84 = [f.geometry_wgs84 for f in features]
    geometries_working = gpd.GeoSeries(geometries_wgs84, crs="EPSG:4326").to_crs(working_crs)
    return list(zip(features, geometries_working, strict=True))


def sample_chainage_bathymetry(
    *,
    chainage_gdf: gpd.GeoDataFrame,
    canonical_raster_path: Path,
    source_reference_features: list[SourceReferenceFeature],
    quality_index_features: list[QualityIndexFeature],
    working_crs: str,
) -> pd.DataFrame:
    """Sample canonical depth + source/quality attribution onto every chainage station.

    Retains all stations, even where depth or attribution is unavailable
    (null rather than dropped or fabricated). Does not derive slope,
    curvature, roughness, or any morphology -- point sampling only.
    """

    with rasterio.open(canonical_raster_path) as src:
        coords = [(pt.x, pt.y) for pt in chainage_gdf.geometry]
        sampled = list(src.sample(coords))
    depths = [float(v[0]) if not np.isnan(v[0]) else None for v in sampled]

    source_ref_pairs = _reproject_features_to_working_crs(source_reference_features, working_crs)
    qi_pairs = _reproject_features_to_working_crs(quality_index_features, working_crs)

    rows = []
    for row, depth in zip(chainage_gdf.itertuples(index=False), depths, strict=True):
        point = row.geometry
        matched_source = next((f for f, geom in source_ref_pairs if geom.contains(point)), None)
        matched_qi = next((f for f, geom in qi_pairs if geom.contains(point)), None)

        rows.append(
            {
                "pipeline_id": row.pipeline_id,
                "station_index": row.station_index,
                "chainage_m": row.chainage_m,
                "kp_label": row.kp_label,
                "depth_lat_m": depth,
                "bathymetry_source_product": "EMODnet Digital Bathymetry (DTM 2024)",
                "source_reference_id": matched_source.identifier if matched_source else None,
                "source_reference_type": matched_source.source_type if matched_source else None,
                "qi_age": matched_qi.age if matched_qi else None,
                "qi_horizontal": matched_qi.horizontal if matched_qi else None,
                "qi_vertical": matched_qi.vertical if matched_qi else None,
                "qi_purpose": matched_qi.purpose if matched_qi else None,
                "qi_combined": matched_qi.combined if matched_qi else None,
            }
        )

    return pd.DataFrame(rows, columns=list(CHAINAGE_OUTPUT_COLUMNS))


def write_chainage_bathymetry(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def print_bathymetry_report(
    dtm_report: CanonicalDtmReport,
    chainage_df: pd.DataFrame,
    *,
    attribution_status: str,
    attribution_notes: str,
    msl_notes: str,
    file: Any = None,
) -> None:
    """Print a concise scientific diagnostics summary."""

    file = file or sys.stdout
    rs = dtm_report.raw_stats
    cs = dtm_report.canonical_stats

    valid_depths = chainage_df["depth_lat_m"].dropna()
    unique_source_refs = chainage_df["source_reference_id"].dropna().unique()

    lines = [
        "=== Raw EMODnet ===",
        f"dimensions:        {rs.width} x {rs.height}",
        f"min/max/mean:       {rs.min:.2f} / {rs.max:.2f} / {rs.mean:.2f}",
        f"sign convention:    {dtm_report.sign_convention_observed}",
        f"nodata value:       {rs.nodata_value}",
        f"finite/total:       {rs.finite_count}/{rs.total_pixels}",
        "",
        "=== Canonical DTM ===",
        f"CRS:                {dtm_report.output_crs}",
        f"pixel size (m):     {dtm_report.output_resolution_m}",
        f"dimensions:         {cs.width} x {cs.height}",
        f"valid / nodata %:   {cs.valid_percent:.1f}% / {100 - cs.valid_percent:.1f}%",
        f"depth min/max:      {cs.depth_min:.2f} / {cs.depth_max:.2f} m",
        f"depth mean/median:  {cs.depth_mean:.2f} / {cs.depth_median:.2f} m",
        f"vertical datum:     {dtm_report.vertical_datum}",
        f"resampling:         {dtm_report.resampling_method}",
        f"output:             {dtm_report.output_raster_path}",
        f"metadata:           {dtm_report.output_metadata_path}",
        "",
        "=== PL854 chainage ===",
        f"stations sampled:   {len(chainage_df)}",
        f"valid depth count:  {len(valid_depths)}",
        f"missing depth:      {len(chainage_df) - len(valid_depths)}",
    ]
    if len(valid_depths):
        lines.append(
            f"depth min/max/mean: {valid_depths.min():.2f} / {valid_depths.max():.2f} / "
            f"{valid_depths.mean():.2f} m"
        )
    lines.append(f"unique source refs: {len(unique_source_refs)}")
    lines.append("")
    lines.append(f"source_attribution_status: {attribution_status}")
    if attribution_notes:
        lines.append(f"  note: {attribution_notes}")
    lines.append(f"MSL availability: {msl_notes}")
    lines.append("")
    lines.append(
        "LIMITATION: ~115 m-class regional baseline; the 100 m grid is an "
        "analysis spacing, not measured resolution. Not sufficient alone "
        "for pipeline-scale scour, sand-wave geometry, or free-span detection."
    )

    print("\n".join(lines), file=file)
