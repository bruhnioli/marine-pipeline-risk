"""Minimal command-line entry point for the marine-engine package."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from marine_engine import __version__
from marine_engine.config import load_study_config
from marine_engine.preprocessing import bathymetry, source_resolution
from marine_engine.preprocessing.aoi import (
    InvalidAoiGeometryError,
    InvalidPipelineInputError,
    build_aoi,
    print_aoi_report,
)
from marine_engine.preprocessing.chainage import (
    ChainageValidationError,
    InvalidPipelineRouteError,
    build_chainage,
    print_chainage_report,
)
from marine_engine.providers.bathymetry import acquisition, bgs, emodnet, inventory, ukho
from marine_engine.providers.nsta import (
    AmbiguousPipelineError,
    InvalidGeometryError,
    PipelineNotFoundError,
    ingest_pipeline,
    print_ingestion_report,
)


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"marine-engine {__version__}")
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    print(f"OK: '{config.study.id}' ({config.study.name})")
    print(f"  CRS (horizontal): {config.crs.horizontal}")
    print(f"  raw data dir:     {config.paths.raw_dir}")
    return 0


def _cmd_ingest_pipeline(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    cache_dir = config.paths.raw_dir / "nsta"
    output_path = config.paths.processed_dir / pipeline_id.lower() / "pipeline.gpkg"

    try:
        report = ingest_pipeline(
            pipeline_id,
            cache_dir=cache_dir,
            output_path=output_path,
            working_crs=config.crs.horizontal,
        )
    except (PipelineNotFoundError, AmbiguousPipelineError, InvalidGeometryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_ingestion_report(report)
    return 0


def _cmd_build_aoi(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    buffer_m = config.area_of_interest.corridor_buffer_m
    if not buffer_m:
        print(
            f"error: '{args.config}' has no area_of_interest.corridor_buffer_m configured",
            file=sys.stderr,
        )
        return 1

    pipeline_gpkg_path = config.paths.processed_dir / pipeline_id.lower() / "pipeline.gpkg"
    output_path = config.paths.processed_dir / pipeline_id.lower() / "aoi.gpkg"

    try:
        report = build_aoi(
            pipeline_gpkg_path=pipeline_gpkg_path,
            pipeline_id=pipeline_id,
            study_id=config.study.id,
            buffer_m=buffer_m,
            working_crs=config.crs.horizontal,
            output_path=output_path,
        )
    except (InvalidPipelineInputError, InvalidAoiGeometryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_aoi_report(report)
    return 0


def _cmd_build_chainage(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    interval_m = config.pipeline.get("chainage_interval_m")
    if not interval_m:
        print(
            f"error: '{args.config}' has no pipeline.chainage_interval_m configured",
            file=sys.stderr,
        )
        return 1

    pipeline_gpkg_path = config.paths.processed_dir / pipeline_id.lower() / "pipeline.gpkg"
    aoi_gpkg_path = config.paths.processed_dir / pipeline_id.lower() / "aoi.gpkg"
    output_path = config.paths.processed_dir / pipeline_id.lower() / "chainage_25m.gpkg"

    try:
        report = build_chainage(
            pipeline_gpkg_path=pipeline_gpkg_path,
            aoi_gpkg_path=aoi_gpkg_path,
            pipeline_id=pipeline_id,
            study_id=config.study.id,
            interval_m=interval_m,
            working_crs=config.crs.horizontal,
            output_path=output_path,
        )
    except (InvalidPipelineRouteError, ChainageValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_chainage_report(report)
    return 0


def _study_paths(config, pipeline_id: str) -> tuple[Path, Path, Path, Path]:
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    interim_dir = config.paths.interim_dir / pipeline_id.lower()
    return (
        study_dir / "pipeline.gpkg",
        study_dir / "aoi.gpkg",
        study_dir / "chainage_25m.gpkg",
        interim_dir,
    )


def _cmd_discover_bathymetry(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )

    try:
        pipeline_gdf = gpd.read_file(pipeline_gpkg_path, layer="pipeline")
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(f"error: could not load canonical pipeline/AOI/chainage: {exc}", file=sys.stderr)
        return 1

    working_crs = config.crs.horizontal
    pipeline_geom = pipeline_gdf.geometry.iloc[0]
    aoi_geom = unary_union(aoi_gdf.geometry)
    aoi_bbox_wgs84 = tuple(float(v) for v in aoi_gdf.to_crs("EPSG:4326").total_bounds)

    ukho_records, ukho_status = ukho.discover_ukho_surveys(aoi_bbox_wgs84)
    bgs_records = bgs.discover_bgs_surveys()
    emodnet_record = emodnet.discover_emodnet_baseline(aoi_bbox_wgs84)
    raw_records = [*ukho_records, *bgs_records, emodnet_record]

    report = inventory.run_discovery(
        raw_records,
        pipeline_geom_working=pipeline_geom,
        aoi_geom_working=aoi_geom,
        chainage_gdf_working=chainage_gdf,
        working_crs=working_crs,
        parquet_path=interim_dir / "bathymetry_inventory.parquet",
        gpkg_path=interim_dir / "bathymetry_inventory.gpkg",
    )

    print(f"UKHO (via MEDIN): {ukho_status.message}")
    print()
    inventory.print_discovery_report(report)
    return 0


def _cmd_fetch_bathymetry(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    _pipeline_gpkg_path, aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    parquet_path = interim_dir / "bathymetry_inventory.parquet"
    manifest_path = interim_dir / "bathymetry_acquisition_manifest.json"

    if not parquet_path.exists():
        print(
            f"error: no inventory found at {parquet_path}; run discover-bathymetry first",
            file=sys.stderr,
        )
        return 1

    try:
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not load AOI: {exc}", file=sys.stderr)
        return 1

    aoi_bbox_wgs84 = tuple(float(v) for v in aoi_gdf.to_crs("EPSG:4326").total_bounds)

    output_path = (
        acquisition.raw_dataset_dir(config.paths.raw_dir, "emodnet", emodnet.COVERAGE_ID)
        / f"{emodnet.COVERAGE_ID}.tif"
    )
    request_parameters = {
        "coverageId": emodnet.COVERAGE_ID,
        "bbox_wgs84": list(aoi_bbox_wgs84),
        "format": "image/tiff;application=geotiff",
    }

    existing = acquisition.already_acquired(
        manifest_path, "EMODnet", emodnet.COVERAGE_ID, request_parameters
    )
    if existing is not None:
        print(f"EMODnet: already acquired ({existing['local_path']}); skipping re-download.")
        emodnet_entry = existing
    else:
        try:
            fetch_result = emodnet.fetch_emodnet_geotiff(aoi_bbox_wgs84, output_path)
        except emodnet.EmodnetUnavailableError as exc:
            print(f"error: EMODnet acquisition failed: {exc}", file=sys.stderr)
            return 1

        emodnet_entry = acquisition.record_acquisition(
            manifest_path,
            source="EMODnet",
            dataset_id=emodnet.COVERAGE_ID,
            source_url_or_service=emodnet.WCS_BASE_URL,
            request_parameters=fetch_result.request_parameters,
            local_path=fetch_result.local_path,
            licence=emodnet.LICENCE,
            acquisition_year=2024,
            horizontal_crs=fetch_result.returned_crs,
            vertical_datum=emodnet.VERTICAL_DATUM,
            nominal_resolution_m=emodnet.NATIVE_RESOLUTION_M,
            acquired_at=datetime.now(UTC),
        )
        print(
            f"EMODnet: acquired {fetch_result.width_px}x{fetch_result.height_px} px "
            f"-> {fetch_result.local_path}"
        )

    df = pd.read_parquet(parquet_path)
    manual_rows = df[df["manual_download_required"].fillna(False)]
    print()
    print("Datasets requiring manual download:")
    if manual_rows.empty:
        print("  (none)")
    else:
        for _, row in manual_rows.iterrows():
            url = row["source_record_url_or_identifier"]
            print(f"  - [{row['source']}] {row['source_dataset_id']}: {url}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"EMODnet sha256: {emodnet_entry['sha256']}")
    print(f"EMODnet file size: {emodnet_entry['file_size_bytes']} bytes")
    return 0


def _cmd_build_bathymetry(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    _pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    manifest_path = interim_dir / "bathymetry_acquisition_manifest.json"
    raw_path = (
        acquisition.raw_dataset_dir(config.paths.raw_dir, "emodnet", emodnet.COVERAGE_ID)
        / f"{emodnet.COVERAGE_ID}.tif"
    )

    if not raw_path.exists():
        print(
            f"error: no raw EMODnet raster found at {raw_path}; run fetch-bathymetry first",
            file=sys.stderr,
        )
        return 1

    try:
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(f"error: could not load canonical AOI/chainage: {exc}", file=sys.stderr)
        return 1

    working_crs = config.crs.horizontal
    aoi_geom_working = unary_union(aoi_gdf.geometry)
    aoi_bbox_wgs84 = tuple(float(v) for v in aoi_gdf.to_crs("EPSG:4326").total_bounds)

    manifest_entries = acquisition.load_manifest(manifest_path)
    raw_manifest_entry = next(
        (
            e
            for e in manifest_entries
            if e.get("source") == "EMODnet" and e.get("dataset_id") == emodnet.COVERAGE_ID
        ),
        None,
    )

    output_dir = config.paths.processed_dir / pipeline_id.lower() / "bathymetry"
    output_raster_path = output_dir / "emodnet_baseline_lat_100m.tif"
    output_metadata_path = output_dir / "emodnet_baseline_lat_100m.json"
    chainage_output_path = output_dir / "chainage_bathymetry.parquet"

    try:
        dtm_report = bathymetry.build_canonical_dtm(
            raw_path=raw_path,
            raw_manifest_entry=raw_manifest_entry,
            aoi_geometry_working=aoi_geom_working,
            working_crs=working_crs,
            aoi_identifier=f"{pipeline_id}_AOI",
            output_raster_path=output_raster_path,
            output_metadata_path=output_metadata_path,
        )
    except (bathymetry.InvalidRawRasterError, bathymetry.AmbiguousSignConventionError) as exc:
        print(f"error: canonical DTM build failed: {exc}", file=sys.stderr)
        return 1

    # Source-reference/quality-index attribution is retrieved best-effort: a live
    # WFS failure here must not fail the DTM build above (Section 11) -- depth
    # processing and source-quality attribution are deliberately separable.
    try:
        source_refs = emodnet.fetch_source_references(aoi_bbox_wgs84)
        qi_features = emodnet.fetch_quality_index(aoi_bbox_wgs84)
        attribution_status = "available"
        attribution_notes = ""
    except emodnet.EmodnetAttributionUnavailableError as exc:
        source_refs = []
        qi_features = []
        attribution_status = "unavailable"
        attribution_notes = str(exc)

    msl_result = emodnet.check_msl_availability(aoi_bbox_wgs84)
    if msl_result.available:
        msl_notes = (
            f"available (tile {msl_result.tile_id}, release {msl_result.dtm_release}, "
            f"format '{msl_result.format_label}'); {msl_result.notes}"
        )
    else:
        msl_notes = f"not available; {msl_result.notes}"

    chainage_df = bathymetry.sample_chainage_bathymetry(
        chainage_gdf=chainage_gdf,
        canonical_raster_path=output_raster_path,
        source_reference_features=source_refs,
        quality_index_features=qi_features,
        working_crs=working_crs,
    )
    bathymetry.write_chainage_bathymetry(chainage_df, chainage_output_path)

    bathymetry.print_bathymetry_report(
        dtm_report,
        chainage_df,
        attribution_status=attribution_status,
        attribution_notes=attribution_notes,
        msl_notes=msl_notes,
    )
    return 0


def _cmd_resolve_bathymetry_sources(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, _interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    chainage_bathymetry_path = study_dir / "bathymetry" / "chainage_bathymetry.parquet"

    if not chainage_bathymetry_path.exists():
        print(
            f"error: no chainage bathymetry attribution found at {chainage_bathymetry_path}; "
            "run build-bathymetry first",
            file=sys.stderr,
        )
        return 1

    try:
        pipeline_gdf = gpd.read_file(pipeline_gpkg_path, layer="pipeline")
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
        chainage_bathymetry = pd.read_parquet(chainage_bathymetry_path)
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(
            f"error: could not load canonical pipeline/AOI/chainage/bathymetry: {exc}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    aoi_bbox_wgs84 = tuple(float(v) for v in aoi_gdf.to_crs("EPSG:4326").total_bounds)

    try:
        source_refs = emodnet.fetch_source_references(aoi_bbox_wgs84)
        qi_features = emodnet.fetch_quality_index(aoi_bbox_wgs84)
    except emodnet.EmodnetAttributionUnavailableError as exc:
        print(
            f"error: could not retrieve EMODnet source-reference attribution: {exc}",
            file=sys.stderr,
        )
        return 1

    df, records, overlaps = source_resolution.resolve_pl854_cdi_sources(
        pipeline_gdf=pipeline_gdf,
        aoi_gdf=aoi_gdf,
        chainage_gdf=chainage_gdf,
        chainage_bathymetry=chainage_bathymetry,
        source_reference_features=source_refs,
        quality_index_features=qi_features,
        working_crs=working_crs,
    )

    parquet_path = study_dir / "bathymetry" / "emodnet_cdi_sources.parquet"
    gpkg_path = study_dir / "bathymetry" / "emodnet_cdi_sources.gpkg"
    source_resolution.write_cdi_sources_parquet(df, parquet_path)
    source_resolution.write_cdi_sources_gpkg(records, working_crs, gpkg_path)

    print(f"Resolved {len(df)} PL854 source-reference record(s).")
    print(f"Output: {parquet_path}")
    print()
    source_resolution.print_source_resolution_report(df, overlaps)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marine-engine",
        description="Seabed-risk modelling engine for subsea pipeline corridors.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the installed package version.")
    version_parser.set_defaults(func=_cmd_version)

    validate_parser = subparsers.add_parser(
        "validate-config", help="Load and validate a study YAML configuration."
    )
    validate_parser.add_argument("config", type=Path, help="Path to a study config YAML file.")
    validate_parser.set_defaults(func=_cmd_validate_config)

    ingest_parser = subparsers.add_parser(
        "ingest-pipeline",
        help="Ingest a study's pipeline geometry from NSTA and write the canonical GeoPackage.",
    )
    ingest_parser.add_argument("config", type=Path, help="Path to a study config YAML file.")
    ingest_parser.set_defaults(func=_cmd_ingest_pipeline)

    aoi_parser = subparsers.add_parser(
        "build-aoi",
        help="Build the pipeline corridor AOI from the canonical pipeline geometry.",
    )
    aoi_parser.add_argument("config", type=Path, help="Path to a study config YAML file.")
    aoi_parser.set_defaults(func=_cmd_build_aoi)

    chainage_parser = subparsers.add_parser(
        "build-chainage",
        help="Build the 25 m chainage/KP linear-reference points along the canonical pipeline.",
    )
    chainage_parser.add_argument("config", type=Path, help="Path to a study config YAML file.")
    chainage_parser.set_defaults(func=_cmd_build_chainage)

    discover_bathymetry_parser = subparsers.add_parser(
        "discover-bathymetry",
        help="Discover, spatially verify, and rank approved bathymetry sources for the AOI.",
    )
    discover_bathymetry_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    discover_bathymetry_parser.set_defaults(func=_cmd_discover_bathymetry)

    fetch_bathymetry_parser = subparsers.add_parser(
        "fetch-bathymetry",
        help="Acquire the mandatory EMODnet baseline and report manual-download-only datasets.",
    )
    fetch_bathymetry_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    fetch_bathymetry_parser.set_defaults(func=_cmd_fetch_bathymetry)

    build_bathymetry_parser = subparsers.add_parser(
        "build-bathymetry",
        help=(
            "Build the canonical EMODnet baseline DTM (positive-down depth, LAT datum, "
            "AOI-clipped) and sample depth/source/quality attribution onto chainage."
        ),
    )
    build_bathymetry_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_bathymetry_parser.set_defaults(func=_cmd_build_bathymetry)

    resolve_sources_parser = subparsers.add_parser(
        "resolve-bathymetry-sources",
        help=(
            "Resolve PL854's EMODnet source-reference ids to their real SeaDataNet CDI "
            "survey provenance (acquisition epoch, instrument, access, recovery potential)."
        ),
    )
    resolve_sources_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    resolve_sources_parser.set_defaults(func=_cmd_resolve_bathymetry_sources)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
