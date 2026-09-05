"""Minimal command-line entry point for the marine-engine package."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.ops import unary_union

from marine_engine import __version__
from marine_engine.config import load_study_config
from marine_engine.metocean import (
    current_map,
    current_normalization,
    wave_orbital,
    wave_orbital_map,
)
from marine_engine.metocean import evidence as metocean_evidence
from marine_engine.morphology import regional
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
    load_pipeline_route,
    print_chainage_report,
)
from marine_engine.providers.bathymetry import acquisition, bgs, emodnet, inventory, ukho
from marine_engine.providers.metocean import acquisition as metocean_acquisition
from marine_engine.providers.metocean import copernicus
from marine_engine.providers.nsta import (
    AmbiguousPipelineError,
    InvalidGeometryError,
    PipelineNotFoundError,
    ingest_pipeline,
    print_ingestion_report,
)
from marine_engine.providers.sediment import bgs as sediment_bgs
from marine_engine.sediment import evidence


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
            # EMODnet DTM 2024 is an aggregate product release, not a survey
            # acquisition -- see acquisition.record_acquisition's docstring.
            acquisition_year=None,
            product_release_year=2024,
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

    pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    # Canonical DTM / chainage-bathymetry stay under processed/<study>/bathymetry/
    # (analysis-ready products); this command's own output is source/provenance
    # *resolution* metadata, not an analysis-ready product, so it belongs under
    # interim/<study>/ instead -- see MAR-006C.
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

    parquet_path = interim_dir / "emodnet_cdi_sources.parquet"
    gpkg_path = interim_dir / "emodnet_cdi_sources.gpkg"
    source_resolution.write_cdi_sources_parquet(df, parquet_path)
    source_resolution.write_cdi_sources_gpkg(records, working_crs, gpkg_path)

    print(f"Resolved {len(df)} PL854 source-reference record(s).")
    print(f"Output: {parquet_path}")
    print()
    source_resolution.print_source_resolution_report(df, overlaps)
    return 0


def _cmd_build_regional_morphology(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    _pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    chainage_bathymetry_path = study_dir / "bathymetry" / "chainage_bathymetry.parquet"
    cdi_sources_path = interim_dir / "emodnet_cdi_sources.parquet"
    canonical_dtm_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"

    if not chainage_bathymetry_path.exists():
        print(
            f"error: no chainage bathymetry attribution found at {chainage_bathymetry_path}; "
            "run build-bathymetry first",
            file=sys.stderr,
        )
        return 1
    if not cdi_sources_path.exists():
        print(
            f"error: no CDI source provenance found at {cdi_sources_path}; "
            "run resolve-bathymetry-sources first",
            file=sys.stderr,
        )
        return 1

    try:
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
        chainage_bathymetry_df = pd.read_parquet(chainage_bathymetry_path)
        cdi_sources_df = pd.read_parquet(cdi_sources_path)
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(
            f"error: could not load canonical AOI/chainage/bathymetry/provenance: {exc}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    aoi_geom_working = unary_union(aoi_gdf.geometry)
    halo_bbox_wgs84 = regional.build_halo_bbox_wgs84(aoi_geom_working, working_crs)

    manifest_path = interim_dir / "bathymetry_acquisition_manifest.json"
    halo_dataset_id = f"{emodnet.COVERAGE_ID}_halo_{pipeline_id.lower()}"
    raw_halo_path = (
        acquisition.raw_dataset_dir(config.paths.raw_dir, "emodnet", halo_dataset_id)
        / f"{halo_dataset_id}.tif"
    )
    # Must match exactly what `fetch_emodnet_geotiff` itself records as
    # `request_parameters` (coverageId/bbox_wgs84/format only) -- any extra
    # key here would never equal the stored entry and defeat idempotency,
    # re-fetching this halo on every single run.
    request_parameters = {
        "coverageId": emodnet.COVERAGE_ID,
        "bbox_wgs84": list(halo_bbox_wgs84),
        "format": "image/tiff;application=geotiff",
    }
    existing = acquisition.already_acquired(
        manifest_path, "EMODnet", halo_dataset_id, request_parameters
    )
    if existing is not None:
        print(f"EMODnet halo: already acquired ({existing['local_path']}); skipping re-download.")
        halo_manifest_entry = existing
    else:
        try:
            fetch_result = emodnet.fetch_emodnet_geotiff(halo_bbox_wgs84, raw_halo_path)
        except emodnet.EmodnetUnavailableError as exc:
            print(f"error: EMODnet halo acquisition failed: {exc}", file=sys.stderr)
            return 1
        halo_manifest_entry = acquisition.record_acquisition(
            manifest_path,
            source="EMODnet",
            dataset_id=halo_dataset_id,
            source_url_or_service=emodnet.WCS_BASE_URL,
            request_parameters=fetch_result.request_parameters,
            local_path=fetch_result.local_path,
            licence=emodnet.LICENCE,
            # EMODnet DTM 2024 is an aggregate product release, not a survey
            # acquisition -- see acquisition.record_acquisition's docstring.
            acquisition_year=None,
            product_release_year=2024,
            horizontal_crs=fetch_result.returned_crs,
            vertical_datum=emodnet.VERTICAL_DATUM,
            nominal_resolution_m=emodnet.NATIVE_RESOLUTION_M,
            acquired_at=datetime.now(UTC),
        )
        print(
            f"EMODnet halo: acquired {fetch_result.width_px}x{fetch_result.height_px} px "
            f"-> {fetch_result.local_path}"
        )

    qa_layer_availability = emodnet.check_native_qa_layers(halo_bbox_wgs84)

    try:
        result = regional.build_regional_morphology(
            aoi_gdf=aoi_gdf,
            chainage_gdf=chainage_gdf,
            chainage_bathymetry_df=chainage_bathymetry_df,
            cdi_sources_df=cdi_sources_df,
            raw_halo_path=raw_halo_path,
            raw_halo_manifest_entry=halo_manifest_entry,
            qa_layer_availability=qa_layer_availability,
            working_crs=working_crs,
            aoi_identifier=f"{pipeline_id}_AOI",
        )
    except (
        bathymetry.InvalidRawRasterError,
        bathymetry.AmbiguousSignConventionError,
        regional.RegionalMorphologyError,
    ) as exc:
        print(f"error: regional morphology build failed: {exc}", file=sys.stderr)
        return 1

    morphology_dir = study_dir / "morphology"
    raster_paths = {}
    for layer in result.layers:
        raster_path = morphology_dir / f"{layer.name}.tif"
        regional.write_morphology_raster(
            layer, working_crs, raster_path, aoi_identifier=f"{pipeline_id}_AOI"
        )
        raster_paths[layer.name] = raster_path

    chainage_output_path = morphology_dir / "chainage_regional_morphology.parquet"
    regional.write_chainage_regional_morphology(result.chainage_df, chainage_output_path)

    metadata_path = morphology_dir / "morphology_metadata.json"
    regional.write_morphology_metadata(
        result,
        input_dtm_path=canonical_dtm_path,
        raster_paths=raster_paths,
        output_path=metadata_path,
    )

    print(f"Wrote {len(raster_paths)} morphology raster(s), chainage table, and metadata to:")
    print(f"  {morphology_dir}")
    print()
    regional.print_regional_morphology_report(result)
    return 0


def _cmd_build_sediment_evidence(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    if not aoi_gpkg_path.exists() or not chainage_gpkg_path.exists():
        print(
            f"error: no canonical AOI/chainage found under {aoi_gpkg_path.parent}; "
            "run build-aoi and build-chainage first",
            file=sys.stderr,
        )
        return 1

    try:
        route_working, _attributes, _source_crs = load_pipeline_route(
            pipeline_gpkg_path, pipeline_id
        )
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(f"error: could not load canonical pipeline/AOI/chainage: {exc}", file=sys.stderr)
        return 1

    working_crs = config.crs.horizontal
    aoi_geometry_working = unary_union(aoi_gdf.geometry)
    aoi_geometry_wgs84 = (
        gpd.GeoSeries([aoi_geometry_working], crs=working_crs).to_crs("EPSG:4326").iloc[0]
    )
    query_timestamp = datetime.now(UTC)

    try:
        psa_features = sediment_bgs.fetch_psa_observations(aoi_geometry_wgs84)
        seabed_250k_features = sediment_bgs.fetch_seabed_sediments_250k(aoi_geometry_wgs84)
        predictive_folk_features = sediment_bgs.fetch_predictive_folk_polygons(aoi_geometry_wgs84)
    except sediment_bgs.BgsSedimentUnavailableError as exc:
        print(f"error: BGS sediment acquisition failed: {exc}", file=sys.stderr)
        return 1

    psa_gdf = evidence.normalize_psa_observations(
        psa_features,
        route_working=route_working,
        working_crs=working_crs,
        aoi_geometry_working=aoi_geometry_working,
        run_timestamp=query_timestamp,
    )
    psa_gdf = evidence.attach_nearest_chainage_station(psa_gdf, chainage_gdf.to_crs(working_crs))

    seabed_250k_gdf = evidence.normalize_seabed_sediments_250k(
        seabed_250k_features, working_crs=working_crs
    )
    seabed_250k_gdf = evidence.compute_250k_intersections(
        seabed_250k_gdf, aoi_geometry_working=aoi_geometry_working, route_working=route_working
    )

    predictive_folk_gdf = evidence.normalize_predictive_folk_polygons(
        predictive_folk_features, working_crs=working_crs
    )

    psa_with_comparisons = evidence.attach_mapped_and_predictive_at_psa_points(
        psa_gdf, seabed_250k_gdf, predictive_folk_gdf
    )

    # Predictive sand/gravel/mud percentages: only at surface PSA points, never
    # at all 941 chainage stations -- see the metadata's
    # predictive_percentage_chainage_note for why (Section 16's own
    # "if not safely queryable, do not fabricate").
    surface_mask = psa_with_comparisons["surface_evidence_class"].isin(
        (evidence.SURFACE_GRAB, evidence.SURFACE_CORE_INTERVAL)
    )
    predictive_percentages_by_psa_id: dict = {}
    for _, row in psa_with_comparisons[surface_mask].iterrows():
        percentages = {}
        for key, layer_id in (
            ("gravel", sediment_bgs.PREDICTIVE_GRAVEL_LAYER_ID),
            ("sand", sediment_bgs.PREDICTIVE_SAND_LAYER_ID),
            ("mud", sediment_bgs.PREDICTIVE_MUD_LAYER_ID),
        ):
            try:
                percentages[key] = sediment_bgs.fetch_predictive_percentage_at_point(
                    row["longitude"], row["latitude"], layer_id
                )
            except sediment_bgs.BgsSedimentUnavailableError:
                percentages[key] = None
        predictive_percentages_by_psa_id[row["psa_data_id"]] = percentages

    predictive_comparison_df = evidence.build_predictive_comparison_table(
        psa_with_comparisons, predictive_percentages_by_psa_id=predictive_percentages_by_psa_id
    )

    chainage_sediment_df = evidence.build_chainage_sediment_evidence(
        chainage_gdf=chainage_gdf,
        psa_gdf_working=psa_with_comparisons,
        seabed_250k_gdf_working=seabed_250k_gdf,
        predictive_folk_gdf_working=predictive_folk_gdf,
        working_crs=working_crs,
    )

    coverage = evidence.compute_coverage_diagnostics(psa_gdf)
    chainage_support = evidence.compute_chainage_support_proportions(chainage_sediment_df)
    agreement = evidence.compute_agreement_diagnostics(psa_with_comparisons, chainage_sediment_df)
    d50_assessment = evidence.assess_d50_spatial_support(chainage_sediment_df, coverage)

    sediment_interim_dir = interim_dir / "sediment"
    sediment_processed_dir = config.paths.processed_dir / pipeline_id.lower() / "sediment"

    psa_parquet_path = sediment_interim_dir / "bgs_psa_observations.parquet"
    psa_gpkg_path = sediment_interim_dir / "bgs_psa_observations.gpkg"
    seabed_250k_gpkg_path = sediment_interim_dir / "bgs_seabed_sediments_250k.gpkg"
    seabed_250k_parquet_path = sediment_interim_dir / "bgs_seabed_sediments_250k.parquet"
    predictive_comparison_path = sediment_interim_dir / "bgs_predictive_sediment_comparison.parquet"
    chainage_sediment_path = sediment_processed_dir / "chainage_sediment_evidence.parquet"
    metadata_path = sediment_processed_dir / "sediment_evidence_metadata.json"

    evidence.write_psa_observations(psa_gdf, psa_parquet_path, psa_gpkg_path)
    evidence.write_seabed_sediments_250k(
        seabed_250k_gdf, seabed_250k_gpkg_path, seabed_250k_parquet_path
    )
    evidence.write_predictive_comparison(predictive_comparison_df, predictive_comparison_path)
    evidence.write_chainage_sediment_evidence(chainage_sediment_df, chainage_sediment_path)

    providers_metadata = {
        "bgs_psa": {
            "provider": "BGS",
            "dataset": sediment_bgs.PSA_DATASET_TITLE,
            "endpoint": sediment_bgs.PSA_SERVICE_URL,
            "evidence_role": "PRIMARY_OBSERVATIONAL",
        },
        "bgs_seabed_sediments_250k": {
            "provider": "BGS",
            "dataset": sediment_bgs.SEABED_SEDIMENTS_250K_DATASET_TITLE,
            "endpoint": sediment_bgs.SEABED_SEDIMENTS_250K_SERVICE_URL,
            "evidence_role": "REGIONAL_MAPPED_SUBSTRATE",
        },
        "bgs_predictive": {
            "provider": "BGS",
            "dataset": sediment_bgs.PREDICTIVE_DATASET_TITLE,
            "endpoint": sediment_bgs.PREDICTIVE_FOLK_SERVICE_URL,
            "percentage_layers_endpoint": sediment_bgs.PREDICTIVE_SERVICE_ROOT_URL,
            "evidence_role": "SECONDARY_MODEL_COMPARISON",
        },
    }
    evidence.write_sediment_evidence_metadata(
        providers=providers_metadata,
        query_timestamp=query_timestamp,
        aoi_identifier=f"{pipeline_id}_AOI",
        coverage=coverage,
        chainage_support=chainage_support,
        agreement=agreement,
        d50_assessment=d50_assessment,
        outputs={
            "psa_observations_parquet": psa_parquet_path,
            "psa_observations_gpkg": psa_gpkg_path,
            "seabed_sediments_250k_gpkg": seabed_250k_gpkg_path,
            "seabed_sediments_250k_parquet": seabed_250k_parquet_path,
            "predictive_comparison_parquet": predictive_comparison_path,
            "chainage_sediment_evidence_parquet": chainage_sediment_path,
        },
        output_path=metadata_path,
    )

    print(f"PSA observations: {len(psa_gdf)} record(s) -> {psa_parquet_path}")
    print(f"Seabed Sediments 250k: {len(seabed_250k_gdf)} polygon(s) -> {seabed_250k_gpkg_path}")
    print(f"Predictive comparison: {len(predictive_comparison_df)} row(s)")
    print(f"  -> {predictive_comparison_path}")
    print(f"Chainage sediment evidence: {len(chainage_sediment_df)} station(s)")
    print(f"  -> {chainage_sediment_path}")
    print(f"Metadata: {metadata_path}")
    print()
    evidence.print_sediment_evidence_report(
        coverage=coverage,
        chainage_support=chainage_support,
        agreement=agreement,
        d50_assessment=d50_assessment,
    )
    return 0


def _read_existing_parquet(path: Path) -> pd.DataFrame:
    """The prior run's canonical output, or an empty frame if there isn't one yet.

    Read BEFORE that same path is overwritten, so the old-vs-corrected
    comparison (MAR-009A Sections 12, 13, 19) reflects what was actually on
    disk rather than a fabricated baseline.
    """

    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _mask_surface_slice(mask: xr.DataArray) -> xr.DataArray:
    """The surface-level slice of a mask variable that may or may not have a depth dimension."""

    if "depth" in mask.dims:
        return mask.isel(depth=0)
    return mask


def _acquire_static_dataset(
    *,
    manifest_path: Path,
    raw_dir: Path,
    product_id: str,
    dataset_id: str,
    variables: list,
    bbox_wgs84: tuple,
    evidence_role: str,
) -> xr.Dataset:
    requested_bbox = list(bbox_wgs84)
    existing = metocean_acquisition.already_acquired(
        manifest_path,
        product_id=product_id,
        dataset_id=dataset_id,
        variables=variables,
        requested_bbox=requested_bbox,
        requested_depths=None,
        requested_start=metocean_acquisition.STATIC_TIME_SENTINEL,
        requested_end=metocean_acquisition.STATIC_TIME_SENTINEL,
    )
    if existing is not None:
        return xr.open_dataset(existing["local_path"])

    result = copernicus.subset_dataset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=bbox_wgs84[0],
        maximum_longitude=bbox_wgs84[2],
        minimum_latitude=bbox_wgs84[1],
        maximum_latitude=bbox_wgs84[3],
        start_datetime=None,
        end_datetime=None,
        minimum_depth=None,
        maximum_depth=None,
        output_directory=raw_dir,
        output_filename=f"{dataset_id}_static.nc",
    )
    metocean_acquisition.record_acquisition(
        manifest_path,
        provider="Copernicus Marine",
        product_id=product_id,
        dataset_id=dataset_id,
        evidence_role=evidence_role,
        variables=variables,
        requested_bbox=requested_bbox,
        requested_depths=None,
        requested_start=None,
        requested_end=None,
        actual_start=None,
        actual_end=None,
        temporal_resolution="static",
        local_path=result.local_path,
        toolbox_version=copernicus.toolbox_version(),
        licence=None,
        downloaded_at=datetime.now(UTC),
    )
    return xr.open_dataset(result.local_path)


def _acquire_chunked_dataset(
    *,
    manifest_path: Path,
    raw_dir: Path,
    product_id: str,
    dataset_id: str,
    variables: list,
    bbox_wgs84: tuple,
    depth_range: tuple | None,
    chunk_ranges: list,
    evidence_role: str,
    temporal_resolution: str,
) -> tuple[xr.Dataset | None, metocean_acquisition.TemporalDeduplicationResult | None]:
    """Acquire every chunk (resuming from the manifest), then open+concat+dedup them.

    Never re-downloads a chunk whose exact identity (product/dataset/
    variables/bbox/depths/start/end) is already manifested with its file
    still on disk (Section 15). Adjacent chunks can share their boundary
    instant (Copernicus subset requests are inclusive of `end_datetime`) --
    `deduplicate_time_coordinate` collapses that overlap here, at chunk
    assembly, before the dataset is ever normalized (MAR-009B).
    """

    requested_bbox = list(bbox_wgs84)
    requested_depths = list(depth_range) if depth_range else None
    chunk_paths: list[Path] = []

    for chunk_start, chunk_end in chunk_ranges:
        existing = metocean_acquisition.already_acquired(
            manifest_path,
            product_id=product_id,
            dataset_id=dataset_id,
            variables=variables,
            requested_bbox=requested_bbox,
            requested_depths=requested_depths,
            requested_start=chunk_start.isoformat(),
            requested_end=chunk_end.isoformat(),
        )
        if existing is not None:
            chunk_paths.append(Path(existing["local_path"]))
            continue

        result = copernicus.subset_dataset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=bbox_wgs84[0],
            maximum_longitude=bbox_wgs84[2],
            minimum_latitude=bbox_wgs84[1],
            maximum_latitude=bbox_wgs84[3],
            start_datetime=chunk_start,
            end_datetime=chunk_end,
            minimum_depth=depth_range[0] if depth_range else None,
            maximum_depth=depth_range[1] if depth_range else None,
            output_directory=raw_dir,
            output_filename=f"{dataset_id}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.nc",
        )
        with xr.open_dataset(result.local_path) as chunk_ds:
            actual_start = (
                str(chunk_ds["time"].min().values) if "time" in chunk_ds.variables else None
            )
            actual_end = (
                str(chunk_ds["time"].max().values) if "time" in chunk_ds.variables else None
            )

        metocean_acquisition.record_acquisition(
            manifest_path,
            provider="Copernicus Marine",
            product_id=product_id,
            dataset_id=dataset_id,
            evidence_role=evidence_role,
            variables=variables,
            requested_bbox=requested_bbox,
            requested_depths=requested_depths,
            requested_start=chunk_start,
            requested_end=chunk_end,
            actual_start=actual_start,
            actual_end=actual_end,
            temporal_resolution=temporal_resolution,
            local_path=result.local_path,
            toolbox_version=copernicus.toolbox_version(),
            licence=None,
            downloaded_at=datetime.now(UTC),
        )
        chunk_paths.append(result.local_path)

    if not chunk_paths:
        return None, None
    if len(chunk_paths) == 1:
        ds = xr.open_dataset(chunk_paths[0])
    else:
        ds = xr.open_mfdataset([str(p) for p in chunk_paths], combine="by_coords")
    deduped_ds, dedup_result = metocean_acquisition.deduplicate_time_coordinate(ds)
    return deduped_ds, dedup_result


def _cmd_build_metocean_evidence(args: argparse.Namespace) -> int:
    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    _pipeline_gpkg_path, aoi_gpkg_path, chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    chainage_bathymetry_path = study_dir / "bathymetry" / "chainage_bathymetry.parquet"

    if not aoi_gpkg_path.exists() or not chainage_gpkg_path.exists():
        print(
            f"error: no canonical AOI/chainage found under {aoi_gpkg_path.parent}; "
            "run build-aoi and build-chainage first",
            file=sys.stderr,
        )
        return 1

    try:
        aoi_gdf = gpd.read_file(aoi_gpkg_path, layer="study_aoi")
        chainage_gdf = gpd.read_file(chainage_gpkg_path, layer="chainage_points")
    except Exception as exc:  # noqa: BLE001 -- one clear message regardless of the underlying cause
        print(f"error: could not load canonical AOI/chainage: {exc}", file=sys.stderr)
        return 1

    canonical_depth_df = None
    if chainage_bathymetry_path.exists():
        canonical_depth_df = pd.read_parquet(chainage_bathymetry_path)[
            ["station_index", "depth_lat_m"]
        ]

    working_crs = config.crs.horizontal
    # A modest extra buffer beyond the AOI so the model bbox request
    # comfortably contains at least one real wet cell even near a coastline.
    request_buffer_m = 5000.0
    aoi_geom_working = unary_union(aoi_gdf.geometry).buffer(request_buffer_m)
    aoi_bbox_wgs84 = tuple(
        float(v)
        for v in gpd.GeoSeries([aoi_geom_working], crs=working_crs).to_crs("EPSG:4326").total_bounds
    )
    chainage_points_working = chainage_gdf.to_crs(working_crs).geometry

    print("Resolving live Copernicus Marine dataset ids...")
    try:
        primary_current_dataset_id = copernicus.confirm_live_dataset_id(
            copernicus.PRIMARY_CURRENT_PRODUCT_ID, copernicus.PRIMARY_CURRENT_DATASET_ID
        )
        long_term_current_dataset_id = copernicus.confirm_live_dataset_id(
            copernicus.LONG_TERM_CURRENT_PRODUCT_ID, copernicus.LONG_TERM_CURRENT_DATASET_ID
        )
        wave_dataset_id = copernicus.confirm_live_dataset_id(
            copernicus.WAVE_PRODUCT_ID, copernicus.WAVE_DATASET_ID
        )
    except copernicus.CopernicusDatasetNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        copernicus.ensure_authenticated()
    except copernicus.CopernicusAuthenticationRequiredError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = config.paths.processed_dir / pipeline_id.lower() / "metocean"
    raw_dir = config.paths.raw_dir / "metocean" / "copernicus"
    manifest_path = metocean_interim_dir / "copernicus_acquisition_manifest.json"

    now_utc = datetime.now(UTC)
    historical_cutoff = metocean_acquisition.compute_historical_cutoff(now_utc)

    # --- static bathymetry/mask for each product -------------------------
    primary_static_ds = _acquire_static_dataset(
        manifest_path=manifest_path,
        raw_dir=raw_dir / "statics",
        product_id=copernicus.PRIMARY_CURRENT_PRODUCT_ID,
        dataset_id=copernicus.PRIMARY_CURRENT_STATIC_DATASET_ID,
        variables=list(copernicus.PRIMARY_CURRENT_STATIC_VARIABLES),
        bbox_wgs84=aoi_bbox_wgs84,
        evidence_role=copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
    )
    long_term_static_ds = _acquire_static_dataset(
        manifest_path=manifest_path,
        raw_dir=raw_dir / "statics",
        product_id=copernicus.LONG_TERM_CURRENT_PRODUCT_ID,
        dataset_id=copernicus.LONG_TERM_CURRENT_STATIC_DATASET_ID,
        variables=["deptho"],
        bbox_wgs84=aoi_bbox_wgs84,
        evidence_role=copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
    )
    wave_static_ds = _acquire_static_dataset(
        manifest_path=manifest_path,
        raw_dir=raw_dir / "statics",
        product_id=copernicus.WAVE_PRODUCT_ID,
        dataset_id=copernicus.WAVE_STATIC_DATASET_ID,
        variables=list(copernicus.WAVE_STATIC_VARIABLES),
        bbox_wgs84=aoi_bbox_wgs84,
        evidence_role=copernicus.PRIMARY_WAVE_CLIMATE_ROLE,
    )

    # --- support-node identification and chainage mapping (Sections 4-5) -
    primary_nodes = metocean_evidence.identify_wet_grid_cells(
        primary_static_ds["longitude"].to_numpy(),
        primary_static_ds["latitude"].to_numpy(),
        _mask_surface_slice(primary_static_ds["mask"]).to_numpy(),
        "current",
    )
    long_term_nodes = metocean_evidence.identify_wet_grid_cells(
        long_term_static_ds["longitude"].to_numpy(),
        long_term_static_ds["latitude"].to_numpy(),
        np.isfinite(long_term_static_ds["deptho"].to_numpy()),
        "current_lt",
    )
    wave_nodes = metocean_evidence.identify_wet_grid_cells(
        wave_static_ds["longitude"].to_numpy(),
        wave_static_ds["latitude"].to_numpy(),
        _mask_surface_slice(wave_static_ds["mask"]).to_numpy(),
        "wave",
    )

    primary_mapping = metocean_evidence.map_points_to_nearest_node(
        chainage_points_working, primary_nodes, working_crs
    )
    long_term_mapping = metocean_evidence.map_points_to_nearest_node(
        chainage_points_working, long_term_nodes, working_crs
    )
    wave_mapping = metocean_evidence.map_points_to_nearest_node(
        chainage_points_working, wave_nodes, working_crs
    )

    primary_bathymetry_by_node = {
        node.node_id: float(
            primary_static_ds["deptho"].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()
        )
        for node in primary_nodes
        if node.node_id in set(primary_mapping["node_id"].dropna())
    }
    primary_deptho_lev_by_node = {
        node.node_id: float(
            primary_static_ds["deptho_lev"]
            .isel(latitude=node.grid_j, longitude=node.grid_i)
            .to_numpy()
        )
        for node in primary_nodes
        if node.node_id in set(primary_mapping["node_id"].dropna())
    }
    wave_bathymetry_by_node = {
        node.node_id: float(
            wave_static_ds["deptho"].isel(latitude=node.grid_j, longitude=node.grid_i).to_numpy()
        )
        for node in wave_nodes
        if node.node_id in set(wave_mapping["node_id"].dropna())
    }

    primary_node_table = metocean_evidence.build_support_node_table(
        primary_nodes,
        primary_mapping["node_id"],
        primary_mapping["distance_m"],
        model_bathymetry_by_node_id=primary_bathymetry_by_node,
        deptho_lev_by_node_id=primary_deptho_lev_by_node,
        source_product=copernicus.PRIMARY_CURRENT_PRODUCT_ID,
        source_dataset=primary_current_dataset_id,
        evidence_role=copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
    )

    # Only chainage-USED nodes are ever normalized into a time series
    # (MAR-009A, Section 7) -- never every wet cell in the request bbox.
    used_primary_nodes = [n for n in primary_nodes if n.node_id in primary_bathymetry_by_node]
    used_long_term_node_ids = set(long_term_mapping["node_id"].dropna())
    used_long_term_nodes = [n for n in long_term_nodes if n.node_id in used_long_term_node_ids]
    used_wave_node_ids = set(wave_mapping["node_id"].dropna())
    used_wave_nodes = [n for n in wave_nodes if n.node_id in used_wave_node_ids]

    # --- capture OLD canonical outputs for the old-vs-corrected comparison
    # (Sections 12, 13, 19) -- read BEFORE anything is overwritten below.
    old_primary_current_df = _read_existing_parquet(
        metocean_interim_dir / "current_primary_hourly.parquet"
    )
    # The node-count regression this ticket fixes (Section 7/8) shows up in
    # the TIME SERIES files (every wet bbox cell was normalized), not the
    # support-node table (already correctly used-only via
    # `build_support_node_table`) -- so the OLD comparison baseline must be
    # read from the same hourly/3-hourly files, not the node table.
    old_wave_df = _read_existing_parquet(metocean_interim_dir / "wave_3hourly.parquet")
    old_long_term_current_df = _read_existing_parquet(
        metocean_interim_dir / "current_long_term_surface_hourly.parquet"
    )

    def _safe_old_stats(compute_fn, df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Old (pre-fix) stats for the comparison report only -- never load-bearing.

        The OLD on-disk data predates MAR-009B's chunk-boundary dedup, so it
        can itself trip the new strict completeness check
        (`TemporalCompletenessError`) -- that is an expected, informative
        outcome (it demonstrates the pre-fix defect), not a reason to abort
        this run. Only the NEW canonical computation must remain a hard
        failure.
        """

        if df.empty:
            return pd.DataFrame()
        try:
            return compute_fn(df)
        except metocean_evidence.TemporalCompletenessError:
            print(
                f"note: old (pre-fix) {label} data itself exceeds 100% completeness "
                "(the exact defect this ticket fixes) -- old comparison values for "
                f"{label} are reported as unavailable rather than computed",
                file=sys.stderr,
            )
            return pd.DataFrame()

    old_current_stats = _safe_old_stats(
        metocean_evidence.compute_current_node_statistics, old_primary_current_df, "primary current"
    )
    old_wave_stats = _safe_old_stats(
        metocean_evidence.compute_wave_node_statistics, old_wave_df, "wave"
    )
    old_long_term_stats = _safe_old_stats(
        metocean_evidence.compute_long_term_surface_current_statistics,
        old_long_term_current_df,
        "long-term surface current",
    )

    # --- primary current acquisition (Section 6, 15) ---------------------
    primary_chunk_ranges = metocean_acquisition.generate_monthly_chunks(
        # The rolling analysis/forecast catalogue's own available start is
        # discovered live, never hard-coded (Section 6).
        _dataset_start_or(
            copernicus.get_dataset_time_range_ms(primary_current_dataset_id), now_utc
        ),
        historical_cutoff,
    )
    try:
        primary_current_ds, primary_temporal_dedup = _acquire_chunked_dataset(
            manifest_path=manifest_path,
            raw_dir=raw_dir / "current_primary",
            product_id=copernicus.PRIMARY_CURRENT_PRODUCT_ID,
            dataset_id=primary_current_dataset_id,
            variables=list(copernicus.PRIMARY_CURRENT_VARIABLES),
            bbox_wgs84=aoi_bbox_wgs84,
            depth_range=None,
            chunk_ranges=primary_chunk_ranges,
            evidence_role=copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
            temporal_resolution="hourly_instantaneous",
        )
    except metocean_acquisition.DuplicateTimestampConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    static_depth_mask_used = False
    static_mask_by_node_id: dict[str, np.ndarray] | None = None
    unreconciled_primary_node_ids: list[str] = []
    primary_current_df = pd.DataFrame()
    if primary_current_ds is not None:
        static_depth_mask_used = "depth" in primary_static_ds.coords and (
            metocean_evidence.check_depth_coordinate_alignment(
                primary_static_ds["depth"].to_numpy(), primary_current_ds["depth"].to_numpy()
            )
        )
        if static_depth_mask_used:
            static_mask_by_node_id = {
                node.node_id: primary_static_ds["mask"]
                .isel(latitude=node.grid_j, longitude=node.grid_i)
                .to_numpy()
                for node in used_primary_nodes
            }
        else:
            print(
                "warning: static mask depth coordinate does not align with the dynamic "
                "current dataset's own depth coordinate -- proceeding with the bathymetry-"
                "depth eligibility constraint alone (Section 3); static_depth_mask_used=False",
                file=sys.stderr,
            )

        primary_current_df, unreconciled_primary_node_ids = (
            metocean_evidence.normalize_primary_current(
                primary_current_ds,
                nodes=used_primary_nodes,
                model_bathymetry_by_node_id=primary_bathymetry_by_node,
                static_mask_by_node_id=static_mask_by_node_id,
                source_dataset=primary_current_dataset_id,
                evidence_role=copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
            )
        )
        if unreconciled_primary_node_ids:
            print(
                f"warning: {len(unreconciled_primary_node_ids)} primary current support "
                "node(s) could not be reconciled against the dynamic dataset grid and were "
                f"excluded from normalization: {unreconciled_primary_node_ids}",
                file=sys.stderr,
            )

    # --- long-term surface current acquisition (Sections 11, 31) ---------
    long_term_start_ms = copernicus.get_dataset_time_range_ms(long_term_current_dataset_id)
    long_term_chunk_ranges = metocean_acquisition.generate_yearly_chunks(
        _dataset_start_or(long_term_start_ms, now_utc), historical_cutoff
    )
    try:
        long_term_current_ds, long_term_temporal_dedup = _acquire_chunked_dataset(
            manifest_path=manifest_path,
            raw_dir=raw_dir / "current_long_term_surface",
            product_id=copernicus.LONG_TERM_CURRENT_PRODUCT_ID,
            dataset_id=long_term_current_dataset_id,
            variables=list(copernicus.LONG_TERM_CURRENT_VARIABLES),
            bbox_wgs84=aoi_bbox_wgs84,
            depth_range=None,
            chunk_ranges=long_term_chunk_ranges,
            evidence_role=copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
            temporal_resolution="hourly_instantaneous",
        )
    except metocean_acquisition.DuplicateTimestampConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    unreconciled_long_term_node_ids: list[str] = []
    long_term_current_df = pd.DataFrame()
    if long_term_current_ds is not None:
        long_term_current_df, unreconciled_long_term_node_ids = (
            metocean_evidence.normalize_long_term_surface_current(
                long_term_current_ds,
                nodes=used_long_term_nodes,
                source_dataset=long_term_current_dataset_id,
                evidence_role=copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
            )
        )
        if unreconciled_long_term_node_ids:
            print(
                f"warning: {len(unreconciled_long_term_node_ids)} long-term surface current "
                "support node(s) could not be reconciled against the dynamic dataset grid and "
                f"were excluded from normalization: {unreconciled_long_term_node_ids}",
                file=sys.stderr,
            )

    # --- wave reanalysis acquisition (Section 12) -------------------------
    wave_start_ms = copernicus.get_dataset_time_range_ms(wave_dataset_id)
    wave_chunk_ranges = metocean_acquisition.generate_yearly_chunks(
        _dataset_start_or(wave_start_ms, now_utc), historical_cutoff
    )
    try:
        wave_ds, wave_temporal_dedup = _acquire_chunked_dataset(
            manifest_path=manifest_path,
            raw_dir=raw_dir / "wave_reanalysis",
            product_id=copernicus.WAVE_PRODUCT_ID,
            dataset_id=wave_dataset_id,
            variables=list(copernicus.WAVE_VARIABLES),
            bbox_wgs84=aoi_bbox_wgs84,
            depth_range=None,
            chunk_ranges=wave_chunk_ranges,
            evidence_role=copernicus.PRIMARY_WAVE_CLIMATE_ROLE,
            temporal_resolution="3hourly_instantaneous",
        )
    except metocean_acquisition.DuplicateTimestampConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    unreconciled_wave_node_ids: list[str] = []
    wave_df = pd.DataFrame()
    if wave_ds is not None:
        wave_df, unreconciled_wave_node_ids = metocean_evidence.normalize_wave(
            wave_ds, nodes=used_wave_nodes, source_dataset=wave_dataset_id
        )
        if unreconciled_wave_node_ids:
            print(
                f"warning: {len(unreconciled_wave_node_ids)} wave support node(s) could not "
                "be reconciled against the dynamic dataset grid and were excluded from "
                f"normalization: {unreconciled_wave_node_ids}",
                file=sys.stderr,
            )

    # --- descriptive statistics (Sections 24, 25, 27) ---------------------
    try:
        current_stats = metocean_evidence.compute_current_node_statistics(primary_current_df)
        long_term_stats = metocean_evidence.compute_long_term_surface_current_statistics(
            long_term_current_df
        )
        wave_stats = metocean_evidence.compute_wave_node_statistics(wave_df)
    except metocean_evidence.TemporalCompletenessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    annual_max_hs = metocean_evidence.compute_annual_max_hs(wave_df)

    primary_start = primary_current_df["time_utc"].min() if not primary_current_df.empty else None
    primary_end = primary_current_df["time_utc"].max() if not primary_current_df.empty else None
    short_window_ratios = metocean_evidence.compute_short_window_surface_context_ratio(
        long_term_current_df, primary_start, primary_end
    )

    # --- new integrity diagnostics (MAR-009A, Sections 4, 9, 10) ----------
    below_bed_diagnostics = metocean_evidence.compute_below_bed_diagnostics(primary_current_df)
    primary_current_route_summary = metocean_evidence.compute_primary_current_route_summary(
        primary_current_df
    )
    primary_distance_diagnostics = metocean_evidence.compute_distance_diagnostics(primary_mapping)
    long_term_distance_diagnostics = metocean_evidence.compute_distance_diagnostics(
        long_term_mapping
    )
    wave_distance_diagnostics = metocean_evidence.compute_distance_diagnostics(wave_mapping)

    # --- temporal integrity QA (MAR-009B, Sections 4, 5) -------------------
    # Merges the chunk-assembly-level dedup diagnostics (raw/unique/removed
    # timestamp counts, identical for every node of a product since they
    # share one dynamic time axis) with a defensive post-normalization
    # re-check (unique/monotonic/no-duplicate-rows) -- reported separately
    # for all three products, never folded together.
    def _temporal_qa(
        dedup_result: metocean_acquisition.TemporalDeduplicationResult | None,
        df: pd.DataFrame,
        *,
        time_column: str,
        node_column: str,
    ) -> dict[str, Any]:
        qa: dict[str, Any] = {
            "raw_time_count": dedup_result.raw_time_count if dedup_result else None,
            "unique_time_count": dedup_result.unique_time_count if dedup_result else None,
            "duplicate_boundary_timestamp_count": (
                dedup_result.duplicate_boundary_timestamp_count if dedup_result else None
            ),
        }
        qa.update(
            metocean_evidence.validate_temporal_integrity(
                df, time_column=time_column, node_column=node_column
            )
        )
        return qa

    primary_temporal_qa = _temporal_qa(
        primary_temporal_dedup,
        primary_current_df,
        time_column="time_utc",
        node_column="current_node_id",
    )
    long_term_temporal_qa = _temporal_qa(
        long_term_temporal_dedup,
        long_term_current_df,
        time_column="time_utc",
        node_column="current_lt_node_id",
    )
    wave_temporal_qa = _temporal_qa(
        wave_temporal_dedup, wave_df, time_column="time_utc", node_column="wave_node_id"
    )

    long_term_node_table = metocean_evidence.build_support_node_table(
        long_term_nodes,
        long_term_mapping["node_id"],
        long_term_mapping["distance_m"],
        source_product=copernicus.LONG_TERM_CURRENT_PRODUCT_ID,
        source_dataset=long_term_current_dataset_id,
        evidence_role=copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
    )
    wave_node_table = metocean_evidence.build_support_node_table(
        wave_nodes,
        wave_mapping["node_id"],
        wave_mapping["distance_m"],
        model_bathymetry_by_node_id=wave_bathymetry_by_node,
        source_product=copernicus.WAVE_PRODUCT_ID,
        source_dataset=wave_dataset_id,
        evidence_role=copernicus.PRIMARY_WAVE_CLIMATE_ROLE,
    )

    # --- chainage evidence assembly (Section 23) --------------------------
    chainage_metocean_df = metocean_evidence.build_chainage_metocean_evidence(
        chainage_gdf=chainage_gdf,
        canonical_depth_df=canonical_depth_df,
        current_mapping=primary_mapping,
        current_stats=current_stats,
        current_node_bathymetry=primary_bathymetry_by_node,
        long_term_mapping=long_term_mapping,
        long_term_stats=long_term_stats,
        wave_mapping=wave_mapping,
        wave_stats=wave_stats,
        wave_node_bathymetry=wave_bathymetry_by_node,
    )

    # --- write outputs -----------------------------------------------------
    primary_nodes_path = metocean_evidence.write_parquet(
        primary_node_table, metocean_interim_dir / "current_primary_support_nodes.parquet"
    )
    primary_current_path = metocean_evidence.write_parquet(
        primary_current_df, metocean_interim_dir / "current_primary_hourly.parquet"
    )
    long_term_nodes_path = metocean_evidence.write_parquet(
        long_term_node_table,
        metocean_interim_dir / "current_long_term_surface_support_nodes.parquet",
    )
    long_term_current_path = metocean_evidence.write_parquet(
        long_term_current_df, metocean_interim_dir / "current_long_term_surface_hourly.parquet"
    )
    wave_nodes_path = metocean_evidence.write_parquet(
        wave_node_table, metocean_interim_dir / "wave_support_nodes.parquet"
    )
    wave_path = metocean_evidence.write_parquet(
        wave_df, metocean_interim_dir / "wave_3hourly.parquet"
    )
    annual_max_hs_path = metocean_evidence.write_parquet(
        annual_max_hs, metocean_interim_dir / "wave_annual_max_hs.parquet"
    )
    chainage_path = metocean_evidence.write_parquet(
        chainage_metocean_df, metocean_processed_dir / "chainage_metocean_evidence.parquet"
    )

    metadata_path = metocean_processed_dir / "metocean_evidence_metadata.json"
    metocean_evidence.write_metocean_evidence_metadata(
        metadata={
            "products": {
                "primary_current": {
                    "product_id": copernicus.PRIMARY_CURRENT_PRODUCT_ID,
                    "dataset_id": primary_current_dataset_id,
                    "static_dataset_id": copernicus.PRIMARY_CURRENT_STATIC_DATASET_ID,
                    "evidence_role": copernicus.PRIMARY_CURRENT_EVIDENCE_ROLE,
                    "temporal_resolution": "hourly_instantaneous",
                    "vertical_semantics": (
                        "deepest_valid_standard_level_current -- NOT native bottom-cell current"
                    ),
                },
                "long_term_surface_current": {
                    "product_id": copernicus.LONG_TERM_CURRENT_PRODUCT_ID,
                    "dataset_id": long_term_current_dataset_id,
                    "static_dataset_id": copernicus.LONG_TERM_CURRENT_STATIC_DATASET_ID,
                    "evidence_role": copernicus.LONG_TERM_SURFACE_CURRENT_CONTEXT_ROLE,
                    "temporal_resolution": "hourly_instantaneous",
                    "forbidden_daily_dataset_id": (
                        copernicus.LONG_TERM_CURRENT_FORBIDDEN_DAILY_DATASET_ID
                    ),
                },
                "wave": {
                    "product_id": copernicus.WAVE_PRODUCT_ID,
                    "dataset_id": wave_dataset_id,
                    "static_dataset_id": copernicus.WAVE_STATIC_DATASET_ID,
                    "evidence_role": copernicus.PRIMARY_WAVE_CLIMATE_ROLE,
                    "temporal_resolution": "3hourly_instantaneous",
                },
            },
            "retrieval_timestamp": now_utc.isoformat(),
            "historical_cutoff": historical_cutoff.isoformat(),
            "current_direction_convention": "current_direction_to_deg: degrees clockwise from "
            "true north, vector points TOWARD that bearing",
            "wave_direction_convention": "wave_mean_direction_from_deg: degrees the waves "
            "travel FROM; wave_mean_direction_to_deg = (from + 180) %% 360 is derived for "
            "convenience only",
            "support_node_mapping_method": "nearest wet model grid cell (no bilinear "
            "interpolation of data or masks)",
            "canonical_model_bathymetry_vertical_datums_not_harmonised": True,
            "dynamic_grid_coordinate_reconciliation_method": (
                "each support node's canonical lon/lat is re-resolved against the dynamic "
                "dataset's own coordinate arrays (nearest cell, refused beyond "
                f"{metocean_evidence.GRID_RECONCILIATION_TOLERANCE_FRACTION:.0%} of that "
                "axis's median grid spacing) -- static dataset grid indices are never reused "
                "directly against a dynamic dataset (MAR-009A Section 6)"
            ),
            "static_dynamic_coordinate_match_status": {
                "primary_current": (
                    "all_used_nodes_reconciled"
                    if not unreconciled_primary_node_ids
                    else f"{len(unreconciled_primary_node_ids)}_node(s)_unreconciled"
                ),
                "long_term_surface_current": (
                    "all_used_nodes_reconciled"
                    if not unreconciled_long_term_node_ids
                    else f"{len(unreconciled_long_term_node_ids)}_node(s)_unreconciled"
                ),
                "wave": (
                    "all_used_nodes_reconciled"
                    if not unreconciled_wave_node_ids
                    else f"{len(unreconciled_wave_node_ids)}_node(s)_unreconciled"
                ),
            },
            "primary_current_vertical_eligibility_rule": (
                "a standard depth is only eligible when uo AND vo are finite AND "
                "depth_m <= model_bathymetry_m + tolerance AND, where the static mask's depth "
                "coordinate is alignable, the mask cell is wet (MAR-009A Sections 2-3)"
            ),
            "static_depth_mask_used": static_depth_mask_used,
            "below_bed_finite_value_diagnostic_summary": {
                "below_model_bed_finite_candidate_count": (
                    int(below_bed_diagnostics["below_model_bed_finite_candidate_count"].sum())
                    if not below_bed_diagnostics.empty
                    else 0
                ),
                "timestamps_with_below_bed_finite_candidates": (
                    int(below_bed_diagnostics["timestamps_with_below_bed_finite_candidates"].sum())
                    if not below_bed_diagnostics.empty
                    else 0
                ),
                "max_below_bed_candidate_depth_m": (
                    float(below_bed_diagnostics["max_below_bed_candidate_depth_m"].max())
                    if not below_bed_diagnostics.empty
                    and below_bed_diagnostics["max_below_bed_candidate_depth_m"].notna().any()
                    else None
                ),
            },
            "only_chainage_used_support_nodes_normalized": True,
            "temporal_qa": {
                "primary_current": primary_temporal_qa,
                "long_term_surface_current": long_term_temporal_qa,
                "wave": wave_temporal_qa,
            },
            "raw_acquisition_manifest_path": str(manifest_path),
            "limitations": [
                "primary current record is only the rolling available historical analysis "
                "period, not a multi-decadal near-bed climatology",
                "deepest valid standard level is not the model native bottom cell",
                "long-term 7 km hourly current is surface current context only",
                "wave data are model reanalysis, not local buoy observations",
                "model bathymetry and canonical LAT bathymetry are not vertically harmonised",
            ],
            "no_physics_yet_statement": (
                "This ticket produces forcing evidence only -- no bed shear stress, Shields "
                "parameter, sediment mobility, erosion/deposition, scour, free-span, fatigue, "
                "or risk scoring is computed anywhere here."
            ),
            "outputs": {
                "current_primary_support_nodes": str(primary_nodes_path),
                "current_primary_hourly": str(primary_current_path),
                "current_long_term_surface_support_nodes": str(long_term_nodes_path),
                "current_long_term_surface_hourly": str(long_term_current_path),
                "wave_support_nodes": str(wave_nodes_path),
                "wave_3hourly": str(wave_path),
                "wave_annual_max_hs": str(annual_max_hs_path),
                "chainage_metocean_evidence": str(chainage_path),
            },
        },
        output_path=metadata_path,
    )

    print(f"Primary current support nodes: {len(primary_node_table)} -> {primary_nodes_path}")
    print(f"Long-term current support nodes: {len(long_term_node_table)} -> {long_term_nodes_path}")
    print(f"Wave support nodes: {len(wave_node_table)} -> {wave_nodes_path}")
    print(f"Chainage metocean evidence: {len(chainage_metocean_df)} station(s) -> {chainage_path}")
    print(f"Metadata: {metadata_path}")
    print()

    # --- old-vs-corrected comparison (Sections 12, 13, 19) ----------------
    # `old_*` values were read from the ON-DISK canonical outputs BEFORE this
    # run overwrote them; a run against an empty/absent prior output simply
    # reports `old=None` rather than fabricating a baseline.
    old_vs_new_comparison: dict[str, tuple[Any, Any]] = {
        "primary_current_speed_mean_m_s": (
            float(old_current_stats["current_speed_mean_m_s"].mean())
            if not old_current_stats.empty
            else None,
            float(current_stats["current_speed_mean_m_s"].mean())
            if not current_stats.empty
            else None,
        ),
        "primary_current_speed_p95_m_s": (
            float(old_current_stats["current_speed_p95_m_s"].max())
            if not old_current_stats.empty
            else None,
            float(current_stats["current_speed_p95_m_s"].max())
            if not current_stats.empty
            else None,
        ),
        "primary_current_speed_p99_m_s": (
            float(old_current_stats["current_speed_p99_m_s"].max())
            if not old_current_stats.empty
            else None,
            float(current_stats["current_speed_p99_m_s"].max())
            if not current_stats.empty
            else None,
        ),
        "primary_current_speed_max_m_s": (
            float(old_current_stats["current_speed_max_m_s"].max())
            if not old_current_stats.empty
            else None,
            float(current_stats["current_speed_max_m_s"].max())
            if not current_stats.empty
            else None,
        ),
        "wave_time_series_distinct_node_count": (
            int(old_wave_df["wave_node_id"].nunique()) if not old_wave_df.empty else None,
            len(wave_stats),
        ),
        "long_term_current_time_series_distinct_node_count": (
            int(old_long_term_current_df["current_lt_node_id"].nunique())
            if not old_long_term_current_df.empty
            else None,
            len(long_term_stats),
        ),
        "long_term_current_speed_p95_m_s": (
            float(old_long_term_stats["surface_current_speed_p95_m_s"].max())
            if not old_long_term_stats.empty
            else None,
            float(long_term_stats["surface_current_speed_p95_m_s"].max())
            if not long_term_stats.empty
            else None,
        ),
        "long_term_current_speed_p99_m_s": (
            float(old_long_term_stats["surface_current_speed_p99_m_s"].max())
            if not old_long_term_stats.empty
            else None,
            float(long_term_stats["surface_current_speed_p99_m_s"].max())
            if not long_term_stats.empty
            else None,
        ),
        "long_term_current_speed_max_m_s": (
            float(old_long_term_stats["surface_current_speed_max_m_s"].max())
            if not old_long_term_stats.empty
            else None,
            float(long_term_stats["surface_current_speed_max_m_s"].max())
            if not long_term_stats.empty
            else None,
        ),
        "wave_hs_mean_m": (
            float(old_wave_stats["hs_mean_m"].mean()) if not old_wave_stats.empty else None,
            float(wave_stats["hs_mean_m"].mean()) if not wave_stats.empty else None,
        ),
        "wave_hs_p95_m": (
            float(old_wave_stats["hs_p95_m"].max()) if not old_wave_stats.empty else None,
            float(wave_stats["hs_p95_m"].max()) if not wave_stats.empty else None,
        ),
        "wave_hs_p99_m": (
            float(old_wave_stats["hs_p99_m"].max()) if not old_wave_stats.empty else None,
            float(wave_stats["hs_p99_m"].max()) if not wave_stats.empty else None,
        ),
        "wave_hs_max_m": (
            float(old_wave_stats["hs_max_m"].max()) if not old_wave_stats.empty else None,
            float(wave_stats["hs_max_m"].max()) if not wave_stats.empty else None,
        ),
        "wave_tp_median_s": (
            float(old_wave_stats["tp_median_s"].median()) if not old_wave_stats.empty else None,
            float(wave_stats["tp_median_s"].median()) if not wave_stats.empty else None,
        ),
        "wave_tp_p95_s": (
            float(old_wave_stats["tp_p95_s"].max()) if not old_wave_stats.empty else None,
            float(wave_stats["tp_p95_s"].max()) if not wave_stats.empty else None,
        ),
    }

    metocean_evidence.print_metocean_evidence_report(
        primary_current_stats=current_stats,
        primary_current_route_summary=primary_current_route_summary,
        primary_current_canonical_row_count=len(primary_current_df),
        primary_temporal_qa=primary_temporal_qa,
        below_bed_diagnostics=below_bed_diagnostics,
        primary_distance_diagnostics=primary_distance_diagnostics,
        long_term_stats=long_term_stats,
        long_term_temporal_qa=long_term_temporal_qa,
        long_term_distance_diagnostics=long_term_distance_diagnostics,
        short_window_ratios=short_window_ratios,
        wave_stats=wave_stats,
        wave_temporal_qa=wave_temporal_qa,
        wave_distance_diagnostics=wave_distance_diagnostics,
        primary_current_actual_start=primary_start,
        primary_current_actual_end=primary_end,
        long_term_actual_start=long_term_current_df["time_utc"].min()
        if not long_term_current_df.empty
        else None,
        long_term_actual_end=long_term_current_df["time_utc"].max()
        if not long_term_current_df.empty
        else None,
        wave_actual_start=wave_df["time_utc"].min() if not wave_df.empty else None,
        wave_actual_end=wave_df["time_utc"].max() if not wave_df.empty else None,
        old_vs_new_comparison=old_vs_new_comparison,
    )
    return 0


def _none_if_nan(value: Any) -> Any:
    """`None` for a missing/NaN scalar, else `float(value)` -- JSON/dataclass-safe."""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _cmd_build_current_normalization(args: argparse.Namespace) -> int:
    """MAR-010: current-only 1 m log-profile normalization sensitivity + reference map.

    Requires the MAR-009B canonical primary-current outputs to already
    exist on disk; performs NO network request and NO Copernicus
    acquisition (Section 12) -- everything it reads was already downloaded
    and normalized by `build-metocean-evidence`.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = study_dir / "metocean"
    maps_dir = study_dir / "maps"

    primary_nodes_path = metocean_interim_dir / "current_primary_support_nodes.parquet"
    primary_hourly_path = metocean_interim_dir / "current_primary_hourly.parquet"
    chainage_metocean_path = metocean_processed_dir / "chainage_metocean_evidence.parquet"
    required_paths = (
        pipeline_gpkg_path,
        primary_nodes_path,
        primary_hourly_path,
        chainage_metocean_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-chainage and "
            f"build-metocean-evidence first: {missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    primary_nodes_df = pd.read_parquet(primary_nodes_path)
    primary_hourly_df = pd.read_parquet(primary_hourly_path)
    chainage_metocean_df = pd.read_parquet(chainage_metocean_path)

    # --- MAR-010 core: log-profile 1 m sensitivity normalization -----------
    try:
        hourly_sensitivity_df = current_normalization.build_current_only_1m_sensitivity_hourly(
            primary_hourly_df
        )
        sensitivity_stats_df = current_normalization.compute_current_only_1m_sensitivity_stats(
            hourly_sensitivity_df
        )
        current_stats = metocean_evidence.compute_current_node_statistics(primary_hourly_df)
    except (
        current_normalization.NormalizationCompletenessError,
        metocean_evidence.TemporalCompletenessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sensitivity_envelope_df = current_normalization.compute_current_only_1m_sensitivity_envelope(
        sensitivity_stats_df
    )
    sensitivity_stats_df = sensitivity_stats_df.merge(
        sensitivity_envelope_df, on="current_node_id", how="left"
    )
    vertical_domain_summary = current_normalization.compute_vertical_domain_summary(
        hourly_sensitivity_df
    )

    # --- per-node reference attributes for segment/map assembly ------------
    node_bathymetry_by_id = primary_nodes_df.set_index("node_id")["model_bathymetry_m"].to_dict()
    current_stats_by_id = current_stats.set_index("current_node_id")
    envelope_by_id = sensitivity_envelope_df.set_index("current_node_id")

    node_attributes_by_id: dict[str, current_map.NodeReferenceAttributes] = {}
    for node_id, stats_row in current_stats_by_id.iterrows():
        bathymetry_m = node_bathymetry_by_id.get(node_id)
        representative_depth_m = stats_row.get("representative_sample_depth_m")
        reference_height_m = (
            bathymetry_m - representative_depth_m
            if bathymetry_m is not None and pd.notna(representative_depth_m)
            else None
        )
        envelope_row = envelope_by_id.loc[node_id] if node_id in envelope_by_id.index else None
        node_attributes_by_id[node_id] = current_map.NodeReferenceAttributes(
            model_bathymetry_m=_none_if_nan(bathymetry_m),
            reference_height_m=_none_if_nan(reference_height_m),
            speed_mean_m_s=_none_if_nan(stats_row.get("current_speed_mean_m_s")),
            speed_p95_m_s=_none_if_nan(stats_row.get("current_speed_p95_m_s")),
            speed_p99_m_s=_none_if_nan(stats_row.get("current_speed_p99_m_s")),
            speed_max_m_s=_none_if_nan(stats_row.get("current_speed_max_m_s")),
            sensitivity_p95_min_m_s=_none_if_nan(envelope_row["speed_1m_p95_sensitivity_min_m_s"])
            if envelope_row is not None
            else None,
            sensitivity_p95_max_m_s=_none_if_nan(envelope_row["speed_1m_p95_sensitivity_max_m_s"])
            if envelope_row is not None
            else None,
            sensitivity_p95_width_m_s=_none_if_nan(
                envelope_row["speed_1m_p95_sensitivity_width_m_s"]
            )
            if envelope_row is not None
            else None,
        )

    # --- contiguous map segments ---------------------------------------------
    chainage_current_df = chainage_metocean_df[
        ["chainage_m", "current_node_id", "current_node_distance_m"]
    ].copy()
    segments_gdf = current_map.build_current_reference_segments(
        pipeline_id=pipeline_id,
        route=route,
        chainage_current_df=chainage_current_df,
        node_attributes_by_id=node_attributes_by_id,
        working_crs=working_crs,
    )
    distance_diagnostics = metocean_evidence.compute_distance_diagnostics(
        chainage_current_df.rename(columns={"current_node_distance_m": "distance_m"})
    )

    # --- write parquet/gpkg outputs ------------------------------------------
    hourly_path = metocean_evidence.write_parquet(
        hourly_sensitivity_df,
        metocean_interim_dir / "current_only_1m_sensitivity_hourly.parquet",
    )
    stats_path = metocean_evidence.write_parquet(
        sensitivity_stats_df, metocean_processed_dir / "current_only_1m_sensitivity_stats.parquet"
    )
    segments_path = current_map.write_current_reference_segments_gpkg(
        segments_gdf, metocean_processed_dir / "current_reference_segments.gpkg"
    )

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    png_path = current_map.render_reference_current_map(
        segments_gdf=segments_gdf,
        route=route,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_reference_current_forcing.png",
        background_raster_path=(
            background_raster_path if background_raster_path.exists() else None
        ),
    )
    png_dimensions = current_map.read_png_dimensions(png_path)

    # --- metadata --------------------------------------------------------------
    metadata_path = metocean_processed_dir / "current_normalization_metadata.json"
    metadata = {
        "scientific_role": current_normalization.SCIENTIFIC_ROLE,
        "target_height_above_model_bed_m": (current_normalization.TARGET_HEIGHT_ABOVE_MODEL_BED_M),
        "equation": (
            "S(z_t,z_r,z0) = [ln(z_t+z0)-ln(z0)] / [ln(z_r+z0)-ln(z0)]; "
            "uo_1m = S * uo_ref; vo_1m = S * vo_ref; "
            "speed_1m = sqrt(uo_1m^2 + vo_1m^2)"
        ),
        "roughness_scenarios_m": dict(current_normalization.ROUGHNESS_SCENARIOS_M),
        "roughness_semantics": "SENSITIVITY_SCENARIOS_NOT_SITE_SPECIFIC_BED_TRUTH",
        "vertical_domain_screen": current_normalization.VERTICAL_DOMAIN_SCREEN_FRACTION,
        "vertical_domain_screen_semantics": (
            "A conservative project data-QA validity screen (z_r_over_h_model <= 0.30) for "
            "applying this simple current-only log-profile formulation at all -- never a "
            "universal physical threshold."
        ),
        "z_r_source": "Copernicus model bathymetry - deepest physically valid standard depth",
        "canonical_LAT_bathymetry_not_used_in_vertical_scaling": True,
        "current_wave_interaction_applied": False,
        "pipeline_directionality_applied": False,
        "current_source_nominal_resolution_m": current_map.SOURCE_GRID_NOMINAL_RESOLUTION_M,
        "map_colour_variable": "current_reference_speed_p95_m_s",
        "wave_interaction_statement": (
            "Surface waves can modify the apparent roughness and mean-current profile in "
            "the bottom boundary layer. MAR-010 intentionally does not model that "
            "interaction; this is a current-only normalization sensitivity product."
        ),
        "limitations": [
            "Roughness scenarios are fixed sensitivity dimensions, not a site-specific "
            "PL854 seabed roughness estimate.",
            "The 0.30 vertical-domain screen is a conservative project heuristic, not a "
            "universal physical threshold.",
            "Wave-current bottom-boundary-layer interaction is not modelled here.",
            "Model bathymetry (used here) and canonical MAR-006 LAT bathymetry remain "
            "deliberately unharmonised.",
            "Map colours represent the native corrected reference-current p95 only, not "
            "any roughness-selected or risk value.",
        ],
        "references": [
            {
                "citation": "Soulsby, R. (1997), Dynamics of Marine Sands.",
                "doi": "10.1680/doms.25844",
            },
            {
                "citation": (
                    "Grant, W.D. & Madsen, O.S. (1979), Combined wave and current "
                    "interaction with a rough bottom."
                ),
                "doi": "10.1029/JC084iC04p01797",
            },
            {
                "citation": (
                    "Warner, J.C. et al. (2008), Development of a three-dimensional, "
                    "regional, coupled wave, current, and sediment-transport model."
                ),
                "doi": "10.1016/j.cageo.2008.02.012",
            },
        ],
        "roughness_reference_note": (
            "The roughness sensitivity values are consistent with long-standing DNV "
            "F105/F109 seabed roughness classes, but this does not claim certified "
            "compliance with the current licensed DNV editions."
        ),
        "vertical_domain_summary": vertical_domain_summary,
        "outputs": {
            "current_only_1m_sensitivity_hourly": str(hourly_path),
            "current_only_1m_sensitivity_stats": str(stats_path),
            "current_reference_segments": str(segments_path),
            "current_reference_map_png": str(png_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(
        f"Current-only 1 m sensitivity (hourly): {len(hourly_sensitivity_df)} row(s) -> "
        f"{hourly_path}"
    )
    print(
        f"Current-only 1 m sensitivity (stats): {len(sensitivity_stats_df)} row(s) -> {stats_path}"
    )
    print(f"Current reference segments: {len(segments_gdf)} section(s) -> {segments_path}")
    print(f"Reference current map: {png_path}")
    print(f"Metadata: {metadata_path}")
    print()
    current_map.print_current_normalization_report(
        vertical_domain_summary=vertical_domain_summary,
        sensitivity_stats_df=sensitivity_stats_df,
        segments_gdf=segments_gdf,
        route_used_node_count=len(current_stats),
        distance_diagnostics=distance_diagnostics,
        segments_path=segments_path,
        png_path=png_path,
        png_dimensions=png_dimensions,
    )
    return 0


def _cmd_build_wave_orbital_forcing(args: argparse.Namespace) -> int:
    """MAR-011: wave-only spectral near-bed orbital velocity + reference map.

    Requires the MAR-009B canonical wave outputs to already exist on disk;
    performs NO network request and NO Copernicus acquisition (Section 18)
    -- everything it reads was already downloaded and normalized by
    `build-metocean-evidence`. Reads no current-related file at all.
    """

    config = load_study_config(args.config)
    pipeline_id = config.pipeline.get("pipeline_id")
    if not pipeline_id:
        print(f"error: '{args.config}' has no pipeline.pipeline_id configured", file=sys.stderr)
        return 1

    pipeline_gpkg_path, _aoi_gpkg_path, _chainage_gpkg_path, interim_dir = _study_paths(
        config, pipeline_id
    )
    study_dir = config.paths.processed_dir / pipeline_id.lower()
    metocean_interim_dir = interim_dir / "metocean"
    metocean_processed_dir = study_dir / "metocean"
    maps_dir = study_dir / "maps"

    wave_hourly_path = metocean_interim_dir / "wave_3hourly.parquet"
    wave_nodes_path = metocean_interim_dir / "wave_support_nodes.parquet"
    chainage_metocean_path = metocean_processed_dir / "chainage_metocean_evidence.parquet"
    required_paths = (
        pipeline_gpkg_path,
        wave_hourly_path,
        wave_nodes_path,
        chainage_metocean_path,
    )
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(
            "error: missing required canonical output(s) -- run build-chainage and "
            f"build-metocean-evidence first: {missing}",
            file=sys.stderr,
        )
        return 1

    working_crs = config.crs.horizontal
    try:
        route, _attributes, source_crs = load_pipeline_route(pipeline_gpkg_path, pipeline_id)
    except InvalidPipelineRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if source_crs != working_crs:
        print(
            f"error: pipeline CRS {source_crs} does not match configured working CRS {working_crs}",
            file=sys.stderr,
        )
        return 1

    wave_df = pd.read_parquet(wave_hourly_path)
    wave_nodes_df = pd.read_parquet(wave_nodes_path)
    chainage_metocean_df = pd.read_parquet(chainage_metocean_path)

    # --- MAR-011 core: Soulsby & Smallman spectral orbital velocity ---------
    # `model_bathymetry_m` comes from the WAVE product's OWN static support-
    # node table (Section 2) -- never the canonical MAR-006 LAT depth, never
    # a current-product bathymetry substitute.
    wave_bathymetry_by_node = wave_nodes_df.set_index("node_id")["model_bathymetry_m"].to_dict()
    wave_df = wave_df.copy()
    wave_df["model_bathymetry_m"] = wave_df["wave_node_id"].map(wave_bathymetry_by_node)

    try:
        hourly_orbital_df = wave_orbital.build_wave_orbital_velocity_3hourly(wave_df)
        stats_df = wave_orbital.compute_wave_orbital_velocity_stats(hourly_orbital_df)
    except wave_orbital.OrbitalVelocityCompletenessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    domain_summary = wave_orbital.compute_wave_orbital_domain_summary(hourly_orbital_df)
    temporal_qa = metocean_evidence.validate_temporal_integrity(
        hourly_orbital_df, time_column="time_utc", node_column="wave_node_id"
    )
    if temporal_qa.get("duplicate_node_time_row_count"):
        print(
            "warning: duplicate (wave_node_id, time_utc) rows detected in the canonical "
            f"wave series: {temporal_qa['duplicate_node_time_row_count']}",
            file=sys.stderr,
        )

    # --- per-node reference attributes for segment/map assembly ------------
    stats_by_id = stats_df.set_index("wave_node_id")
    node_attributes_by_id: dict[str, wave_orbital_map.WaveNodeReferenceAttributes] = {}
    for node_id, row in stats_by_id.iterrows():
        node_attributes_by_id[node_id] = wave_orbital_map.WaveNodeReferenceAttributes(
            model_bathymetry_m=_none_if_nan(row.get("model_bathymetry_m")),
            hs_p95_m=_none_if_nan(row.get("hs_p95_m")),
            hs_p99_m=_none_if_nan(row.get("hs_p99_m")),
            hs_max_m=_none_if_nan(row.get("hs_max_m")),
            tm02_median_s=_none_if_nan(row.get("tm02_median_s")),
            tm02_p95_s=_none_if_nan(row.get("tm02_p95_s")),
            orbital_rms_mean_m_s=_none_if_nan(row.get("orbital_rms_mean_m_s")),
            orbital_rms_p95_m_s=_none_if_nan(row.get("orbital_rms_p95_m_s")),
            orbital_rms_p99_m_s=_none_if_nan(row.get("orbital_rms_p99_m_s")),
            orbital_rms_max_m_s=_none_if_nan(row.get("orbital_rms_max_m_s")),
            orbital_amplitude_p95_m_s=_none_if_nan(row.get("orbital_amplitude_p95_m_s")),
            orbital_amplitude_p99_m_s=_none_if_nan(row.get("orbital_amplitude_p99_m_s")),
            orbital_amplitude_max_m_s=_none_if_nan(row.get("orbital_amplitude_max_m_s")),
        )

    # --- contiguous map segments ---------------------------------------------
    chainage_wave_df = chainage_metocean_df[
        ["chainage_m", "wave_node_id", "wave_node_distance_m"]
    ].copy()
    segments_gdf = wave_orbital_map.build_wave_orbital_reference_segments(
        pipeline_id=pipeline_id,
        route=route,
        chainage_wave_df=chainage_wave_df,
        node_attributes_by_id=node_attributes_by_id,
        working_crs=working_crs,
    )
    distance_diagnostics = metocean_evidence.compute_distance_diagnostics(
        chainage_wave_df.rename(columns={"wave_node_distance_m": "distance_m"})
    )

    # --- write parquet/gpkg outputs ------------------------------------------
    hourly_path = metocean_evidence.write_parquet(
        hourly_orbital_df, metocean_interim_dir / "wave_orbital_velocity_3hourly.parquet"
    )
    stats_path = metocean_evidence.write_parquet(
        stats_df, metocean_processed_dir / "wave_orbital_velocity_stats.parquet"
    )
    segments_path = wave_orbital_map.write_wave_orbital_reference_segments_gpkg(
        segments_gdf, metocean_processed_dir / "wave_orbital_reference_segments.gpkg"
    )

    background_raster_path = study_dir / "bathymetry" / "emodnet_baseline_lat_100m.tif"
    png_path = wave_orbital_map.render_wave_orbital_map(
        segments_gdf=segments_gdf,
        route=route,
        working_crs=working_crs,
        output_path=maps_dir / "pl854_wave_orbital_forcing.png",
        background_raster_path=(
            background_raster_path if background_raster_path.exists() else None
        ),
    )
    png_dimensions = wave_orbital_map.read_png_dimensions(png_path)

    # --- metadata --------------------------------------------------------------
    metadata_path = metocean_processed_dir / "wave_orbital_velocity_metadata.json"
    metadata = {
        "scientific_role": wave_orbital.SCIENTIFIC_ROLE,
        "gravity_m_s2": wave_orbital.GRAVITY_M_S2,
        "hs_source": "VHM0",
        "tz_source": wave_orbital.TZ_SOURCE,
        "tp_observed_source": "VTPK",
        "energy_period_source": "VTM10",
        "water_depth_source": ("Copernicus wave-product static deptho at same support node"),
        "canonical_LAT_bathymetry_used_in_orbital_calculation": False,
        "method": "Soulsby & Smallman irregular-wave spectral approximation",
        "equations": (
            "Tn = sqrt(h/g); t = Tn/Tz; A = [6500 + (0.56 + 15.54*t)^6]^(1/6); "
            "Urms = 0.25*Hs / [Tn * (1 + A*t^2)^3]; "
            "equivalent_amplitude = sqrt(2)*Urms; "
            "equivalent_peak_period = 1.28*Tz"
        ),
        "calibration_domain": f"0 <= sqrt(h/g)/Tz <= {wave_orbital.CALIBRATION_DOMAIN_MAX_T}",
        "calibration_domain_semantics": ("METHOD_ACCURACY_DOMAIN_NOT_UNIVERSAL_PHYSICAL_THRESHOLD"),
        "equivalent_amplitude_definition": "sqrt(2) * Urms",
        "equivalent_peak_period_definition": "1.28 * Tz",
        "equivalent_peak_period_is_diagnostic_not_observed_tp": True,
        "current_effect_on_wave_dispersion_applied": False,
        "wave_current_bottom_boundary_layer_applied": False,
        "directional_spreading_correction_applied": False,
        "nonbreaking_wave_assumption_applied": True,
        "explicit_breaking_wave_classifier_applied": False,
        "source_grid_resolution_note": wave_orbital_map.SOURCE_GRID_RESOLUTION_NOTE,
        "map_colour_variable": "orbital_rms_p95_m_s",
        "limitations": [
            "The Soulsby & Smallman approximation is only accepted within its own "
            "calibration domain (t <= 0.54) -- a method accuracy domain, not a universal "
            "physical threshold.",
            "Non-breaking waves are assumed; no breaking-wave classifier is applied here.",
            "Wave-current interaction and bed shear stress are not modelled in this ticket.",
            "The equivalent amplitude and equivalent peak period are derived helpers, "
            "never more canonical than Urms/observed Tp respectively.",
            "Wave model bathymetry and canonical MAR-006 LAT bathymetry remain "
            "deliberately unharmonised; only wave model bathymetry is used here.",
        ],
        "references": [
            {
                "citation": (
                    "Soulsby, R.L. (2006). Simplified calculation of wave orbital "
                    "velocities. HR Wallingford Report TR155."
                )
            },
            {
                "citation": (
                    "Soulsby, R.L. & Smallman, J.V. (1986). A direct method of "
                    "calculating bottom orbital velocity under waves. Hydraulics "
                    "Research Report SR76."
                )
            },
            {
                "citation": (
                    "Wiberg, P.L. & Sherwood, C.R. (2008). Calculating wave-generated "
                    "bottom orbital velocities from surface-wave parameters. Computers "
                    "& Geosciences 34, 1243-1262."
                ),
                "doi": "10.1016/j.cageo.2008.02.010",
            },
            {
                "citation": (
                    "Wilson, R.J. et al. (2018). A synthetic map of the north-west "
                    "European Shelf sedimentary environment for applications in marine "
                    "science. Earth System Science Data 10, 109-130."
                ),
                "doi": "10.5194/essd-10-109-2018",
            },
            {"citation": "Copernicus Marine: NWSHELF_REANALYSIS_WAV_004_015 PUM / QUID."},
        ],
        "vertical_domain_summary": domain_summary,
        "outputs": {
            "wave_orbital_velocity_3hourly": str(hourly_path),
            "wave_orbital_velocity_stats": str(stats_path),
            "wave_orbital_reference_segments": str(segments_path),
            "wave_orbital_map_png": str(png_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"Wave orbital velocity (3-hourly): {len(hourly_orbital_df)} row(s) -> {hourly_path}")
    print(f"Wave orbital velocity (stats): {len(stats_df)} row(s) -> {stats_path}")
    print(f"Wave orbital reference segments: {len(segments_gdf)} section(s) -> {segments_path}")
    print(f"Wave orbital map: {png_path}")
    print(f"Metadata: {metadata_path}")
    print()
    wave_orbital_map.print_wave_orbital_report(
        domain_summary=domain_summary,
        stats_df=stats_df,
        segments_gdf=segments_gdf,
        route_used_node_count=len(stats_df),
        distance_diagnostics=distance_diagnostics,
        segments_path=segments_path,
        png_path=png_path,
        png_dimensions=png_dimensions,
    )
    return 0


def _dataset_start_or(time_range_ms: tuple | None, fallback_now: datetime) -> datetime:
    """The live dataset's own start timestamp, or `fallback_now` if it could not be discovered."""

    if time_range_ms is None:
        return fallback_now
    return datetime.fromtimestamp(time_range_ms[0] / 1000.0, tz=UTC)


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

    build_morphology_parser = subparsers.add_parser(
        "build-regional-morphology",
        help=(
            "Build broad (500/1000/2000 m-scale) regional seabed morphology context "
            "(slope, TPI, local relief) from the EMODnet baseline, with an analysis halo "
            "to avoid AOI-edge bias, and join MAR-006B/C source-age provenance onto chainage."
        ),
    )
    build_morphology_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_morphology_parser.set_defaults(func=_cmd_build_regional_morphology)

    build_sediment_evidence_parser = subparsers.add_parser(
        "build-sediment-evidence",
        help=(
            "Build the PL854 seabed sediment/substrate evidence base from BGS PSA "
            "observations, BGS Seabed Sediments 250k, and the BGS predictive product "
            "(comparison only) -- no sediment mobility physics."
        ),
    )
    build_sediment_evidence_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_sediment_evidence_parser.set_defaults(func=_cmd_build_sediment_evidence)

    build_metocean_evidence_parser = subparsers.add_parser(
        "build-metocean-evidence",
        help=(
            "Build the PL854 metocean forcing evidence base from Copernicus Marine: primary "
            "1.5 km 3D hourly current, 7 km long-term surface current context, and wave "
            "reanalysis -- forcing evidence only, no bed-shear physics."
        ),
    )
    build_metocean_evidence_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_metocean_evidence_parser.set_defaults(func=_cmd_build_metocean_evidence)

    build_current_normalization_parser = subparsers.add_parser(
        "build-current-normalization",
        help=(
            "Build the current-only 1 m log-profile near-bed normalization sensitivity "
            "(MAR-010) and render the reference-current map -- no network, requires "
            "build-metocean-evidence to have already run."
        ),
    )
    build_current_normalization_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_current_normalization_parser.set_defaults(func=_cmd_build_current_normalization)

    build_wave_orbital_forcing_parser = subparsers.add_parser(
        "build-wave-orbital-forcing",
        help=(
            "Build the wave-only spectral near-bed orbital velocity (MAR-011, Soulsby & "
            "Smallman) and render the wave-orbital reference map -- no network, requires "
            "build-metocean-evidence to have already run."
        ),
    )
    build_wave_orbital_forcing_parser.add_argument(
        "config", type=Path, help="Path to a study config YAML file."
    )
    build_wave_orbital_forcing_parser.set_defaults(func=_cmd_build_wave_orbital_forcing)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
