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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
