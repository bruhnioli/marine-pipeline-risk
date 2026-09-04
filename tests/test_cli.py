"""Tests for the marine_engine CLI entry point."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from marine_engine.cli import main


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["version"])

    assert exit_code == 0
    assert "marine-engine" in capsys.readouterr().out


def test_validate_config_command(
    pl854_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["validate-config", str(pl854_config_path)])

    assert exit_code == 0
    assert "PL854" in capsys.readouterr().out


def test_ingest_pipeline_command_requires_pipeline_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "no_pipeline.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:4326'\n",
        encoding="utf-8",
    )

    exit_code = main(["ingest-pipeline", str(config_path)])

    assert exit_code == 1
    assert "pipeline.pipeline_id" in capsys.readouterr().err


def test_ingest_pipeline_command_success(
    pl854_config_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marine_engine.providers.nsta import IngestionReport, ValidationCheck

    fake_report = IngestionReport(
        pipeline_number="PL854",
        source_label="active",
        source_title="NSTA UKCS offshore infrastructure pipeline linear (WGS84)",
        source_service_url="https://example.invalid/FeatureServer/1",
        pipeline_number_field="NSTAPIPNO",
        source_feature_id="test-id",
        status="NOT IN USE",
        diameter_mm=304.8,
        source_length_m=23489.0,
        geometry_length_m=23480.0,
        source_crs="EPSG:4326",
        working_crs="EPSG:32631",
        bbox_wgs84=(1.65, 53.36, 2.00, 53.39),
        output_path=tmp_path / "pl854" / "pipeline.gpkg",
        raw_cache_path=tmp_path / "PL854_active.geojson",
        validation_checks=[
            ValidationCheck(
                name="pipeline_number", reference_value="PL854", source_value="PL854", note="match"
            )
        ],
    )

    monkeypatch.setattr("marine_engine.cli.ingest_pipeline", lambda *a, **k: fake_report)

    exit_code = main(["ingest-pipeline", str(pl854_config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PL854" in out
    assert "NOT IN USE" in out


def test_build_aoi_command_requires_buffer_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "no_buffer.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n"
        "pipeline:\n  pipeline_id: PL854\n",
        encoding="utf-8",
    )

    exit_code = main(["build-aoi", str(config_path)])

    assert exit_code == 1
    assert "corridor_buffer_m" in capsys.readouterr().err


def test_build_aoi_command_success(
    pl854_config_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marine_engine.preprocessing.aoi import AoiBuildReport

    fake_report = AoiBuildReport(
        study_id="PL854",
        pipeline_id="PL854",
        corridor_buffer_m=5000,
        working_crs="EPSG:32631",
        pipeline_source_crs="EPSG:4326",
        area_km2=312.5,
        min_boundary_distance_m=4999.8,
        bounds_working_crs=(400000.0, 5900000.0, 430000.0, 5920000.0),
        bounds_wgs84=(1.6, 53.3, 2.05, 53.45),
        centroid_wgs84=(1.8, 53.38),
        pipeline_gpkg_path=tmp_path / "pl854" / "pipeline.gpkg",
        output_path=tmp_path / "pl854" / "aoi.gpkg",
    )

    monkeypatch.setattr("marine_engine.cli.build_aoi", lambda *a, **k: fake_report)

    exit_code = main(["build-aoi", str(pl854_config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PL854" in out
    assert "312.5" in out


def test_build_chainage_command_requires_interval_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "no_interval.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n"
        "pipeline:\n  pipeline_id: PL854\n",
        encoding="utf-8",
    )

    exit_code = main(["build-chainage", str(config_path)])

    assert exit_code == 1
    assert "chainage_interval_m" in capsys.readouterr().err


def test_build_chainage_command_success(
    pl854_config_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marine_engine.preprocessing.chainage import ChainageBuildReport

    fake_report = ChainageBuildReport(
        study_id="PL854",
        pipeline_id="PL854",
        working_crs="EPSG:32631",
        interval_m=25,
        pipeline_length_m=23480.67,
        regular_station_count=940,
        total_station_count=941,
        terminal_residual_m=5.67,
        chainage_origin_basis="source_geometry_start",
        origin_basis_note="No authoritative from/to installation metadata.",
        origin_working_crs=(413365.0, 5906432.0),
        origin_wgs84=(1.70, 53.30),
        terminus_working_crs=(426693.0, 5906208.0),
        terminus_wgs84=(1.90, 53.30),
        max_point_to_route_distance_m=1e-9,
        all_within_aoi=True,
        output_path=tmp_path / "pl854" / "chainage_25m.gpkg",
    )

    monkeypatch.setattr("marine_engine.cli.build_chainage", lambda *a, **k: fake_report)

    exit_code = main(["build-chainage", str(pl854_config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PL854" in out
    assert "941" in out
    assert "unresolved" in out


def test_discover_bathymetry_command_requires_pipeline_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "no_pipeline.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n",
        encoding="utf-8",
    )

    exit_code = main(["discover-bathymetry", str(config_path)])

    assert exit_code == 1
    assert "pipeline.pipeline_id" in capsys.readouterr().err


def _write_minimal_study_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "study.yaml"
    processed_dir = tmp_path / "processed"
    interim_dir = tmp_path / "interim"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n"
        f"paths:\n  processed_dir: {processed_dir}\n  interim_dir: {interim_dir}\n"
        "pipeline:\n  pipeline_id: PL854\n",
        encoding="utf-8",
    )
    return config_path


def test_discover_bathymetry_command_reports_missing_canonical_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_minimal_study_config(tmp_path)

    exit_code = main(["discover-bathymetry", str(config_path)])

    assert exit_code == 1
    assert "could not load canonical pipeline/AOI/chainage" in capsys.readouterr().err


def test_fetch_bathymetry_command_requires_inventory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_minimal_study_config(tmp_path)

    exit_code = main(["fetch-bathymetry", str(config_path)])

    assert exit_code == 1
    assert "discover-bathymetry first" in capsys.readouterr().err


def test_build_bathymetry_command_requires_pipeline_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "no_pipeline.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n",
        encoding="utf-8",
    )

    exit_code = main(["build-bathymetry", str(config_path)])

    assert exit_code == 1
    assert "pipeline.pipeline_id" in capsys.readouterr().err


def test_build_bathymetry_command_requires_raw_raster(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A dedicated config (not _write_minimal_study_config) that isolates
    # raw_dir to an empty tmp location -- otherwise it defaults to the
    # repo's real relative data/raw, which may genuinely hold a
    # previously-fetched EMODnet file and mask the missing-raster path.
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n"
        f"paths:\n  raw_dir: {tmp_path / 'raw'}\n"
        "pipeline:\n  pipeline_id: PL854\n",
        encoding="utf-8",
    )

    exit_code = main(["build-bathymetry", str(config_path)])

    assert exit_code == 1
    assert "fetch-bathymetry first" in capsys.readouterr().err


def _write_build_bathymetry_fixture(tmp_path: Path) -> Path:
    """A minimal but complete on-disk study: config + AOI + chainage + a raw raster stub."""

    processed_dir = tmp_path / "processed"
    interim_dir = tmp_path / "interim"
    raw_dir = tmp_path / "raw"
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n"
        f"paths:\n  raw_dir: {raw_dir}\n  processed_dir: {processed_dir}\n"
        f"  interim_dir: {interim_dir}\n"
        "pipeline:\n  pipeline_id: PL854\n",
        encoding="utf-8",
    )

    study_dir = processed_dir / "pl854"
    study_dir.mkdir(parents=True, exist_ok=True)
    aoi_gdf = gpd.GeoDataFrame(
        [{"study_id": "PL854"}],
        geometry=[LineString([(500000.0, 5900000.0), (500100.0, 5900000.0)]).buffer(50.0)],
        crs="EPSG:32631",
    )
    aoi_gdf.to_file(study_dir / "aoi.gpkg", driver="GPKG", layer="study_aoi")

    chainage_gdf = gpd.GeoDataFrame(
        [
            {"pipeline_id": "PL854", "station_index": 0, "chainage_m": 0.0, "kp_label": "KP 0.000"},
            {
                "pipeline_id": "PL854",
                "station_index": 1,
                "chainage_m": 25.0,
                "kp_label": "KP 0.025",
            },
        ],
        geometry=[Point(500000.0, 5900000.0), Point(500025.0, 5900000.0)],
        crs="EPSG:32631",
    )
    chainage_gdf.to_file(study_dir / "chainage_25m.gpkg", driver="GPKG", layer="chainage_points")

    raw_raster_dir = raw_dir / "bathymetry" / "emodnet" / "emodnet__mean"
    raw_raster_dir.mkdir(parents=True, exist_ok=True)
    (raw_raster_dir / "emodnet__mean.tif").write_bytes(
        b"not a real tiff -- build_canonical_dtm is mocked"
    )

    return config_path


def _fake_dtm_report(tmp_path: Path, source_sha256: str | None = "deadbeef"):
    from marine_engine.preprocessing.bathymetry import (
        CanonicalDtmReport,
        CanonicalRasterStats,
        RawRasterStats,
    )

    raw_stats = RawRasterStats(
        width=10,
        height=10,
        source_crs="EPSG:4326",
        nodata_value=None,
        total_pixels=100,
        nan_count=0,
        finite_count=100,
        min=-30.0,
        max=-20.0,
        mean=-25.0,
        median=-25.0,
    )
    canonical_stats = CanonicalRasterStats(
        width=5,
        height=5,
        valid_pixel_count=20,
        nodata_pixel_count=5,
        valid_percent=80.0,
        depth_min=20.0,
        depth_max=30.0,
        depth_mean=25.0,
        depth_median=25.0,
    )
    return CanonicalDtmReport(
        source_path=tmp_path / "raw.tif",
        source_sha256=source_sha256,
        source_crs="EPSG:4326",
        source_nominal_resolution_m=115.0,
        raw_stats=raw_stats,
        sign_convention_observed="negative_elevation",
        output_crs="EPSG:32631",
        output_resolution_m=100.0,
        resampling_method="bilinear",
        vertical_datum="LAT",
        canonical_stats=canonical_stats,
        output_raster_path=tmp_path / "canonical.tif",
        output_metadata_path=tmp_path / "canonical.json",
        processing_timestamp="2026-09-04T00:00:00+00:00",
    )


def _fake_chainage_df(with_attribution: bool = True) -> pd.DataFrame:
    from marine_engine.preprocessing.bathymetry import CHAINAGE_OUTPUT_COLUMNS

    row = {
        "pipeline_id": "PL854",
        "station_index": 0,
        "chainage_m": 0.0,
        "kp_label": "KP 0.000",
        "depth_lat_m": 25.0,
        "bathymetry_source_product": "EMODnet Digital Bathymetry (DTM 2024)",
        "source_reference_id": "121954" if with_attribution else None,
        "source_reference_type": "CDI" if with_attribution else None,
        "qi_age": 3 if with_attribution else None,
        "qi_horizontal": 2 if with_attribution else None,
        "qi_vertical": 3 if with_attribution else None,
        "qi_purpose": 3 if with_attribution else None,
        "qi_combined": 76.9 if with_attribution else None,
    }
    return pd.DataFrame([row], columns=list(CHAINAGE_OUTPUT_COLUMNS))


def test_build_bathymetry_command_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from marine_engine.preprocessing import bathymetry
    from marine_engine.providers.bathymetry import emodnet

    config_path = _write_build_bathymetry_fixture(tmp_path)

    monkeypatch.setattr(
        bathymetry, "build_canonical_dtm", lambda *a, **k: _fake_dtm_report(tmp_path)
    )
    monkeypatch.setattr(emodnet, "fetch_source_references", lambda *a, **k: [])
    monkeypatch.setattr(emodnet, "fetch_quality_index", lambda *a, **k: [])
    monkeypatch.setattr(
        emodnet,
        "check_msl_availability",
        lambda *a, **k: emodnet.MslAvailabilityResult(
            available=False,
            dtm_release="2024",
            tile_id=None,
            format_label=None,
            download_url=None,
            notes="No tile found for this AOI.",
        ),
    )
    monkeypatch.setattr(
        bathymetry, "sample_chainage_bathymetry", lambda *a, **k: _fake_chainage_df()
    )
    monkeypatch.setattr(bathymetry, "write_chainage_bathymetry", lambda df, path: path)

    exit_code = main(["build-bathymetry", str(config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Canonical DTM" in out
    assert "source_attribution_status: available" in out


def test_build_bathymetry_command_attribution_unavailable_does_not_fail_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from marine_engine.preprocessing import bathymetry
    from marine_engine.providers.bathymetry import emodnet

    config_path = _write_build_bathymetry_fixture(tmp_path)

    def raise_unavailable(*a, **k):
        raise emodnet.EmodnetAttributionUnavailableError("simulated WFS outage")

    monkeypatch.setattr(
        bathymetry, "build_canonical_dtm", lambda *a, **k: _fake_dtm_report(tmp_path)
    )
    monkeypatch.setattr(emodnet, "fetch_source_references", raise_unavailable)
    monkeypatch.setattr(emodnet, "fetch_quality_index", raise_unavailable)
    monkeypatch.setattr(
        emodnet,
        "check_msl_availability",
        lambda *a, **k: emodnet.MslAvailabilityResult(
            available=False,
            dtm_release="2024",
            tile_id=None,
            format_label=None,
            download_url=None,
            notes="Could not query the download-tiles index.",
        ),
    )
    monkeypatch.setattr(
        bathymetry,
        "sample_chainage_bathymetry",
        lambda *a, **k: _fake_chainage_df(with_attribution=False),
    )
    monkeypatch.setattr(bathymetry, "write_chainage_bathymetry", lambda df, path: path)

    exit_code = main(["build-bathymetry", str(config_path)])

    out = capsys.readouterr().out
    # A live-attribution failure must NOT fail the whole command -- depth
    # processing and source-quality attribution are deliberately separable.
    assert exit_code == 0
    assert "source_attribution_status: unavailable" in out
    assert "simulated WFS outage" in out


def test_build_bathymetry_command_reports_dtm_build_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from marine_engine.preprocessing import bathymetry

    config_path = _write_build_bathymetry_fixture(tmp_path)

    def raise_ambiguous(*a, **k):
        raise bathymetry.AmbiguousSignConventionError("simulated ambiguous sign convention")

    monkeypatch.setattr(bathymetry, "build_canonical_dtm", raise_ambiguous)

    exit_code = main(["build-bathymetry", str(config_path)])

    assert exit_code == 1
    assert "canonical DTM build failed" in capsys.readouterr().err


def test_resolve_bathymetry_sources_command_requires_pipeline_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "no_pipeline.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n",
        encoding="utf-8",
    )

    exit_code = main(["resolve-bathymetry-sources", str(config_path)])

    assert exit_code == 1
    assert "pipeline.pipeline_id" in capsys.readouterr().err


def test_resolve_bathymetry_sources_command_requires_chainage_bathymetry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_minimal_study_config(tmp_path)

    exit_code = main(["resolve-bathymetry-sources", str(config_path)])

    assert exit_code == 1
    assert "build-bathymetry first" in capsys.readouterr().err


def _write_resolve_sources_fixture(tmp_path: Path) -> Path:
    """Extends the build-bathymetry fixture with pipeline.gpkg and chainage_bathymetry.parquet."""

    from marine_engine.preprocessing.bathymetry import CHAINAGE_OUTPUT_COLUMNS

    config_path = _write_build_bathymetry_fixture(tmp_path)
    study_dir = tmp_path / "processed" / "pl854"

    pipeline_gdf = gpd.GeoDataFrame(
        [{"pipeline_id": "PL854", "source": "test", "status": "ACTIVE"}],
        geometry=[LineString([(500000.0, 5900000.0), (500100.0, 5900000.0)])],
        crs="EPSG:32631",
    )
    pipeline_gdf.to_file(study_dir / "pipeline.gpkg", driver="GPKG", layer="pipeline")

    chainage_bathymetry_dir = study_dir / "bathymetry"
    chainage_bathymetry_dir.mkdir(parents=True, exist_ok=True)
    chainage_bathymetry_df = pd.DataFrame(
        [
            {
                "pipeline_id": "PL854",
                "station_index": 0,
                "chainage_m": 0.0,
                "kp_label": "KP 0.000",
                "depth_lat_m": 25.0,
                "bathymetry_source_product": "Test",
                "source_reference_id": "121954",
                "source_reference_type": "CDI",
                "qi_age": 0,
                "qi_horizontal": 3,
                "qi_vertical": 4,
                "qi_purpose": 3,
                "qi_combined": 76.9,
            }
        ],
        columns=list(CHAINAGE_OUTPUT_COLUMNS),
    )
    chainage_bathymetry_df.to_parquet(
        chainage_bathymetry_dir / "chainage_bathymetry.parquet", index=False
    )

    return config_path


def test_resolve_bathymetry_sources_command_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from marine_engine.preprocessing import source_resolution
    from marine_engine.providers.bathymetry import emodnet

    config_path = _write_resolve_sources_fixture(tmp_path)

    fake_df = pd.DataFrame(
        [{"source_reference_id": "121954"}], columns=list(source_resolution.CDI_SOURCES_COLUMNS)
    )
    monkeypatch.setattr(emodnet, "fetch_source_references", lambda *a, **k: [])
    monkeypatch.setattr(emodnet, "fetch_quality_index", lambda *a, **k: [])
    monkeypatch.setattr(
        source_resolution,
        "resolve_pl854_cdi_sources",
        lambda **k: (fake_df, [], []),
    )
    monkeypatch.setattr(source_resolution, "write_cdi_sources_parquet", lambda df, path: path)
    monkeypatch.setattr(source_resolution, "write_cdi_sources_gpkg", lambda *a, **k: None)

    exit_code = main(["resolve-bathymetry-sources", str(config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Resolved 1 PL854 source-reference record" in out


def test_resolve_bathymetry_sources_command_writes_output_under_interim_not_processed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAR-006C: this is provenance-resolution metadata, not an analysis-ready
    product -- it belongs under data/interim/<study>/, not data/processed/."""

    from marine_engine.preprocessing import source_resolution
    from marine_engine.providers.bathymetry import emodnet

    config_path = _write_resolve_sources_fixture(tmp_path)

    fake_df = pd.DataFrame(
        [{"source_reference_id": "121954"}], columns=list(source_resolution.CDI_SOURCES_COLUMNS)
    )
    captured_paths: dict[str, Path] = {}
    monkeypatch.setattr(emodnet, "fetch_source_references", lambda *a, **k: [])
    monkeypatch.setattr(emodnet, "fetch_quality_index", lambda *a, **k: [])
    monkeypatch.setattr(
        source_resolution, "resolve_pl854_cdi_sources", lambda **k: (fake_df, [], [])
    )

    def fake_write_parquet(df, path):
        captured_paths["parquet"] = path
        return path

    def fake_write_gpkg(records, working_crs, path):
        captured_paths["gpkg"] = path
        return None

    monkeypatch.setattr(source_resolution, "write_cdi_sources_parquet", fake_write_parquet)
    monkeypatch.setattr(source_resolution, "write_cdi_sources_gpkg", fake_write_gpkg)

    exit_code = main(["resolve-bathymetry-sources", str(config_path)])

    assert exit_code == 0
    interim_dir = tmp_path / "interim" / "pl854"
    processed_dir = tmp_path / "processed" / "pl854"
    assert captured_paths["parquet"] == interim_dir / "emodnet_cdi_sources.parquet"
    assert captured_paths["gpkg"] == interim_dir / "emodnet_cdi_sources.gpkg"
    assert processed_dir not in captured_paths["parquet"].parents

    # The canonical DTM / chainage-bathymetry inputs are still read from
    # processed/<study>/bathymetry/ -- this command only redirects its OWN
    # output, never the products it reads.
    assert (processed_dir / "bathymetry" / "chainage_bathymetry.parquet").exists()


def test_resolve_bathymetry_sources_command_reports_attribution_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from marine_engine.providers.bathymetry import emodnet

    config_path = _write_resolve_sources_fixture(tmp_path)

    def raise_unavailable(*a, **k):
        raise emodnet.EmodnetAttributionUnavailableError("simulated WFS outage")

    monkeypatch.setattr(emodnet, "fetch_source_references", raise_unavailable)

    exit_code = main(["resolve-bathymetry-sources", str(config_path)])

    # Unlike build-bathymetry, resolving sources IS this command's whole job --
    # a WFS outage here is a hard failure, not something to gracefully degrade.
    assert exit_code == 1
    assert "simulated WFS outage" in capsys.readouterr().err


def test_build_regional_morphology_command_requires_pipeline_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "no_pipeline.yaml"
    config_path.write_text(
        "study:\n  id: X\n  name: Test\ncrs:\n  horizontal: 'EPSG:32631'\n",
        encoding="utf-8",
    )

    exit_code = main(["build-regional-morphology", str(config_path)])

    assert exit_code == 1
    assert "pipeline.pipeline_id" in capsys.readouterr().err


def test_build_regional_morphology_command_requires_chainage_bathymetry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_minimal_study_config(tmp_path)

    exit_code = main(["build-regional-morphology", str(config_path)])

    assert exit_code == 1
    assert "build-bathymetry first" in capsys.readouterr().err


def _write_regional_morphology_fixture(tmp_path: Path) -> Path:
    """Extends the resolve-sources fixture with emodnet_cdi_sources.parquet under interim/."""

    from marine_engine.preprocessing import source_resolution

    config_path = _write_resolve_sources_fixture(tmp_path)
    interim_dir = tmp_path / "interim" / "pl854"
    interim_dir.mkdir(parents=True, exist_ok=True)

    cdi_sources_df = pd.DataFrame(
        [
            {
                "source_reference_id": "121954",
                "acquisition_year": 1991,
                "acquisition_start": "1991-04-24",
                "acquisition_end": "1991-08-16",
                "survey_age_at_product_release_year": 33,
            }
        ],
        columns=list(source_resolution.CDI_SOURCES_COLUMNS),
    )
    cdi_sources_df.to_parquet(interim_dir / "emodnet_cdi_sources.parquet", index=False)

    return config_path


def test_build_regional_morphology_command_requires_cdi_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_resolve_sources_fixture(tmp_path)  # no emodnet_cdi_sources.parquet written

    exit_code = main(["build-regional-morphology", str(config_path)])

    assert exit_code == 1
    assert "resolve-bathymetry-sources first" in capsys.readouterr().err


def test_build_regional_morphology_command_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    import rasterio

    from marine_engine.morphology import regional
    from marine_engine.providers.bathymetry import acquisition, emodnet
    from marine_engine.providers.bathymetry.emodnet import (
        EmodnetFetchResult,
        NativeQaLayerAvailability,
    )

    config_path = _write_regional_morphology_fixture(tmp_path)

    fake_fetch_result = EmodnetFetchResult(
        local_path=tmp_path / "halo.tif",
        request_parameters={"coverageId": "emodnet__mean"},
        returned_crs="EPSG:4326",
        width_px=10,
        height_px=10,
        content_type="image/tiff",
    )
    monkeypatch.setattr(emodnet, "fetch_emodnet_geotiff", lambda *a, **k: fake_fetch_result)
    monkeypatch.setattr(
        acquisition,
        "record_acquisition",
        lambda *a, **k: {"sha256": "deadbeef", "local_path": str(tmp_path / "halo.tif")},
    )
    monkeypatch.setattr(
        emodnet,
        "check_native_qa_layers",
        lambda *a, **k: NativeQaLayerAvailability(
            wcs_coverage_ids=(),
            wcs_matches={},
            download_tile_formats=(),
            download_tile_matches={},
            notes="n/a",
        ),
    )

    fake_layer = regional.MorphologyLayer(
        name="slope_500m_deg",
        array=np.array([[1.0]]),
        transform=rasterio.transform.from_origin(0.0, 1.0, 100.0, 100.0),
        radius_m=500.0,
        unit="degrees",
        description="test",
    )
    fake_halo_grid = regional.HaloElevationGrid(
        elevation=np.array([[1.0]]),
        valid_mask=np.array([[True]]),
        transform=fake_layer.transform,
        crs="EPSG:32631",
        raw_stats=None,
        sign_convention_observed="negative_elevation",
        source_sha256="deadbeef",
    )
    fake_result = regional.RegionalMorphologyResult(
        chainage_df=pd.DataFrame([{"pipeline_id": "PL854", "station_index": 0}]),
        layers=[fake_layer],
        halo_grid=fake_halo_grid,
        qa_layer_availability=NativeQaLayerAvailability(
            wcs_coverage_ids=(),
            wcs_matches={},
            download_tile_formats=(),
            download_tile_matches={},
            notes="n/a",
        ),
        aoi_identifier="PL854_AOI",
        working_crs="EPSG:32631",
        processing_timestamp="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(regional, "build_regional_morphology", lambda **k: fake_result)
    monkeypatch.setattr(regional, "write_morphology_raster", lambda *a, **k: None)
    monkeypatch.setattr(regional, "write_chainage_regional_morphology", lambda *a, **k: None)
    monkeypatch.setattr(regional, "write_morphology_metadata", lambda *a, **k: None)
    monkeypatch.setattr(regional, "print_regional_morphology_report", lambda *a, **k: None)

    exit_code = main(["build-regional-morphology", str(config_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "morphology raster" in out
