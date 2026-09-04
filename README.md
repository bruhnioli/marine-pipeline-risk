# marine-engine

Research PoC for estimating how seabed morphodynamics affect a subsea
pipeline over its design life: erosion, deposition, sediment mobility,
burial/exposure, scour susceptibility, free-span susceptibility, and
lifetime risk scenarios.

**First study case:** PL854, Anglia A -> LOGGS pipeline corridor, Southern
North Sea (see [`configs/pl854.yaml`](configs/pl854.yaml)).

This is a personal research project, not a production engineering tool.

## Architecture

The scientific engine is a plain, importable Python package
(`marine_engine`), not notebooks or one-off scripts, so it can later be
wrapped by an API and consumed by a GIS/web application. Each stage of the
target workflow is its own subpackage:

```
open-data providers -> preprocessing -> seabed/morphology features
    -> metocean features -> sediment mobility -> erosion/deposition
    -> pipeline interaction -> lifetime risk scenarios -> validation
    -> export (GeoTIFF / GeoPackage / Parquet / JSON)
```

| Package        | Responsibility                                                        |
|-----------------|------------------------------------------------------------------------|
| `providers`     | Open-data provider clients (bathymetry, metocean, sediment, route, ...)|
| `preprocessing` | Cleaning, reprojection, resampling, harmonisation of raw provider data  |
| `morphology`    | Seabed morphology features (slope, roughness, bedforms, mobility)      |
| `sediment`      | Sediment mobility modelling (grain size, shear stress, transport)      |
| `metocean`      | Metocean features (waves, currents, tides) driving seabed mobility     |
| `pipeline`      | Pipeline interaction (burial/exposure, free-span, scour susceptibility)|
| `risk`          | Lifetime risk scenarios combining the above                            |
| `validation`    | Validation of model outputs against observed/survey data               |
| `export`        | Export of results to GeoTIFF, GeoPackage, Parquet, and JSON            |

`config.py` defines the schema (via pydantic) and loader for study-specific
YAML configuration files under `configs/`. `cli.py` is a thin argparse
entry point over that config system.

Beyond the NSTA provider (MAR-002), AOI/chainage preprocessing
(MAR-003/004), bathymetry source discovery (MAR-005), the canonical
EMODnet baseline DTM (MAR-006), and CDI source-survey resolution
(MAR-006B), the stage packages contain no algorithms yet — see "Status"
below.

## Project layout

```
configs/            Study-specific YAML configs (e.g. pl854.yaml)
data/raw/            Unmodified downloaded datasets (gitignored, not committed)
data/interim/        Intermediate/derived data (gitignored, not committed)
data/processed/      Analysis-ready outputs (gitignored, not committed)
src/marine_engine/   The engine package (see table above)
tests/               pytest suite
```

## Getting started

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                # create .venv and install dependencies
uv run pytest                          # run the test suite
uv run ruff check .                    # lint
uv run marine-engine version
uv run marine-engine validate-config configs/pl854.yaml
uv run marine-engine ingest-pipeline configs/pl854.yaml
uv run marine-engine build-aoi configs/pl854.yaml
uv run marine-engine build-chainage configs/pl854.yaml
uv run marine-engine discover-bathymetry configs/pl854.yaml
uv run marine-engine fetch-bathymetry configs/pl854.yaml
uv run marine-engine build-bathymetry configs/pl854.yaml
uv run marine-engine resolve-bathymetry-sources configs/pl854.yaml
```

`ingest-pipeline`, `discover-bathymetry`, `fetch-bathymetry`,
`build-bathymetry`, and `resolve-bathymetry-sources` require network access
to public services (NSTA, MEDIN, BGS GeoNetwork, EMODnet, SeaDataNet CDI).
Their live-source smoke tests are excluded from the default test run; opt
in with `uv run pytest -m live`.

## Status

- `MAR-001`: project scaffold — structure, config system, CLI, and test
  infrastructure. No scientific algorithms, no dataset downloads, no ML, no
  web application.
- `MAR-002`: first real provider (`providers/nsta.py`) ingests the PL854
  pipeline geometry from the authoritative NSTA UKCS offshore infrastructure
  dataset and normalizes it to `data/processed/pl854/pipeline.gpkg`.
- `MAR-003`: AOI preprocessing (`preprocessing/aoi.py`) buffers the canonical
  pipeline by the configured `area_of_interest.corridor_buffer_m` (5000 m)
  into the study's spatial extent, `data/processed/pl854/aoi.gpkg`.
- `MAR-004`: chainage/KP preprocessing (`preprocessing/chainage.py`)
  generates the linear-reference point system at the configured
  `pipeline.chainage_interval_m` (25 m) plus the exact route terminus, into
  `data/processed/pl854/chainage_25m.gpkg`. Chainage direction is recorded
  honestly as `source_geometry_start` — semantic endpoint identity
  (Anglia A vs LOGGS) remains unresolved by design.
- `MAR-005`: bathymetry source discovery (`providers/bathymetry/`) queries
  the approved UKHO (via MEDIN), BGS, and EMODnet sources, spatially
  verifies each against the real canonical pipeline/AOI/chainage, ranks
  candidates, and acquires the mandatory EMODnet 2024 baseline into
  `data/interim/pl854/bathymetry_inventory.{parquet,gpkg}` and
  `data/raw/bathymetry/`. As of this run, no automatically-downloadable
  dataset intersects the pipeline itself (the historical BGS surveys are
  metadata/bbox-only and restricted-access) — the primary analysis
  candidate is honestly reported as unresolved rather than defaulting to
  EMODnet.
- `MAR-006`: canonical EMODnet baseline DTM (`preprocessing/bathymetry.py`)
  transforms the raw EMODnet 2024 raster into a reproducible baseline:
  empirically-verified sign convention (observed `negative_elevation` for
  this raster, converted to positive-down `depth_lat_m`), reprojected to
  `EPSG:32631` at a 100 m analysis grid using bilinear resampling, clipped
  to the real AOI polygon (not just its bounding box), and written to
  `data/processed/pl854/bathymetry/emodnet_baseline_lat_100m.tif` with a
  JSON provenance sidecar. Source-reference and quality-index attribution
  (`providers/bathymetry/emodnet.py`'s WFS functions, queried server-side
  via `CQL_FILTER`) is sampled onto all 941 chainage stations into
  `data/processed/pl854/bathymetry/chainage_bathymetry.parquet`. Depth
  processing and source/quality attribution are kept deliberately
  separable — a WFS attribution outage is recorded as
  `source_attribution_status = unavailable` but never fails the DTM build.
  An official Mean Sea Level product was confirmed reproducibly acquirable
  (tile D4, 2024 release, via `emodnet:download_tiles`) but is not
  downloaded or merged with the LAT baseline in this ticket. Still no
  morphology (slope/curvature/roughness/BPI), erosion/deposition,
  scour/free-span, or risk calculations — those are later tickets.

  **Scientific limitations of this baseline** (apply to every downstream
  use of `emodnet_baseline_lat_100m.tif` / `chainage_bathymetry.parquet`
  until a higher-resolution survey supersedes it):
  1. EMODnet 2024 is approximately 115 m-class regional bathymetry.
  2. The 100 m projected grid is an analysis grid, not true 100 m
     measurement resolution.
  3. Appropriate for regional seabed context and broad morphology only.
  4. NOT sufficient by itself for pipeline-scale local scour, metre-scale
     sand-wave geometry, or free-span detection.
  5. High-resolution MBES can replace/augment this baseline later without
     changing the canonical pipeline/chainage architecture.

  Still no sediment/metocean providers, erosion/deposition or risk science,
  no ML, no web app — those arrive in later tickets.
- `MAR-006B`: PL854 EMODnet CDI source-survey resolution
  (`providers/bathymetry/cdi.py`, `preprocessing/source_resolution.py`)
  follows each of the three real 2024-release `source_references` ids that
  cross the pipeline (`110153`, `121953`, `121954`) through their official
  SeaDataNet CDI `metadata_url` to real survey provenance: `110153` is a
  1992 single-beam-echosounder survey ("Haddock Bank", UKHO cruise HI560,
  covering the first ~1% of the route); `121953`/`121954` are 1991 surveys
  (UKHO cruise HI524-HI525-HI531, instrument unstated) covering the
  remaining ~99%. All three are 32-33 years old at the 2024 release and
  require registration plus owner negotiation (via OceanWise/UKHO) to
  request the original data -- none is directly downloadable, and **none of
  the three states a numeric spatial resolution**, so whether the original
  source data is actually finer than the ~115 m EMODnet composite remains
  unverified (see `MAR-006C` below) -- they are requestable/negotiable, not
  confirmed higher-resolution products. Output:
  `data/interim/pl854/emodnet_cdi_sources.parquet`. The CDI report host
  fronts every request with a client-side proof-of-work challenge that a
  plain HTTP client cannot pass, so live automated resolution falls back to
  a manually browser-verified snapshot for these three known ids and
  reports that honestly rather than either fabricating data or crashing.

  **`EMODnet DTM 2024` is a product release/version, not a seabed survey
  date -- do not read "2024" as when this bathymetry was measured.** This
  ticket's audit fix makes that explicit in the schema itself:
  `SurveyRecord` now has a separate `product_release_year` field: the
  EMODnet composite sets `product_release_year=2024` and leaves
  `acquisition_year=None`, while each CDI-resolved underlying survey
  carries its own real `acquisition_year` (1991/1992 here) and
  `survey_age_at_product_release_year`.

  Still no morphology, sediment/metocean, erosion/deposition,
  scour/free-span, or risk science, no ML, no web app, no LAT/MSL
  conversion, and no SeaDataNet data request was submitted.
- `MAR-006C`: provenance-semantics correction. MAR-006B's
  `classify_recovery_potential()` had conflated "a request path exists"
  with "higher resolution is confirmed" -- a source requestable via owner
  negotiation was reported as `HIGH_RES_SOURCE_REQUESTABLE` even though
  none of PL854's three CDI records state a numeric resolution (QI
  instrument class, e.g. QI_Vertical=4 suggesting MBES, is not proof of
  exported/grid resolution on its own). Fixed: `recovery_potential` now
  requires CDI itself to state a real numeric resolution finer than
  EMODnet's ~115 m baseline before returning a `HIGH_RES_*` value; for the
  current three PL854 records it correctly reports
  `SOURCE_RESOLUTION_UNKNOWN`, while `access_class`
  (`OWNER_PERMISSION_REQUIRED` for all three) still separately and
  accurately conveys that a real request path exists. Also moved this
  ticket's own output from `data/processed/pl854/bathymetry/` to
  `data/interim/pl854/` -- it is provenance-resolution metadata, not an
  analysis-ready product; the canonical DTM and chainage-bathymetry outputs
  are unaffected. No further ticket has started.
