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
EMODnet baseline DTM (MAR-006), CDI source-survey resolution (MAR-006B),
broad regional seabed morphology (MAR-007), and the PL854 sediment
evidence base (MAR-008), the stage packages contain no algorithms yet —
see "Status" below.

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
uv run marine-engine build-regional-morphology configs/pl854.yaml
uv run marine-engine build-sediment-evidence configs/pl854.yaml
```

`ingest-pipeline`, `discover-bathymetry`, `fetch-bathymetry`,
`build-bathymetry`, `resolve-bathymetry-sources`, `build-regional-morphology`,
and `build-sediment-evidence` require network access to public services
(NSTA, MEDIN, BGS GeoNetwork/ArcGIS REST, EMODnet, SeaDataNet CDI).
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
  are unaffected.
- `MAR-006D`: a further recovery-potential fix. `SOURCE_RESOLUTION_UNKNOWN`
  vs `HIGH_RES_SOURCE_*` had been checking both `horizontal_resolution_note`
  and `vertical_resolution_note`, but vertical resolution/accuracy (how
  precisely depth is measured) does not establish horizontal spatial/grid
  resolution (how densely the seabed is sampled) -- a sub-metre vertical
  figure could have incorrectly triggered a `HIGH_RES_SOURCE_*` result with
  no horizontal resolution stated at all. Fixed to consider only
  `horizontal_resolution_note`; `vertical_resolution_note` remains
  preserved as metadata everywhere else. PL854's three records are
  unaffected (`SOURCE_RESOLUTION_UNKNOWN`, as before).
- `MAR-007`: broad (500/1000/2000 m-scale) regional seabed morphology
  context (`morphology/regional.py`) from the EMODnet baseline --
  local-plane-fit slope, Topographic Position Index, local relief, and
  terrain variability, computed on a 2.2 km analysis halo beyond the
  canonical AOI (to avoid edge bias) and then clipped back to it. Outputs:
  `data/processed/pl854/morphology/{slope_500m_deg,slope_1000m_deg,
  tpi_1000m_m,tpi_2000m_m,local_relief_1000m_m,local_relief_2000m_m,
  terrain_std_1000m_m,terrain_std_2000m_m}.tif`,
  `chainage_regional_morphology.parquet` (941 stations), and
  `morphology_metadata.json`. No curvature, rugosity, TRI, or aspect; no
  sand-wave crest/trough/wavelength detection; no scour, free-span, or risk
  scoring -- none of that is supported by this baseline (see below).

  **These are broad regional morphology derivatives, not present-day local
  pipeline survey products.** The source bathymetry underlying 100% of
  PL854 is from CDI surveys acquired in 1991 (~99% of the route) and 1992
  (~1%) -- confirmed from the actual joined MAR-006B/C provenance, not
  assumed. EMODnet 2024 is the DTM *product release* year, never the
  acquisition year. These broad, kilometre-scale features are appropriate
  for regional gradient and bank/flank/channel-scale context only; they
  must NOT be read as current sand-wave crests/troughs, local pipeline
  scour, metre-scale roughness, embedment, or free-span condition -- no
  such mapping has been performed. EMODnet's official per-cell QA
  attributes (min/max/std depth, sample count, interpolation flag) were
  checked live and found unavailable as a small/queryable coverage for
  this release (only a whole-tile, non-AOI-clipped "SD" archive exists);
  the corresponding chainage fields are recorded as null rather than
  fabricated.
- `MAR-008`: PL854 seabed sediment/substrate evidence base
  (`providers/sediment/bgs.py`, `sediment/{evidence,grain_size}.py`) from
  three separate, never-blended BGS evidence tiers, each queried spatially
  against the real AOI polygon (not a bounding box or text match):

  1. **Observed** -- BGS "Offshore samples: particle size analysis": 27
     real point samples intersect the PL854 AOI (all confirmed
     `SURFACE_GRAB`; sample years 1979-2009). Distance to the pipeline
     ranges 141-4986 m (median ~2928 m) -- proximity is always reported
     alongside the observation, never treated as proof the sample
     represents seabed conditions at the pipe. `surface_evidence_class`
     (e.g. `SURFACE_GRAB`) is a vertical-position/sampling-relationship
     classification at collection time only -- it does NOT establish that
     an observation represents present-day seabed conditions merely
     because it was taken at the seabed surface; a 1979 grab is still
     `SURFACE_GRAB`, with its real age carried separately and explicitly in
     `sample_date`/`sample_year`/`sample_age_years_at_run` (`MAR-008A`).
  2. **Mapped** -- BGS Seabed Sediments 250k (1:250,000 regional geological
     mapping, never site-specific ground truth): 8 polygons intersect the
     AOI, covering all 941 chainage stations (`mapped_250k_*` fields).
  3. **Predictive** -- BGS Predictive Seabed Sediments UK (Distributional
     Random Forest, ~38,000 training observations, covariates including
     bathymetry/morphometry/currents/tides): `evidence_role =
     SECONDARY_MODEL_COMPARISON` always, with an explicit
     `circularity_warning` on every predictive-field record -- never
     treated as ground truth, never blended with the observed or mapped
     tiers, never used to fill a missing observed value.

  Grain-size percentiles (D10/D50/D90) are derived only from a PSA
  record's own internally consistent, non-overlapping phi bins (never from
  Folk class, GSM percentages, or the predictive product); of the 27
  observed samples, 5 have a valid whole-sample D10/D50/D90 (D50 range
  0.21-0.38 mm, pure/near-pure sand), 6 are `AMBIGUOUS_BIN_SCHEME`, and 16
  are `INSUFFICIENT_BINS` -- most of those because their phi-bin breakdown
  covers only the sand fraction while gravel is materially present (a real
  correctness guard, not conservatism for its own sake: computing a
  whole-sample D50 from a partial-fraction breakdown would misrepresent
  it). No sediment mobility, Shields parameter, critical shear stress,
  bedload/suspended transport, erosion/deposition, or cohesive/noncohesive
  classification is computed anywhere in this ticket.

  **D50 spatial support assessment: `VERY_SPARSE`** (descriptive only, an
  explicit project heuristic for planning purposes -- never a
  physical/statistical threshold). Only 34.1% of chainage stations have a
  surface PSA sample within 1000 m, and only 5 samples yield a usable D50;
  whether to build a continuous pipeline D50 field in a later ticket is
  left to the external scientific reviewer.
- `MAR-008A`: external review found that `derive_grain_percentiles`
  (`sediment/grain_size.py`) validated a percent-unit phi-bin total against
  the sample's `WEIGHT` only for mass-unit (`grams`) bins -- a percent-unit
  distribution's own populated-bin total was never checked against 100%,
  so a materially incomplete distribution (e.g. bins summing to 80%) could
  have been silently renormalized into a false whole-sample D10/D50/D90.
  Fixed: a new `PHI_PERCENT_TOTAL_TOLERANCE_PCT` (2 percentage points, an
  explicit project data-QA heuristic, never a physical threshold) gates
  percent-unit bins -- a total outside `100% +/- 2pp` now returns
  `INVALID_TOTAL` with null D10/D50/D90, while the original
  (non-renormalized) total remains recorded in
  `phi_total_before_normalization`. This check is additional to, not a
  replacement for, the existing gravel/sand/mud whole-sample coverage guard
  and the mass-bin/`WEIGHT` check, both of which are unchanged and continue
  to dominate the partial-fraction case (a sand-only distribution that
  itself sums to ~100% is still correctly rejected as `INSUFFICIENT_BINS`
  before this new check would ever run).

  Independently re-running `build-sediment-evidence` against the real
  PL854 data confirms the previous 5 valid-D50 records (`65218674,
  65222864, 65235166, 65243220, 65247338`) are unaffected and byte-for-byte
  identical -- all 5 use `PHI_UNITS=grams`, not `percent`, so this fix
  never touches them. Of PL854's 27 real PSA records, 6 use
  `PHI_UNITS=percent`; all 6 already failed as `AMBIGUOUS_BIN_SCHEME` for
  an unrelated, pre-existing reason (non-uniform phi-bin spacing) before
  ever reaching the new total check, so the new validation had zero
  observable effect on this specific dataset -- a genuinely verified
  outcome, not one forced to preserve the prior count. D50 spatial support
  assessment is unchanged: `VERY_SPARSE`. This ticket also corrected
  wording in `sediment/evidence.py` and this README that could be read as
  "surface evidence = present-day seabed sediment" -- `surface_evidence_class`
  is a vertical-position/sampling-relationship classification at collection
  time only (see the `MAR-008` bullet above); no age cutoff was introduced,
  and real surface/subsurface classifications were not changed. No further
  ticket has started.
