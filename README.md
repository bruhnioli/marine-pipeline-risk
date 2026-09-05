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
broad regional seabed morphology (MAR-007), the PL854 sediment evidence
base (MAR-008), and the PL854 metocean forcing evidence base (MAR-009),
the stage packages contain no algorithms yet — see "Status" below.

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
uv run marine-engine build-metocean-evidence configs/pl854.yaml
```

`ingest-pipeline`, `discover-bathymetry`, `fetch-bathymetry`,
`build-bathymetry`, `resolve-bathymetry-sources`, `build-regional-morphology`,
`build-sediment-evidence`, and `build-metocean-evidence` require network
access to public services (NSTA, MEDIN, BGS GeoNetwork/ArcGIS REST,
EMODnet, SeaDataNet CDI, Copernicus Marine). Their live-source smoke tests
are excluded from the default test run; opt in with `uv run pytest -m live`.

`build-metocean-evidence` additionally requires a (free) Copernicus Marine
account: catalogue metadata is public, but real data acquisition needs
`uv run copernicusmarine login` (or the `COPERNICUSMARINE_SERVICE_USERNAME`
/ `COPERNICUSMARINE_SERVICE_PASSWORD` environment variables) run once,
outside of any AI assistant, before this command's real acquisition steps
can proceed -- it stops with a clear `CopernicusAuthenticationRequiredError`
otherwise, never an interactive credential prompt.

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
  and real surface/subsurface classifications were not changed.
- `MAR-009`: PL854 metocean forcing evidence base
  (`providers/metocean/{copernicus,acquisition}.py`,
  `metocean/{current,wave,evidence}.py`) from three separate, never-blended
  Copernicus Marine products, each mapped from the 941 dense chainage
  stations onto a much smaller set of real model grid cells ("support
  nodes") via nearest-wet-cell assignment -- never 941 fabricated
  independent time series, never bilinear interpolation of data or masks:

  1. **Primary current** -- `NWSHELF_ANALYSISFORECAST_PHY_004_013`
     (`cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i`, confirmed live): ~1.5 km
     3D hourly instantaneous current. Per support node/hour, the **deepest
     valid standard level** with finite `uo`/`vo` is selected and stored as
     `deepest_valid_standard_level_current` -- explicitly **NOT** the
     model's native terrain-following bottom cell, and never called
     "bottom current"/"seabed current" anywhere in the codebase (naming
     enforced by a schema-regression test). `height_above_model_bed_m` is
     carried alongside and flagged if negative.
  2. **Long-term surface current context** --
     `NWSHELF_MULTIYEAR_PHY_004_009`
     (`cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i`, confirmed live, 1993
     onward): `LONG_TERM_SURFACE_CURRENT_CONTEXT` role only, hourly
     instantaneous 2D surface current -- never the daily 3D mean on the
     same product (`cmems_mod_nws_phy-uv_my_7km-3D_P1D-m`, a 25-hour
     tide-removing average, explicitly forbidden as a hard rule), never
     used to fill a missing primary-current value, never downscaled.
  3. **Wave climate** -- `NWSHELF_REANALYSIS_WAV_004_015`
     (`MetO-NWS-WAV-RAN`, confirmed live, 1980 onward): 3-hourly `VHM0`/
     `VTPK`/`VTM02`/`VTM10`/`VMDR` (+ Stokes drift where available). `VMDR`
     is preserved as the raw wave FROM-direction
     (`wave_mean_direction_from_deg`); a TO-direction is only ever derived
     for convenience, never replacing the original.

  Current direction (`current_direction_to_deg`) and wave direction
  (`wave_mean_direction_from_deg`) use opposite conventions by design (TO
  vs FROM) and are never confused or arithmetic-averaged -- directional
  summaries use proper circular statistics. Model bathymetry (`deptho` on
  each product's own static dataset) and the canonical MAR-006
  `depth_lat_m` (LAT datum) are carried side by side, never subtracted or
  compared as an "error"
  (`canonical_model_bathymetry_vertical_datums_not_harmonised = true` in
  the metadata). Historical forcing evidence is cut off at least 48 hours
  behind the live analysis/forecast boundary, computed dynamically from
  the actual dataset time coordinates each run, never hard-coded.
  Acquisition uses the official Copernicus Marine Toolbox
  (`copernicusmarine`) with resumable, idempotent monthly/yearly chunks; a
  short-window surface-current-context ratio is reported as a descriptive
  diagnostic only, never a scale factor or bias correction.

  Real execution against PL854 (2026-09-04): all three dataset ids were
  confirmed live against the current Copernicus Marine catalogue
  successfully. Real data acquisition then stopped cleanly at
  `CopernicusAuthenticationRequiredError` -- no Copernicus Marine
  credentials are configured in this environment -- printing the exact
  `copernicusmarine login` steps an operator needs to run, per this
  ticket's explicit requirement to never attempt an interactive credential
  prompt or ask for a password/token in chat. The full pipeline (support-
  node mapping, deepest-valid-level selection, statistics, chainage
  assembly, metadata) is implemented and verified with 88 new offline
  tests plus opt-in live catalogue-reachability smoke tests; it has not
  yet processed real Copernicus current/wave data end-to-end pending that
  one-time authentication step. No bed shear stress, Shields parameter,
  sediment mobility, wave orbital velocity, erosion/deposition, scour,
  free-span, fatigue, or risk scoring is computed anywhere in this ticket.
  No further ticket has started.
- `MAR-009A`: the real MAR-009 acquisition subsequently completed and
  external review of its actual output found a genuine vertical-eligibility
  integrity failure: `select_deepest_valid_standard_level`
  (`metocean/current.py`) checked only that `uo`/`vo` were finite, never
  the Copernicus model's own bathymetry. On **100% of the 260,764 real
  primary-current hourly rows across all 14 route-used support nodes**,
  standard levels well below that cell's own model bathymetry (up to 75 m
  deep against a bathymetry of ~23-30 m) still carried finite `uo`/`vo` --
  direct inspection showed an identically-repeated "held/padded" fill
  pattern below where the real profile stops varying, never a genuine
  measurement. Fixed: a depth candidate is now eligible only when `uo`/`vo`
  are finite AND `depth_m <= model_bathymetry_m + tolerance` AND, where the
  static 3D `mask`'s own depth coordinate is confirmed exactly aligned to
  the dynamic dataset's (`check_depth_coordinate_alignment` -- true for
  this real product pair), that mask cell is wet; neither condition alone
  would have caught every contaminated case at the one real node directly
  inspected, so both are required and share one numerical tolerance with
  `height_above_model_bed_m`'s own validity check so the two can never
  disagree.

  A second, independent real bug (this ticket's Section 7): long-term
  surface current and wave normalization were called against every wet grid
  cell in the request bounding box, not just the cells actually assigned to
  a PL854 chainage station -- the real MAR-009 report showed 330 wave and
  18 long-term-current time-series node ids against only 14/4 actually
  route-used (the support-node *tables* were already correctly filtered;
  only the time-series normalization calls were not). Fixed:
  `_cmd_build_metocean_evidence` (`cli.py`) now builds
  `used_primary_nodes`/`used_long_term_nodes`/`used_wave_nodes` and
  normalizes only those. A third, defensive fix (Section 6): static and
  dynamic dataset grid indices are no longer assumed identical -- each
  node's canonical lon/lat is re-resolved against the dynamic dataset's own
  coordinate arrays (`reconcile_node_grid_indices`, refused beyond 10% of
  that axis's median grid spacing, never a guessed index) before every
  sample. For the real PL854 primary-current product pair, static and
  dynamic datasets happen to share identical coordinate arrays, so this
  path was not itself the cause of the observed contamination, but is
  exercised by dedicated index-shift, reversed-ordering, and
  unreconcilable-coordinate tests.

  Independently re-running `build-metocean-evidence` against the
  already-downloaded raw Copernicus chunks (idempotent manifest -- zero
  re-download confirmed across two full reruns, byte-identical raw file
  mtimes and an unchanged 111-entry manifest) and reopening all 9 canonical
  outputs directly confirms **ALL 260,764 canonical primary-current rows
  are now within the Copernicus model water column**
  (`height_above_model_bed_m` min=0.243 m, median=3.195 m, p95/max=4.868 m;
  zero violations beyond the 1e-6 m numerical tolerance). Current speed
  statistics changed materially now that the contaminating deep levels are
  excluded (mean 0.245 -> 0.338 m/s, p95 0.671 -> 0.698, p99 0.770 -> 0.814,
  max 0.954 -> 0.983 m/s; the corrected values are canonical, the old
  values are not preserved by design) and the time-series node counts
  collapsed to the correct route-used set (wave 330 -> 14, long-term
  current 18 -> 4). Station-to-node distance diagnostics (min/median/p95/
  max, never a confidence score) are now reported for all three products.
  Model bathymetry and the canonical MAR-006 LAT bathymetry remain
  deliberately unharmonised and unsubtracted
  (`canonical_model_bathymetry_vertical_datums_not_harmonised = true`); new
  metadata additionally records
  `dynamic_grid_coordinate_reconciliation_method`,
  `static_dynamic_coordinate_match_status` (all three products' used nodes
  fully reconciled), `primary_current_vertical_eligibility_rule`,
  `static_depth_mask_used = true` (the real static/dynamic depth
  coordinates align exactly for this product), and a below-model-bed
  finite-candidate diagnostic summary (QA only, 1,266,568 excluded
  candidates across all 260,764 timestamps -- never entering canonical
  statistics). 25 new offline tests were added across `test_current.py` and
  `test_metocean_evidence.py`; the full offline suite (498 tests) and
  repo-wide `ruff format`/`ruff check` pass clean. No bed shear stress,
  Shields parameter, sediment mobility, erosion/deposition, scour,
  free-span, fatigue, or risk scoring is computed anywhere in this ticket.
  Per this ticket's explicit instruction, external scientific review of the
  corrected height-above-bed, model bathymetry, current statistics, and
  support-node distances is required before MAR-010 (near-bed hydrodynamic
  formulation) begins -- no further ticket has started.
- `MAR-009B`: the real re-run confirmed a second, independent integrity
  finding: the Copernicus Marine Toolbox's `subset()` treats `end_datetime`
  as INCLUSIVE, so two adjacent monthly/yearly acquisition chunks each
  return their shared boundary instant, and
  `xr.open_mfdataset(..., combine="by_coords")` does not itself deduplicate
  -- the real primary-current record carried **18,626 raw hourly
  timestamps per node against the physically correct 18,600** (26
  duplicated internal monthly-chunk boundaries), pushing completeness to a
  physically-impossible ~100.1%; the same class of duplication was
  confirmed in the yearly long-term-current (33 duplicate boundaries) and
  wave (46 duplicate boundaries) acquisitions. Fixed at the CHUNK ASSEMBLY
  boundary, not inside any statistics function:
  `deduplicate_time_coordinate` (`providers/metocean/acquisition.py`)
  detects every duplicated timestamp, requires every data variable to
  agree at every duplicate occurrence (NaN treated as equal to NaN) before
  ever collapsing it to one canonical row, and raises
  `DuplicateTimestampConflictError` -- never silently picking a side -- the
  moment any duplicate's data disagrees; it never rewrites or deletes the
  raw NetCDF chunk files. Every `normalize_*` function now always receives
  an already-unique, already-monotonic time coordinate.
  `validate_temporal_integrity` (`metocean/evidence.py`) defensively
  re-confirms uniqueness/monotonicity/no-duplicate-`(node, time)`-rows on
  the normalized output itself, and a new `_completeness_pct` helper
  refuses (`TemporalCompletenessError`) to ever report completeness above
  100% for any of the three products -- a hard failure, never a silent
  clamp. `compute_long_term_surface_current_statistics` gained a
  `completeness_pct` field for the first time (Section 7 of the ticket).

  Independently re-running `build-metocean-evidence` against the
  already-downloaded raw chunks (zero re-download confirmed: byte-identical
  raw file mtimes and an unchanged 111-entry manifest) and reopening all 9
  canonical outputs directly confirms **primary current: exactly 260,400
  canonical rows (18,600/node x 14 nodes), 100.0% completeness at every
  node, zero duplicate `(node, time)` rows, ALL CANONICAL METOCEAN TIME
  COORDINATES ARE UNIQUE AND MONOTONIC** -- matching the ticket's
  independently-derived expected count exactly. Long-term surface current
  (1,174,464 rows, 4 nodes) and wave (1,895,264 rows, 14 nodes) show the
  same zero-duplicate, 100.0%-completeness result. Removing exact
  duplicate rows left the aggregate current-speed/Hs/Tp statistics
  materially unchanged (duplicates carried the same values as their
  originals, so only the row/completeness accounting was wrong, not the
  distribution) -- the old (MAR-009A-era) on-disk data itself still
  exceeded 100% completeness, so old-vs-new comparison values for the
  absolute statistics are reported as `n/a` (unavailable) rather than
  computed from data the new strict check correctly refuses to trust; the
  two node-count comparisons (wave 14, long-term current 4) are unchanged,
  as expected. 19 new offline tests were added across
  `test_metocean_acquisition.py` and `test_metocean_evidence.py`, including
  realistic monthly-current, yearly-long-term-current, and yearly-wave
  boundary-overlap cases and an explicit NaN-consistency case; the full
  offline suite (517 tests) and repo-wide `ruff format`/`ruff check` pass
  clean. No bed shear stress, Shields parameter, sediment mobility,
  erosion/deposition, scour, free-span, fatigue, or risk scoring is
  computed anywhere in this ticket -- no further ticket has started.
- `MAR-010`: current-only near-bed normalization
  (`metocean/{current_normalization,current_map}.py`) -- the first step
  beyond forcing evidence, and the first static map in this project.
  Normalizes the MAR-009B corrected primary-current reference sample
  (`deepest_valid_standard_level_current`, 0.243-4.868 m above the
  Copernicus model bed) to a standard 1.0 m above that SAME model bed,
  using the ticket's fixed logarithmic velocity-profile ratio
  `S(z_t,z_r,z0) = [ln(z_t+z0)-ln(z0)]/[ln(z_r+z0)-ln(z0)]` --
  `uo_1m/vo_1m = S * uo_ref/vo_ref` preserves direction exactly (`S` is a
  positive scalar shared by both components). Canonical role name
  `CURRENT_ONLY_LOG_PROFILE_SENSITIVITY` throughout -- never "bed
  current"/"seabed current"/"combined near-bed current" (this is
  current-only; wave-current bottom-boundary-layer interaction is
  explicitly deferred, `current_wave_interaction_applied = false`).
  Roughness is run as five FIXED sensitivity scenarios (SILT 5e-6 m
  through GRAVEL 3e-4 m, consistent with long-standing DNV F105/F109
  roughness classes without claiming certified compliance) -- never a
  canonical PL854 seabed roughness choice, never a BGS-Folk mapping, never
  a D50-derived field, never averaged into a "best estimate"; the
  sensitivity envelope (min/max across the five) is itself the output.
  Every row also carries `z_r_over_h_model` and an explicitly-named
  `log_profile_vertical_domain_status` against a conservative 0.30
  project screening heuristic (never a universal physical threshold) --
  rows outside it get null normalized values, never a silent
  extrapolation. `NormalizationCompletenessError` mirrors MAR-009B's own
  completeness invariant for this derived product.

  Contiguous current-support map sections (`current_reference_segments.gpkg`)
  dissolve runs of chainage stations sharing one real support node using
  `shapely.ops.substring` on the true canonical route geometry -- never a
  straight chord between chainage points, never 941 independently-coloured
  25 m cells (honest ~1.5 km model spatial support). The required static
  map (`maps/pl854_reference_current_forcing.png`, matplotlib + rasterio,
  `Agg` backend, a newly-declared project dependency) colours the route by
  the assumption-minimal `current_reference_speed_p95_m_s` only -- never
  one arbitrary roughness scenario's 1 m value, never a risk judgement --
  with a muted EMODnet bathymetry background, KP labels, scale bar, north
  arrow, and the top-3 native-p95 sections called out; endpoints stay
  "Source geometry start"/"Source geometry terminus" per the project's
  established direction-honesty stance.

  Real execution against PL854 confirms the vertical screen independently
  (never hard-coded): **0 of 260,400 canonical rows fall outside the 0.30
  screen** (z_r/h_model max = 0.163), producing exactly 1,302,000 hourly
  sensitivity rows (260,400 x 5) and a 70-row (14 nodes x 5 scenarios)
  stats table, 14 contiguous map sections (one per real route-used node,
  tiling the full 23,480.67 m route with zero gaps), and a rendered PNG.
  40 new offline tests were added across `test_current_normalization.py`,
  `test_current_map.py`, and `test_cli.py` (including a full synthetic
  end-to-end CLI fixture); the full offline suite (557 tests) and
  repo-wide `ruff format`/`ruff check` pass clean. No bed shear stress,
  Shields parameter, sediment mobility, scour, free-span, erosion/
  deposition, or risk scoring is computed anywhere in this ticket -- no
  further ticket has started.
- `MAR-011`: wave-only spectral near-bed orbital velocity
  (`metocean/{wave_orbital,wave_orbital_map}.py`) -- converts the MAR-009B
  canonical wave evidence into a near-bed orbital-velocity forcing product
  using the FIXED Soulsby & Smallman irregular-wave spectral approximation
  (`Tn = sqrt(h/g)`, `t = Tn/Tz`, `A = [6500 + (0.56 + 15.54*t)^6]^(1/6)`,
  `Urms = 0.25*Hs / [Tn*(1+A*t^2)^3]`). Canonical role name
  `WAVE_ONLY_SPECTRAL_NEAR_BED_ORBITAL_VELOCITY` throughout -- never "bed
  current"/"combined wave-current velocity"/"bed shear stress". `Tz` is
  always Copernicus `VTM02` (`tm02_s`) -- changing observed `VTPK`
  (`tp_s`, preserved as a diagnostic) or `VTM10` (`tm10_s`, preserved as
  context) alone never changes the canonical Urms, confirmed by dedicated
  regression tests. Water depth `h` comes from the WAVE product's own
  static `deptho` at the same real wave support node -- never the
  canonical MAR-006 LAT depth, never a current-product bathymetry
  substitute, and no current data are read anywhere in this ticket. Every
  row also carries an explicitly-named `orbital_velocity_method_status`
  against a 0.30->0.54 method-accuracy calibration domain (never called a
  universal physical threshold) -- rows outside it keep their raw
  Hs/Tm02/Tp/depth values but get null canonical Urms/equivalent
  amplitude, never a silent out-of-domain extrapolation.
  `hs_over_model_depth` is reported purely as a non-breaking-assumption QA
  diagnostic (min/median/p95/p99/max) and never gates/rejects a row. A
  real edge case worth naming: an exactly-zero depth yields a
  mathematically finite (not NaN) `Tn`/`t` via plain propagation, which
  would otherwise slip through as spuriously "within domain" -- caught by
  an explicit `is_depth_and_period_valid` check before classification, not
  relied-upon incidental NaN propagation alone.

  Contiguous wave-support map sections
  (`wave_orbital_reference_segments.gpkg`) mirror MAR-010's honest-
  spatial-support approach independently (a self-contained module, so
  MAR-010's already-shipped map is never put at risk by this ticket's
  changes) -- true route geometry via `shapely.ops.substring`, never 941
  independently-coloured 25 m cells. The required static map
  (`maps/pl854_wave_orbital_forcing.png`) colours the route by the
  assumption-minimal `orbital_rms_p95_m_s` only -- never Hs/Tp/equivalent
  amplitude/direction/risk -- and applies presentation fixes from the
  MAR-010 map review: a landscape canvas sized from the TRUE displayed
  content (route + background raster) rather than the route alone, at
  most 3 hotspot labels stacked to avoid collisions, and simplified
  km-precision hotspot KP labels (`KP 6.14-8.16`) rather than the
  survey-grade `+metres` form still used for the canonical `kp_start`/
  `kp_end` segment attributes themselves.

  Real execution against PL854 confirms a genuine, expected finding
  (never hidden): **275,434 of 1,895,264 canonical rows (14.5%) fall
  outside the 0.54 calibration domain** (Tn/Tz median=0.428, max=1.301) --
  physically explained by short-period local wind-sea conditions (real
  Tm02 median 3.93 s) pushing `t` above the method's own accuracy domain;
  every one of those rows independently verified to retain its raw Hs
  while its canonical Urms is null, and every within-domain row (with
  valid Hs) has a real Urms (mean=0.029, p95=0.132, p99=0.238,
  max=0.717 m/s). Zero duplicate `(wave_node_id, time_utc)` rows, 14
  route-used wave nodes, 14 contiguous map sections tiling the full
  23,480.67 m route exactly as MAR-010's current sections did. 45 new
  offline tests were added across `test_wave_orbital.py`,
  `test_wave_orbital_map.py`, and `test_cli.py`; the full offline suite
  (602 tests) and repo-wide `ruff format`/`ruff check` pass clean. No bed
  shear stress, friction factor, Shields parameter, sediment mobility,
  erosion/deposition, scour, free-span, or risk scoring is computed
  anywhere in this ticket -- no further ticket has started.
