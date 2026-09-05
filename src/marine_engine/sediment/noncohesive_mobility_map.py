"""Hydro-pair route segments, capacity map, and chainage profile (MAR-013).

Map-ready spatial support reuses MAR-012's hydro pairs (Section 18)
-----------------------------------------------------------------------
Route sections are the SAME contiguous hydro-pair runs MAR-012 already
built and verified -- this module never re-derives node pairing, it only
attaches mobility-capacity attributes to that existing segmentation. Kept
as an independent, self-contained module (rather than importing
`combined_bed_shear_map`) so that already-shipped map is never put at risk
by a change made for this ticket.

Map honesty: discrete tested scenarios, never continuous D50 (Sections 19-22)
-----------------------------------------------------------------------------------
The primary map's route colour is
`largest_tested_d50_with_p95_mobility_ratio_ge_1_mm` -- one of exactly nine
FIXED tested values, rendered with a discrete (never continuously
interpolated) colour lookup so two segments can never visually imply an
untested grain size between them. A route section coloured e.g. 1.0 mm
means "of the tested scenarios, 1.0 mm is the largest whose p95 mobility
ratio reaches the incipient-motion threshold" -- it does NOT mean the
actual local D50 is 1.0 mm, that the whole seabed is mobile, that 1.0 mm
sediment exists there, or that erosion/scour/pipeline exposure will occur.
This interpretation is carried in the map footer and in metadata. Valid
observed BGS PSA D50 points are overlaid as small, visually distinct point
markers -- they are NEVER used to choose or shift the route's own colour.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from marine_engine.preprocessing.chainage import format_kp_label
from marine_engine.sediment.noncohesive_mobility import SCIENTIFIC_ROLE, TESTED_D50_SCENARIOS_MM

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NONCOHESIVE_MOBILITY_CAPACITY_SEGMENTS_COLUMNS = (
    "pipeline_id",
    "segment_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "hydro_pair_id",
    "largest_tested_d50_with_p90_mobility_ratio_ge_1_mm",
    "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm",
    "largest_tested_d50_with_p99_mobility_ratio_ge_1_mm",
    "largest_tested_d50_with_any_exceedance_mm",
    "p95_mobility_sequence_monotonic_nonincreasing",
    "monotonicity_violation_count",
    "mobility_ratio_p95_d50_125um",
    "mobility_ratio_p95_d50_250um",
    "mobility_ratio_p95_d50_500um",
    "mobility_ratio_p95_d50_1000um",
    "mobility_ratio_p95_d50_2000um",
    "mapped_250k_folk_class",
    "mapped_250k_nominal_scale",
    "nearest_valid_psa_id",
    "nearest_valid_psa_d50_mm",
    "nearest_valid_psa_distance_m",
    "nearest_valid_psa_sample_year",
    "scientific_role",
)

_REFERENCE_COLUMN_BY_MM = {
    0.125: "mobility_ratio_p95_d50_125um",
    0.250: "mobility_ratio_p95_d50_250um",
    0.500: "mobility_ratio_p95_d50_500um",
    1.000: "mobility_ratio_p95_d50_1000um",
    2.000: "mobility_ratio_p95_d50_2000um",
}


@dataclass(frozen=True)
class CapacityAttributes:
    """One hydro pair's already-computed capacity/QA/reference attributes.

    Everything here is looked up from `noncohesive_mobility` outputs
    already computed -- this module never recomputes mobility statistics
    itself.
    """

    largest_tested_d50_with_p90_mobility_ratio_ge_1_mm: float | None
    largest_tested_d50_with_p95_mobility_ratio_ge_1_mm: float | None
    largest_tested_d50_with_p99_mobility_ratio_ge_1_mm: float | None
    largest_tested_d50_with_any_exceedance_mm: float | None
    p95_mobility_sequence_monotonic_nonincreasing: bool | None
    monotonicity_violation_count: int | None
    reference_p95_ratios_by_mm: dict[float, float | None]


def _contiguous_runs(ids: pd.Series) -> pd.Series:
    """A run-id per row: increments only where the id actually changes.

    NaN-safe -- a run of consecutive missing assignments is still treated
    as one contiguous run rather than splitting on every row.
    """

    previous = ids.shift()
    changed = ~((ids == previous) | (ids.isna() & previous.isna()))
    changed.iloc[0] = True
    return changed.cumsum()


def build_noncohesive_mobility_capacity_segments(
    *,
    pipeline_id: str,
    route: LineString,
    chainage_hydro_df: pd.DataFrame,
    capacity_by_pair_id: dict[str, CapacityAttributes],
    observed_d50_context_df: pd.DataFrame,
    working_crs: str,
) -> gpd.GeoDataFrame:
    """Dissolve contiguous chainage-station runs sharing one hydro pair into map sections.

    `chainage_hydro_df` is one row per real chainage station: `chainage_m`,
    `hydro_pair_id`, `mapped_250k_folk_class`, `mapped_250k_nominal_scale`,
    sorted by chainage. Regional Folk-class context is the most common
    (mode) mapped class among a run's stations -- never converted to a
    numeric D50. `nearest_valid_psa_*` fields are looked up by comparing
    each segment's own chainage MIDPOINT against the five valid observed
    PSA points' own `nearest_pipeline_chainage_m` -- context only, never
    used to change a segment's own colour or capacity value (Section 18).
    """

    if chainage_hydro_df.empty:
        return gpd.GeoDataFrame(
            columns=list(NONCOHESIVE_MOBILITY_CAPACITY_SEGMENTS_COLUMNS), geometry=[]
        )

    ordered = chainage_hydro_df.sort_values("chainage_m").reset_index(drop=True)
    run_id = _contiguous_runs(ordered["hydro_pair_id"])
    total_length_m = route.length

    psa_chainages = (
        observed_d50_context_df["nearest_pipeline_chainage_m"].to_numpy(dtype=float)
        if not observed_d50_context_df.empty
        else np.array([])
    )

    runs = []
    for _, group in ordered.groupby(run_id, sort=True):
        folk_mode = group["mapped_250k_folk_class"].mode(dropna=True)
        scale_mode = group["mapped_250k_nominal_scale"].mode(dropna=True)
        runs.append(
            {
                "hydro_pair_id": group["hydro_pair_id"].iloc[0],
                "first_chainage_m": float(group["chainage_m"].iloc[0]),
                "last_chainage_m": float(group["chainage_m"].iloc[-1]),
                "mapped_250k_folk_class": folk_mode.iloc[0] if len(folk_mode) else None,
                "mapped_250k_nominal_scale": scale_mode.iloc[0] if len(scale_mode) else None,
            }
        )

    records = []
    geometries = []
    for i, run in enumerate(runs):
        start_chainage_m = (
            0.0 if i == 0 else (runs[i - 1]["last_chainage_m"] + run["first_chainage_m"]) / 2.0
        )
        end_chainage_m = (
            total_length_m
            if i == len(runs) - 1
            else (run["last_chainage_m"] + runs[i + 1]["first_chainage_m"]) / 2.0
        )
        segment_geom = substring(route, start_chainage_m, end_chainage_m, normalized=False)

        pair_id = run["hydro_pair_id"]
        attrs = capacity_by_pair_id.get(pair_id) if pd.notna(pair_id) else None
        attrs = attrs or CapacityAttributes(None, None, None, None, None, None, {})

        nearest_psa_id = nearest_psa_d50 = nearest_psa_distance = nearest_psa_year = None
        if len(psa_chainages):
            midpoint = (start_chainage_m + end_chainage_m) / 2.0
            distances = np.abs(psa_chainages - midpoint)
            idx = int(np.argmin(distances))
            psa_row = observed_d50_context_df.iloc[idx]
            nearest_psa_id = psa_row["psa_data_id"]
            nearest_psa_d50 = float(psa_row["d50_mm"])
            nearest_psa_distance = float(distances[idx])
            nearest_psa_year = psa_row["sample_year"]

        record = {
            "pipeline_id": pipeline_id,
            "segment_id": i,
            "start_chainage_m": start_chainage_m,
            "end_chainage_m": end_chainage_m,
            "kp_start": format_kp_label(start_chainage_m),
            "kp_end": format_kp_label(end_chainage_m),
            "hydro_pair_id": pair_id if pd.notna(pair_id) else None,
            "largest_tested_d50_with_p90_mobility_ratio_ge_1_mm": (
                attrs.largest_tested_d50_with_p90_mobility_ratio_ge_1_mm
            ),
            "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm": (
                attrs.largest_tested_d50_with_p95_mobility_ratio_ge_1_mm
            ),
            "largest_tested_d50_with_p99_mobility_ratio_ge_1_mm": (
                attrs.largest_tested_d50_with_p99_mobility_ratio_ge_1_mm
            ),
            "largest_tested_d50_with_any_exceedance_mm": (
                attrs.largest_tested_d50_with_any_exceedance_mm
            ),
            "p95_mobility_sequence_monotonic_nonincreasing": (
                attrs.p95_mobility_sequence_monotonic_nonincreasing
            ),
            "monotonicity_violation_count": attrs.monotonicity_violation_count,
            "mapped_250k_folk_class": run["mapped_250k_folk_class"],
            "mapped_250k_nominal_scale": run["mapped_250k_nominal_scale"],
            "nearest_valid_psa_id": nearest_psa_id,
            "nearest_valid_psa_d50_mm": nearest_psa_d50,
            "nearest_valid_psa_distance_m": nearest_psa_distance,
            "nearest_valid_psa_sample_year": nearest_psa_year,
            "scientific_role": SCIENTIFIC_ROLE,
        }
        for mm, column_name in _REFERENCE_COLUMN_BY_MM.items():
            record[column_name] = attrs.reference_p95_ratios_by_mm.get(mm)
        records.append(record)
        geometries.append(segment_geom)

    return gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs=working_crs,
        columns=list(NONCOHESIVE_MOBILITY_CAPACITY_SEGMENTS_COLUMNS),
    )


def write_noncohesive_mobility_capacity_segments_gpkg(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    layer: str = "noncohesive_mobility_capacity_segments",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


# --- Discrete grain-size colour scale (Section 19) ------------------------------------


def _discrete_d50_cmap_and_norm() -> tuple[Any, BoundaryNorm, list[float]]:
    """A discrete colormap + boundary norm with one colour bin per tested scenario.

    Boundaries sit at the LOG-space midpoint between consecutive tested
    values (the nine scenarios are geometrically, not linearly, spaced) --
    never a continuous interpolation between scenarios (Section 19).
    """

    values = sorted(TESTED_D50_SCENARIOS_MM)
    log_values = np.log10(values)
    boundaries = [log_values[0] - (log_values[1] - log_values[0]) / 2.0]
    for i in range(len(log_values) - 1):
        boundaries.append((log_values[i] + log_values[i + 1]) / 2.0)
    boundaries.append(log_values[-1] + (log_values[-1] - log_values[-2]) / 2.0)
    boundaries_linear = [10.0**b for b in boundaries]

    cmap = plt.get_cmap("viridis", len(values))
    norm = BoundaryNorm(boundaries_linear, cmap.N)
    return cmap, norm, values


def _colour_for_value(cmap: Any, norm: BoundaryNorm, value: float | None) -> tuple:
    if value is None or pd.isna(value):
        return (0.6, 0.6, 0.6, 1.0)
    return cmap(norm(value))


def _build_psa_overlay_points(
    psa_observations_df: pd.DataFrame, *, working_crs: str
) -> gpd.GeoDataFrame:
    """The five (or however many) valid observed PSA points, reprojected for overlay only.

    Never used to change route colour -- purely a visual, separate point
    layer (Section 21).
    """

    from marine_engine.sediment.noncohesive_mobility import build_observed_d50_context

    valid = build_observed_d50_context(psa_observations_df)
    if valid.empty or psa_observations_df.empty:
        return gpd.GeoDataFrame(columns=["psa_data_id", "d50_mm"], geometry=[], crs=working_crs)

    lookup = psa_observations_df.set_index("psa_data_id")[["longitude", "latitude"]]
    points = []
    for psa_id in valid["psa_data_id"]:
        row = lookup.loc[psa_id]
        points.append(Point(float(row["longitude"]), float(row["latitude"])))
    points_working = gpd.GeoSeries(points, crs="EPSG:4326").to_crs(working_crs)
    return gpd.GeoDataFrame(
        {"psa_data_id": valid["psa_data_id"].to_numpy(), "d50_mm": valid["d50_mm"].to_numpy()},
        geometry=list(points_working),
        crs=working_crs,
    )


# --- Static PNG map rendering (Sections 19-23) -----------------------------------------


def _kp_tick_chainages_m(total_length_m: float, interval_km: float = 5.0) -> list[float]:
    interval_m = interval_km * 1000.0
    ticks = list(np.arange(0.0, total_length_m, interval_m))
    if not ticks or not np.isclose(ticks[-1], total_length_m):
        ticks.append(total_length_m)
    return ticks


def _content_bounds(
    route: LineString, background_raster_path: Path | None
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = route.bounds
    if background_raster_path is not None and Path(background_raster_path).exists():
        try:
            import rasterio

            with rasterio.open(background_raster_path) as src:
                bounds = src.bounds
            minx, miny = min(minx, bounds.left), min(miny, bounds.bottom)
            maxx, maxy = max(maxx, bounds.right), max(maxy, bounds.top)
        except Exception:  # noqa: BLE001 -- background context is optional, never fatal
            pass
    return minx, miny, maxx, maxy


def _format_capacity_hotspot_label(row: pd.Series) -> str:
    kp_start_km = row["start_chainage_m"] / 1000.0
    kp_end_km = row["end_chainage_m"] / 1000.0
    value = row["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"]
    value_text = f"{value:g} mm" if pd.notna(value) else "none tested"
    return f"KP {kp_start_km:.1f}-{kp_end_km:.1f} | p95 capacity: {value_text}"


def render_noncohesive_mobility_capacity_map(
    *,
    segments_gdf: gpd.GeoDataFrame,
    route: LineString,
    working_crs: str,
    output_path: Path,
    psa_observations_df: pd.DataFrame | None = None,
    background_raster_path: Path | None = None,
    route_start_label: str = "Source geometry start",
    route_end_label: str = "Source geometry terminus",
    title: str = "PL854 — Noncohesive Sediment Mobility Capacity",
    subtitle: str = (
        "Soulsby wave–current skin stress + Soulsby–Whitehouse incipient-motion threshold"
    ),
    dpi: int = 150,
) -> Path:
    """Render the required static PL854 noncohesive-mobility-capacity PNG.

    Colours the route by `largest_tested_d50_with_p95_mobility_ratio_ge_1_mm`
    using a DISCRETE, stepped scale matching the nine tested grain-size
    scenarios -- never a continuously interpolated value between them.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = _content_bounds(route, background_raster_path)
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    fig_height = 7.0
    fig_width = max(8.0, min(18.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pad_x, pad_y = span_x * 0.06, span_y * 0.10
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if background_raster_path is not None and Path(background_raster_path).exists():
        _plot_background_raster(ax, Path(background_raster_path))

    value_column = "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"
    has_value = value_column in segments_gdf.columns and segments_gdf[value_column].notna().any()
    if has_value:
        cmap, norm, tick_values = _discrete_d50_cmap_and_norm()
        colours = [_colour_for_value(cmap, norm, v) for v in segments_gdf[value_column]]
        segments_gdf.plot(color=colours, linewidth=4, ax=ax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, ticks=tick_values)
        cbar.ax.set_yticklabels([f"{v:g}" for v in tick_values])
        cbar.set_label("Largest tested D50 with p95 mobility ratio ≥ 1 (mm)")
    else:
        segments_gdf.plot(color="0.4", linewidth=4, ax=ax)

    if psa_observations_df is not None and not psa_observations_df.empty:
        overlay = _build_psa_overlay_points(psa_observations_df, working_crs=working_crs)
        if not overlay.empty:
            ax.scatter(
                overlay.geometry.x,
                overlay.geometry.y,
                marker="^",
                s=60,
                facecolor="red",
                edgecolor="black",
                linewidth=0.8,
                zorder=6,
                label="Valid BGS PSA D50 observation — point evidence only",
            )
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    _add_kp_labels(ax, route)
    _add_endpoint_labels(ax, route, route_start_label, route_end_label)
    _add_scale_bar(ax)
    _add_north_arrow(ax)
    _label_top_sections(ax, segments_gdf)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=9, style="italic", pad=14)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    footer_lines = [
        "Route: NSTA | Hydrodynamics: Copernicus Marine | Sediment observations/mapping: "
        "BGS | Background: EMODnet",
        "Colours show hydrodynamic capacity for tested noncohesive grain sizes; no "
        "continuous route D50 has been assigned.",
        "BGS PSA D50 points are observations only and are not spatially interpolated.",
        "Cohesive/mixed-bed erosion, transport rate, scour and pipeline risk are not yet modelled.",
    ]
    ax.text(
        0.0,
        -0.10,
        "\n".join(footer_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.2",
        wrap=True,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_background_raster(ax, raster_path: Path) -> None:
    try:
        import rasterio

        with rasterio.open(raster_path) as src:
            band = src.read(1, masked=True)
            bounds = src.bounds
        ax.imshow(
            band,
            extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
            cmap="gray",
            alpha=0.35,
            zorder=0,
        )
    except Exception as exc:  # noqa: BLE001 -- background context is optional, never fatal
        print(f"note: could not render background bathymetry context: {exc}", file=sys.stderr)


def _add_kp_labels(ax, route: LineString) -> None:
    for chainage_m in _kp_tick_chainages_m(route.length):
        point = route.interpolate(chainage_m)
        ax.plot(point.x, point.y, marker="o", markersize=3, color="black", zorder=5)
        ax.annotate(
            format_kp_label(chainage_m),
            (point.x, point.y),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=7,
        )


def _add_endpoint_labels(ax, route: LineString, start_label: str, end_label: str) -> None:
    start, end = Point(route.coords[0]), Point(route.coords[-1])
    for point, label, offset in ((start, start_label, (8, -26)), (end, end_label, (8, -26))):
        ax.plot(point.x, point.y, marker="s", markersize=5, color="black", zorder=5)
        ax.annotate(
            label,
            (point.x, point.y),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            fontweight="bold",
        )


def _add_scale_bar(ax) -> None:
    xlim = ax.get_xlim()
    view_width_m = abs(xlim[1] - xlim[0])
    bar_km = 0.5
    for candidate_km in (0.5, 1, 2, 5, 10, 20, 50):
        if candidate_km * 1000.0 <= view_width_m * 0.35:
            bar_km = candidate_km
        else:
            break
    bar_m = bar_km * 1000.0

    x0 = xlim[0] + view_width_m * 0.05
    y0 = ax.get_ylim()[0] + abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
    ax.add_line(Line2D([x0, x0 + bar_m], [y0, y0], color="black", linewidth=2))
    ax.annotate(
        f"{bar_km:g} km",
        (x0 + bar_m / 2.0, y0),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=8,
    )


def _add_north_arrow(ax) -> None:
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = xlim[1] - abs(xlim[1] - xlim[0]) * 0.08
    y0 = ylim[0] + abs(ylim[1] - ylim[0]) * 0.08
    y1 = y0 + abs(ylim[1] - ylim[0]) * 0.08
    ax.annotate(
        "N",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.5},
        ha="center",
        fontsize=9,
        fontweight="bold",
    )


def _label_top_sections(ax, segments_gdf: gpd.GeoDataFrame, top_n: int = 3) -> None:
    value_column = "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"
    if value_column not in segments_gdf.columns:
        return
    ranked = segments_gdf.dropna(subset=[value_column]).nlargest(top_n, value_column)
    for rank, (_, row) in enumerate(ranked.iterrows()):
        centroid = row.geometry.centroid
        ax.annotate(
            _format_capacity_hotspot_label(row),
            (centroid.x, centroid.y),
            textcoords="offset points",
            xytext=(0, 16 + rank * 14),
            fontsize=7,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.5", "alpha": 0.85},
        )


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


# --- Secondary chainage profile PNG (Section 24) ---------------------------------------


def render_mobility_capacity_profile(
    *,
    segments_gdf: gpd.GeoDataFrame,
    psa_observations_df: pd.DataFrame | None,
    working_crs: str,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Chainage/KP profile of the tested-scenario mobility capacity (Section 24).

    Stepped per SEGMENT support (never a 25 m interpolated pseudo-
    resolution); valid PSA D50 observations are overlaid at their own
    nearest-pipeline chainage as visually distinct POINT context only.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    value_column = "largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"

    for _, row in segments_gdf.iterrows():
        value = row.get(value_column)
        if pd.isna(value):
            continue
        ax.plot(
            [row["start_chainage_m"] / 1000.0, row["end_chainage_m"] / 1000.0],
            [value, value],
            color="tab:blue",
            linewidth=3,
            solid_capstyle="butt",
        )

    if psa_observations_df is not None and not psa_observations_df.empty:
        overlay = _build_psa_overlay_points(psa_observations_df, working_crs=working_crs)
        if not overlay.empty:
            from marine_engine.sediment.noncohesive_mobility import build_observed_d50_context

            valid = build_observed_d50_context(psa_observations_df).set_index("psa_data_id")
            chainages_km = [
                valid.loc[psa_id, "nearest_pipeline_chainage_m"] / 1000.0
                for psa_id in overlay["psa_data_id"]
            ]
            ax.scatter(
                chainages_km,
                overlay["d50_mm"],
                marker="^",
                s=70,
                facecolor="red",
                edgecolor="black",
                linewidth=0.8,
                zorder=6,
                label="Valid BGS PSA D50 observation — point evidence only",
            )

    ax.set_yscale("log")
    ax.set_yticks(sorted(TESTED_D50_SCENARIOS_MM))
    ax.set_yticklabels([f"{v:g}" for v in sorted(TESTED_D50_SCENARIOS_MM)])
    ax.set_xlabel("Chainage (km)")
    ax.set_ylabel("D50 (mm)")
    ax.set_title(
        "PL854 — Noncohesive Mobility Capacity Profile\n"
        "Largest tested D50 with p95 mobility ratio ≥ 1, by route section",
        fontsize=11,
    )
    ax.grid(True, which="both", axis="y", alpha=0.3)
    if psa_observations_df is not None and not psa_observations_df.empty:
        ax.legend(loc="upper right", fontsize=8)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- Final scientific report (Section 29) -----------------------------------------------


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def print_noncohesive_mobility_report(
    *,
    stats_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    observed_d50_context_df: pd.DataFrame,
    segments_gdf: gpd.GeoDataFrame,
    segments_path: Path,
    png_path: Path,
    png_dimensions: tuple[int, int],
    profile_path: Path,
    profile_dimensions: tuple[int, int],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 Noncohesive Sediment Mobility Capacity (MAR-013) ===", ""]

    lines.append("## Threshold physics")
    for d50_mm in sorted(TESTED_D50_SCENARIOS_MM):
        rows = stats_df[stats_df["tested_d50_mm"] == d50_mm]
        tau_cr = rows["tau_critical_pa"].iloc[0] if len(rows) else None
        from marine_engine.sediment import noncohesive_mobility as ncm

        d50_m = d50_mm / 1000.0
        z0_skin_mm = ncm.compute_z0_skin_m(d50_m) * 1000.0
        d_star = float(ncm.compute_dimensionless_grain_size(d50_m))
        theta_cr = float(ncm.compute_soulsby_whitehouse_critical_shields_parameter(d_star))
        lines.append(
            f"  D50={d50_mm:g} mm | z0_skin={float(z0_skin_mm):.5f} mm | D*={d_star:.3f} | "
            f"theta_cr={theta_cr:.4f} | tau_cr={_fmt(tau_cr)} Pa"
        )
    lines.append("")

    lines.append("## Mobility (route-wide, across all hydro pairs)")
    if not stats_df.empty:
        for d50_mm, group in stats_df.groupby("tested_d50_mm", sort=True):
            ratio = group["mobility_ratio_p95"].dropna()
            exceedance_pct = group["threshold_exceedance_pct"].dropna()
            lines.append(
                f"  D50={d50_mm:g} mm: mobility_ratio p50/p90/p95/p99/max="
                f"{_fmt(group['mobility_ratio_p50'].median(), '.3f')}/"
                f"{_fmt(group['mobility_ratio_p90'].median(), '.3f')}/"
                f"{_fmt(ratio.median() if len(ratio) else None, '.3f')}/"
                f"{_fmt(group['mobility_ratio_p99'].median(), '.3f')}/"
                f"{_fmt(group['mobility_ratio_max'].max(), '.3f')} | "
                f"exceedance={_fmt(exceedance_pct.mean() if len(exceedance_pct) else None, '.1f')}%"
            )
    lines.append("")

    lines.append("## Capacity")
    if not capacity_df.empty:
        p95_values = capacity_df["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"].dropna()
        if len(p95_values):
            lines.append(
                f"  p95 capacity (mm) across sections: min={p95_values.min():g} "
                f"median={p95_values.median():g} max={p95_values.max():g}"
            )
        counts = capacity_df["largest_tested_d50_with_p95_mobility_ratio_ge_1_mm"].value_counts(
            dropna=False
        )
        for value, count in counts.items():
            label = f"{value:g} mm" if pd.notna(value) else "no tested scenario passes"
            lines.append(f"    sections at capacity {label}: {count}")
        violations = int(capacity_df["monotonicity_violation_count"].fillna(0).sum())
        lines.append(f"  Monotonicity violations (route-wide total): {violations}")
    lines.append("")

    lines.append("## Observed sediment context")
    if not observed_d50_context_df.empty:
        d50 = observed_d50_context_df["d50_mm"]
        dist = observed_d50_context_df["distance_to_pipeline_m"]
        years = observed_d50_context_df["sample_year"].dropna()
        lines.append(f"  Valid D50 PSA observations: {len(observed_d50_context_df)}")
        lines.append(
            f"  D50 (mm): min={d50.min():.3f} median={d50.median():.3f} max={d50.max():.3f}"
        )
        lines.append(
            "  Distance to pipeline (m): "
            f"min={dist.min():.1f} median={dist.median():.1f} max={dist.max():.1f}"
        )
        if len(years):
            lines.append(f"  Sample-year range: {int(years.min())}-{int(years.max())}")
    lines.append("  OBSERVED D50 POINTS WERE NOT INTERPOLATED OR ASSIGNED TO THE PIPELINE.")
    lines.append("")

    lines.append("## Regional mapped context")
    if not segments_gdf.empty:
        folk_classes = sorted(segments_gdf["mapped_250k_folk_class"].dropna().unique().tolist())
        lines.append(f"  Folk classes occurring along the route: {folk_classes}")
    lines.append("  BGS 250K FOLK CLASSES WERE NOT CONVERTED TO NUMERIC D50.")
    lines.append("")

    lines.append("## Map")
    lines.append(f"  Segment count: {len(segments_gdf)}")
    lines.append(f"  Map path: {png_path}")
    lines.append(f"  Map dimensions: {png_dimensions[0]} x {png_dimensions[1]} px")
    lines.append(f"  Profile path: {profile_path}")
    lines.append(f"  Profile dimensions: {profile_dimensions[0]} x {profile_dimensions[1]} px")
    lines.append("")

    lines.append(
        "MAP COLOURS SHOW THE LARGEST TESTED NONCOHESIVE GRAIN SIZE WHOSE P95 MOBILITY "
        "RATIO REACHES THE INCIPIENT-MOTION THRESHOLD."
    )
    lines.append(
        "THIS IS A HYDRODYNAMIC MOBILITY-CAPACITY PRODUCT, NOT A CONTINUOUS SITE-SPECIFIC "
        "SEDIMENT-MOBILITY OR RISK MAP."
    )
    lines.append(
        "COHESIVE/MIXED-BED THRESHOLDS, TRANSPORT RATE, EROSION, SCOUR AND PIPELINE "
        "EXPOSURE ARE NOT YET MODELLED."
    )

    print("\n".join(lines), file=file)
