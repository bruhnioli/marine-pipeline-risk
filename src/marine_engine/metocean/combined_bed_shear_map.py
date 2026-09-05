"""Contiguous hydro-pair route segments + the PL854 combined bed-shear map (MAR-012).

Map-ready spatial support built from BOTH nodes (Section 25)
------------------------------------------------------------------
A hydro pair combines one `current_node_id` and one `wave_node_id` --
segments are built from that COMBINED identity (`hydro_pair_id`), never
from either node id alone, so a hypothetical future run where the two
products do NOT share a support grid still produces honest segment
boundaries. Kept as an independent, self-contained module (rather than
importing from `current_map`/`wave_orbital_map`) so those already-shipped,
real-data-verified maps are never put at risk by a change made for this
ticket.

Map honesty: the colour is an upper bound, not a best estimate (Section 26)
--------------------------------------------------------------------------------
The main map colour is `tau_max_p95_sensitivity_max_pa` -- deliberately the
UPPER BOUND of the five roughness-sensitivity scenarios' own p95 combined
maximum stress, never a best estimate, probability, risk score, or
sediment-mobility threshold. Each of the (at most 3) hotspot annotations
shows the FULL envelope (`p95 min-max`), never merely the upper value used
for colouring.
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

from marine_engine.metocean.combined_bed_shear import SCIENTIFIC_ROLE
from marine_engine.preprocessing.chainage import format_kp_label

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COMBINED_BED_SHEAR_SEGMENTS_COLUMNS = (
    "pipeline_id",
    "segment_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "hydro_pair_id",
    "current_node_id",
    "wave_node_id",
    "source_node_distance_min_m",
    "source_node_distance_median_m",
    "source_node_distance_max_m",
    "tau_max_p95_sensitivity_min_pa",
    "tau_max_p95_sensitivity_max_pa",
    "tau_max_p95_sensitivity_width_pa",
    "tau_max_p99_sensitivity_min_pa",
    "tau_max_p99_sensitivity_max_pa",
    "tau_max_p99_sensitivity_width_pa",
    "overlap_start_time_utc",
    "overlap_end_time_utc",
    "scientific_role",
)


@dataclass(frozen=True)
class HydroPairReferenceAttributes:
    """One real hydro pair's already-computed sensitivity-envelope + alignment attributes.

    Everything here is looked up from `combined_bed_shear` outputs already
    computed -- this module never recomputes stress statistics itself.
    """

    tau_max_p95_sensitivity_min_pa: float | None
    tau_max_p95_sensitivity_max_pa: float | None
    tau_max_p95_sensitivity_width_pa: float | None
    tau_max_p99_sensitivity_min_pa: float | None
    tau_max_p99_sensitivity_max_pa: float | None
    tau_max_p99_sensitivity_width_pa: float | None
    overlap_start_time_utc: Any
    overlap_end_time_utc: Any


def _contiguous_runs(ids: pd.Series) -> pd.Series:
    """A run-id per row: increments only where the id actually changes.

    NaN-safe -- a run of consecutive missing assignments is still treated
    as one contiguous run rather than splitting on every row.
    """

    previous = ids.shift()
    changed = ~((ids == previous) | (ids.isna() & previous.isna()))
    changed.iloc[0] = True
    return changed.cumsum()


def build_combined_bed_shear_segments(
    *,
    pipeline_id: str,
    route: LineString,
    chainage_hydro_df: pd.DataFrame,
    node_attributes_by_id: dict[str, HydroPairReferenceAttributes],
    working_crs: str,
) -> gpd.GeoDataFrame:
    """Dissolve contiguous chainage-station runs sharing one hydro pair into map sections.

    `chainage_hydro_df` is one row per real chainage station: `chainage_m`,
    `hydro_pair_id`, `current_node_id`, `wave_node_id`,
    `current_node_distance_m`, `wave_node_distance_m`, sorted by chainage.
    Every station is consumed into exactly one segment -- never dropped,
    never duplicated. `source_node_distance_*` pools BOTH the current and
    wave station-to-node distances for the run, since a hydro pair is the
    combination of both source nodes, not either alone (Section 25).
    """

    if chainage_hydro_df.empty:
        return gpd.GeoDataFrame(columns=list(COMBINED_BED_SHEAR_SEGMENTS_COLUMNS), geometry=[])

    ordered = chainage_hydro_df.sort_values("chainage_m").reset_index(drop=True)
    run_id = _contiguous_runs(ordered["hydro_pair_id"])
    total_length_m = route.length

    runs = []
    for _, group in ordered.groupby(run_id, sort=True):
        pooled_distances = pd.concat(
            [group["current_node_distance_m"], group["wave_node_distance_m"]]
        ).dropna()
        runs.append(
            {
                "hydro_pair_id": group["hydro_pair_id"].iloc[0],
                "current_node_id": group["current_node_id"].iloc[0],
                "wave_node_id": group["wave_node_id"].iloc[0],
                "first_chainage_m": float(group["chainage_m"].iloc[0]),
                "last_chainage_m": float(group["chainage_m"].iloc[-1]),
                "distance_min_m": float(pooled_distances.min()) if len(pooled_distances) else None,
                "distance_median_m": float(pooled_distances.median())
                if len(pooled_distances)
                else None,
                "distance_max_m": float(pooled_distances.max()) if len(pooled_distances) else None,
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
        attrs = node_attributes_by_id.get(pair_id) if pd.notna(pair_id) else None
        attrs = attrs or HydroPairReferenceAttributes(*([None] * 8))

        records.append(
            {
                "pipeline_id": pipeline_id,
                "segment_id": i,
                "start_chainage_m": start_chainage_m,
                "end_chainage_m": end_chainage_m,
                "kp_start": format_kp_label(start_chainage_m),
                "kp_end": format_kp_label(end_chainage_m),
                "hydro_pair_id": pair_id if pd.notna(pair_id) else None,
                "current_node_id": run["current_node_id"]
                if pd.notna(run["current_node_id"])
                else None,
                "wave_node_id": run["wave_node_id"] if pd.notna(run["wave_node_id"]) else None,
                "source_node_distance_min_m": run["distance_min_m"],
                "source_node_distance_median_m": run["distance_median_m"],
                "source_node_distance_max_m": run["distance_max_m"],
                "tau_max_p95_sensitivity_min_pa": attrs.tau_max_p95_sensitivity_min_pa,
                "tau_max_p95_sensitivity_max_pa": attrs.tau_max_p95_sensitivity_max_pa,
                "tau_max_p95_sensitivity_width_pa": attrs.tau_max_p95_sensitivity_width_pa,
                "tau_max_p99_sensitivity_min_pa": attrs.tau_max_p99_sensitivity_min_pa,
                "tau_max_p99_sensitivity_max_pa": attrs.tau_max_p99_sensitivity_max_pa,
                "tau_max_p99_sensitivity_width_pa": attrs.tau_max_p99_sensitivity_width_pa,
                "overlap_start_time_utc": attrs.overlap_start_time_utc,
                "overlap_end_time_utc": attrs.overlap_end_time_utc,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
        geometries.append(segment_geom)

    return gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs=working_crs,
        columns=list(COMBINED_BED_SHEAR_SEGMENTS_COLUMNS),
    )


def write_combined_bed_shear_segments_gpkg(
    gdf: gpd.GeoDataFrame, output_path: Path, layer: str = "combined_bed_shear_segments"
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


# --- Static PNG map rendering (Section 26-27) -------------------------------------------


def _format_kp_km_range(start_chainage_m: float, end_chainage_m: float) -> str:
    """Simplified km-precision KP range for a MAP LABEL only -- carries forward the MAR-011A
    visual standard, never the survey-grade `+metres` form used for `kp_start`/`kp_end`."""

    return f"KP {start_chainage_m / 1000.0:.2f}–{end_chainage_m / 1000.0:.2f}"


def _format_combined_hotspot_label(row: pd.Series) -> str:
    """FULL envelope label, e.g. 'KP 12.30-14.10 | p95 0.18-0.24 Pa' -- never the upper value
    alone."""

    kp_range = _format_kp_km_range(row["start_chainage_m"], row["end_chainage_m"])
    low = row["tau_max_p95_sensitivity_min_pa"]
    high = row["tau_max_p95_sensitivity_max_pa"]
    return f"{kp_range} | p95 {low:.2f}-{high:.2f} Pa"


def _kp_tick_chainages_m(total_length_m: float, interval_km: float = 5.0) -> list[float]:
    """Chainages (m) at ~0, 5, 10, ... km plus the exact terminus."""

    interval_m = interval_km * 1000.0
    ticks = list(np.arange(0.0, total_length_m, interval_m))
    if not ticks or not np.isclose(ticks[-1], total_length_m):
        ticks.append(total_length_m)
    return ticks


def _content_bounds(
    route: LineString, background_raster_path: Path | None
) -> tuple[float, float, float, float]:
    """The TRUE displayed extent (route + background raster if any), for a figure aspect
    ratio that matches what is actually shown -- mirrors MAR-011A's own map sizing fix."""

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


def render_combined_bed_shear_map(
    *,
    segments_gdf: gpd.GeoDataFrame,
    route: LineString,
    working_crs: str,
    output_path: Path,
    background_raster_path: Path | None = None,
    route_start_label: str = "Source geometry start",
    route_end_label: str = "Source geometry terminus",
    title: str = "PL854 — Combined Wave–Current Bed Shear Stress",
    subtitle: str = "Soulsby algebraic interaction; contemporaneous current–wave overlap",
    dpi: int = 150,
) -> Path:
    """Render the required static PL854 combined bed-shear-sensitivity PNG.

    Colours the route by `tau_max_p95_sensitivity_max_pa` -- the UPPER
    BOUND across the five roughness-sensitivity scenarios, never a best
    estimate, probability, risk score, or sediment-mobility threshold.
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

    has_value = "tau_max_p95_sensitivity_max_pa" in segments_gdf.columns and (
        segments_gdf["tau_max_p95_sensitivity_max_pa"].notna().any()
    )
    if has_value:
        segments_gdf.plot(
            column="tau_max_p95_sensitivity_max_pa",
            cmap="inferno",
            linewidth=4,
            legend=True,
            legend_kwds={
                "label": "Combined maximum bed shear p95 — roughness sensitivity upper bound (Pa)",
                "shrink": 0.6,
            },
            ax=ax,
        )
    else:
        segments_gdf.plot(color="0.4", linewidth=4, ax=ax)

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
        "Route: NSTA | Current/Waves: Copernicus Marine | Background: EMODnet",
        "Colours show upper p95 bound across five seabed-roughness sensitivity scenarios; "
        "labels show the full p95 range.",
        "Combined record uses only contemporaneous 2024–2026 current-wave overlap; no "
        "sediment threshold or risk model is applied.",
    ]
    ax.text(
        0.0,
        -0.08,
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
    """Muted grayscale EMODnet bathymetry context, best-effort only."""

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
    """KP tick labels sit BELOW the route line; top-section labels sit above it
    (`_label_top_sections`) -- the two families are kept on opposite sides so they
    never overlap even when a KP tick and a labelled section share a location."""

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
    """Placed further below the line than the KP ticks so the KP 0/terminus tick
    label and the endpoint's own name never sit on top of each other."""

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
    """A simple, dependency-free scale bar sized to a round fraction of the view width."""

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
    """A simple fixed vertical north arrow (valid at this route's regional scale/CRS)."""

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
    """Label at most 3 sections by the same upper-bound colour variable -- each showing the
    FULL p95 envelope (min-max), never merely the upper value (Section 26). Always placed
    ABOVE the route line (opposite side from the KP ticks), stacked at increasing height per
    rank so nearby top sections never collide with each other either."""

    if "tau_max_p95_sensitivity_max_pa" not in segments_gdf.columns:
        return
    ranked = segments_gdf.dropna(subset=["tau_max_p95_sensitivity_max_pa"]).nlargest(
        top_n, "tau_max_p95_sensitivity_max_pa"
    )
    for rank, (_, row) in enumerate(ranked.iterrows()):
        centroid = row.geometry.centroid
        ax.annotate(
            _format_combined_hotspot_label(row),
            (centroid.x, centroid.y),
            textcoords="offset points",
            xytext=(0, 16 + rank * 14),
            fontsize=7,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.5", "alpha": 0.85},
        )


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    """(width_px, height_px), read back from the actual saved file.

    Never assumed from the figure's own inches x dpi -- `bbox_inches="tight"`
    crops on save, so the true pixel size must come from reopening the file.
    """

    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


# --- Final scientific report (Section 32) ------------------------------------------------


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def print_combined_bed_shear_report(
    *,
    alignment_summary: dict[str, Any],
    hydro_pairs_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    long_term_stats_df: pd.DataFrame,
    segments_gdf: gpd.GeoDataFrame,
    segments_path: Path,
    png_path: Path,
    png_dimensions: tuple[int, int],
    distance_diagnostics: dict[str, float | None] | None = None,
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 Combined Wave-Current Bed Shear Sensitivity (MAR-012) ===", ""]

    # --- Alignment ------------------------------------------------------------------
    lines.append("## Alignment")
    lines.append(f"  Current nodes: {hydro_pairs_df['current_node_id'].nunique()}")
    lines.append(f"  Wave nodes: {hydro_pairs_df['wave_node_id'].nunique()}")
    lines.append(f"  Reconciled hydro pairs: {len(hydro_pairs_df)}")
    if not hydro_pairs_df.empty:
        sep = hydro_pairs_df["coordinate_separation_m"]
        lines.append(
            "  Coordinate separation (m): min="
            f"{sep.min():.6f} median={sep.median():.6f} max={sep.max():.6f}"
        )
    lines.append(
        f"  Overlap: {alignment_summary.get('overlap_start_time_utc')} .. "
        f"{alignment_summary.get('overlap_end_time_utc')}"
    )
    lines.append(
        "  Expected 3-hour timestamps: "
        f"{alignment_summary.get('expected_3hour_count')} | Matched: "
        f"{alignment_summary.get('matched_timestamp_count')} | Completeness: "
        f"{_fmt(alignment_summary.get('completeness_pct'), '.1f')}%"
    )
    lines.append("")

    # --- Roughness scenarios ----------------------------------------------------------
    lines.append("## Roughness scenarios")
    if not stats_df.empty:
        for scenario, group in stats_df.groupby("roughness_scenario", sort=False):
            lines.append(
                f"  {scenario}: tau_current p95/p99/max="
                f"{_fmt(group['tau_current_p95_pa'].max())}/"
                f"{_fmt(group['tau_current_p99_pa'].max())}/"
                f"{_fmt(group['tau_current_max_pa'].max())} Pa | "
                f"tau_wave p95/p99/max="
                f"{_fmt(group['tau_wave_p95_pa'].max())}/"
                f"{_fmt(group['tau_wave_p99_pa'].max())}/"
                f"{_fmt(group['tau_wave_max_pa'].max())} Pa"
            )
            lines.append(
                "      tau_mean_combined p95/p99/max="
                f"{_fmt(group['tau_mean_combined_p95_pa'].max())}/"
                f"{_fmt(group['tau_mean_combined_p99_pa'].max())}/"
                f"{_fmt(group['tau_mean_combined_max_pa'].max())} Pa | "
                f"tau_max_combined p95/p99/max="
                f"{_fmt(group['tau_max_combined_p95_pa'].max())}/"
                f"{_fmt(group['tau_max_combined_p99_pa'].max())}/"
                f"{_fmt(group['tau_max_combined_max_pa'].max())} Pa"
            )
    lines.append("")

    # --- Interaction --------------------------------------------------------------------
    lines.append("## Interaction")
    if not stats_df.empty:
        angle_median = stats_df["wave_current_axis_angle_median_deg"].dropna()
        angle_p05 = stats_df["wave_current_axis_angle_p05_deg"].dropna()
        angle_p95 = stats_df["wave_current_axis_angle_p95_deg"].dropna()
        ratio_median = stats_df["tau_max_to_max_single_component_median_ratio"].dropna()
        ratio_p95 = stats_df["tau_max_to_max_single_component_p95_ratio"].dropna()
        lines.append(
            "  Wave-current axis angle (deg): median="
            f"{_fmt(angle_median.median() if len(angle_median) else None, '.1f')} "
            f"p05={_fmt(angle_p05.median() if len(angle_p05) else None, '.1f')} "
            f"p95={_fmt(angle_p95.median() if len(angle_p95) else None, '.1f')}"
        )
        lines.append(
            "  tau_max / max(tau_current,tau_wave): median="
            f"{_fmt(ratio_median.median() if len(ratio_median) else None, '.3f')} "
            f"p95={_fmt(ratio_p95.median() if len(ratio_p95) else None, '.3f')}"
        )
    lines.append("")

    # --- Sensitivity ----------------------------------------------------------------------
    lines.append("## Sensitivity")
    if not envelope_df.empty:
        widths = envelope_df["tau_max_p95_sensitivity_width_pa"].dropna()
        if len(widths):
            lines.append(
                "  p95 envelope width (Pa) across segments: "
                f"min={widths.min():.4f} median={widths.median():.4f} max={widths.max():.4f}"
            )
    if not stats_df.empty:
        min_counts = stats_df.loc[
            stats_df.groupby("hydro_pair_id")["tau_max_combined_p95_pa"].idxmin()
        ]["roughness_scenario"].value_counts()
        max_counts = stats_df.loc[
            stats_df.groupby("hydro_pair_id")["tau_max_combined_p95_pa"].idxmax()
        ]["roughness_scenario"].value_counts()
        if len(min_counts):
            lines.append(f"  Scenario giving minimum p95 most often: {min_counts.idxmax()}")
        if len(max_counts):
            lines.append(f"  Scenario giving maximum p95 most often: {max_counts.idxmax()}")
    lines.append("")

    # --- Long-term wave context -----------------------------------------------------------
    lines.append("## Long-term wave context")
    if not long_term_stats_df.empty:
        lines.append(
            "  Full-record tau_wave p95 (Pa): "
            f"{_fmt(long_term_stats_df['long_term_tau_wave_p95_pa'].max())}"
        )
        lines.append(
            "  Overlap tau_wave p95 (Pa): "
            f"{_fmt(long_term_stats_df['overlap_tau_wave_p95_pa'].max())}"
        )
        ratios = long_term_stats_df["overlap_to_long_term_tau_wave_p95_ratio"].dropna()
        ratio_median = ratios.median() if len(ratios) else None
        lines.append(f"  Overlap/full p95 ratio: median={_fmt(ratio_median, '.3f')}")
    lines.append("")

    # --- Map -----------------------------------------------------------------------------
    lines.append("## Map")
    lines.append(f"  Hydro-pair sections: {len(segments_gdf)}")
    if distance_diagnostics:
        lines.append(
            "  Station-to-source-node distance (m): min="
            f"{_fmt(distance_diagnostics.get('min_m'))} "
            f"median={_fmt(distance_diagnostics.get('median_m'))} "
            f"p95={_fmt(distance_diagnostics.get('p95_m'))} "
            f"max={_fmt(distance_diagnostics.get('max_m'))}"
        )
    lines.append(f"  GeoPackage: {segments_path}")
    lines.append(f"  PNG: {png_path}")
    lines.append(f"  PNG dimensions: {png_dimensions[0]} x {png_dimensions[1]} px")
    lines.append("")

    lines.append(
        "MAP COLOURS REPRESENT THE UPPER P95 BOUND ACROSS FIVE ROUGHNESS SENSITIVITY "
        "SCENARIOS; THIS IS NOT A BEST ESTIMATE OR RISK VALUE."
    )
    lines.append(
        "MAR-012 COMPUTES HYDRODYNAMIC BED SHEAR STRESS ONLY. NO SHIELDS THRESHOLD OR "
        "SEDIMENT MOBILITY HAS YET BEEN APPLIED."
    )
    lines.append("COMBINED STATISTICS USE ONLY THE CONTEMPORANEOUS PRIMARY-CURRENT / WAVE OVERLAP.")
    lines.append("")
    lines.append(
        "COMBINED BED-SHEAR STATISTICS ARE BASED ON THE CONTEMPORANEOUS PRIMARY-CURRENT / "
        "WAVE OVERLAP, NOT THE FULL 1980–2026 WAVE RECORD AND NOT A 25-YEAR RETURN-PERIOD "
        "ANALYSIS."
    )

    print("\n".join(lines), file=file)
