"""Contiguous current-support route segments + the PL854 reference-current map (MAR-010).

Honest spatial support (Section 8)
--------------------------------------
The primary current product has ~1.5 km model support and only a handful
of route-used support nodes -- this module never draws one independently
coloured feature per 25 m chainage station (that would misrepresent 941
fabricated independent observations as real spatial resolution). Instead
it walks the chainage-to-current-node assignment in route order and
dissolves every contiguous run of stations sharing one `current_node_id`
into a single map section, using `shapely.ops.substring` to extract the
TRUE pipeline geometry between two chainage distances -- never a straight
chord between chainage points. A segment boundary is placed at the
midpoint between the last station of one run and the first station of the
next, so sections tile the route with no gaps or overlaps.

Map honesty (Section 9)
---------------------------
The main map colour is `current_reference_speed_p95_m_s` -- the
assumption-minimal, corrected MAR-009B reference-current p95, never one
arbitrary roughness scenario's 1 m-normalized value and never a risk
judgement. Endpoint labels stay "Source geometry start"/"Source geometry
terminus" (`preprocessing/chainage.py`'s own direction-honesty stance: the
canonical schema carries no authoritative from/to field, so this module
never relabels them Anglia/LOGGS).
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

from marine_engine.metocean import current_normalization
from marine_engine.metocean.current_normalization import SCIENTIFIC_ROLE
from marine_engine.preprocessing.chainage import format_kp_label

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SOURCE_GRID_NOMINAL_RESOLUTION_M = 1500.0

CURRENT_REFERENCE_SEGMENTS_COLUMNS = (
    "pipeline_id",
    "segment_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "current_node_id",
    "current_node_distance_min_m",
    "current_node_distance_median_m",
    "current_node_distance_max_m",
    "current_model_bathymetry_m",
    "current_reference_height_m",
    "current_reference_speed_mean_m_s",
    "current_reference_speed_p95_m_s",
    "current_reference_speed_p99_m_s",
    "current_reference_speed_max_m_s",
    "current_only_1m_p95_sensitivity_min_m_s",
    "current_only_1m_p95_sensitivity_max_m_s",
    "current_only_1m_p95_sensitivity_width_m_s",
    "source_grid_nominal_resolution_m",
    "scientific_role",
)


@dataclass(frozen=True)
class NodeReferenceAttributes:
    """One real current support node's already-computed reference attributes.

    Everything here is looked up from MAR-009B/MAR-010 outputs already on
    disk -- this module never recomputes current statistics itself.
    """

    model_bathymetry_m: float | None
    reference_height_m: float | None
    speed_mean_m_s: float | None
    speed_p95_m_s: float | None
    speed_p99_m_s: float | None
    speed_max_m_s: float | None
    sensitivity_p95_min_m_s: float | None
    sensitivity_p95_max_m_s: float | None
    sensitivity_p95_width_m_s: float | None


def _contiguous_runs(node_ids: pd.Series) -> pd.Series:
    """A run-id per row: increments only where the node id actually changes.

    NaN-safe -- a run of consecutive missing assignments is still treated
    as one contiguous run rather than splitting on every row (pandas'
    plain `!=` is NOT NaN-safe: `NaN != NaN` is `True`, which would
    otherwise fragment every unassigned station into its own segment).
    """

    previous = node_ids.shift()
    changed = ~((node_ids == previous) | (node_ids.isna() & previous.isna()))
    changed.iloc[0] = True
    return changed.cumsum()


def build_current_reference_segments(
    *,
    pipeline_id: str,
    route: LineString,
    chainage_current_df: pd.DataFrame,
    node_attributes_by_id: dict[str, NodeReferenceAttributes],
    working_crs: str,
) -> gpd.GeoDataFrame:
    """Dissolve contiguous chainage-station runs sharing one node into map sections.

    `chainage_current_df` is one row per real chainage station (as in
    `chainage_metocean_evidence.parquet`): `chainage_m`, `current_node_id`,
    `current_node_distance_m`, sorted by chainage. Every station is
    consumed into exactly one segment -- never dropped, never duplicated.
    """

    if chainage_current_df.empty:
        return gpd.GeoDataFrame(columns=list(CURRENT_REFERENCE_SEGMENTS_COLUMNS), geometry=[])

    ordered = chainage_current_df.sort_values("chainage_m").reset_index(drop=True)
    run_id = _contiguous_runs(ordered["current_node_id"])
    total_length_m = route.length

    runs = []
    for _, group in ordered.groupby(run_id, sort=True):
        runs.append(
            {
                "node_id": group["current_node_id"].iloc[0],
                "first_chainage_m": float(group["chainage_m"].iloc[0]),
                "last_chainage_m": float(group["chainage_m"].iloc[-1]),
                "distance_min_m": float(group["current_node_distance_m"].min())
                if group["current_node_distance_m"].notna().any()
                else None,
                "distance_median_m": float(group["current_node_distance_m"].median())
                if group["current_node_distance_m"].notna().any()
                else None,
                "distance_max_m": float(group["current_node_distance_m"].max())
                if group["current_node_distance_m"].notna().any()
                else None,
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

        node_id = run["node_id"]
        attrs = node_attributes_by_id.get(node_id) if pd.notna(node_id) else None
        attrs = attrs or NodeReferenceAttributes(*([None] * 9))

        records.append(
            {
                "pipeline_id": pipeline_id,
                "segment_id": i,
                "start_chainage_m": start_chainage_m,
                "end_chainage_m": end_chainage_m,
                "kp_start": format_kp_label(start_chainage_m),
                "kp_end": format_kp_label(end_chainage_m),
                "current_node_id": node_id if pd.notna(node_id) else None,
                "current_node_distance_min_m": run["distance_min_m"],
                "current_node_distance_median_m": run["distance_median_m"],
                "current_node_distance_max_m": run["distance_max_m"],
                "current_model_bathymetry_m": attrs.model_bathymetry_m,
                "current_reference_height_m": attrs.reference_height_m,
                "current_reference_speed_mean_m_s": attrs.speed_mean_m_s,
                "current_reference_speed_p95_m_s": attrs.speed_p95_m_s,
                "current_reference_speed_p99_m_s": attrs.speed_p99_m_s,
                "current_reference_speed_max_m_s": attrs.speed_max_m_s,
                "current_only_1m_p95_sensitivity_min_m_s": attrs.sensitivity_p95_min_m_s,
                "current_only_1m_p95_sensitivity_max_m_s": attrs.sensitivity_p95_max_m_s,
                "current_only_1m_p95_sensitivity_width_m_s": attrs.sensitivity_p95_width_m_s,
                "source_grid_nominal_resolution_m": SOURCE_GRID_NOMINAL_RESOLUTION_M,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
        geometries.append(segment_geom)

    return gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs=working_crs,
        columns=list(CURRENT_REFERENCE_SEGMENTS_COLUMNS),
    )


def write_current_reference_segments_gpkg(
    gdf: gpd.GeoDataFrame, output_path: Path, layer: str = "current_reference_segments"
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


# --- Static PNG map rendering (Section 9-10) ----------------------------------------


def _kp_tick_chainages_m(total_length_m: float, interval_km: float = 5.0) -> list[float]:
    """Chainages (m) at ~0, 5, 10, ... km plus the exact terminus."""

    interval_m = interval_km * 1000.0
    ticks = list(np.arange(0.0, total_length_m, interval_m))
    if not ticks or not np.isclose(ticks[-1], total_length_m):
        ticks.append(total_length_m)
    return ticks


def render_reference_current_map(
    *,
    segments_gdf: gpd.GeoDataFrame,
    route: LineString,
    working_crs: str,
    output_path: Path,
    background_raster_path: Path | None = None,
    route_start_label: str = "Source geometry start",
    route_end_label: str = "Source geometry terminus",
    title: str = "PL854 — Reference Current Forcing",
    subtitle: str = (
        "Copernicus ~1.5 km primary current; p95 speed at deepest physically valid standard level"
    ),
    dpi: int = 150,
) -> Path:
    """Render the required static PL854 reference-current-forcing PNG.

    Colours the route by `current_reference_speed_p95_m_s` only -- never a
    roughness-selected 1 m sensitivity value, never a risk judgement.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = route.bounds
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    fig_height = 10.0
    fig_width = max(6.0, min(16.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if background_raster_path is not None and Path(background_raster_path).exists():
        _plot_background_raster(ax, Path(background_raster_path))

    has_speed = "current_reference_speed_p95_m_s" in segments_gdf.columns and (
        segments_gdf["current_reference_speed_p95_m_s"].notna().any()
    )
    if has_speed:
        segments_gdf.plot(
            column="current_reference_speed_p95_m_s",
            cmap="viridis",
            linewidth=4,
            legend=True,
            legend_kwds={"label": "Reference current speed p95 (m/s)", "shrink": 0.6},
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

    note_lines = [
        "Model support ~1.5 km — colours must not be interpreted as 25 m hydrodynamic resolution.",
        "1 m current-only log-profile sensitivity is available in the segment attributes; "
        "wave-current interaction is not yet applied.",
    ]
    ax.text(
        0.0,
        -0.06,
        "\n".join(note_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.25",
        wrap=True,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_background_raster(ax, raster_path: Path) -> None:
    """Muted grayscale EMODnet bathymetry context, best-effort only (Section 9)."""

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
    """Placed further below the line than the KP ticks (Section 9) so the KP 0/terminus
    tick label and the endpoint's own name never sit on top of each other."""

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
    """Label only the top-N sections by native p95 current -- KP range + value (Section 9).

    Always placed ABOVE the route line (opposite side from the KP ticks),
    and stacked at increasing height per rank so two nearby top sections
    never overlap each other either.
    """

    if "current_reference_speed_p95_m_s" not in segments_gdf.columns:
        return
    ranked = segments_gdf.dropna(subset=["current_reference_speed_p95_m_s"]).nlargest(
        top_n, "current_reference_speed_p95_m_s"
    )
    for rank, (_, row) in enumerate(ranked.iterrows()):
        centroid = row.geometry.centroid
        ax.annotate(
            f"{row['kp_start']}–{row['kp_end']}: {row['current_reference_speed_p95_m_s']:.3f} m/s",
            (centroid.x, centroid.y),
            textcoords="offset points",
            xytext=(0, 16 + rank * 14),
            fontsize=7,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.5", "alpha": 0.85},
        )


def read_png_dimensions(png_path: Path) -> tuple[int, int]:
    """(width_px, height_px), read back from the actual saved file (Section 14/15).

    Never assumed from the figure's own inches x dpi -- `bbox_inches="tight"`
    crops on save, so the true pixel size must come from reopening the file.
    """

    image = plt.imread(png_path)
    height_px, width_px = image.shape[0], image.shape[1]
    return width_px, height_px


# --- Final scientific report (Section 15) -------------------------------------------


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def print_current_normalization_report(
    *,
    vertical_domain_summary: dict[str, Any],
    sensitivity_stats_df: pd.DataFrame,
    segments_gdf: gpd.GeoDataFrame,
    route_used_node_count: int,
    distance_diagnostics: dict[str, float | None],
    segments_path: Path,
    png_path: Path,
    png_dimensions: tuple[int, int],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 Current-Only Near-Bed Normalization (MAR-010) ===", ""]

    lines.append("## Vertical-domain QA")
    s = vertical_domain_summary
    lines.append(
        f"  z_r (m): min={_fmt(s['z_r_min'])} median={_fmt(s['z_r_median'])} "
        f"p95={_fmt(s['z_r_p95'])} max={_fmt(s['z_r_max'])}"
    )
    lines.append(
        "  Model bathymetry (m): min="
        f"{_fmt(s['model_bathymetry_m_min'])} median={_fmt(s['model_bathymetry_m_median'])} "
        f"max={_fmt(s['model_bathymetry_m_max'])}"
    )
    lines.append(
        f"  z_r/h_model: min={_fmt(s['z_r_over_h_min'])} median={_fmt(s['z_r_over_h_median'])} "
        f"p95={_fmt(s['z_r_over_h_p95'])} max={_fmt(s['z_r_over_h_max'])}"
    )
    lines.append(
        f"  Rows outside {current_normalization.VERTICAL_DOMAIN_SCREEN_FRACTION:.2f} screen: "
        f"{s['rows_outside_screen']} / {s['total_reference_rows']}"
    )
    lines.append("")

    lines.append("## 1 m normalization sensitivity")
    for scenario_name, z0 in current_normalization.ROUGHNESS_SCENARIOS_M:
        scenario_rows = sensitivity_stats_df[
            sensitivity_stats_df["roughness_scenario"] == scenario_name
        ]
        if scenario_rows.empty:
            lines.append(f"  {scenario_name} (z0={z0:g} m): no valid rows")
            continue
        lines.append(
            f"  {scenario_name} (z0={z0:g} m): scale factor "
            f"min={_fmt(scenario_rows['scale_factor_min'].min(), '.4f')} "
            f"median={_fmt(scenario_rows['scale_factor_median'].median(), '.4f')} "
            f"p95={_fmt(scenario_rows['scale_factor_p95'].max(), '.4f')} "
            f"max={_fmt(scenario_rows['scale_factor_max'].max(), '.4f')} | speed (m/s) "
            f"mean={_fmt(scenario_rows['speed_1m_mean_m_s'].mean())} "
            f"p95={_fmt(scenario_rows['speed_1m_p95_m_s'].max())} "
            f"p99={_fmt(scenario_rows['speed_1m_p99_m_s'].max())} "
            f"max={_fmt(scenario_rows['speed_1m_max_m_s'].max())}"
        )
    if "speed_1m_p95_sensitivity_width_m_s" in sensitivity_stats_df.columns:
        widths = sensitivity_stats_df.drop_duplicates("current_node_id")[
            "speed_1m_p95_sensitivity_width_m_s"
        ].dropna()
        if len(widths):
            lines.append(
                "  Across scenarios -- p95 sensitivity envelope width (m/s): "
                f"min={widths.min():.3f} median={widths.median():.3f} max={widths.max():.3f}"
            )
    lines.append("")

    lines.append("## Map support")
    lines.append(f"  Route-used current nodes: {route_used_node_count}")
    lines.append(f"  Contiguous rendered current sections: {len(segments_gdf)}")
    lines.append(
        "  Station-to-node distance (m): min="
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
        "MAP COLOURS REPRESENT NATIVE CORRECTED REFERENCE-CURRENT P95, NOT A "
        "ROUGHNESS-SELECTED OR RISK VALUE"
    )
    lines.append("MAR-010 IS CURRENT-ONLY; WAVE-CURRENT INTERACTION HAS NOT YET BEEN APPLIED")

    print("\n".join(lines), file=file)
