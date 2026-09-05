"""Contiguous wave-support route segments + the PL854 wave-orbital map (MAR-011).

Honest spatial support (Section 13)
--------------------------------------
The wave product has ~1.9 km longitude x 1.5 km latitude model support --
this module never draws one independently coloured feature per 25 m
chainage station. It dissolves every contiguous run of chainage stations
sharing one `wave_node_id` into a single map section using
`shapely.ops.substring` on the TRUE canonical pipeline geometry, exactly
mirroring MAR-010's `current_map.py` approach -- kept as an independent,
self-contained module (rather than importing from `current_map`) so
MAR-010's already-shipped, real-data-verified map is never put at risk by
a change made for this ticket.

Map presentation (Section 14-16, applying the MAR-010 review in Section 15)
---------------------------------------------------------------------------------
The main map colour is always `orbital_rms_p95_m_s` -- the scalar
spectral RMS near-bed orbital velocity, never Hs/Tp/equivalent amplitude/
direction/risk. A landscape canvas sized from the TRUE displayed content
(route + background raster, not the route alone) minimises empty canvas
space; at most 3 hotspot labels are shown, stacked at increasing height so
nearby ones never collide, and hotspot KP labels use simple km precision
(`KP 2.09-4.11`) rather than the survey-grade `+metres` form used for the
canonical `kp_start`/`kp_end` segment attributes themselves.
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

from marine_engine.metocean import wave_orbital
from marine_engine.metocean.wave_orbital import SCIENTIFIC_ROLE
from marine_engine.preprocessing.chainage import format_kp_label

matplotlib.use("Agg")  # deterministic, non-interactive, headless-safe -- must precede the
# pyplot import below, so it sits after the sorted import block rather than before it.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SOURCE_GRID_RESOLUTION_NOTE = "1.9±0.4 km longitude x 1.5 km latitude"

WAVE_ORBITAL_REFERENCE_SEGMENTS_COLUMNS = (
    "pipeline_id",
    "segment_id",
    "start_chainage_m",
    "end_chainage_m",
    "kp_start",
    "kp_end",
    "wave_node_id",
    "wave_node_distance_min_m",
    "wave_node_distance_median_m",
    "wave_node_distance_max_m",
    "wave_model_bathymetry_m",
    "hs_p95_m",
    "hs_p99_m",
    "hs_max_m",
    "tm02_median_s",
    "tm02_p95_s",
    "orbital_rms_mean_m_s",
    "orbital_rms_p95_m_s",
    "orbital_rms_p99_m_s",
    "orbital_rms_max_m_s",
    "orbital_amplitude_p95_m_s",
    "orbital_amplitude_p99_m_s",
    "orbital_amplitude_max_m_s",
    "source_grid_resolution_note",
    "scientific_role",
)


@dataclass(frozen=True)
class WaveNodeReferenceAttributes:
    """One real wave support node's already-computed reference attributes.

    Everything here is looked up from MAR-009B/MAR-011 outputs already on
    disk -- this module never recomputes wave statistics itself.
    """

    model_bathymetry_m: float | None
    hs_p95_m: float | None
    hs_p99_m: float | None
    hs_max_m: float | None
    tm02_median_s: float | None
    tm02_p95_s: float | None
    orbital_rms_mean_m_s: float | None
    orbital_rms_p95_m_s: float | None
    orbital_rms_p99_m_s: float | None
    orbital_rms_max_m_s: float | None
    orbital_amplitude_p95_m_s: float | None
    orbital_amplitude_p99_m_s: float | None
    orbital_amplitude_max_m_s: float | None


def _contiguous_runs(node_ids: pd.Series) -> pd.Series:
    """A run-id per row: increments only where the node id actually changes.

    NaN-safe -- a run of consecutive missing assignments is still treated
    as one contiguous run rather than splitting on every row.
    """

    previous = node_ids.shift()
    changed = ~((node_ids == previous) | (node_ids.isna() & previous.isna()))
    changed.iloc[0] = True
    return changed.cumsum()


def build_wave_orbital_reference_segments(
    *,
    pipeline_id: str,
    route: LineString,
    chainage_wave_df: pd.DataFrame,
    node_attributes_by_id: dict[str, WaveNodeReferenceAttributes],
    working_crs: str,
) -> gpd.GeoDataFrame:
    """Dissolve contiguous chainage-station runs sharing one wave node into map sections.

    `chainage_wave_df` is one row per real chainage station: `chainage_m`,
    `wave_node_id`, `wave_node_distance_m`, sorted by chainage. Every
    station is consumed into exactly one segment -- never dropped, never
    duplicated. Section boundaries fall at the midpoint between the last
    station of one run and the first station of the next, so sections
    tile the route with no gaps or overlaps.
    """

    if chainage_wave_df.empty:
        return gpd.GeoDataFrame(columns=list(WAVE_ORBITAL_REFERENCE_SEGMENTS_COLUMNS), geometry=[])

    ordered = chainage_wave_df.sort_values("chainage_m").reset_index(drop=True)
    run_id = _contiguous_runs(ordered["wave_node_id"])
    total_length_m = route.length

    runs = []
    for _, group in ordered.groupby(run_id, sort=True):
        distances = group["wave_node_distance_m"]
        runs.append(
            {
                "node_id": group["wave_node_id"].iloc[0],
                "first_chainage_m": float(group["chainage_m"].iloc[0]),
                "last_chainage_m": float(group["chainage_m"].iloc[-1]),
                "distance_min_m": float(distances.min()) if distances.notna().any() else None,
                "distance_median_m": float(distances.median()) if distances.notna().any() else None,
                "distance_max_m": float(distances.max()) if distances.notna().any() else None,
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
        attrs = attrs or WaveNodeReferenceAttributes(*([None] * 13))

        records.append(
            {
                "pipeline_id": pipeline_id,
                "segment_id": i,
                "start_chainage_m": start_chainage_m,
                "end_chainage_m": end_chainage_m,
                "kp_start": format_kp_label(start_chainage_m),
                "kp_end": format_kp_label(end_chainage_m),
                "wave_node_id": node_id if pd.notna(node_id) else None,
                "wave_node_distance_min_m": run["distance_min_m"],
                "wave_node_distance_median_m": run["distance_median_m"],
                "wave_node_distance_max_m": run["distance_max_m"],
                "wave_model_bathymetry_m": attrs.model_bathymetry_m,
                "hs_p95_m": attrs.hs_p95_m,
                "hs_p99_m": attrs.hs_p99_m,
                "hs_max_m": attrs.hs_max_m,
                "tm02_median_s": attrs.tm02_median_s,
                "tm02_p95_s": attrs.tm02_p95_s,
                "orbital_rms_mean_m_s": attrs.orbital_rms_mean_m_s,
                "orbital_rms_p95_m_s": attrs.orbital_rms_p95_m_s,
                "orbital_rms_p99_m_s": attrs.orbital_rms_p99_m_s,
                "orbital_rms_max_m_s": attrs.orbital_rms_max_m_s,
                "orbital_amplitude_p95_m_s": attrs.orbital_amplitude_p95_m_s,
                "orbital_amplitude_p99_m_s": attrs.orbital_amplitude_p99_m_s,
                "orbital_amplitude_max_m_s": attrs.orbital_amplitude_max_m_s,
                "source_grid_resolution_note": SOURCE_GRID_RESOLUTION_NOTE,
                "scientific_role": SCIENTIFIC_ROLE,
            }
        )
        geometries.append(segment_geom)

    return gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs=working_crs,
        columns=list(WAVE_ORBITAL_REFERENCE_SEGMENTS_COLUMNS),
    )


def write_wave_orbital_reference_segments_gpkg(
    gdf: gpd.GeoDataFrame, output_path: Path, layer: str = "wave_orbital_reference_segments"
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer)
    return output_path


# --- Static PNG map rendering (Section 14-16) ---------------------------------------


def _format_kp_km_range(start_chainage_m: float, end_chainage_m: float) -> str:
    """Simplified km-precision KP range for a MAP LABEL only (Section 15-C) --
    never the survey-grade `+metres` form used for the canonical `kp_start`/
    `kp_end` segment attributes."""

    return f"KP {start_chainage_m / 1000.0:.2f}–{end_chainage_m / 1000.0:.2f}"


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
    """The TRUE displayed extent (route + background raster if any), for a
    figure aspect ratio that matches what is actually shown (Section 15-A) --
    sizing from the route alone can under-count a wider background raster
    and leave unnecessary blank canvas space."""

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


def render_wave_orbital_map(
    *,
    segments_gdf: gpd.GeoDataFrame,
    route: LineString,
    working_crs: str,
    output_path: Path,
    background_raster_path: Path | None = None,
    route_start_label: str = "Source geometry start",
    route_end_label: str = "Source geometry terminus",
    title: str = "PL854 — Wave-Induced Near-Bed Orbital Forcing",
    subtitle: str = "Copernicus WWIII-AMM15; spectral Hm0 + Tm02; wave-only",
    dpi: int = 150,
) -> Path:
    """Render the required static PL854 wave-orbital-forcing PNG.

    Colours the route by `orbital_rms_p95_m_s` only -- never Hs alone,
    Tp, equivalent amplitude, direction, or a risk/classification value.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = _content_bounds(route, background_raster_path)
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    # A landscape canvas capped well below MAR-010's tall default (Section 15-A).
    fig_height = 7.0
    fig_width = max(8.0, min(18.0, fig_height * (span_x / span_y)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pad_x, pad_y = span_x * 0.06, span_y * 0.10
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if background_raster_path is not None and Path(background_raster_path).exists():
        _plot_background_raster(ax, Path(background_raster_path))

    has_speed = "orbital_rms_p95_m_s" in segments_gdf.columns and (
        segments_gdf["orbital_rms_p95_m_s"].notna().any()
    )
    if has_speed:
        segments_gdf.plot(
            column="orbital_rms_p95_m_s",
            cmap="magma",
            linewidth=4,
            legend=True,
            legend_kwds={"label": "Wave orbital velocity RMS p95 (m/s)", "shrink": 0.6},
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
        "Route: NSTA | Waves: Copernicus Marine NWSHELF_REANALYSIS_WAV_004_015 | "
        "Background: EMODnet",
        "Wave model support ~1.5–2 km; colours do not represent 25 m hydrodynamic resolution.",
        "Wave-current interaction and bed shear stress are not yet applied.",
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
    """Muted grayscale EMODnet bathymetry context, best-effort only (Section 14)."""

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
    """Label at most 3 sections by native p95 orbital RMS -- KP range + value
    (Section 14, 15-B/C). Always placed ABOVE the route line (opposite side
    from the KP ticks), stacked at increasing height per rank so nearby top
    sections never collide with each other either."""

    if "orbital_rms_p95_m_s" not in segments_gdf.columns:
        return
    ranked = segments_gdf.dropna(subset=["orbital_rms_p95_m_s"]).nlargest(
        top_n, "orbital_rms_p95_m_s"
    )
    for rank, (_, row) in enumerate(ranked.iterrows()):
        centroid = row.geometry.centroid
        kp_range = _format_kp_km_range(row["start_chainage_m"], row["end_chainage_m"])
        ax.annotate(
            f"{kp_range}: {row['orbital_rms_p95_m_s']:.3f} m/s",
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


# --- Final scientific report (Section 21) -------------------------------------------


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def print_wave_orbital_report(
    *,
    domain_summary: dict[str, Any],
    stats_df: pd.DataFrame,
    segments_gdf: gpd.GeoDataFrame,
    route_used_node_count: int,
    distance_diagnostics: dict[str, float | None],
    segments_path: Path,
    png_path: Path,
    png_dimensions: tuple[int, int],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 Wave-Only Spectral Near-Bed Orbital Velocity (MAR-011) ===", ""]

    lines.append("## Wave-input QA")
    s = domain_summary
    lines.append(f"  Route-used wave nodes: {route_used_node_count}")
    lines.append(
        "  Model bathymetry (m): min="
        f"{_fmt(s['model_bathymetry_m_min'])} median={_fmt(s['model_bathymetry_m_median'])} "
        f"max={_fmt(s['model_bathymetry_m_max'])}"
    )
    lines.append(
        f"  Hs (m): mean={_fmt(s['hs_m_mean'])} p95={_fmt(s['hs_m_p95'])} "
        f"p99={_fmt(s['hs_m_p99'])} max={_fmt(s['hs_m_max'])}"
    )
    lines.append(f"  Tm02 (s): median={_fmt(s['tm02_s_median'])} p95={_fmt(s['tm02_s_p95'])}")
    lines.append(f"  Observed Tp (s): median={_fmt(s['tp_s_median'])} p95={_fmt(s['tp_s_p95'])}")
    lines.append(
        "  Hs/model-depth: min="
        f"{_fmt(s['hs_over_model_depth_min'])} median={_fmt(s['hs_over_model_depth_median'])} "
        f"p95={_fmt(s['hs_over_model_depth_p95'])} p99={_fmt(s['hs_over_model_depth_p99'])} "
        f"max={_fmt(s['hs_over_model_depth_max'])}"
    )
    lines.append(
        "  Tn/Tz: min="
        f"{_fmt(s['t_parameter_min'], '.4f')} median={_fmt(s['t_parameter_median'], '.4f')} "
        f"p95={_fmt(s['t_parameter_p95'], '.4f')} max={_fmt(s['t_parameter_max'], '.4f')}"
    )
    lines.append(
        f"  Rows outside {wave_orbital.CALIBRATION_DOMAIN_MAX_T:.2f} calibration domain: "
        f"{s['rows_outside_calibration_domain']} / {s['total_rows']}"
    )
    lines.append("")

    lines.append("## Spectral near-bed orbital velocity")
    if not stats_df.empty:
        lines.append(
            "  Urms (m/s): mean="
            f"{_fmt(stats_df['orbital_rms_mean_m_s'].mean())} "
            f"p95={_fmt(stats_df['orbital_rms_p95_m_s'].max())} "
            f"p99={_fmt(stats_df['orbital_rms_p99_m_s'].max())} "
            f"max={_fmt(stats_df['orbital_rms_max_m_s'].max())}"
        )
        lines.append(
            "  Equivalent amplitude (m/s): mean="
            f"{_fmt(stats_df['orbital_amplitude_mean_m_s'].mean())} "
            f"p95={_fmt(stats_df['orbital_amplitude_p95_m_s'].max())} "
            f"p99={_fmt(stats_df['orbital_amplitude_p99_m_s'].max())} "
            f"max={_fmt(stats_df['orbital_amplitude_max_m_s'].max())}"
        )
        lines.append(
            "  Peak-period diagnostic (observed Tp / (1.28*Tz)): "
            f"p05={_fmt(stats_df['tp_observed_to_equivalent_p05_ratio'].mean(), '.3f')} "
            f"median={_fmt(stats_df['tp_observed_to_equivalent_median_ratio'].mean(), '.3f')} "
            f"p95={_fmt(stats_df['tp_observed_to_equivalent_p95_ratio'].mean(), '.3f')}"
        )
    lines.append("")

    lines.append("## Map support")
    lines.append(f"  Route-used wave nodes: {route_used_node_count}")
    lines.append(f"  Contiguous rendered sections: {len(segments_gdf)}")
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
        "MAP COLOURS REPRESENT SPECTRAL RMS WAVE-ORBITAL VELOCITY P95, NOT SIGNIFICANT "
        "WAVE HEIGHT, BED SHEAR STRESS, OR RISK."
    )
    lines.append(
        "MAR-011 IS WAVE-ONLY; CURRENT EFFECTS AND WAVE-CURRENT BOTTOM BOUNDARY-LAYER "
        "INTERACTION HAVE NOT YET BEEN APPLIED."
    )

    print("\n".join(lines), file=file)
