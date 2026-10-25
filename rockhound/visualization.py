"""Matplotlib visualization helpers for a TerrainGrid and a planned path."""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

from .grid import TerrainGrid
from .lighting import SunPosition, cast_shadow_mask, hillshade

Cell = Tuple[int, int]


def plot_terrain(
    grid: TerrainGrid,
    path: Optional[List[Cell]] = None,
    texture: Optional[np.ndarray] = None,
    show_contours: bool = False,
    sun: Optional[SunPosition] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot elevation, no-go zones, rocks, and an optional planned path.

    `texture`, if given, is drawn as a faint overlay purely for visual
    reference -- it plays no role in traversability or planning.

    `sun`, if given, renders the elevation layer shaded for that Sun
    position (hillshade plus cast shadows) instead of the plain
    elevation colormap -- see rockhound.lighting for why this matters.
    Ignored when `show_contours` is True.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    extent = (0, grid.size_m, 0, grid.size_m)

    if show_contours:
        ax.contourf(grid.x_coords, grid.y_coords, grid.elevation, levels=20, cmap="terrain")
    elif sun is not None:
        cmap = plt.get_cmap("terrain")
        vmin, vmax = grid.elevation.min(), grid.elevation.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
        rgba = cmap(norm(grid.elevation))
        # A pure Lambertian shade (no ambient term) goes to near-black at
        # low Sun elevation -- physically real, but it makes the shadow
        # contrast this is meant to illustrate hard to actually see. This
        # floor is a display-only choice; hillshade() itself stays a clean
        # signal for anything (like the lidar-vs-stereo demo) that wants
        # the raw contrast trend rather than a rendering of it.
        display_shade = 0.15 + 0.85 * hillshade(grid, sun)
        rgba[..., :3] *= display_shade[..., None]
        rgba[..., :3][cast_shadow_mask(grid, sun)] *= 0.4
        ax.imshow(rgba, origin="lower", extent=extent)
    else:
        ax.imshow(grid.elevation, origin="lower", extent=extent, cmap="terrain")

    if texture is not None:
        ax.imshow(texture, origin="lower", extent=extent, cmap="gray", alpha=0.15)

    no_go = ~grid.traversable
    if np.any(no_go):
        overlay = np.zeros((*no_go.shape, 4))
        overlay[no_go] = (1.0, 0.0, 0.0, 0.45)
        ax.imshow(overlay, origin="lower", extent=extent)

    for rock in grid.rocks:
        circle = plt.Circle(
            (rock.x_m, rock.y_m),
            rock.radius_m,
            facecolor="saddlebrown",
            edgecolor="black",
            fill=True,
            alpha=0.6,
            linewidth=0.5,
        )
        ax.add_patch(circle)

    for crater in grid.craters:
        circle = plt.Circle(
            (crater.x_m, crater.y_m),
            crater.radius_m,
            facecolor="none",
            edgecolor="slategray",
            linewidth=1.2,
            linestyle=":",
        )
        ax.add_patch(circle)

    if np.any(no_go):
        # Outline no-go cells on top of everything else (rocks included) so
        # the impassable boundary is visible regardless of draw order.
        ax.contour(
            grid.x_coords, grid.y_coords, no_go.astype(float),
            levels=[0.5], colors="red", linewidths=1.8, linestyles="--",
        )

        # A no-go region a couple of cells wide is real (and A* correctly
        # routes around it) but can be all but invisible at map scale.
        # Boost anything smaller than a minimum on-screen size with a
        # solid marker so it's never missed.
        labeled, n_regions = ndimage.label(no_go)
        min_radius_m = grid.size_m * 0.012
        for label_id in range(1, n_regions + 1):
            rows, cols = np.where(labeled == label_id)
            area_m2 = len(rows) * grid.resolution_m**2
            actual_radius_m = np.sqrt(area_m2 / np.pi)
            if actual_radius_m < min_radius_m:
                cx, cy = grid.index_to_world(rows.mean(), cols.mean())
                marker = plt.Circle(
                    (cx, cy), min_radius_m,
                    facecolor="red", edgecolor="darkred", alpha=0.9, zorder=5,
                )
                ax.add_patch(marker)

    if path:
        xs = [grid.index_to_world(r, c)[0] for r, c in path]
        ys = [grid.index_to_world(r, c)[1] for r, c in path]
        ax.plot(xs, ys, color="cyan", linewidth=2.0, marker="o", markersize=2)

    legend_handles = []
    if grid.rocks:
        legend_handles.append(Patch(facecolor="saddlebrown", edgecolor="black", alpha=0.6, label="rock"))
    if grid.craters:
        legend_handles.append(Line2D([], [], color="slategray", linestyle=":", linewidth=1.2, label="crater rim"))
    if np.any(no_go):
        legend_handles.append(Line2D([], [], color="red", linestyle="--", linewidth=1.5, label="no-go zone"))
    if path:
        legend_handles.append(Line2D([], [], color="cyan", linewidth=2.0, label="planned path"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Terrain map (Sun el={sun.elevation_deg:.0f}°)" if sun is not None else "Terrain map")
    ax.set_xlim(0, grid.size_m)
    ax.set_ylim(0, grid.size_m)
    ax.set_aspect("equal")
    return ax
