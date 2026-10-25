"""Grid-based terrain representation.

TerrainGrid is a numpy-backed 2D map of square cells. It exposes a
generic read/write interface and has no knowledge of how its cell data
was produced -- procedural generation, a lidar scan, or a rosbag
replay are all just "populator" functions/callables that write into a
TerrainGrid through this interface (see rockhound.rocks for one such
populator). Keep it that way: no generation logic belongs in this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Rock:
    """Metadata for one rock recorded in a TerrainGrid's rock list.

    This is a plain data record, not tied to any particular populator:
    a procedural generator and a future lidar-based populator both
    produce these.
    """

    x_m: float
    y_m: float
    radius_m: float
    height_m: float
    source: str = "unknown"


@dataclass
class Crater:
    """Metadata for one crater recorded in a TerrainGrid's crater list.

    A second, independent hazard type alongside Rock: negative-relief
    rather than positive-relief, but stored the same way -- a plain
    data record any populator (procedural or sensor-based) can produce.
    """

    x_m: float
    y_m: float
    radius_m: float
    depth_m: float
    source: str = "unknown"


class TerrainGrid:
    """A square, uniformly-spaced 2D terrain map.

    Backed by three co-indexed numpy arrays of shape (n, n):
      - elevation:   float64, meters
      - traversable: bool, True = go, False = no-go
      - rock_id:     int32, -1 if no rock occupies the cell, else an
                     index into `self.rocks`

    Array index [row, col] maps to world position via
    `index_to_world`/`world_to_index`; row increases with y, column
    increases with x.
    """

    def __init__(self, size_m: float, resolution_m: float) -> None:
        if size_m <= 0 or resolution_m <= 0:
            raise ValueError("size_m and resolution_m must be positive")
        if resolution_m > size_m:
            raise ValueError("resolution_m must not exceed size_m")

        self.size_m = float(size_m)
        self.resolution_m = float(resolution_m)
        self.n = int(round(size_m / resolution_m))

        self.elevation: np.ndarray = np.zeros((self.n, self.n), dtype=np.float64)
        self.traversable: np.ndarray = np.ones((self.n, self.n), dtype=bool)
        self.rock_id: np.ndarray = np.full((self.n, self.n), -1, dtype=np.int32)

        self.rocks: List[Rock] = []

        self.crater_id: np.ndarray = np.full((self.n, self.n), -1, dtype=np.int32)
        self.craters: List[Crater] = []

        self._x_coords: np.ndarray | None = None
        self._y_coords: np.ndarray | None = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape

    @property
    def x_coords(self) -> np.ndarray:
        """(n, n) array of each cell's center x-coordinate, in meters."""
        if self._x_coords is None:
            self._build_coord_grids()
        return self._x_coords  # type: ignore[return-value]

    @property
    def y_coords(self) -> np.ndarray:
        """(n, n) array of each cell's center y-coordinate, in meters."""
        if self._y_coords is None:
            self._build_coord_grids()
        return self._y_coords  # type: ignore[return-value]

    def _build_coord_grids(self) -> None:
        centers = (np.arange(self.n) + 0.5) * self.resolution_m
        col_x, row_y = np.meshgrid(centers, centers)
        self._x_coords = col_x
        self._y_coords = row_y

    # --- coordinate conversion -------------------------------------------------

    def world_to_index(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert world (x, y) meters to a (row, col) grid index.

        Does not bounds-check; use `in_bounds` on the result if the
        point may fall outside the grid.
        """
        col = int(x_m / self.resolution_m)
        row = int(y_m / self.resolution_m)
        return row, col

    def index_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Return the world (x, y) coordinate of a cell's center, in meters."""
        x_m = (col + 0.5) * self.resolution_m
        y_m = (row + 0.5) * self.resolution_m
        return x_m, y_m

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n and 0 <= col < self.n

    # --- write interface used by populators -------------------------------------

    def add_rock(self, rock: Rock) -> int:
        """Register a rock's metadata and return its index into self.rocks.

        Populators are expected to then write `rock_id`, `elevation`,
        and `traversable` for the cells the rock occupies directly via
        the corresponding public arrays (e.g. `grid.rock_id[mask] = idx`).
        """
        self.rocks.append(rock)
        return len(self.rocks) - 1

    def add_crater(self, crater: Crater) -> int:
        """Register a crater's metadata and return its index into self.craters.

        Same pattern as `add_rock`: populators write `crater_id`,
        `elevation`, and `traversable` for the cells the crater
        occupies directly via the corresponding public arrays.
        """
        self.craters.append(crater)
        return len(self.craters) - 1

    def get_cell(self, row: int, col: int) -> Tuple[float, bool, int, int]:
        """Return (elevation, traversable, rock_id, crater_id) for a single cell."""
        return (
            float(self.elevation[row, col]),
            bool(self.traversable[row, col]),
            int(self.rock_id[row, col]),
            int(self.crater_id[row, col]),
        )

    def reset(self) -> None:
        """Clear all cell data and the rock/crater lists back to an empty grid."""
        self.elevation.fill(0.0)
        self.traversable.fill(True)
        self.rock_id.fill(-1)
        self.rocks.clear()
        self.crater_id.fill(-1)
        self.craters.clear()
