"""A* path planning over a traversability grid."""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

Cell = Tuple[int, int]

_NEIGHBORS_8: List[Tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    (0, -1, 1.0),                          (0, 1, 1.0),
    (1, -1, math.sqrt(2)),  (1, 0, 1.0),  (1, 1, math.sqrt(2)),
]
_NEIGHBORS_4: List[Tuple[int, int, float]] = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]


def _neighbor_costs(
    traversable: np.ndarray, cell: Cell, allow_diagonal: bool = True
) -> Iterator[Tuple[Cell, float]]:
    """Yield (neighbor, cost) for every cell reachable from `cell` in one move.

    Shared by astar() and DStarLite so both use exactly the same movement
    model. Symmetric by construction (a move is valid based only on the
    traversability of the two endpoints and, for a diagonal step, the two
    corner cells between them -- the same set regardless of which endpoint
    is queried from), which DStarLite relies on to treat predecessors and
    successors as the same set.
    """
    n_rows, n_cols = traversable.shape
    r, c = cell
    if not (0 <= r < n_rows and 0 <= c < n_cols) or not traversable[r, c]:
        return
    for dr, dc, cost in _NEIGHBORS_8 if allow_diagonal else _NEIGHBORS_4:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n_rows and 0 <= nc < n_cols) or not traversable[nr, nc]:
            continue
        if dr != 0 and dc != 0:
            # Disallow cutting a corner between two blocked cells.
            if not traversable[r + dr, c] or not traversable[r, c + dc]:
                continue
        yield (nr, nc), cost


def _octile_heuristic(a: Cell, b: Cell) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return math.sqrt(2) * min(dr, dc) + abs(dr - dc)


def astar(
    traversable: np.ndarray,
    start: Cell,
    goal: Cell,
    allow_diagonal: bool = True,
    return_expansions: bool = False,
) -> Union[Optional[List[Cell]], Tuple[Optional[List[Cell]], int]]:
    """Find a shortest path from `start` to `goal` over a boolean grid.

    `traversable` is a 2D bool array (True = go, False = no-go/impassable),
    e.g. `grid.traversable` from a TerrainGrid. Returns a list of
    (row, col) cells from start to goal inclusive, or None if no path
    exists. Raises ValueError if start/goal are out of bounds or sit
    on a no-go cell.

    If `return_expansions` is True, returns `(path, expansions)` instead,
    where `expansions` is the number of cells popped off the open set --
    a proxy for how much work this from-scratch search did, useful for
    comparing against DStarLite's incremental replanning cost.
    """
    n_rows, n_cols = traversable.shape

    def in_bounds(c: Cell) -> bool:
        return 0 <= c[0] < n_rows and 0 <= c[1] < n_cols

    if not in_bounds(start) or not in_bounds(goal):
        raise ValueError("start/goal out of grid bounds")
    if not traversable[start]:
        raise ValueError("start cell is not traversable")
    if not traversable[goal]:
        raise ValueError("goal cell is not traversable")

    open_heap: List[Tuple[float, Cell]] = [(0.0, start)]
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = _reconstruct_path(came_from, current)
            return (path, len(visited)) if return_expansions else path
        if current in visited:
            continue
        visited.add(current)

        for neighbor, cost in _neighbor_costs(traversable, current, allow_diagonal):
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _octile_heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return (None, len(visited)) if return_expansions else None


def _reconstruct_path(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _line_of_sight(traversable: np.ndarray, a: Cell, b: Cell) -> bool:
    """True if every cell on the straight grid line from a to b is
    traversable, and no diagonal step along it cuts a blocked corner."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while (r, c) != (r1, c1):
        e2 = 2 * err
        step_r = step_c = 0
        if e2 > -dc:
            err -= dc
            r += sr
            step_r = sr
        if e2 < dr:
            err += dr
            c += sc
            step_c = sc
        if not traversable[r, c]:
            return False
        if step_r != 0 and step_c != 0:
            if not traversable[r - step_r, c] or not traversable[r, c - step_c]:
                return False
    return True


def smooth_path(traversable: np.ndarray, path: List[Cell]) -> List[Cell]:
    """Collapse a grid path's staircase into the minimal set of straight
    segments, via greedy line-of-sight shortcutting ("string pulling").

    A* on a grid graph only guarantees the shortest path among
    8-connected moves -- when several such paths tie in cost (a common
    occurrence), it returns whichever the tie-break happens to prefer,
    which can look like an unnecessary jog even though the cost is
    optimal. This walks the path forward, keeps extending the current
    straight run as far as line-of-sight allows, and only adds a
    waypoint where the run is actually blocked -- so the result hugs
    no-go boundaries where required and is a straight line everywhere
    else.
    """
    if len(path) <= 2:
        return list(path)

    smoothed = [path[0]]
    anchor_idx = 0
    candidate_idx = 2
    while candidate_idx < len(path):
        if not _line_of_sight(traversable, path[anchor_idx], path[candidate_idx]):
            smoothed.append(path[candidate_idx - 1])
            anchor_idx = candidate_idx - 1
        candidate_idx += 1
    smoothed.append(path[-1])
    return smoothed
