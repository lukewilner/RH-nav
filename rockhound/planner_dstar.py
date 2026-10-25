"""D* Lite incremental replanning over a traversability grid.

astar() is a one-shot search: every call recomputes a path from
nothing. A rover with limited sensing doesn't have that luxury -- it
discovers rocks and craters only as it drives near them, and
replanning from scratch after every such discovery redoes work on the
entire map even though only a small region actually changed.

D* Lite (Koenig & Likhachev, 2002) keeps its search state (g/rhs
values and a priority queue) alive across calls so that, when cells
are reported changed, it repairs only the part of the plan those
cells could have affected. It searches backward from the goal for
exactly this reason: as the rover's start position moves, the
already-settled region behind it stays valid and is never revisited.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .planner import Cell, _neighbor_costs, _octile_heuristic

Key = Tuple[float, float]


class DStarLite:
    """Incremental replanner for a single, fixed goal on a changing grid.

    `traversable` is held by reference, not copied -- the caller owns it
    and is expected to mutate it directly (e.g. as sensing reveals
    previously-unknown cells) and then call `update_cells()` with exactly
    the cells whose value changed, followed by `replan()`.
    """

    def __init__(
        self,
        traversable: np.ndarray,
        start: Cell,
        goal: Cell,
        allow_diagonal: bool = True,
    ) -> None:
        self.traversable = traversable
        self.start = start
        self.goal = goal
        self.allow_diagonal = allow_diagonal

        self._g: Dict[Cell, float] = {}
        self._rhs: Dict[Cell, float] = {goal: 0.0}
        self._km = 0.0
        self._open: List[Tuple[Key, int, Cell]] = []
        self._open_keys: Dict[Cell, Key] = {}
        self._counter = itertools.count()
        self.last_expansions = 0

        self._push(goal)
        self.compute_shortest_path()

    # --- value accessors (default to +inf, per the D* Lite convention for
    # any vertex that hasn't been touched by the search yet) --------------

    def _g_of(self, s: Cell) -> float:
        return self._g.get(s, math.inf)

    def _rhs_of(self, s: Cell) -> float:
        return self._rhs.get(s, math.inf)

    def _calculate_key(self, s: Cell) -> Key:
        m = min(self._g_of(s), self._rhs_of(s))
        return (m + _octile_heuristic(s, self.start) + self._km, m)

    def _push(self, s: Cell) -> None:
        key = self._calculate_key(s)
        self._open_keys[s] = key
        heapq.heappush(self._open, (key, next(self._counter), s))

    def _update_vertex(self, u: Cell) -> None:
        if u != self.goal:
            best = math.inf
            for v, cost in _neighbor_costs(self.traversable, u, self.allow_diagonal):
                best = min(best, cost + self._g_of(v))
            self._rhs[u] = best

        if self._g_of(u) != self._rhs_of(u):
            self._push(u)
        else:
            # Locally consistent: if a stale heap entry for u exists it
            # will be skipped as stale when popped (see compute_shortest_path).
            self._open_keys.pop(u, None)

    def compute_shortest_path(self) -> None:
        while self._open:
            key, _, u = self._open[0]
            if key >= self._calculate_key(self.start) and self._rhs_of(self.start) == self._g_of(self.start):
                break

            heapq.heappop(self._open)
            if self._open_keys.get(u) != key:
                continue  # stale entry from an earlier, now-superseded push
            self._open_keys.pop(u, None)
            self.last_expansions += 1

            k_new = self._calculate_key(u)
            if key < k_new:
                self._push(u)
            elif self._g_of(u) > self._rhs_of(u):
                # u improved: lock in its new value and propagate to neighbors.
                self._g[u] = self._rhs_of(u)
                for v, _ in _neighbor_costs(self.traversable, u, self.allow_diagonal):
                    self._update_vertex(v)
            else:
                # u got worse (or was never reachable): invalidate it and
                # let anything that depended on it recompute.
                self._g[u] = math.inf
                self._update_vertex(u)
                for v, _ in _neighbor_costs(self.traversable, u, self.allow_diagonal):
                    self._update_vertex(v)

    def update_cells(self, cells: Iterable[Cell]) -> None:
        """Mark the given cells (and their neighborhoods) dirty after a
        traversability change. Call `replan()` afterward to actually
        repair the path -- this only queues the necessary recomputation.
        """
        n_rows, n_cols = self.traversable.shape
        touched = set()
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols:
                        touched.add((nr, nc))
        for u in touched:
            self._update_vertex(u)

    def update_start(self, new_start: Cell) -> None:
        """Call as the rover actually moves, so future heuristics stay valid."""
        self._km += _octile_heuristic(self.start, new_start)
        self.start = new_start

    def replan(self) -> None:
        """Repair the plan after update_cells()/update_start() calls."""
        self.last_expansions = 0
        self.compute_shortest_path()

    def get_path(self) -> Optional[List[Cell]]:
        """Greedily follow the current g-values from `start` to `goal`.

        Returns None if `start` has no known route to the goal.
        """
        if math.isinf(self._g_of(self.start)):
            return None

        path = [self.start]
        current = self.start
        visited = {current}
        while current != self.goal:
            best_cost = math.inf
            best_next: Optional[Cell] = None
            for v, cost in _neighbor_costs(self.traversable, current, self.allow_diagonal):
                total = cost + self._g_of(v)
                if total < best_cost:
                    best_cost = total
                    best_next = v
            if best_next is None or best_next in visited:
                return None
            path.append(best_next)
            visited.add(best_next)
            current = best_next
        return path
