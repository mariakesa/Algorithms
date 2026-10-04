#!/usr/bin/env python3
"""
Parallel metric-first symmetry-constrained solver for Jane Street's 11x11 puzzle.

Pipeline
--------
1. metric:
   Derive a relaxed metric from clue inequalities and certify forced edge-weight
   ranges using shortest-path separation.

2. candidates:
   Generate COMPLETE connected symmetric state candidates of exact size.
   Candidate generation is parallelized across affine D4 symmetry tasks using
   multiple processes.

3. search:
   Solve the board as an exact-cover / set-partition problem over complete state
   candidates, with exact metric pruning and final Dijkstra verification.

Key design choices
------------------
- We do NOT trust one arbitrary relaxed LP solution as the hidden geography.
- We compute per-edge feasible lower/upper bounds under the relaxed metric model.
- We generate whole symmetric states, not cell-by-cell shapes.
- Intrinsic shapes may be deduplicated as templates, but distinct placements are
  preserved.
- Search branches on complete state candidates.
- Failed exact shortest-path checks can produce path certificates for diagnosis.

Dependencies
------------
    numpy
    scipy

Example usage
-------------
    python metric_first_symmetry_solver_parallel.py metric --only-strongest 40
    python metric_first_symmetry_solver_parallel.py candidates --workers 12 --max-size 30
    python metric_first_symmetry_solver_parallel.py search --max-nodes 100000

This file was created but not executed by ChatGPT.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


# ============================================================================
# PUZZLE INPUT
# ============================================================================

GRID = [
    ['X', 8, 4, 'X', 'X', 11, 'X', 'X', 'X', 5, 'X'],
    ['X', 'X', 'X', 'X', 'X', 'X', 'X', 11, 'X', 'X', 'X'],
    ['X', 'X', 7, 'X', 'X', 1, 'X', 'X', 'X', 'X', 14],
    [28, 'X', 'X', 51, 'X', 'X', 1, 'X', 6, 'X', 'X'],
    ['X', 22, 'X', 'X', 'X', 'X', 'X', 'X', 4, 'X', 1],
    ['X', 'X', 'X', 15, 'X', 10, 'X', 0, 'X', 'X', 'X'],
    [11, 'X', 11, 'X', 'X', 'X', 'X', 'X', 'X', 9, 'X'],
    ['X', 'X', 17, 'X', 14, 'X', 'X', 10, 'X', 'X', 13],
    [30, 'X', 'X', 'X', 'X', 6, 'X', 'X', 45, 'X', 'X'],
    ['X', 'X', 'X', 10, 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
    ['X', 26, 'X', 'X', 'X', 0, 'X', 'X', 77, 61, 'X'],
]

R = len(GRID)
C = len(GRID[0])
assert R == C == 11
assert all(len(row) == C for row in GRID)

Cell = Tuple[int, int]
Affine = Tuple[Tuple[int, int, int, int], Tuple[int, int]]

NODES: List[Cell] = [(r, c) for r in range(R) for c in range(C)]
NODE_ID: Dict[Cell, int] = {p: i for i, p in enumerate(NODES)}
ID_NODE: Dict[int, Cell] = {i: p for p, i in NODE_ID.items()}
N = len(NODES)

CLUES_BY_CELL: Dict[Cell, int] = {
    (r, c): int(v)
    for r, row in enumerate(GRID)
    for c, v in enumerate(row)
    if v != 'X'
}
CLUES: Dict[int, int] = {NODE_ID[p]: v for p, v in CLUES_BY_CELL.items()}

ZERO_CELLS: Set[int] = {v for v, y in CLUES.items() if y == 0}
POSITIVE_CLUE_CELLS: Set[int] = {v for v, y in CLUES.items() if y > 0}
ONE_CELLS: Set[int] = {v for v, y in CLUES.items() if y == 1}

EDGES: List[Tuple[int, int]] = []
EDGE_ID: Dict[Tuple[int, int], int] = {}
ADJ: List[List[Tuple[int, int]]] = [[] for _ in range(N)]

for r in range(R):
    for c in range(C):
        u = NODE_ID[(r, c)]
        for dr, dc in ((1, 0), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C:
                v = NODE_ID[(rr, cc)]
                eid = len(EDGES)
                EDGES.append((u, v))
                EDGE_ID[tuple(sorted((u, v)))] = eid
                ADJ[u].append((v, eid))
                ADJ[v].append((u, eid))

E = len(EDGES)


# ============================================================================
# GRAPH UTILITIES
# ============================================================================

def dijkstra(
    weights: Sequence[float],
    source: int,
    target: Optional[int] = None,
) -> Tuple[List[float], Dict[int, Tuple[int, int]]]:
    dist = [math.inf] * N
    prev: Dict[int, Tuple[int, int]] = {}
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if target is not None and u == target:
            break
        for v, eid in ADJ[u]:
            nd = d + float(weights[eid])
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, eid)
                heapq.heappush(pq, (nd, v))
    return dist, prev


def multi_source_dijkstra(
    weights: Sequence[float],
    sources: Iterable[int],
) -> Tuple[List[float], Dict[int, Tuple[int, int]], List[Optional[int]]]:
    dist = [math.inf] * N
    prev: Dict[int, Tuple[int, int]] = {}
    owner: List[Optional[int]] = [None] * N
    pq: List[Tuple[float, int, int]] = []

    for s in sources:
        dist[s] = 0.0
        owner[s] = s
        heapq.heappush(pq, (0.0, s, s))

    while pq:
        d, src, u = heapq.heappop(pq)
        if d != dist[u] or owner[u] != src:
            continue
        for v, eid in ADJ[u]:
            nd = d + float(weights[eid])
            if nd < dist[v]:
                dist[v] = nd
                owner[v] = src
                prev[v] = (u, eid)
                heapq.heappush(pq, (nd, src, v))
    return dist, prev, owner


def recover_path_edges(
    prev: Dict[int, Tuple[int, int]],
    source: int,
    target: int,
) -> List[int]:
    out: List[int] = []
    cur = target
    while cur != source:
        if cur not in prev:
            return []
        p, eid = prev[cur]
        out.append(eid)
        cur = p
    out.reverse()
    return out


def connected(cells: Set[int]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v, _ in ADJ[u]:
            if v in cells and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen == cells


# ============================================================================
# METRIC LOWER BOUNDS
# ============================================================================

@dataclass(frozen=True)
class CluePairConstraint:
    i: int
    j: int
    lower: int


def clue_pair_constraints() -> List[CluePairConstraint]:
    out: List[CluePairConstraint] = []
    items = sorted(CLUES.items())
    for (i, yi), (j, yj) in itertools.combinations(items, 2):
        L = abs(yi - yj)
        if L > 0:
            out.append(CluePairConstraint(i, j, L))
    return out


PAIR_CONSTRAINTS = clue_pair_constraints()


# ============================================================================
# RELAXED METRIC + FORCED EDGE RANGES
# ============================================================================

@dataclass
class PathCut:
    edge_ids: Tuple[int, ...]
    lower: float


@dataclass
class MetricRelaxationResult:
    weights: np.ndarray
    cuts: List[PathCut]
    objective: float
    feasible: bool


class RelaxedMetricModel:
    """
    Independent positive edge weights with shortest-path lower bounds
    d(i,j) >= |y_i-y_j|.

    Cutting-plane loop:
      1) optimize current LP/MILP
      2) recompute shortest paths
      3) if a clue pair is too close, add that shortest path as a cut
      4) repeat
    """

    def __init__(
        self,
        integer_weights: bool = True,
        edge_min: int = 1,
        edge_max: int = 121,
    ):
        self.integer_weights = integer_weights
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.cuts: List[PathCut] = []
        self._cut_keys: Set[Tuple[Tuple[int, ...], float]] = set()

    def add_cut(self, edge_ids: Sequence[int], lower: float) -> bool:
        key = (tuple(sorted(edge_ids)), float(lower))
        if key in self._cut_keys:
            return False
        self._cut_keys.add(key)
        self.cuts.append(PathCut(tuple(edge_ids), float(lower)))
        return True

    def _solve_linear_objective(
        self,
        objective: np.ndarray,
        maximize: bool = False,
        time_limit: Optional[float] = None,
    ):
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        lb: List[float] = []
        ub: List[float] = []

        for cut in self.cuts:
            rr = len(lb)
            for eid in cut.edge_ids:
                rows.append(rr)
                cols.append(eid)
                data.append(1.0)
            lb.append(cut.lower)
            ub.append(np.inf)

        constraints = None
        if lb:
            A = coo_matrix(
                (data, (rows, cols)),
                shape=(len(lb), E),
            ).tocsr()
            constraints = LinearConstraint(
                A,
                np.asarray(lb, dtype=float),
                np.asarray(ub, dtype=float),
            )

        c = -objective if maximize else objective
        integ = (
            np.ones(E, dtype=int)
            if self.integer_weights
            else np.zeros(E, dtype=int)
        )

        options = {}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)

        return milp(
            c=c,
            integrality=integ,
            bounds=Bounds(
                np.full(E, self.edge_min, dtype=float),
                np.full(E, self.edge_max, dtype=float),
            ),
            constraints=constraints,
            options=options,
        )

    def separate(self, weights: Sequence[float]) -> int:
        source_cache: Dict[
            int, Tuple[List[float], Dict[int, Tuple[int, int]]]
        ] = {}
        new_cuts = 0

        for pc in PAIR_CONSTRAINTS:
            if pc.i not in source_cache:
                source_cache[pc.i] = dijkstra(weights, pc.i)
            dist, prev = source_cache[pc.i]

            if dist[pc.j] + 1e-9 < pc.lower:
                path = recover_path_edges(prev, pc.i, pc.j)
                if path and self.add_cut(path, pc.lower):
                    new_cuts += 1

        return new_cuts

    def solve_with_separation(
        self,
        objective: Optional[np.ndarray] = None,
        maximize: bool = False,
        max_rounds: int = 100,
        time_limit_per_round: Optional[float] = None,
    ) -> MetricRelaxationResult:
        if objective is None:
            objective = np.ones(E, dtype=float)

        last = None
        for _ in range(max_rounds):
            res = self._solve_linear_objective(
                objective,
                maximize=maximize,
                time_limit=time_limit_per_round,
            )
            last = res

            if res.x is None:
                return MetricRelaxationResult(
                    weights=np.zeros(E),
                    cuts=list(self.cuts),
                    objective=math.inf,
                    feasible=False,
                )

            weights = np.asarray(res.x, dtype=float)
            nnew = self.separate(weights)

            if nnew == 0:
                return MetricRelaxationResult(
                    weights=weights,
                    cuts=list(self.cuts),
                    objective=float(np.dot(objective, weights)),
                    feasible=True,
                )

        if last is None or last.x is None:
            return MetricRelaxationResult(
                weights=np.zeros(E),
                cuts=list(self.cuts),
                objective=math.inf,
                feasible=False,
            )

        return MetricRelaxationResult(
            weights=np.asarray(last.x, dtype=float),
            cuts=list(self.cuts),
            objective=float(np.dot(objective, last.x)),
            feasible=False,
        )

    def forced_edge_ranges(
        self,
        edge_ids: Optional[Iterable[int]] = None,
        max_rounds: int = 100,
        time_limit_per_round: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if edge_ids is None:
            edge_ids = range(E)

        lo = np.full(E, np.nan)
        hi = np.full(E, np.nan)

        for eid in edge_ids:
            objective = np.zeros(E, dtype=float)
            objective[eid] = 1.0

            rmin = self.solve_with_separation(
                objective=objective,
                maximize=False,
                max_rounds=max_rounds,
                time_limit_per_round=time_limit_per_round,
            )
            if not rmin.feasible:
                raise RuntimeError(
                    f"Could not certify lower bound for edge {eid}"
                )
            lo[eid] = rmin.weights[eid]

            rmax = self.solve_with_separation(
                objective=objective,
                maximize=True,
                max_rounds=max_rounds,
                time_limit_per_round=time_limit_per_round,
            )
            if not rmax.feasible:
                raise RuntimeError(
                    f"Could not certify upper bound for edge {eid}"
                )
            hi[eid] = rmax.weights[eid]

        return lo, hi


# ============================================================================
# D4 / AFFINE SYMMETRY
# ============================================================================

D4_LINEAR: Dict[str, Tuple[int, int, int, int]] = {
    "R90": (0, -1, 1, 0),
    "R180": (-1, 0, 0, -1),
    "R270": (0, 1, -1, 0),
    "REF_X": (1, 0, 0, -1),
    "REF_Y": (-1, 0, 0, 1),
    "REF_DIAG": (0, 1, 1, 0),
    "REF_ANTI": (0, -1, -1, 0),
}
IDENTITY = (1, 0, 0, 1)


def apply_linear(
    A: Tuple[int, int, int, int],
    p: Cell,
) -> Cell:
    a, b, c, d = A
    r, col = p
    return a * r + b * col, c * r + d * col


def apply_affine(g: Affine, p: Cell) -> Cell:
    A, t = g
    x, y = apply_linear(A, p)
    return x + t[0], y + t[1]


def compose_affine(g: Affine, h: Affine) -> Affine:
    Ag, tg = g
    Ah, th = h

    a, b, c, d = Ag
    e, f, g2, h2 = Ah

    A = (
        a * e + b * g2,
        a * f + b * h2,
        c * e + d * g2,
        c * f + d * h2,
    )

    Ath = apply_linear(Ag, th)
    t = (Ath[0] + tg[0], Ath[1] + tg[1])
    return A, t


def affine_power(g: Affine, k: int) -> Affine:
    out: Affine = (IDENTITY, (0, 0))
    for _ in range(k):
        out = compose_affine(g, out)
    return out


def is_finite_d4_action(g: Affine) -> bool:
    A, _ = g
    order = (
        4
        if A in (D4_LINEAR["R90"], D4_LINEAR["R270"])
        else 2
    )
    return affine_power(g, order) == (IDENTITY, (0, 0))


def enumerate_board_relevant_affine_symmetries(
) -> List[Tuple[str, Affine]]:
    out: Dict[Affine, str] = {}
    board = list(NODES)

    for name, A in D4_LINEAR.items():
        for p in board:
            Ap = apply_linear(A, p)
            for q in board:
                t = (q[0] - Ap[0], q[1] - Ap[1])
                g: Affine = (A, t)
                if g in out:
                    continue
                if is_finite_d4_action(g):
                    out[g] = name

    return [(name, g) for g, name in out.items()]


def orbit_of_cell(
    g: Affine,
    cell_id: int,
) -> Optional[Tuple[int, ...]]:
    start = ID_NODE[cell_id]
    seen: List[Cell] = []
    cur = start

    for _ in range(8):
        if cur in seen:
            break

        r, c = cur
        if not (0 <= r < R and 0 <= c < C):
            return None

        seen.append(cur)
        cur = apply_affine(g, cur)

    if cur != start:
        return None

    return tuple(sorted(NODE_ID[p] for p in seen))


def full_symmetry_group(
    cells: Set[int],
    all_actions: Sequence[Tuple[str, Affine]],
) -> List[Tuple[str, Affine]]:
    pts = {ID_NODE[v] for v in cells}
    syms: List[Tuple[str, Affine]] = []

    for name, g in all_actions:
        mapped = {apply_affine(g, p) for p in pts}
        if mapped == pts:
            syms.append((name, g))

    return syms


def fixed_cells_of_action(
    g: Affine,
    cells: Set[int],
) -> Set[int]:
    return {
        v
        for v in cells
        if apply_affine(g, ID_NODE[v]) == ID_NODE[v]
    }


def candidate_capitol(
    cells: Set[int],
    all_actions: Sequence[Tuple[str, Affine]],
) -> Optional[int]:
    syms = full_symmetry_group(cells, all_actions)
    if not syms:
        return None

    common: Optional[Set[int]] = None
    for _, g in syms:
        F = fixed_cells_of_action(g, cells)
        common = F if common is None else (common & F)

    if common is not None and len(common) == 1:
        return next(iter(common))

    return None


# ============================================================================
# COMPLETE STATE CANDIDATES
# ============================================================================

@dataclass(frozen=True)
class StateCandidate:
    id: int
    cells: frozenset
    size: int
    symmetry_name: str
    transform: Affine
    capitol: Optional[int]
    incident_edges: frozenset
    internal_edges: frozenset

    def contains(self, v: int) -> bool:
        return v in self.cells


def incident_and_internal_edges(
    cells: Set[int],
) -> Tuple[Set[int], Set[int]]:
    incident: Set[int] = set()
    internal: Set[int] = set()

    for u in cells:
        for v, eid in ADJ[u]:
            incident.add(eid)
            if v in cells:
                internal.add(eid)

    return incident, internal


def orbit_graph(
    g: Affine,
) -> Tuple[List[Tuple[int, ...]], Dict[int, Set[int]]]:
    unique: Dict[Tuple[int, ...], int] = {}

    for v in range(N):
        orb = orbit_of_cell(g, v)
        if orb is None:
            continue
        if orb not in unique:
            unique[orb] = len(unique)

    orbits = list(unique.keys())
    G: Dict[int, Set[int]] = {
        i: set() for i in range(len(orbits))
    }

    cell_to_oid: Dict[int, int] = {}
    for i, orb in enumerate(orbits):
        for v in orb:
            cell_to_oid[v] = i

    for u, v in EDGES:
        if u in cell_to_oid and v in cell_to_oid:
            a = cell_to_oid[u]
            b = cell_to_oid[v]
            if a != b:
                G[a].add(b)
                G[b].add(a)

    return orbits, G


def enumerate_connected_orbit_unions_for_roots(
    orbits: Sequence[Tuple[int, ...]],
    orbit_adj: Dict[int, Set[int]],
    roots: Sequence[int],
    min_size: int,
    max_size: int,
    max_results: int,
) -> Iterator[Set[int]]:
    """
    Enumerate connected orbit unions starting only from selected root orbits.

    This is the parallelization hook: different workers get disjoint root sets.
    """
    produced = 0
    seen_sets: Set[Tuple[int, ...]] = set()

    for root in roots:
        root_cells = set(orbits[root])
        if len(root_cells) > max_size:
            continue

        stack: List[Tuple[frozenset, frozenset]] = [
            (
                frozenset({root}),
                frozenset(
                    x for x in orbit_adj[root]
                    if x > root
                ),
            )
        ]

        while stack:
            chosen, frontier = stack.pop()

            cells: Set[int] = set()
            for oid in chosen:
                cells.update(orbits[oid])

            n = len(cells)

            if min_size <= n <= max_size:
                key = tuple(sorted(cells))
                if key not in seen_sets:
                    seen_sets.add(key)
                    yield cells
                    produced += 1

                    if produced >= max_results:
                        return

            if n >= max_size:
                continue

            for oid in sorted(frontier, reverse=True):
                new_chosen = set(chosen)
                new_chosen.add(oid)

                new_frontier = set(frontier)
                new_frontier.discard(oid)

                for nb in orbit_adj[oid]:
                    if (
                        nb >= root
                        and nb not in new_chosen
                    ):
                        new_frontier.add(nb)

                stack.append(
                    (
                        frozenset(new_chosen),
                        frozenset(new_frontier),
                    )
                )


def intrinsic_shape_signature(
    cells: Set[int],
) -> Tuple[Cell, ...]:
    pts = [ID_NODE[v] for v in cells]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)

    return tuple(
        sorted(
            (r - r0, c - c0)
            for r, c in pts
        )
    )


# ============================================================================
# PARALLEL CANDIDATE WORKER
# ============================================================================

@dataclass(frozen=True)
class CandidateTask:
    symmetry_name: str
    transform: Affine
    roots: Tuple[int, ...]
    edge_lower_bounds: Tuple[float, ...]
    min_state_size: int
    max_state_size: int
    max_results: int


def _candidate_worker(
    task: CandidateTask,
) -> List[Tuple[
    Tuple[int, ...],
    int,
    str,
    Affine,
    Optional[int],
    Tuple[int, ...],
    Tuple[int, ...],
]]:
    """
    Process worker.

    Returns only compact tuples.  The parent process constructs StateCandidate
    objects and handles global deduplication.
    """
    all_actions = enumerate_board_relevant_affine_symmetries()
    g = task.transform

    orbits, oadj = orbit_graph(g)
    if not orbits:
        return []

    out = []
    local_seen: Set[Tuple[int, ...]] = set()

    for cells in enumerate_connected_orbit_unions_for_roots(
        orbits=orbits,
        orbit_adj=oadj,
        roots=task.roots,
        min_size=task.min_state_size,
        max_size=task.max_state_size,
        max_results=task.max_results,
    ):
        if not connected(cells):
            continue

        key = tuple(sorted(cells))
        if key in local_seen:
            continue
        local_seen.add(key)

        size = len(cells)
        incident, internal = incident_and_internal_edges(cells)

        if any(
            task.edge_lower_bounds[e] > size + 1e-9
            for e in incident
        ):
            continue

        cap = candidate_capitol(cells, all_actions)

        zero_inside = ZERO_CELLS & cells
        if zero_inside:
            if len(zero_inside) != 1:
                continue
            if cap not in zero_inside:
                continue

        if cap is not None and cap in POSITIVE_CLUE_CELLS:
            continue

        if size == 1 and cap is None:
            continue

        out.append(
            (
                key,
                size,
                task.symmetry_name,
                task.transform,
                cap,
                tuple(sorted(incident)),
                tuple(sorted(internal)),
            )
        )

    return out


def chunk_roots(
    roots: Sequence[int],
    chunk_size: int,
) -> List[Tuple[int, ...]]:
    return [
        tuple(roots[i:i + chunk_size])
        for i in range(0, len(roots), chunk_size)
    ]


def generate_state_candidates_parallel(
    edge_lower_bounds: Sequence[float],
    workers: int,
    root_chunk_size: int = 2,
    min_state_size: int = 1,
    max_state_size: int = 40,
    max_results_per_task: int = 10000,
    global_candidate_cap: int = 250000,
    verbose: bool = True,
) -> Tuple[
    List[StateCandidate],
    Dict[Tuple[Cell, ...], List[int]],
]:
    """
    Parallel candidate generation.

    Parallelization unit:
      affine symmetry action + a small subset of root orbits

    This creates many medium-sized tasks instead of one huge task per symmetry,
    reducing load imbalance between workers.
    """
    all_actions = enumerate_board_relevant_affine_symmetries()
    lb_tuple = tuple(float(x) for x in edge_lower_bounds)

    tasks: List[CandidateTask] = []

    for sym_name, g in all_actions:
        orbits, _ = orbit_graph(g)
        if not orbits:
            continue

        root_ids = list(range(len(orbits)))

        for roots in chunk_roots(
            root_ids,
            max(1, root_chunk_size),
        ):
            tasks.append(
                CandidateTask(
                    symmetry_name=sym_name,
                    transform=g,
                    roots=roots,
                    edge_lower_bounds=lb_tuple,
                    min_state_size=min_state_size,
                    max_state_size=max_state_size,
                    max_results=max_results_per_task,
                )
            )

    if verbose:
        print(
            f"Candidate tasks: {len(tasks)} "
            f"across {len(all_actions)} affine symmetry actions"
        )
        print(f"Workers: {workers}")

    candidates: List[StateCandidate] = []
    templates: Dict[
        Tuple[Cell, ...],
        List[int],
    ] = defaultdict(list)

    seen_cellsets: Set[frozenset] = set()

    with ProcessPoolExecutor(
        max_workers=workers
    ) as ex:
        future_to_task = {
            ex.submit(_candidate_worker, task): task
            for task in tasks
        }

        completed = 0

        for fut in as_completed(future_to_task):
            completed += 1
            task = future_to_task[fut]

            batch = fut.result()

            for (
                cell_tuple,
                size,
                sym_name,
                transform,
                cap,
                incident_tuple,
                internal_tuple,
            ) in batch:

                fs = frozenset(cell_tuple)

                if fs in seen_cellsets:
                    continue

                cid = len(candidates)

                cand = StateCandidate(
                    id=cid,
                    cells=fs,
                    size=size,
                    symmetry_name=sym_name,
                    transform=transform,
                    capitol=cap,
                    incident_edges=frozenset(
                        incident_tuple
                    ),
                    internal_edges=frozenset(
                        internal_tuple
                    ),
                )

                candidates.append(cand)
                seen_cellsets.add(fs)

                sig = intrinsic_shape_signature(
                    set(cell_tuple)
                )
                templates[sig].append(cid)

                if (
                    len(candidates)
                    >= global_candidate_cap
                ):
                    if verbose:
                        print(
                            "Reached global candidate cap; "
                            "stopping collection."
                        )

                    for pending in future_to_task:
                        pending.cancel()

                    return candidates, templates

            if verbose and (
                completed % 25 == 0
                or completed == len(tasks)
            ):
                print(
                    f"[{completed}/{len(tasks)} tasks] "
                    f"unique candidates={len(candidates)}"
                )

    return candidates, templates


# ============================================================================
# EXACT-COVER / SET-PARTITION SEARCH
# ============================================================================

@dataclass
class SearchStats:
    nodes: int = 0
    prunes_overlap: int = 0
    prunes_dead_cell: int = 0
    prunes_metric: int = 0
    prunes_clue: int = 0
    complete_partitions: int = 0


@dataclass
class ExactSolution:
    candidate_ids: List[int]
    cell_to_state: List[int]
    edge_weights: np.ndarray
    capitols: List[int]
    predicted: List[float]


class ExactCoverMetricSearch:
    def __init__(
        self,
        candidates: Sequence[StateCandidate],
        metric_edge_lb: Sequence[float],
    ):
        self.candidates = list(candidates)
        self.metric_edge_lb = np.asarray(
            metric_edge_lb,
            dtype=float,
        )
        self.stats = SearchStats()

        self.by_cell: List[List[int]] = [
            [] for _ in range(N)
        ]

        for cand in self.candidates:
            for v in cand.cells:
                self.by_cell[v].append(cand.id)

        self.candidate_score = np.zeros(
            len(self.candidates),
            dtype=float,
        )

        for c in self.candidates:
            support = sum(
                self.metric_edge_lb[e]
                for e in c.incident_edges
            )
            self.candidate_score[c.id] = (
                support + 0.25 * c.size
            )

    def _choose_uncovered_cell(
        self,
        covered: Set[int],
        blocked_candidates: Set[int],
    ) -> Optional[int]:
        best_v = None
        best_count = math.inf

        for v in range(N):
            if v in covered:
                continue

            viable = sum(
                1
                for cid in self.by_cell[v]
                if cid not in blocked_candidates
            )

            if viable < best_count:
                best_count = viable
                best_v = v

                if viable <= 1:
                    break

        return best_v

    def _edge_exact_if_assigned(
        self,
        u: int,
        v: int,
        cell_state: Sequence[Optional[int]],
    ) -> Optional[int]:
        su = cell_state[u]
        sv = cell_state[v]

        if su is None or sv is None:
            return None

        cu = self.candidates[su]
        cv = self.candidates[sv]

        return min(cu.size, cv.size)

    def _partial_metric_consistent(
        self,
        cell_state: Sequence[Optional[int]],
    ) -> bool:
        for eid, (u, v) in enumerate(EDGES):
            w = self._edge_exact_if_assigned(
                u,
                v,
                cell_state,
            )

            if (
                w is not None
                and w + 1e-9
                < self.metric_edge_lb[eid]
            ):
                return False

        return True

    def _fixed_cheap_path_contradiction(
        self,
        cell_state: Sequence[Optional[int]],
        selected: Sequence[int],
    ) -> bool:
        fixed_caps = [
            self.candidates[cid].capitol
            for cid in selected
            if self.candidates[cid].capitol
            is not None
        ]

        fixed_caps = [
            c for c in fixed_caps
            if c is not None
        ]

        if not fixed_caps:
            return False

        weights = [math.inf] * E

        for eid, (u, v) in enumerate(EDGES):
            w = self._edge_exact_if_assigned(
                u,
                v,
                cell_state,
            )

            if w is not None:
                weights[eid] = float(w)

        dist, _, _ = multi_source_dijkstra(
            weights,
            fixed_caps,
        )

        for v, y in CLUES.items():
            if dist[v] < y - 1e-9:
                return True

        return False

    def _build_complete_partition(
        self,
        selected: Sequence[int],
    ) -> ExactSolution:
        cell_to_state: List[int] = [-1] * N

        for cid in selected:
            for v in self.candidates[cid].cells:
                if cell_to_state[v] != -1:
                    raise RuntimeError(
                        "Overlap in complete partition"
                    )
                cell_to_state[v] = cid

        if any(x == -1 for x in cell_to_state):
            raise RuntimeError(
                "Incomplete partition"
            )

        weights = np.zeros(E, dtype=float)

        for eid, (u, v) in enumerate(EDGES):
            su = self.candidates[
                cell_to_state[u]
            ].size
            sv = self.candidates[
                cell_to_state[v]
            ].size

            weights[eid] = min(su, sv)

        capitols = [
            self.candidates[cid].capitol
            for cid in selected
            if self.candidates[cid].capitol
            is not None
        ]

        capitols = [
            c for c in capitols
            if c is not None
        ]

        predicted, _, _ = multi_source_dijkstra(
            weights,
            capitols,
        )

        return ExactSolution(
            candidate_ids=list(selected),
            cell_to_state=cell_to_state,
            edge_weights=weights,
            capitols=capitols,
            predicted=predicted,
        )

    def verify_complete_solution(
        self,
        sol: ExactSolution,
    ) -> Tuple[
        bool,
        Optional[
            Tuple[int, int, List[int], float]
        ],
    ]:
        dist, prev, owner = multi_source_dijkstra(
            sol.edge_weights,
            sol.capitols,
        )

        for v, target in CLUES.items():
            got = dist[v]

            if abs(got - target) > 1e-9:
                if got < target and owner[v] is not None:
                    src = owner[v]
                    path = []
                    cur = v

                    while (
                        cur != src
                        and cur in prev
                    ):
                        p, eid = prev[cur]
                        path.append(eid)
                        cur = p

                    return False, (
                        v,
                        target,
                        path,
                        got,
                    )

                return False, (
                    v,
                    target,
                    [],
                    got,
                )

        capset = set(sol.capitols)

        if not ZERO_CELLS <= capset:
            return False, None

        if POSITIVE_CLUE_CELLS & capset:
            return False, None

        return True, None

    def search(
        self,
        max_nodes: Optional[int] = None,
    ) -> Iterator[ExactSolution]:
        selected: List[int] = []
        covered: Set[int] = set()
        blocked: Set[int] = set()
        cell_state: List[
            Optional[int]
        ] = [None] * N

        def dfs() -> Iterator[ExactSolution]:
            if (
                max_nodes is not None
                and self.stats.nodes
                >= max_nodes
            ):
                return

            self.stats.nodes += 1

            if len(covered) == N:
                self.stats.complete_partitions += 1

                sol = self._build_complete_partition(
                    selected
                )

                ok, _ = self.verify_complete_solution(
                    sol
                )

                if ok:
                    yield sol
                else:
                    self.stats.prunes_clue += 1

                return

            v = self._choose_uncovered_cell(
                covered,
                blocked,
            )

            if v is None:
                return

            viable = [
                cid
                for cid in self.by_cell[v]
                if (
                    cid not in blocked
                    and not (
                        self.candidates[cid].cells
                        & covered
                    )
                )
            ]

            if not viable:
                self.stats.prunes_dead_cell += 1
                return

            viable.sort(
                key=lambda cid:
                    self.candidate_score[cid],
                reverse=True,
            )

            for cid in viable:
                cand = self.candidates[cid]

                if cand.cells & covered:
                    self.stats.prunes_overlap += 1
                    continue

                newly_covered = list(cand.cells)

                newly_blocked: Set[int] = set()

                for x in newly_covered:
                    for other in self.by_cell[x]:
                        if (
                            other not in blocked
                            and other != cid
                        ):
                            blocked.add(other)
                            newly_blocked.add(other)

                selected.append(cid)

                for x in newly_covered:
                    covered.add(x)
                    cell_state[x] = cid

                good = self._partial_metric_consistent(
                    cell_state
                )

                if not good:
                    self.stats.prunes_metric += 1

                elif self._fixed_cheap_path_contradiction(
                    cell_state,
                    selected,
                ):
                    self.stats.prunes_clue += 1
                    good = False

                if good:
                    yield from dfs()

                for x in newly_covered:
                    covered.remove(x)
                    cell_state[x] = None

                selected.pop()

                for other in newly_blocked:
                    blocked.remove(other)

        yield from dfs()


# ============================================================================
# PICKLE HELPERS
# ============================================================================

def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("wb") as f:
        pickle.dump(
            obj,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


# ============================================================================
# REPORTING
# ============================================================================

def print_edge_bound_grid(
    lo: Sequence[float],
    hi: Sequence[float],
) -> None:
    ranked = sorted(
        range(E),
        key=lambda e: (
            lo[e],
            -(hi[e] - lo[e]),
        ),
        reverse=True,
    )

    print("\nStrongest forced edge bounds:")

    for eid in ranked[:40]:
        u, v = EDGES[eid]

        print(
            f"  e{eid:03d} "
            f"{ID_NODE[u]}--{ID_NODE[v]} : "
            f"[{lo[eid]:.0f}, "
            f"{hi[eid]:.0f}]"
        )


def print_solution(
    sol: ExactSolution,
    candidates: Sequence[StateCandidate],
) -> None:
    print("\n=== EXACT SOLUTION ===")
    print("states:", len(sol.candidate_ids))
    print(
        "capitols:",
        [ID_NODE[c] for c in sol.capitols],
    )
    print()

    labels = [
        ["." for _ in range(C)]
        for _ in range(R)
    ]

    for k, cid in enumerate(
        sol.candidate_ids
    ):
        ch = str(k % 10)

        for v in candidates[cid].cells:
            r, c = ID_NODE[v]
            labels[r][c] = ch

    for row in labels:
        print(" ".join(row))

    print("\nSelected states:")

    for k, cid in enumerate(
        sol.candidate_ids
    ):
        c = candidates[cid]

        print(
            f"  {k:2d}: "
            f"size={c.size:3d} "
            f"sym={c.symmetry_name:8s} "
            f"capitol="
            f"{None if c.capitol is None else ID_NODE[c.capitol]} "
            f"cells="
            f"{[ID_NODE[v] for v in sorted(c.cells)]}"
        )


# ============================================================================
# COMMANDS
# ============================================================================

def command_metric(args) -> None:
    model = RelaxedMetricModel(
        integer_weights=not args.continuous,
        edge_min=1,
        edge_max=121,
    )

    base = model.solve_with_separation(
        objective=np.ones(E),
        max_rounds=args.max_rounds,
        time_limit_per_round=args.time_limit,
    )

    if not base.feasible:
        raise RuntimeError(
            "Metric relaxation did not converge."
        )

    print(
        f"Initial separated metric feasible "
        f"with {len(model.cuts)} path cuts."
    )

    if args.only_strongest is None:
        edge_ids = list(range(E))
    else:
        score = np.zeros(E, dtype=float)

        for pc in PAIR_CONSTRAINTS:
            a = ID_NODE[pc.i]
            b = ID_NODE[pc.j]

            r, c = a
            rr, cc = b

            path_cells = [pc.i]
            cur = (r, c)

            while cur[0] != rr:
                cur = (
                    cur[0] + (
                        1
                        if rr > cur[0]
                        else -1
                    ),
                    cur[1],
                )
                path_cells.append(
                    NODE_ID[cur]
                )

            while cur[1] != cc:
                cur = (
                    cur[0],
                    cur[1] + (
                        1
                        if cc > cur[1]
                        else -1
                    ),
                )
                path_cells.append(
                    NODE_ID[cur]
                )

            for x, y in zip(
                path_cells,
                path_cells[1:],
            ):
                eid = EDGE_ID[
                    tuple(sorted((x, y)))
                ]
                score[eid] += pc.lower

        edge_ids = list(
            np.argsort(-score)[
                :args.only_strongest
            ]
        )

    lo = np.full(E, 1.0)
    hi = np.full(E, 121.0)

    part_lo, part_hi = (
        model.forced_edge_ranges(
            edge_ids=edge_ids,
            max_rounds=args.max_rounds,
            time_limit_per_round=args.time_limit,
        )
    )

    for eid in edge_ids:
        lo[eid] = part_lo[eid]
        hi[eid] = part_hi[eid]

    out = {
        "edge_lb": lo,
        "edge_ub": hi,
        "cuts": model.cuts,
        "edge_ids_certified": edge_ids,
    }

    save_pickle(
        out,
        Path(args.output),
    )

    print_edge_bound_grid(
        lo,
        hi,
    )

    print(
        f"\nSaved metric bounds "
        f"to {args.output}"
    )


def command_candidates(args) -> None:
    data = load_pickle(
        Path(args.metric)
    )

    lo = np.asarray(
        data["edge_lb"],
        dtype=float,
    )

    workers = (
        args.workers
        if args.workers is not None
        else max(1, os.cpu_count() or 1)
    )

    candidates, templates = (
        generate_state_candidates_parallel(
            edge_lower_bounds=lo,
            workers=workers,
            root_chunk_size=args.root_chunk_size,
            min_state_size=args.min_size,
            max_state_size=args.max_size,
            max_results_per_task=args.max_results_per_task,
            global_candidate_cap=args.global_candidate_cap,
            verbose=True,
        )
    )

    out = {
        "candidates": candidates,
        "templates": templates,
        "metric_edge_lb": lo,
    }

    save_pickle(
        out,
        Path(args.output),
    )

    print(
        f"\nGenerated "
        f"{len(candidates)} "
        f"complete state candidates."
    )

    print(
        f"Intrinsic shape templates: "
        f"{len(templates)}"
    )

    print(
        "Distinct placements were preserved."
    )

    print(
        f"Saved candidates "
        f"to {args.output}"
    )


def command_search(args) -> None:
    data = load_pickle(
        Path(args.candidates)
    )

    candidates: List[
        StateCandidate
    ] = data["candidates"]

    lo = np.asarray(
        data["metric_edge_lb"],
        dtype=float,
    )

    solver = ExactCoverMetricSearch(
        candidates=candidates,
        metric_edge_lb=lo,
    )

    found = 0

    for sol in solver.search(
        max_nodes=args.max_nodes,
    ):
        found += 1

        print_solution(
            sol,
            candidates,
        )

        if args.solution_output:
            save_pickle(
                sol,
                Path(args.solution_output),
            )

            print(
                f"\nSaved exact solution "
                f"to {args.solution_output}"
            )

        if not args.all_solutions:
            break

    print("\nSearch stats:")
    print(solver.stats)
    print(
        "solutions found:",
        found,
    )


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Parallel metric-first "
            "symmetry-constrained "
            "Jane Street solver"
        )
    )

    sub = p.add_subparsers(
        dest="command",
        required=True,
    )

    pm = sub.add_parser(
        "metric",
        help=(
            "derive relaxed "
            "forced edge-weight ranges"
        ),
    )

    pm.add_argument(
        "--output",
        default="metric_bounds.pkl",
    )

    pm.add_argument(
        "--continuous",
        action="store_true",
    )

    pm.add_argument(
        "--max-rounds",
        type=int,
        default=100,
    )

    pm.add_argument(
        "--time-limit",
        type=float,
        default=None,
    )

    pm.add_argument(
        "--only-strongest",
        type=int,
        default=None,
        help=(
            "certify only this many "
            "highest-priority edges first"
        ),
    )

    pm.set_defaults(
        func=command_metric
    )

    pc = sub.add_parser(
        "candidates",
        help=(
            "generate complete "
            "symmetric state candidates "
            "in parallel"
        ),
    )

    pc.add_argument(
        "--metric",
        default="metric_bounds.pkl",
    )

    pc.add_argument(
        "--output",
        default="state_candidates.pkl",
    )

    pc.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "number of worker processes; "
            "default = all CPU cores"
        ),
    )

    pc.add_argument(
        "--root-chunk-size",
        type=int,
        default=2,
        help=(
            "root orbits per worker task; "
            "smaller improves load balancing"
        ),
    )

    pc.add_argument(
        "--min-size",
        type=int,
        default=1,
    )

    pc.add_argument(
        "--max-size",
        type=int,
        default=40,
    )

    pc.add_argument(
        "--max-results-per-task",
        type=int,
        default=10000,
    )

    pc.add_argument(
        "--global-candidate-cap",
        type=int,
        default=250000,
    )

    pc.set_defaults(
        func=command_candidates
    )

    ps = sub.add_parser(
        "search",
        help=(
            "exact-cover search "
            "over complete symmetric states"
        ),
    )

    ps.add_argument(
        "--candidates",
        default="state_candidates.pkl",
    )

    ps.add_argument(
        "--max-nodes",
        type=int,
        default=None,
    )

    ps.add_argument(
        "--all-solutions",
        action="store_true",
    )

    ps.add_argument(
        "--solution-output",
        default="exact_solution.pkl",
    )

    ps.set_defaults(
        func=command_search
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
