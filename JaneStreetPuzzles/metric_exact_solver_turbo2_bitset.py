#!/usr/bin/env python3

"""

Metric-first symmetry-constrained solver for Jane Street's 11x11 puzzle.



This script implements an upgraded version of the algorithm sketched in the

user-provided plan:



    clues

      -> metric lower bounds

      -> forced edge-weight ranges

      -> complete symmetric state candidates

      -> exact-cover / set-partition search

      -> exact Dijkstra verification



Key upgrades over the earlier sketch

-------------------------------------

1. Do not trust one arbitrary relaxed LP solution as a "wall map".

   Instead, compute forced edge lower/upper bounds by optimizing each edge

   over the relaxed feasible metric polytope.



2. Search over COMPLETE symmetric state candidates of exact size k rather than

   growing states cell-by-cell.  Orbit unions are used only to generate legal

   state candidates.



3. Deduplicate intrinsic shapes only as templates.  Distinct placements on the

   clue board remain distinct search candidates.



4. Every accepted state candidate has exact size = number of cells in the

   state.  That immediately fixes all internal edge costs and strongly

   constrains boundary edge costs.



5. A failed exact Dijkstra check returns an explicit cheap path certificate,

   which can be used as a cut / branch-ordering signal.



Dependencies

------------

    numpy

    scipy



No external solver is required; scipy.optimize.milp uses HiGHS.



WARNING

-------

Candidate generation can still be combinatorially large.  The command line

offers size and candidate-count limits.  Start with small limits, inspect the

forced metric bounds, then expand.



This file is intentionally NOT auto-executed by its creator.

"""



from __future__ import annotations



import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import heapq

import itertools

import math

import pickle

import time

from collections import defaultdict, deque

from dataclasses import dataclass

from pathlib import Path

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple



import numpy as np

from scipy.optimize import Bounds, LinearConstraint, milp

from scipy.sparse import coo_matrix, csr_matrix





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

Edge = Tuple[int, int]

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

# BASIC GRAPH UTILITIES

# ============================================================================



def dijkstra(

    weights: Sequence[float],

    source: int,

    target: Optional[int] = None,

) -> Tuple[List[float], Dict[int, Tuple[int, int]]]:

    """Shortest paths on the fixed 11x11 grid."""

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

    """Distance to nearest source, with one nearest-source label."""

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

# PHASE 1: METRIC LOWER BOUNDS

# ============================================================================



@dataclass(frozen=True)

class CluePairConstraint:

    i: int

    j: int

    lower: int





def clue_pair_constraints() -> List[CluePairConstraint]:

    """|f(i)-f(j)| <= d(i,j), hence d(i,j) >= |y_i-y_j|."""

    out: List[CluePairConstraint] = []

    items = sorted(CLUES.items())

    for (i, yi), (j, yj) in itertools.combinations(items, 2):

        L = abs(yi - yj)

        if L > 0:

            out.append(CluePairConstraint(i, j, L))

    return out





PAIR_CONSTRAINTS = clue_pair_constraints()





# ============================================================================

# PHASE 2/3: RELAXED METRIC + FORCED EDGE RANGES

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

    Independent positive edge weights with all clue-pair metric lower bounds.



    The model is solved by cutting planes:

      - solve current LP/MILP

      - Dijkstra every clue pair

      - if shortest path < |yi-yj|, add that path inequality

      - repeat



    This model deliberately ignores states.  It is only a NECESSARY relaxation.

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

            A = coo_matrix((data, (rows, cols)), shape=(len(lb), E)).tocsr()

            constraints = LinearConstraint(

                A,

                np.asarray(lb, dtype=float),

                np.asarray(ub, dtype=float),

            )



        c = -objective if maximize else objective

        integ = np.ones(E, dtype=int) if self.integer_weights else np.zeros(E, dtype=int)



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



    def separate(

        self,

        weights: Sequence[float],

    ) -> int:

        """Add violated shortest-path cuts. Returns number of new cuts."""

        source_cache: Dict[int, Tuple[List[float], Dict[int, Tuple[int, int]]]] = {}

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

                obj = float(np.dot(objective, weights))

                return MetricRelaxationResult(

                    weights=weights,

                    cuts=list(self.cuts),

                    objective=obj,

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

        """

        Compute true feasible min/max for chosen edges under the relaxed model.



        Important:

            We keep every cut discovered while optimizing previous edges.

            The separation loop guarantees each reported bound satisfies ALL

            clue-pair shortest-path inequalities, not merely the currently

            materialized path constraints.

        """

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

                raise RuntimeError(f"Could not certify lower bound for edge {eid}")

            lo[eid] = rmin.weights[eid]



            rmax = self.solve_with_separation(

                objective=objective,

                maximize=True,

                max_rounds=max_rounds,

                time_limit_per_round=time_limit_per_round,

            )

            if not rmax.feasible:

                raise RuntimeError(f"Could not certify upper bound for edge {eid}")

            hi[eid] = rmax.weights[eid]



        return lo, hi





# ============================================================================

# D4 / AFFINE SYMMETRY MACHINERY

# ============================================================================



# Linear parts of D4 on Z^2.

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





def apply_linear(A: Tuple[int, int, int, int], p: Cell) -> Cell:

    a, b, c, d = A

    r, col = p

    return a * r + b * col, c * r + d * col





def apply_affine(g: Affine, p: Cell) -> Cell:

    A, t = g

    x, y = apply_linear(A, p)

    return x + t[0], y + t[1]





def compose_affine(g: Affine, h: Affine) -> Affine:

    """

    Return g∘h.

    g(x)=Ag x+tg; h(x)=Ah x+th.

    """

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

    order = 4 if A in (D4_LINEAR["R90"], D4_LINEAR["R270"]) else 2

    return affine_power(g, order) == (IDENTITY, (0, 0))





def enumerate_board_relevant_affine_symmetries() -> List[Tuple[str, Affine]]:

    """

    Enumerate finite-order affine D4 actions that map at least one board cell to

    another board cell.  These include local rotations/reflections whose center

    or axis can lie on integer or half-integer coordinates.

    """

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





def orbit_of_cell(g: Affine, cell_id: int) -> Optional[Tuple[int, ...]]:

    """

    Return the finite orbit if every orbit point lies inside the 11x11 board.

    Otherwise return None.

    """

    start = ID_NODE[cell_id]

    seen: List[Cell] = []

    cur = start

    for _ in range(8):  # D4 order <= 4; 8 is a safe guard.

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





def full_symmetry_group(cells: Set[int], all_actions: Sequence[Tuple[str, Affine]]) -> List[Tuple[str, Affine]]:

    pts = {ID_NODE[v] for v in cells}

    syms: List[Tuple[str, Affine]] = []

    for name, g in all_actions:

        mapped = {apply_affine(g, p) for p in pts}

        if mapped == pts:

            syms.append((name, g))

    return syms





def fixed_cells_of_action(g: Affine, cells: Set[int]) -> Set[int]:

    return {

        v for v in cells

        if apply_affine(g, ID_NODE[v]) == ID_NODE[v]

    }





def candidate_capitol(

    cells: Set[int],

    all_actions: Sequence[Tuple[str, Affine]],

) -> Optional[int]:

    """

    Conservative interpretation:

    - find every nontrivial affine D4 symmetry of the state;

    - if the common fixed-cell set is exactly one board cell, that is the capitol;

    - otherwise the state has no cell capitol.



    This matches the algorithmic interpretation used in the supplied plan.

    """

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







def full_symmetry_group_fast(cells: Set[int]) -> List[Tuple[str, Affine]]:
    """Find all nontrivial D4 affine symmetries of one candidate state.

    For a fixed linear D4 action A and anchor cell p0, any symmetry mapping
    the finite cell set to itself must send p0 to some q in the same set.
    That uniquely determines the translation t = q - A p0.

    This reduces candidate symmetry checking from scanning ~1400 board-wide
    affine actions to at most 7*|S| candidate transforms.
    """
    if not cells:
        return []
    pts = {ID_NODE[v] for v in cells}
    p0 = min(pts)
    out: List[Tuple[str, Affine]] = []
    seen: Set[Affine] = set()

    for name, A in D4_LINEAR.items():
        # R90 and R270 generate the same orbit group, but both can be actual
        # state symmetries; retain both here for correct full-group/fixed-point
        # calculation.
        Ap0 = apply_linear(A, p0)
        for q in pts:
            t = (q[0] - Ap0[0], q[1] - Ap0[1])
            g: Affine = (A, t)
            if g in seen:
                continue
            seen.add(g)
            if not is_finite_d4_action(g):
                continue
            mapped = {apply_affine(g, p) for p in pts}
            if mapped == pts:
                out.append((name, g))
    return out


def candidate_capitol_fast(cells: Set[int]) -> Optional[int]:
    """Exact candidate capitol test using only candidate-derived symmetries."""
    syms = full_symmetry_group_fast(cells)
    if not syms:
        return None

    common: Optional[Set[int]] = None
    for _, g in syms:
        F = {
            v for v in cells
            if apply_affine(g, ID_NODE[v]) == ID_NODE[v]
        }
        common = F if common is None else (common & F)

    if common is not None and len(common) == 1:
        return next(iter(common))
    return None


def cell_required_sizes(edge_lower_bounds: Sequence[float]) -> np.ndarray:
    """Necessary lower bound on the size of any state containing each cell.

    Every edge incident to a state of size k has puzzle weight <= k:
      internal edge: k
      boundary edge: min(k, neighbor_size) <= k

    Therefore if an incident edge has a forced metric lower bound B, any state
    containing that endpoint must have size at least B.
    """
    req = np.ones(N, dtype=float)
    for eid, (u, v) in enumerate(EDGES):
        b = float(edge_lower_bounds[eid])
        req[u] = max(req[u], b)
        req[v] = max(req[v], b)
    return req



# ============================================================================
# TURBO CLUE-SIZE SIEVE
# ============================================================================

ALL_NODE_MASK = (1 << N) - 1

NODE_BIT: List[int] = [1 << v for v in range(N)]

NEIGHBOR_MASK: List[int] = [0] * N
for _v in range(N):
    _m = 0
    for _nb, _eid in ADJ[_v]:
        _m |= NODE_BIT[_nb]
    NEIGHBOR_MASK[_v] = _m

POSITIVE_CLUE_ITEMS: Tuple[Tuple[int, int], ...] = tuple(
    sorted((v, y) for v, y in CLUES.items() if y > 0)
)

POSITIVE_CLUE_IDS: Set[int] = {v for v, _ in POSITIVE_CLUE_ITEMS}


# A singleton capitol cannot sit on a positive clue cell.  For each clue-1
# position, these are the neighboring cells that could still serve as its
# required singleton capitol.
ONE_SINGLETON_ELIGIBLE_NEIGHBORS: Dict[int, int] = {}
for _v in ONE_CELLS:
    _mask = 0
    for _nb, _eid in ADJ[_v]:
        if _nb not in POSITIVE_CLUE_CELLS:
            _mask |= NODE_BIT[_nb]
    ONE_SINGLETON_ELIGIBLE_NEIGHBORS[_v] = _mask


def _manhattan_shortest_cell_paths(a: int, b: int, max_paths: int = 1000):
    """Enumerate cell masks for short Manhattan-shortest paths."""
    ra, ca = ID_NODE[a]
    rb, cb = ID_NODE[b]
    out = []

    def rec(r: int, c: int, mask: int):
        if len(out) >= max_paths:
            return
        v = NODE_ID[(r, c)]
        mask2 = mask | NODE_BIT[v]
        if (r, c) == (rb, cb):
            out.append(mask2)
            return
        if r != rb:
            nr = r + (1 if rb > r else -1)
            rec(nr, c, mask2)
        if c != cb:
            nc = c + (1 if cb > c else -1)
            rec(r, nc, mask2)

    rec(ra, ca, 0)
    return out


# If a candidate state contains every cell of one concrete path P between two
# clue cells, that path costs |P_edges| * |S| because every path edge is
# internal. Triangle inequality gives
#
#     |S| >= ceil(|y_i-y_j| / |P_edges|).
#
# We materialize only very short Manhattan paths; these are cheap, local,
# strong constraints.  Each rule is (cell_mask, minimum_state_size).
CLUE_INTERNAL_PATH_RULES: List[Tuple[int, int]] = []
for _pc in PAIR_CONSTRAINTS:
    _a = ID_NODE[_pc.i]
    _b = ID_NODE[_pc.j]
    _m = abs(_a[0] - _b[0]) + abs(_a[1] - _b[1])
    if not (1 <= _m <= 4):
        continue
    _need = int(math.ceil(_pc.lower / _m))
    if _need <= 1:
        continue
    for _mask in _manhattan_shortest_cell_paths(_pc.i, _pc.j):
        CLUE_INTERNAL_PATH_RULES.append((_mask, _need))

# Deduplicate identical masks by keeping the strongest requirement.
_rule_best: Dict[int, int] = {}
for _mask, _need in CLUE_INTERNAL_PATH_RULES:
    _rule_best[_mask] = max(_rule_best.get(_mask, 1), _need)
CLUE_INTERNAL_PATH_RULES = sorted(_rule_best.items())

# Index path rules by cells so adding an orbit only checks rules that could
# have become newly complete.
PATH_RULE_IDS_BY_CELL: List[List[int]] = [[] for _ in range(N)]
for _rid, (_mask, _need) in enumerate(CLUE_INTERNAL_PATH_RULES):
    _x = _mask
    while _x:
        _lsb = _x & -_x
        _v = _lsb.bit_length() - 1
        PATH_RULE_IDS_BY_CELL[_v].append(_rid)
        _x ^= _lsb


def tighten_internal_path_requirement(
    state_mask: int,
    newly_added_cells: Iterable[int],
    current_req: float,
) -> float:
    """Raise minimum final state size using newly completed clue paths."""
    req = current_req
    rule_ids = set()
    for v in newly_added_cells:
        rule_ids.update(PATH_RULE_IDS_BY_CELL[v])
    for rid in rule_ids:
        path_mask, need = CLUE_INTERNAL_PATH_RULES[rid]
        if need > req and (path_mask & state_mask) == path_mask:
            req = float(need)
    return req


def clue_one_still_possible(state_mask: int) -> bool:
    """Necessary monotone rule for clue value 1.

    Every clue-1 cell needs an adjacent singleton capitol.  If a candidate
    state contains the clue-1 cell and has already swallowed every adjacent
    cell that could legally be that singleton capitol, extending the state
    cannot repair the branch.
    """
    for v in ONE_CELLS:
        if not (state_mask & NODE_BIT[v]):
            continue
        eligible = ONE_SINGLETON_ELIGIBLE_NEIGHBORS[v]
        if eligible == 0:
            return False
        if (eligible & ~state_mask) == 0:
            return False
    return True


def action_zero_compatible(g: Affine, orbit_cells: Sequence[int]) -> bool:
    """If an orbit contains a known zero, the generating symmetry must fix it.

    Any symmetry used to generate a valid state is an actual symmetry of that
    state.  A capitol must be fixed by every nontrivial state symmetry under
    the conservative interpretation used by this solver.
    """
    for z in ZERO_CELLS:
        if z in orbit_cells and apply_affine(g, ID_NODE[z]) != ID_NODE[z]:
            return False
    return True


def cells_to_mask(cells: Iterable[int]) -> int:
    m = 0
    for v in cells:
        m |= NODE_BIT[v]
    return m


def newly_buried_clue_cap(
    state_mask: int,
    clue_ids: Iterable[int],
    current_cap: int,
) -> int:
    """Tighten the maximum possible FINAL state size.

    If a positive clue v with value y is an interior cell of a state S, then
    every route from v must take at least one internal step before leaving S
    or reaching a different cell-capitol in S.  That first internal step costs
    |S|.  Positive clues cannot themselves be capitols, hence necessarily

        |S| <= y.

    "Interior" means every grid neighbor of v is also in the state.  Once a
    clue becomes interior while a state is being grown, later additions can
    never make it non-interior, so this is a monotone and safe DFS prune.
    """
    cap = current_cap
    for v in clue_ids:
        y = CLUES.get(v)
        if y is None or y <= 0:
            continue
        if not (state_mask & NODE_BIT[v]):
            continue
        if (NEIGHBOR_MASK[v] & ~state_mask) == 0:
            if y < cap:
                cap = y
    return cap


def distances_inside_cells(cells: Set[int], sources: Iterable[int]) -> Dict[int, int]:
    """Unweighted grid distances restricted to a candidate state."""
    src = list(sources)
    if not src:
        return {}
    dist = {s: 0 for s in src}
    q = deque(src)
    while q:
        u = q.popleft()
        du = dist[u]
        for v, _ in ADJ[u]:
            if v in cells and v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist


def clue_geometry_consistent(
    cells: Set[int],
    size: int,
    capitol: Optional[int],
) -> bool:
    """Necessary clue tests for one complete state candidate.

    1) Escape/capitol lower bound:
       If a clue is d internal edges from the state boundary, reaching any
       OUTSIDE capitol costs at least d*size + 1.  Reaching this state's own
       capitol costs d_cap*size.  If the minimum of these unavoidable costs is
       already larger than the published clue, reject the state.

    2) Same-state clue-pair upper path:
       If two clues inside this state have an internal path of l steps, then
       d(i,j) <= l*size.  Triangle inequality requires
       d(i,j) >= |y_i-y_j|.  Therefore l*size must be at least the clue gap.

    Both tests are necessary conditions only, so they are safe pruning rules.
    """
    clues_here = [(v, CLUES[v]) for v in cells if v in CLUES]
    if not clues_here:
        return True

    boundary = {
        u for u in cells
        if any(v not in cells for v, _ in ADJ[u])
    }

    # A state equal to the whole board has no grid boundary.
    dist_boundary = distances_inside_cells(cells, boundary)
    dist_cap = (
        distances_inside_cells(cells, [capitol])
        if capitol is not None else {}
    )

    for v, y in clues_here:
        if y == 0:
            # Zero handling is enforced separately by candidate capitol logic.
            continue

        lb_exit = math.inf
        if v in dist_boundary:
            lb_exit = dist_boundary[v] * size + 1

        lb_cap = math.inf
        if capitol is not None and v in dist_cap:
            lb_cap = dist_cap[v] * size

        if min(lb_exit, lb_cap) > y + 1e-9:
            return False

    # Internal paths give upper bounds on the true graph metric.
    positive_here = [(v, y) for v, y in clues_here if y > 0]
    for i in range(len(positive_here)):
        u, yu = positive_here[i]
        dist_u = distances_inside_cells(cells, [u])
        for j in range(i + 1, len(positive_here)):
            v, yv = positive_here[j]
            l = dist_u.get(v)
            if l is None:
                continue
            if l * size + 1e-9 < abs(yu - yv):
                return False

    return True


def enumerate_connected_orbit_unions_fast(
    orbits: Sequence[Tuple[int, ...]],
    orbit_adj: Dict[int, Set[int]],
    cell_req: Sequence[float],
    min_size: int,
    max_size: int,
    max_results: int,
) -> Iterator[Set[int]]:
    """Turbo connected orbit-union enumeration.

    Each DFS branch maintains a feasible interval for the FINAL state size:

        required_min <= |S_final| <= allowed_max

    required_min increases from:
        * forced metric edge lower bounds,
        * short fully-internal clue paths.

    allowed_max decreases when:
        * a positive clue becomes interior.

    A branch dies immediately when the interval becomes empty.
    """
    if not orbits:
        return

    orbit_size = [len(o) for o in orbits]
    orbit_req = [max(float(cell_req[v]) for v in o) for o in orbits]
    orbit_mask = [cells_to_mask(o) for o in orbits]

    orbit_affected_clues: List[Tuple[int, ...]] = []
    for orb in orbits:
        affected = set()
        for x in orb:
            if x in POSITIVE_CLUE_IDS:
                affected.add(x)
            for nb, _ in ADJ[x]:
                if nb in POSITIVE_CLUE_IDS:
                    affected.add(nb)
        orbit_affected_clues.append(tuple(affected))

    produced = 0
    seen_sets: Set[Tuple[int, ...]] = set()

    for root in range(len(orbits)):
        n0 = orbit_size[root]
        req0 = max(float(min_size), orbit_req[root])
        if n0 > max_size or req0 > max_size + 1e-9:
            continue

        mask0 = orbit_mask[root]
        if not clue_one_still_possible(mask0):
            continue

        req0 = tighten_internal_path_requirement(
            mask0, orbits[root], req0
        )
        cap0 = newly_buried_clue_cap(
            mask0,
            orbit_affected_clues[root],
            max_size,
        )

        if n0 > cap0 or req0 > cap0 + 1e-9:
            continue

        initial_frontier = frozenset(
            x for x in orbit_adj[root]
            if x > root and orbit_req[x] <= cap0 + 1e-9
        )

        # chosen, frontier, n, required_min, allowed_max, state_mask
        stack: List[Tuple[frozenset, frozenset, int, float, int, int]] = [
            (frozenset({root}), initial_frontier, n0, req0, cap0, mask0)
        ]

        while stack:
            chosen, frontier, n, req, allowed_max, state_mask = stack.pop()

            if n > allowed_max or req > allowed_max + 1e-9:
                continue

            if n >= req - 1e-9 and n >= min_size:
                cells: Set[int] = set()
                for oid in chosen:
                    cells.update(orbits[oid])
                key = tuple(sorted(cells))
                if key not in seen_sets:
                    seen_sets.add(key)
                    yield cells
                    produced += 1
                    if produced >= max_results:
                        return

            if n >= allowed_max:
                continue

            frontier_list = sorted(
                frontier,
                key=lambda oid: (
                    orbit_req[oid],
                    len(orbit_affected_clues[oid]),
                    orbit_size[oid],
                ),
                reverse=True,
            )

            for oid in frontier_list:
                new_n = n + orbit_size[oid]
                if new_n > allowed_max:
                    continue

                new_mask = state_mask | orbit_mask[oid]

                # Clue 1 must retain an outside singleton-capitol option.
                if not clue_one_still_possible(new_mask):
                    continue

                new_req = max(req, orbit_req[oid])
                new_req = tighten_internal_path_requirement(
                    new_mask, orbits[oid], new_req
                )

                new_allowed_max = newly_buried_clue_cap(
                    new_mask,
                    orbit_affected_clues[oid],
                    allowed_max,
                )

                if new_n > new_allowed_max:
                    continue
                if new_req > new_allowed_max + 1e-9:
                    continue

                new_chosen = set(chosen)
                new_chosen.add(oid)

                new_frontier = set(frontier)
                new_frontier.discard(oid)

                for nb in orbit_adj[oid]:
                    if nb < root or nb in new_chosen:
                        continue
                    if orbit_req[nb] <= new_allowed_max + 1e-9:
                        new_frontier.add(nb)

                stack.append((
                    frozenset(new_chosen),
                    frozenset(new_frontier),
                    new_n,
                    new_req,
                    new_allowed_max,
                    new_mask,
                ))

# ============================================================================

# COMPLETE SYMMETRIC STATE CANDIDATES

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





def incident_and_internal_edges(cells: Set[int]) -> Tuple[Set[int], Set[int]]:

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

    """

    Build valid board-contained orbits and adjacency between orbits.

    """

    orbit_map: Dict[int, Tuple[int, ...]] = {}

    unique: Dict[Tuple[int, ...], int] = {}



    for v in range(N):

        orb = orbit_of_cell(g, v)

        if orb is None:

            continue

        if orb not in unique:

            unique[orb] = len(unique)

        oid = unique[orb]

        for x in orb:

            orbit_map[x] = orb



    orbits = list(unique.keys())

    oid_by_orbit = {orb: i for i, orb in enumerate(orbits)}

    G: Dict[int, Set[int]] = {i: set() for i in range(len(orbits))}



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





def enumerate_connected_orbit_unions(

    orbits: Sequence[Tuple[int, ...]],

    orbit_adj: Dict[int, Set[int]],

    min_size: int,

    max_size: int,

    max_results: int,

) -> Iterator[Set[int]]:

    """

    Enumerate connected orbit-unions.



    Uses a canonical expansion rule to reduce duplicate subset generation.

    """

    produced = 0

    seen_sets: Set[Tuple[int, ...]] = set()



    for root in range(len(orbits)):

        root_cells = set(orbits[root])

        if len(root_cells) > max_size:

            continue



        stack: List[Tuple[frozenset, frozenset]] = [

            (frozenset({root}), frozenset(x for x in orbit_adj[root] if x > root))

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



            frontier_list = sorted(frontier, reverse=True)

            for oid in frontier_list:

                new_chosen = set(chosen)

                new_chosen.add(oid)



                # Canonical growth: never add an orbit index below the root.

                new_frontier = set(frontier)

                new_frontier.discard(oid)

                for nb in orbit_adj[oid]:

                    if nb >= root and nb not in new_chosen:

                        new_frontier.add(nb)



                stack.append((frozenset(new_chosen), frozenset(new_frontier)))





def intrinsic_shape_signature(cells: Set[int]) -> Tuple[Cell, ...]:

    """

    Translation-normalized shape only.

    Used for template bookkeeping; NOT for eliminating distinct placements.

    """

    pts = [ID_NODE[v] for v in cells]

    r0 = min(r for r, _ in pts)

    c0 = min(c for _, c in pts)

    return tuple(sorted((r - r0, c - c0) for r, c in pts))





def _generate_candidates_for_actions_worker(payload):
    (action_chunk, edge_lower_bounds, min_state_size, max_state_size,
     max_unions_per_symmetry, per_worker_candidate_cap) = payload
    edge_lower_bounds = np.asarray(edge_lower_bounds, dtype=float)
    cell_req = cell_required_sizes(edge_lower_bounds)
    records = []
    seen_local: Set[frozenset] = set()
    for sym_name, g in action_chunk:
        orbits, oadj = orbit_graph(g)
        if not orbits:
            continue
        keep_old = [
            i for i, orb in enumerate(orbits)
            if max(float(cell_req[v]) for v in orb) <= max_state_size + 1e-9
            and action_zero_compatible(g, orb)
        ]
        if not keep_old:
            continue
        remap = {old: new for new, old in enumerate(keep_old)}
        orbits2 = [orbits[old] for old in keep_old]
        oadj2 = {i: set() for i in range(len(orbits2))}
        for old in keep_old:
            a = remap[old]
            for old_nb in oadj.get(old, set()):
                if old_nb in remap:
                    oadj2[a].add(remap[old_nb])
        for cells in enumerate_connected_orbit_unions_fast(
            orbits2, oadj2, cell_req=cell_req,
            min_size=min_state_size, max_size=max_state_size,
            max_results=max_unions_per_symmetry):
            if not connected(cells):
                continue
            fs = frozenset(cells)
            if fs in seen_local:
                continue
            size = len(cells)
            incident, internal = incident_and_internal_edges(cells)
            if any(edge_lower_bounds[e] > size + 1e-9 for e in incident):
                continue
            cap = candidate_capitol_fast(cells)

            # Strong candidate-specific clue/size sieve.
            if not clue_geometry_consistent(cells, size, cap):
                continue

            zero_inside = ZERO_CELLS & cells
            if zero_inside and (len(zero_inside) != 1 or cap not in zero_inside):
                continue
            if cap is not None and cap in POSITIVE_CLUE_CELLS:
                continue
            if size == 1 and cap is None:
                continue
            records.append((fs, size, sym_name, g, cap,
                            frozenset(incident), frozenset(internal)))
            seen_local.add(fs)
            if len(records) >= per_worker_candidate_cap:
                return records
    return records


def _chunk_actions(actions, nchunks: int):
    nchunks = max(1, min(nchunks, len(actions)))
    chunks = [[] for _ in range(nchunks)]
    for i, action in enumerate(actions):
        chunks[i % nchunks].append(action)
    return [c for c in chunks if c]


def generate_state_candidates(
    edge_lower_bounds: Sequence[float], min_state_size: int = 1,
    max_state_size: int = 40, max_unions_per_symmetry: int = 5000,
    global_candidate_cap: int = 100000, workers: int = 1,
    worker_chunks_per_cpu: int = 2, progress: bool = True,
) -> Tuple[List[StateCandidate], Dict[Tuple[Cell, ...], List[int]]]:
    """Parallel candidate generation over independent affine D4 actions."""
    all_actions = enumerate_board_relevant_affine_symmetries()
    generator_actions = [(name, g) for name, g in all_actions if name != "R270"]
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    workers = max(1, min(int(workers), len(generator_actions)))
    nchunks = max(1, workers * max(1, int(worker_chunks_per_cpu)))
    chunks = _chunk_actions(generator_actions, nchunks)
    per_worker_cap = max(global_candidate_cap,
                         math.ceil(global_candidate_cap / max(1, workers)) * 4)
    edge_lb_arr = np.asarray(edge_lower_bounds, dtype=float)
    payloads = [(chunk, edge_lb_arr, min_state_size, max_state_size,
                 max_unions_per_symmetry, per_worker_cap) for chunk in chunks]
    if progress:
        print(f"Candidate generation TURBO: {len(generator_actions)} actions, "
              f"{len(chunks)} chunks, {workers} workers")
        print("  sieves: metric-size + buried-clue upper caps + short clue-path "
              "lower bounds + clue-1 singleton escape + zero/symmetry + "
              "complete-candidate clue geometry")
    raw_records = []
    start_time = time.monotonic()

    def _fmt_seconds(sec: float) -> str:
        sec = max(0, int(round(sec)))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _print_progress(done: int, total: int, new_count: int) -> None:
        if not progress:
            return
        elapsed = time.monotonic() - start_time
        pct = 100.0 * done / max(1, total)
        avg = elapsed / max(1, done)
        remaining = avg * max(0, total - done)
        print(
            f"  progress {done}/{total} ({pct:6.2f}%) | "
            f"+{new_count:,} raw | total_raw={len(raw_records):,} | "
            f"elapsed={_fmt_seconds(elapsed)} | "
            f"avg/chunk={avg:.1f}s | ETA={_fmt_seconds(remaining)}",
            flush=True,
        )

    if workers == 1:
        for i, payload in enumerate(payloads, 1):
            recs = _generate_candidates_for_actions_worker(payload)
            raw_records.extend(recs)
            _print_progress(i, len(payloads), len(recs))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(_generate_candidates_for_actions_worker, p): i
                       for i, p in enumerate(payloads, 1)}
            done = 0
            for fut in as_completed(futures):
                recs = fut.result()
                raw_records.extend(recs)
                done += 1
                _print_progress(done, len(payloads), len(recs))
    by_cells = {}
    for rec in raw_records:
        by_cells.setdefault(rec[0], rec)
    records = list(by_cells.values())
    records.sort(key=lambda r: (r[1], tuple(sorted(r[0])), r[2]))
    records = records[:global_candidate_cap]
    candidates = []
    templates = defaultdict(list)
    for cid, rec in enumerate(records):
        fs, size, sym_name, g, cap, incident, internal = rec
        cand = StateCandidate(id=cid, cells=fs, size=size,
                              symmetry_name=sym_name, transform=g,
                              capitol=cap, incident_edges=incident,
                              internal_edges=internal)
        candidates.append(cand)
        templates[intrinsic_shape_signature(set(fs))].append(cid)
    if progress:
        print(f"Parallel generation finished: {len(raw_records)} raw -> "
              f"{len(candidates)} unique candidates")
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

        checkpoint_path: Optional[Path] = None,

    ):

        self.candidates = list(candidates)

        self.metric_edge_lb = np.asarray(metric_edge_lb, dtype=float)

        self.checkpoint_path = checkpoint_path

        self.stats = SearchStats()
        self._search_start_time = None
        self._last_progress_nodes = 0
        self._last_progress_time = None
        self._time_choose = 0.0
        self._time_metric = 0.0
        self._time_clue = 0.0

        # Memoize expensive partial clue checks.
        # Keyed by the currently selected complete-state candidate IDs.
        self._cheap_path_cache: Dict[Tuple[int, ...], bool] = {}
        self._cheap_path_cache_hits = 0
        self._cheap_path_cache_misses = 0
        self._incremental_relaxations = 0
        self._incremental_updates = 0

        # Cache edge weights for candidate pairs.  Whenever both endpoint
        # states are known, the edge cost is simply min(size_a, size_b).
        self._pair_weight_cache: Dict[Tuple[int, int], int] = {}



        self.by_cell: List[List[int]] = [[] for _ in range(N)]

        for cand in self.candidates:

            for v in cand.cells:

                self.by_cell[v].append(cand.id)

        # Bitset incidence: CELL_CAND_MASK[v] has bit cid set iff candidate cid
        # covers cell v. Python big-int AND/OR/bit_count execute in optimized C.
        self.cell_cand_mask: List[int] = [0] * N
        for v in range(N):
            m = 0
            for cid in self.by_cell[v]:
                m |= (1 << cid)
            self.cell_cand_mask[v] = m

        self.all_candidate_mask = (1 << len(self.candidates)) - 1

        # Precompute, for each candidate, the union of incidence bitsets of all
        # cells it occupies. This is the full set of conflicting candidates.
        # Unlike Python sets-of-sets, one big-int per candidate is compact.
        self.conflict_mask: List[int] = [0] * len(self.candidates)
        for cand in self.candidates:
            m = 0
            for v in cand.cells:
                m |= self.cell_cand_mask[v]
            self.conflict_mask[cand.id] = m



        # Prefer small candidate domains and candidates that cover high-LB edges.

        self.candidate_score = np.zeros(len(self.candidates), dtype=float)

        for c in self.candidates:

            support = sum(self.metric_edge_lb[e] for e in c.incident_edges)

            self.candidate_score[c.id] = support + 0.25 * c.size



    def _choose_uncovered_cell(
        self,
        covered_mask: int,
        active_mask: int,
    ) -> Tuple[Optional[int], int]:
        """MRV branching using big-int bitsets.

        Returns:
            (cell, viable_candidate_mask_for_that_cell)

        Complexity is O(121) big-int AND + popcount operations.
        """
        best_v = None
        best_count = math.inf
        best_mask = 0

        for v in range(N):
            if covered_mask & NODE_BIT[v]:
                continue

            vm = self.cell_cand_mask[v] & active_mask
            cnt = vm.bit_count()

            if cnt < best_count:
                best_count = cnt
                best_v = v
                best_mask = vm
                if cnt <= 1:
                    break

        return best_v, best_mask


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

        key = (su, sv) if su <= sv else (sv, su)
        cached = self._pair_weight_cache.get(key)
        if cached is not None:
            return cached

        cu = self.candidates[su]
        cv = self.candidates[sv]
        w = min(cu.size, cv.size)
        self._pair_weight_cache[key] = w
        return w


    def _partial_metric_consistent(

        self,

        cell_state: Sequence[Optional[int]],

    ) -> bool:

        """

        Exact assigned edge weights must respect the globally forced relaxed

        edge lower bounds.

        """

        for eid, (u, v) in enumerate(EDGES):

            w = self._edge_exact_if_assigned(u, v, cell_state)

            if w is not None and w + 1e-9 < self.metric_edge_lb[eid]:

                return False

        return True



    def _extend_fixed_distances(
        self,
        parent_dist: Sequence[float],
        cell_state: Sequence[Optional[int]],
        newly_covered: Sequence[int],
        new_capitol: Optional[int],
    ) -> Tuple[bool, List[float]]:
        """Incrementally update shortest distances to fixed capitols.

        Parent invariant:
            parent_dist[v] is the shortest distance from v to any capitol
            using only edges whose endpoint states were fixed in the parent.

        Adding one complete state only introduces:
            * its internal edges,
            * boundary edges to already-fixed neighboring states,
            * possibly one new capitol.

        We therefore seed Dijkstra only where new information enters instead
        of recomputing from every capitol over the entire partial graph.

        Returns:
            (contradiction, new_dist)

        A contradiction means a published clue now has a permanently fixed
        route to a capitol cheaper than its target.
        """
        self._incremental_updates += 1
        dist = list(parent_dist)
        pq: List[Tuple[float, int]] = []

        new_set = set(newly_covered)

        # A newly fixed capitol is a new zero-distance source.
        if new_capitol is not None and dist[new_capitol] > 0.0:
            dist[new_capitol] = 0.0
            heapq.heappush(pq, (0.0, new_capitol))

        # Existing fixed distances can enter the newly added state only through
        # its boundary edges. Seed those relaxations.
        for u in newly_covered:
            su = cell_state[u]
            if su is None:
                continue
            for v, _eid in ADJ[u]:
                if v in new_set:
                    continue
                sv = cell_state[v]
                if sv is None:
                    continue
                dv = parent_dist[v]
                if math.isinf(dv):
                    continue

                pair = (su, sv) if su <= sv else (sv, su)
                w = self._pair_weight_cache.get(pair)
                if w is None:
                    w = min(self.candidates[su].size, self.candidates[sv].size)
                    self._pair_weight_cache[pair] = w

                nd = dv + float(w)
                if nd < dist[u]:
                    dist[u] = nd
                    heapq.heappush(pq, (nd, u))

        # If nothing can reach a fixed capitol yet, this branch cannot have a
        # cheap-path contradiction.
        if not pq:
            return False, dist

        # Parent had already been checked and was contradiction-free, so only
        # nodes whose distance IMPROVES can newly violate a clue.
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue

            target = CLUES.get(u)
            if target is not None and d < target - 1e-9:
                return True, dist

            su = cell_state[u]
            if su is None:
                continue

            for v, _eid in ADJ[u]:
                sv = cell_state[v]
                if sv is None:
                    continue

                pair = (su, sv) if su <= sv else (sv, su)
                w = self._pair_weight_cache.get(pair)
                if w is None:
                    w = min(self.candidates[su].size, self.candidates[sv].size)
                    self._pair_weight_cache[pair] = w

                nd = d + float(w)
                if nd < dist[v]:
                    dist[v] = nd
                    self._incremental_relaxations += 1
                    heapq.heappush(pq, (nd, v))

        return False, dist


    def _fixed_cheap_path_contradiction(
        self,
        cell_state: Sequence[Optional[int]],
        selected: Sequence[int],
    ) -> bool:
        """Safe partial clue pruning with memoization.

        Only edges whose endpoint states are already fixed can participate.
        The result depends solely on the selected complete-state candidates,
        so repeated search states can reuse it exactly.
        """
        key = tuple(sorted(selected))
        cached = self._cheap_path_cache.get(key)
        if cached is not None:
            self._cheap_path_cache_hits += 1
            return cached

        self._cheap_path_cache_misses += 1

        fixed_caps = [
            self.candidates[cid].capitol
            for cid in selected
            if self.candidates[cid].capitol is not None
        ]
        fixed_caps = [c for c in fixed_caps if c is not None]

        if not fixed_caps:
            return self._cheap_cache_store(key, False)

        # Dijkstra directly over currently fixed edges instead of allocating a
        # full E-length weight array and then calling the generic routine.
        dist = [math.inf] * N
        pq = []

        for s in fixed_caps:
            if dist[s] > 0.0:
                dist[s] = 0.0
                heapq.heappush(pq, (0.0, s))

        # Early stopping threshold: once the cheapest unsettled node exceeds
        # every clue target, no future popped node can create a cheap-path
        # contradiction.
        max_target = max(CLUES.values())

        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            if d >= max_target:
                break

            # If u itself is a clue and it is already too cheap, we can stop.
            target = CLUES.get(u)
            if target is not None and d < target - 1e-9:
                return self._cheap_cache_store(key, True)

            su = cell_state[u]
            if su is None:
                continue

            for v, _eid in ADJ[u]:
                sv = cell_state[v]
                if sv is None:
                    continue

                pair = (su, sv) if su <= sv else (sv, su)
                w = self._pair_weight_cache.get(pair)
                if w is None:
                    w = min(self.candidates[su].size, self.candidates[sv].size)
                    self._pair_weight_cache[pair] = w

                nd = d + float(w)
                if nd < dist[v] and nd < max_target:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        # Some clues may be fixed-capitol nodes themselves (distance 0), so
        # explicit final scan is still cheap and safe.
        for v, y in CLUES.items():
            if dist[v] < y - 1e-9:
                return self._cheap_cache_store(key, True)

        return self._cheap_cache_store(key, False)


    def _cheap_cache_store(self, key: Tuple[int, ...], value: bool) -> bool:
        # Keep memory bounded.  Clear wholesale rather than maintaining an LRU;
        # this is rare and much cheaper than per-access bookkeeping.
        if len(self._cheap_path_cache) >= 500_000:
            self._cheap_path_cache.clear()
            self._cheap_path_cache_hits = 0
            self._cheap_path_cache_misses = 0
        self._cheap_path_cache[key] = value
        return value

    def _build_complete_partition(

        self,

        selected: Sequence[int],

    ) -> ExactSolution:

        cell_to_state: List[int] = [-1] * N

        for cid in selected:

            for v in self.candidates[cid].cells:

                if cell_to_state[v] != -1:

                    raise RuntimeError("Overlap in complete partition")

                cell_to_state[v] = cid



        if any(x == -1 for x in cell_to_state):

            raise RuntimeError("Incomplete partition")



        weights = np.zeros(E, dtype=float)

        for eid, (u, v) in enumerate(EDGES):

            su = self.candidates[cell_to_state[u]].size

            sv = self.candidates[cell_to_state[v]].size

            weights[eid] = min(su, sv)



        capitols = [

            self.candidates[cid].capitol

            for cid in selected

            if self.candidates[cid].capitol is not None

        ]

        capitols = [c for c in capitols if c is not None]



        predicted, _, _ = multi_source_dijkstra(weights, capitols)



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

    ) -> Tuple[bool, Optional[Tuple[int, int, List[int], float]]]:

        """

        Exact verifier.



        Returns

        -------

        ok

        certificate



        If a clue is too small in the candidate:

            certificate = (cell, target, path_edges, predicted)

        where path_edges is an explicit forbidden cheap path to a selected

        capitol whenever recoverable.



        If a clue is too large:

            path_edges is empty; the branch needs a cheaper route.

        """

        # Recompute with predecessors for certificate extraction.

        dist, prev, owner = multi_source_dijkstra(sol.edge_weights, sol.capitols)



        for v, target in CLUES.items():

            got = dist[v]

            if abs(got - target) > 1e-9:

                if got < target and owner[v] is not None:

                    src = owner[v]

                    # prev points from node toward predecessor closer to source.

                    path = []

                    cur = v

                    while cur != src and cur in prev:

                        p, eid = prev[cur]

                        path.append(eid)

                        cur = p

                    return False, (v, target, path, got)

                return False, (v, target, [], got)



        # 0 clues must be capitols, positive clues must not be capitols.

        capset = set(sol.capitols)

        if not ZERO_CELLS <= capset:

            return False, None

        if POSITIVE_CLUE_CELLS & capset:

            return False, None



        return True, None



    def search(

        self,

        max_nodes: Optional[int] = None,

        stop_after_first: bool = True,

        progress_every: int = 10000,

        heartbeat_seconds: float = 10.0,

    ) -> Iterator[ExactSolution]:

        """

        Depth-first exact-cover search.



        Branches on a complete state candidate covering the most constrained

        currently-uncovered cell.

        """

        selected: List[int] = []

        covered: Set[int] = set()
        covered_mask = 0

        # Candidate viability as one Python big-int bitset.
        active_mask = self.all_candidate_mask

        cell_state: List[Optional[int]] = [None] * N

        self._search_start_time = time.monotonic()
        self._last_progress_time = self._search_start_time

        def _fmt_seconds(sec: float) -> str:
            sec = max(0, int(round(sec)))
            h, rem = divmod(sec, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def _print_progress() -> None:
            now = time.monotonic()
            elapsed = now - self._search_start_time
            nodes = self.stats.nodes
            rate = nodes / elapsed if elapsed > 0 else 0.0

            if max_nodes is not None and rate > 0:
                remaining_nodes = max(0, max_nodes - nodes)
                eta = remaining_nodes / rate
                pct = 100.0 * nodes / max_nodes if max_nodes else 100.0
                node_part = f"{nodes:,}/{max_nodes:,} ({pct:6.2f}%)"
                eta_part = _fmt_seconds(eta)
            else:
                node_part = f"{nodes:,}"
                eta_part = "n/a"

            min_domain = math.inf
            for _v in range(N):
                if covered_mask & NODE_BIT[_v]:
                    continue
                _cnt = (self.cell_cand_mask[_v] & active_mask).bit_count()
                if _cnt < min_domain:
                    min_domain = _cnt
            if min_domain is math.inf:
                min_domain = 0

            print(
                f"[search] nodes={node_part} | "
                f"rate={rate:,.1f}/s | "
                f"depth={len(selected)} | "
                f"min_domain={min_domain} | "
                f"elapsed={_fmt_seconds(elapsed)} | "
                f"ETA={eta_part} | "
                f"dead={self.stats.prunes_dead_cell:,} | "
                f"metric={self.stats.prunes_metric:,} | "
                f"clue={self.stats.prunes_clue:,} | "
                f"complete={self.stats.complete_partitions:,} | "
                f"t_choose={self._time_choose:.1f}s | "
                f"t_metric={self._time_metric:.1f}s | "
                f"t_clue={self._time_clue:.1f}s | "
                f"cache={self._cheap_path_cache_hits:,}/"
                f"{self._cheap_path_cache_hits + self._cheap_path_cache_misses:,} | "
                f"inc_updates={self._incremental_updates:,} | "
                f"relax={self._incremental_relaxations:,} | "
                f"cands={len(self.candidates):,} | "
                f"active={active_mask.bit_count():,}",
                flush=True,
            )

        root_fixed_dist = [math.inf] * N

        def dfs(
            fixed_dist: Sequence[float],
            covered_mask_local: int,
            active_mask_local: int,
        ) -> Iterator[ExactSolution]:
            nonlocal covered_mask, active_mask
            covered_mask = covered_mask_local
            active_mask = active_mask_local

            if max_nodes is not None and self.stats.nodes >= max_nodes:

                return

            self.stats.nodes += 1

            now = time.monotonic()
            if (
                heartbeat_seconds is not None
                and heartbeat_seconds > 0
                and now - self._last_progress_time >= heartbeat_seconds
            ):
                _print_progress()
                self._last_progress_time = now
            elif (
                progress_every is not None
                and progress_every > 0
                and self.stats.nodes % progress_every == 0
            ):
                _print_progress()
                self._last_progress_time = now



            if covered_mask_local == ALL_NODE_MASK:

                self.stats.complete_partitions += 1

                sol = self._build_complete_partition(selected)

                ok, cert = self.verify_complete_solution(sol)

                if ok:

                    yield sol

                else:

                    self.stats.prunes_clue += 1

                return



            _t0 = time.monotonic()
            v, viable_mask = self._choose_uncovered_cell(
                covered_mask_local,
                active_mask_local,
            )
            self._time_choose += time.monotonic() - _t0

            if v is None:

                return



            if viable_mask == 0:
                self.stats.prunes_dead_cell += 1
                return

            viable = []
            _bits = viable_mask
            while _bits:
                _lsb = _bits & -_bits
                cid = _lsb.bit_length() - 1
                viable.append(cid)
                _bits ^= _lsb



            viable.sort(

                key=lambda cid: self.candidate_score[cid],

                reverse=True,

            )



            for cid in viable:

                cand = self.candidates[cid]

                cand_mask = 0
                for x in cand.cells:
                    cand_mask |= NODE_BIT[x]

                # Should already be guaranteed by active-mask filtering.
                if cand_mask & covered_mask_local:
                    self.stats.prunes_overlap += 1
                    continue

                newly_covered = list(cand.cells)

                # Child candidate activity is one big-int operation:
                # remove every candidate conflicting with selected cid.
                child_active_mask = active_mask_local & ~self.conflict_mask[cid]
                # Keep selected candidate bit irrelevant/cleared; it cannot be
                # chosen again because its cells become covered.

                child_covered_mask = covered_mask_local | cand_mask

                selected.append(cid)

                for x in newly_covered:
                    covered.add(x)
                    cell_state[x] = cid



                _t0 = time.monotonic()
                good = self._partial_metric_consistent(cell_state)
                self._time_metric += time.monotonic() - _t0

                if not good:

                    self.stats.prunes_metric += 1

                else:
                    _t0 = time.monotonic()
                    cheap_bad, child_fixed_dist = self._extend_fixed_distances(
                        fixed_dist,
                        cell_state,
                        newly_covered,
                        cand.capitol,
                    )
                    self._time_clue += time.monotonic() - _t0
                    if cheap_bad:
                        self.stats.prunes_clue += 1
                        good = False



                if good:

                    yield from dfs(
                        child_fixed_dist,
                        child_covered_mask,
                        child_active_mask,
                    )

                    if stop_after_first:

                        # If recursion yielded a solution, the caller will stop

                        # iteration; no special mutable flag is needed here.

                        pass



                # undo mutable state; bitset masks are immutable integers
                # carried by recursion, so they need no restoration.
                for x in newly_covered:
                    covered.remove(x)
                    cell_state[x] = None

                selected.pop()



        yield from dfs(
            root_fixed_dist,
            0,
            self.all_candidate_mask,
        )
        _print_progress()





# ============================================================================

# PHASE 11/12: CHEAP-PATH CERTIFICATE SUPPORT

# ============================================================================



def cheap_path_repair_scores(

    certificate: Tuple[int, int, List[int], float],

    candidates: Sequence[StateCandidate],

) -> List[Tuple[float, int]]:

    """

    Rank state candidates by how much they touch an explicit forbidden cheap

    path.  This is a branch-ordering heuristic, not a correctness condition.

    """

    _, target, path_edges, got = certificate

    if not path_edges:

        return []



    deficit = target - got

    path_edge_set = set(path_edges)

    scored: List[Tuple[float, int]] = []



    for cand in candidates:

        touched = path_edge_set & set(cand.incident_edges)

        if not touched:

            continue

        # Larger exact state size can potentially raise more touched edges.

        score = len(touched) * cand.size + max(0.0, deficit)

        scored.append((score, cand.id))



    scored.sort(reverse=True)

    return scored





# ============================================================================

# CHECKPOINT I/O

# ============================================================================



def save_pickle(obj, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:

        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)





def load_pickle(path: Path):

    with path.open("rb") as f:

        return pickle.load(f)





# ============================================================================

# REPORTING

# ============================================================================



def print_edge_bound_grid(lo: Sequence[float], hi: Sequence[float]) -> None:

    """

    Compact textual report of the strongest forced lower bounds.

    """

    ranked = sorted(

        range(E),

        key=lambda e: (lo[e], -(hi[e] - lo[e])),

        reverse=True,

    )

    print("\nStrongest forced edge bounds:")

    for eid in ranked[:40]:

        u, v = EDGES[eid]

        print(

            f"  e{eid:03d} {ID_NODE[u]}--{ID_NODE[v]} : "

            f"[{lo[eid]:.0f}, {hi[eid]:.0f}]"

        )





def print_solution(sol: ExactSolution, candidates: Sequence[StateCandidate]) -> None:

    print("\n=== EXACT SOLUTION ===")

    print("states:", len(sol.candidate_ids))

    print("capitols:", [ID_NODE[c] for c in sol.capitols])

    print()



    labels = [["." for _ in range(C)] for _ in range(R)]

    for k, cid in enumerate(sol.candidate_ids):

        ch = str(k % 10)

        for v in candidates[cid].cells:

            r, c = ID_NODE[v]

            labels[r][c] = ch



    for row in labels:

        print(" ".join(row))



    print("\nSelected states:")

    for k, cid in enumerate(sol.candidate_ids):

        c = candidates[cid]

        print(

            f"  {k:2d}: size={c.size:3d} "

            f"sym={c.symmetry_name:8s} "

            f"capitol={None if c.capitol is None else ID_NODE[c.capitol]} "

            f"cells={[ID_NODE[v] for v in sorted(c.cells)]}"

        )





# ============================================================================

# CLI

# ============================================================================



def command_metric(args) -> None:

    model = RelaxedMetricModel(

        integer_weights=not args.continuous,

        edge_min=1,

        edge_max=121,

    )



    # Seed with a sum-minimizing separated solution to accumulate useful cuts.

    base = model.solve_with_separation(

        objective=np.ones(E),

        max_rounds=args.max_rounds,

        time_limit_per_round=args.time_limit,

    )

    if not base.feasible:

        raise RuntimeError("Metric relaxation did not converge.")



    print(f"Initial separated metric feasible with {len(model.cuts)} path cuts.")



    if args.only_strongest is None:

        edge_ids = list(range(E))

    else:

        # Cheap pre-screen: edges on high-gap Manhattan corridors first.

        score = np.zeros(E, dtype=float)

        for pc in PAIR_CONSTRAINTS:

            # Use one Manhattan path as a ranking heuristic only.

            a = ID_NODE[pc.i]

            b = ID_NODE[pc.j]

            r, c = a

            rr, cc = b

            path_cells = [pc.i]

            cur = (r, c)

            while cur[0] != rr:

                cur = (cur[0] + (1 if rr > cur[0] else -1), cur[1])

                path_cells.append(NODE_ID[cur])

            while cur[1] != cc:

                cur = (cur[0], cur[1] + (1 if cc > cur[1] else -1))

                path_cells.append(NODE_ID[cur])

            for x, y in zip(path_cells, path_cells[1:]):

                score[EDGE_ID[tuple(sorted((x, y)))]] += pc.lower

        edge_ids = list(np.argsort(-score)[:args.only_strongest])



    lo = np.full(E, 1.0)

    hi = np.full(E, 121.0)



    part_lo, part_hi = model.forced_edge_ranges(

        edge_ids=edge_ids,

        max_rounds=args.max_rounds,

        time_limit_per_round=args.time_limit,

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

    save_pickle(out, Path(args.output))

    print_edge_bound_grid(lo, hi)

    print(f"\nSaved metric bounds to {args.output}")





def command_candidates(args) -> None:

    data = load_pickle(Path(args.metric))

    lo = np.asarray(data["edge_lb"], dtype=float)



    candidates, templates = generate_state_candidates(

        edge_lower_bounds=lo,

        min_state_size=args.min_size,

        max_state_size=args.max_size,

        max_unions_per_symmetry=args.max_unions_per_symmetry,

        global_candidate_cap=args.global_candidate_cap,

    workers=args.workers,
        worker_chunks_per_cpu=args.worker_chunks_per_cpu,
        progress=not args.quiet,
    )



    out = {

        "candidates": candidates,

        "templates": templates,

        "metric_edge_lb": lo,

    }

    save_pickle(out, Path(args.output))



    print(f"Generated {len(candidates)} complete state candidates.")

    print(f"Intrinsic shape templates: {len(templates)}")

    print("Distinct placements were preserved.")

    print(f"Saved candidates to {args.output}")





def command_search(args) -> None:

    data = load_pickle(Path(args.candidates))

    candidates: List[StateCandidate] = data["candidates"]

    lo = np.asarray(data["metric_edge_lb"], dtype=float)



    solver = ExactCoverMetricSearch(

        candidates=candidates,

        metric_edge_lb=lo,

        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,

    )



    found = 0

    for sol in solver.search(

        max_nodes=args.max_nodes,

        stop_after_first=not args.all_solutions,

        progress_every=args.progress_every,

        heartbeat_seconds=args.heartbeat_seconds,

    ):

        found += 1

        print_solution(sol, candidates)

        if args.solution_output:

            save_pickle(sol, Path(args.solution_output))

            print(f"\nSaved exact solution to {args.solution_output}")

        if not args.all_solutions:

            break



    print("\nSearch stats:")

    print(solver.stats)

    print("solutions found:", found)





def build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(

        description="Metric-first symmetry-constrained Jane Street solver"

    )

    sub = p.add_subparsers(dest="command", required=True)



    pm = sub.add_parser(

        "metric",

        help="derive relaxed forced edge-weight ranges",

    )

    pm.add_argument("--output", default="metric_bounds.pkl")

    pm.add_argument("--continuous", action="store_true")

    pm.add_argument("--max-rounds", type=int, default=100)

    pm.add_argument("--time-limit", type=float, default=None)

    pm.add_argument(

        "--only-strongest",

        type=int,

        default=None,

        help="certify only this many highest-priority edges first",

    )

    pm.set_defaults(func=command_metric)



    pc = sub.add_parser(

        "candidates",

        help="generate complete symmetric state candidates",

    )

    pc.add_argument("--metric", default="metric_bounds.pkl")

    pc.add_argument("--output", default="state_candidates.pkl")

    pc.add_argument("--min-size", type=int, default=1)

    pc.add_argument("--max-size", type=int, default=40)

    pc.add_argument("--max-unions-per-symmetry", type=int, default=5000)

    pc.add_argument("--global-candidate-cap", type=int, default=100000)

    pc.add_argument("--workers", type=int, default=0,
                    help="candidate worker processes; 0 = all CPU cores")
    pc.add_argument("--worker-chunks-per-cpu", type=int, default=2,
                    help="coarse action chunks per worker")
    pc.add_argument("--quiet", action="store_true",
                    help="suppress candidate-generation progress")

    pc.set_defaults(func=command_candidates)



    ps = sub.add_parser(

        "search",

        help="exact-cover search over complete symmetric states",

    )

    ps.add_argument("--candidates", default="state_candidates.pkl")

    ps.add_argument("--max-nodes", type=int, default=None)

    ps.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="print live search progress every N visited nodes (0 disables)",
    )

    ps.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=10.0,
        help="print a live heartbeat at least this often while DFS is advancing",
    )

    ps.add_argument("--all-solutions", action="store_true")

    ps.add_argument("--checkpoint", default=None)

    ps.add_argument("--solution-output", default="exact_solution.pkl")

    ps.set_defaults(func=command_search)



    return p





def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    args.func(args)





if __name__ == "__main__":

    main()
