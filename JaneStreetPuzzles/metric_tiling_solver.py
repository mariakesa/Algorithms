#!/usr/bin/env python3
"""
metric_tiling_solver.py

Integrated solver skeleton for Jane Street October 2026 "It's a Metric, Too".

It combines:
  1) symmetry-generated connected state candidates,
  2) exact-cover / packing,
  3) clue-pair triangle/path pruning,
  4) cheap-path-to-capitol pruning,
  5) exact multi-source Dijkstra verification.

Important modeling assumption used here:
  We count only states that have a capitol, i.e. their complete nontrivial
  symmetry group fixes exactly one square in common. This matches the
  "candidate capitol" treatment in the study notes.

The script is exact under that assumption: it does NOT use arbitrary
state-size cutoffs.

Run:
    python metric_tiling_solver.py

Useful knobs near the top:
    PRINT_PROGRESS
    MAX_TILINGS_TO_COUNT   # None means count all
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from itertools import product
import math
import sys
import time


# ============================================================
# INPUT
# ============================================================

GRID = [
    ['X',8,4,'X','X',11,'X','X','X',5,'X'],
    ['X','X','X','X','X','X','X',11,'X','X','X'],
    ['X','X',7,'X','X',1,'X','X','X','X',14],
    [28,'X','X',51,'X','X',1,'X',6,'X','X'],
    ['X',22,'X','X','X','X','X','X',4,'X',1],
    ['X','X','X',15,'X',10,'X',0,'X','X','X'],
    [11,'X',11,'X','X','X','X','X','X',9,'X'],
    ['X','X',17,'X',14,'X','X',10,'X','X',13],
    [30,'X','X','X','X',6,'X','X',45,'X','X'],
    ['X','X','X',10,'X','X','X','X','X','X','X'],
    ['X',26,'X','X','X',0,'X','X',77,61,'X']
]

R = len(GRID)
C = len(GRID[0])
N = R * C

PRINT_PROGRESS = True

# None => count every exact tiling found.
# Set to a small integer (e.g. 100) while debugging.
MAX_TILINGS_TO_COUNT = None


# ============================================================
# BASIC GRID UTILITIES
# ============================================================

def vid(r, c):
    return r * C + c

def rc(v):
    return divmod(v, C)

CELLS = tuple(range(N))

NEI = [[] for _ in CELLS]
EDGES = []

for r in range(R):
    for c in range(C):
        u = vid(r, c)
        if r + 1 < R:
            v = vid(r + 1, c)
            NEI[u].append(v)
            NEI[v].append(u)
            EDGES.append((min(u, v), max(u, v)))
        if c + 1 < C:
            v = vid(r, c + 1)
            NEI[u].append(v)
            NEI[v].append(u)
            EDGES.append((min(u, v), max(u, v)))

EDGE_ID = {e: i for i, e in enumerate(EDGES)}

CLUES = {}
for r in range(R):
    for c in range(C):
        if GRID[r][c] != 'X':
            CLUES[vid(r, c)] = int(GRID[r][c])

ZERO_CELLS = frozenset(v for v, x in CLUES.items() if x == 0)
NONZERO_CLUE_CELLS = frozenset(v for v, x in CLUES.items() if x != 0)

CLUE_PAIRS = []
clue_items = list(CLUES.items())
for i in range(len(clue_items)):
    x, fx = clue_items[i]
    for j in range(i + 1, len(clue_items)):
        y, fy = clue_items[j]
        CLUE_PAIRS.append((x, y, abs(fx - fy)))


# ============================================================
# AFFINE D4 SYMMETRIES
#
# A state can be symmetric around a local center/axis, not only
# around the center of the whole 11x11 board.
#
# Represent coordinates doubled:
#     X = 2*r, Y = 2*c
# so centers/axes may lie on integers or half-integers without
# floating-point arithmetic.
#
# g(p) = A p + t
#
# A is one of the seven non-identity D4 linear maps.
# We enumerate all translations t that yield at least one finite
# orbit wholly inside the board.
# ============================================================

D4_A = {
    "R90":  ((0,-1),(1,0)),
    "R180": ((-1,0),(0,-1)),
    "R270": ((0,1),(-1,0)),
    "REF_X": ((1,0),(0,-1)),
    "REF_Y": ((-1,0),(0,1)),
    "REF_D": ((0,1),(1,0)),
    "REF_A": ((0,-1),(-1,0)),
}

ORDER = {
    "R90": 4, "R180": 2, "R270": 4,
    "REF_X": 2, "REF_Y": 2, "REF_D": 2, "REF_A": 2,
}

def mat_apply(A, p):
    x, y = p
    return (A[0][0]*x + A[0][1]*y,
            A[1][0]*x + A[1][1]*y)

def affine_apply(A, t, p):
    q = mat_apply(A, p)
    return (q[0] + t[0], q[1] + t[1])

def cell_to_doubled(v):
    r, c = rc(v)
    return (2*r, 2*c)

DOUBLED_TO_CELL = {cell_to_doubled(v): v for v in CELLS}

@dataclass(frozen=True)
class Action:
    name: str
    A: tuple
    t: tuple
    order: int

    def apply_cell(self, v):
        p = cell_to_doubled(v)
        q = affine_apply(self.A, self.t, p)
        return DOUBLED_TO_CELL.get(q, None)


def enumerate_actions():
    """
    Enumerate affine D4 actions relevant to the finite board.

    We can derive candidate translations t from pairs p -> q of board-cell
    centers: t = q - A p.  Deduplicate.
    """
    seen = set()
    actions = []

    doubled = [cell_to_doubled(v) for v in CELLS]

    for name, A in D4_A.items():
        ordg = ORDER[name]
        for p in doubled:
            Ap = mat_apply(A, p)
            for q in doubled:
                t = (q[0] - Ap[0], q[1] - Ap[1])
                key = (name, t)
                if key in seen:
                    continue
                seen.add(key)

                act = Action(name, A, t, ordg)

                # Keep only actions with at least one complete orbit inside board.
                good = False
                for v in CELLS:
                    cur = v
                    ok = True
                    for _ in range(ordg):
                        cur = act.apply_cell(cur)
                        if cur is None:
                            ok = False
                            break
                    if ok and cur == v:
                        good = True
                        break
                if good:
                    actions.append(act)

    return actions


# ============================================================
# ORBITS AND CANDIDATE STATES
# ============================================================

def action_orbits(act: Action):
    """
    Return complete board-contained orbits of this action.
    Cells whose orbit leaves the board cannot belong to an invariant state.
    """
    used = set()
    orbits = []

    for v in CELLS:
        if v in used:
            continue

        orbit = []
        cur = v
        ok = True

        for _ in range(act.order):
            if cur is None:
                ok = False
                break
            orbit.append(cur)
            cur = act.apply_cell(cur)

        if not ok or cur != v:
            continue

        orbit = tuple(sorted(set(orbit)))

        # Verify closure exactly.
        if all(act.apply_cell(x) in orbit for x in orbit):
            for x in orbit:
                used.add(x)
            orbits.append(orbit)

    return orbits


def cells_connected(cells):
    cells = set(cells)
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in NEI[u]:
            if v in cells and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(cells)


def induced_shortest_unweighted(cells, src, dst):
    """
    Shortest number of grid edges inside candidate state.
    """
    allowed = set(cells)
    if src not in allowed or dst not in allowed:
        return math.inf
    q = [src]
    dist = {src: 0}
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        if u == dst:
            return dist[u]
        for v in NEI[u]:
            if v in allowed and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return math.inf


def all_self_symmetries(cells):
    """
    Determine every nontrivial affine D4 symmetry that maps this candidate
    state to itself.  We only need to test actions already known to map
    board cells to board cells.
    """
    S = frozenset(cells)
    syms = []
    for act in ACTIONS:
        image = []
        ok = True
        for v in S:
            w = act.apply_cell(v)
            if w is None:
                ok = False
                break
            image.append(w)
        if ok and frozenset(image) == S:
            syms.append(act)
    return syms


def candidate_capitol(cells):
    """
    Puzzle rule used here:
      candidate must have at least one nontrivial self-symmetry, and
      the set of squares fixed by EVERY nontrivial self-symmetry
      must be exactly one cell.
    """
    syms = all_self_symmetries(cells)
    if not syms:
        return None

    common_fixed = set(cells)

    for act in syms:
        fixed = {v for v in cells if act.apply_cell(v) == v}
        common_fixed &= fixed
        if not common_fixed:
            return None

    if len(common_fixed) != 1:
        return None

    return next(iter(common_fixed))


def candidate_internal_metric_ok(cells):
    """
    Necessary triangle/path check for a COMPLETE candidate state S of size s.

    Every edge wholly inside S has weight s.
    Therefore, for any two clue cells x,y in S, a shortest path *inside S*
    of L steps has concrete cost L*s. Since every concrete path must cost
    at least |f(x)-f(y)|, reject if L*s < clue gap.
    """
    S = frozenset(cells)
    s = len(S)
    clue_cells = [v for v in S if v in CLUES]

    for i in range(len(clue_cells)):
        x = clue_cells[i]
        for j in range(i + 1, len(clue_cells)):
            y = clue_cells[j]
            gap = abs(CLUES[x] - CLUES[y])
            L = induced_shortest_unweighted(S, x, y)
            if L < math.inf and L * s < gap:
                return False
    return True


@dataclass(frozen=True)
class Candidate:
    cells: frozenset
    size: int
    capitol: int


def enumerate_candidates_for_action(act: Action):
    """
    Enumerate connected invariant unions of action-orbits.

    We root each search at a fixed-point orbit (size 1), because under the
    capitol assumption a legal state must ultimately have one common fixed
    square. This is a major pruning step.
    """
    orbits = action_orbits(act)
    if not orbits:
        return []

    cell_to_orbit = {}
    for oi, orb in enumerate(orbits):
        for v in orb:
            cell_to_orbit[v] = oi

    # Orbit graph: two orbit nodes touch if any member cells are grid-adjacent.
    OADJ = [set() for _ in orbits]
    for u, v in EDGES:
        if u in cell_to_orbit and v in cell_to_orbit:
            a = cell_to_orbit[u]
            b = cell_to_orbit[v]
            if a != b:
                OADJ[a].add(b)
                OADJ[b].add(a)

    root_orbits = [i for i, orb in enumerate(orbits) if len(orb) == 1]
    out = []

    # Canonical connected-subset enumeration:
    # each connected union is generated once by enforcing a minimum orbit id.
    for root in root_orbits:
        chosen = frozenset([root])
        frontier = frozenset(OADJ[root])

        seen_states = set()

        def rec(chosen, frontier):
            key = chosen
            if key in seen_states:
                return
            seen_states.add(key)

            cells = frozenset(v for oi in chosen for v in orbits[oi])

            # Fast observed-zero constraints:
            zeros_inside = ZERO_CELLS & cells
            if len(zeros_inside) > 1:
                return

            # A nonzero published clue cannot itself be the capitol.
            # We only know the final capitol after checking all symmetries,
            # so defer that exact check.

            # Every current union is connected by construction.
            # Test as a possible complete state.
            cap = candidate_capitol(cells)
            if cap is not None:
                if cap in NONZERO_CLUE_CELLS:
                    pass
                elif zeros_inside and cap not in zeros_inside:
                    pass
                elif candidate_internal_metric_ok(cells):
                    out.append(Candidate(cells, len(cells), cap))

            # Extend.
            # No arbitrary size cap: exact enumeration.
            for oi in sorted(frontier):
                new_chosen = chosen | {oi}
                new_frontier = (frontier | OADJ[oi]) - new_chosen
                rec(frozenset(new_chosen), frozenset(new_frontier))

        rec(chosen, frontier)

    return out


# ============================================================
# EXACT-COVER SEARCH WITH METRIC PROPAGATION
# ============================================================

def dijkstra_on_fixed_edges(assigned_state, state_sizes, sources):
    """
    Dijkstra using only edges whose endpoint states are both already assigned.
    Unknown edges are absent.

    assigned_state[v] = state-id or -1.
    """
    INF = 10**18
    dist = [INF] * N
    pq = []

    for s in sources:
        if assigned_state[s] != -1:
            dist[s] = 0
            heappush(pq, (0, s))

    while pq:
        du, u = heappop(pq)
        if du != dist[u]:
            continue
        su = assigned_state[u]
        if su == -1:
            continue

        for v in NEI[u]:
            sv = assigned_state[v]
            if sv == -1:
                continue

            w = min(state_sizes[su], state_sizes[sv])
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                heappush(pq, (nd, v))

    return dist


def partial_metric_ok(assigned_state, state_sizes, selected_caps):
    """
    Two monotone checks:

    A) clue-pair:
       any currently fixed path with cost < |Δf| is impossible.

    B) clue-to-known-capitol:
       any currently fixed path from clue x to a selected capitol with
       cost < published f(x) is impossible.
    """
    INF = 10**18

    # Run from each assigned clue cell; graph is tiny (121 nodes).
    for x, fx in CLUES.items():
        if assigned_state[x] == -1:
            continue

        dist = dijkstra_on_fixed_edges(
            assigned_state, state_sizes, [x]
        )

        # A) pairwise metric lower bounds
        for y, fy in CLUES.items():
            if y <= x or assigned_state[y] == -1:
                continue
            if dist[y] < abs(fx - fy):
                return False

        # B) published distance cannot exceed an already-existing
        #    fixed path to a known capitol.
        for c in selected_caps:
            if dist[c] < fx:
                return False

    return True


def full_dijkstra(state_of, state_sizes, capitols):
    INF = 10**18
    dist = [INF] * N
    pq = []

    for c in capitols:
        dist[c] = 0
        heappush(pq, (0, c))

    while pq:
        du, u = heappop(pq)
        if du != dist[u]:
            continue

        su = state_of[u]

        for v in NEI[u]:
            sv = state_of[v]
            w = min(state_sizes[su], state_sizes[sv])
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                heappush(pq, (nd, v))

    return dist


def exact_verify(selected):
    state_of = [-1] * N
    state_sizes = {}
    capitols = []

    for sid, cand in enumerate(selected):
        state_sizes[sid] = cand.size
        capitols.append(cand.capitol)
        for v in cand.cells:
            if state_of[v] != -1:
                return False
            state_of[v] = sid

    if any(x == -1 for x in state_of):
        return False

    dist = full_dijkstra(state_of, state_sizes, capitols)

    return all(dist[v] == value for v, value in CLUES.items())


def count_exact_tilings(candidates):
    by_cell = [[] for _ in CELLS]

    for i, cand in enumerate(candidates):
        for v in cand.cells:
            by_cell[v].append(i)

    full_mask = (1 << N) - 1
    cand_masks = []
    for cand in candidates:
        m = 0
        for v in cand.cells:
            m |= 1 << v
        cand_masks.append(m)

    total = 0
    by_k = defaultdict(int)
    nodes = 0
    pruned_metric = 0

    assigned_state = [-1] * N
    state_sizes = {}
    selected_caps = []
    selected = []

    t0 = time.time()

    def recurse(covered_mask):
        nonlocal total, nodes, pruned_metric
        nodes += 1

        if PRINT_PROGRESS and nodes % 10000 == 0:
            print(
                f"[search] nodes={nodes:,} tilings={total:,} "
                f"metric_prunes={pruned_metric:,} "
                f"elapsed={time.time()-t0:.1f}s"
            )

        if MAX_TILINGS_TO_COUNT is not None and total >= MAX_TILINGS_TO_COUNT:
            return

        if covered_mask == full_mask:
            if exact_verify(selected):
                total += 1
                by_k[len(selected)] += 1
                print(f"*** exact tiling #{total} with {len(selected)} states")
            return

        # MRV: choose uncovered cell with fewest non-overlapping candidates.
        best_cell = None
        best_opts = None

        for v in CELLS:
            if covered_mask >> v & 1:
                continue

            opts = [
                ci for ci in by_cell[v]
                if cand_masks[ci] & covered_mask == 0
            ]

            if not opts:
                return

            if best_opts is None or len(opts) < len(best_opts):
                best_cell = v
                best_opts = opts
                if len(best_opts) == 1:
                    break

        # Try tighter / larger candidates first: often fixes more edges sooner.
        best_opts.sort(key=lambda ci: -candidates[ci].size)

        for ci in best_opts:
            cand = candidates[ci]
            mask = cand_masks[ci]

            sid = len(selected)

            # Assign.
            for v in cand.cells:
                assigned_state[v] = sid
            state_sizes[sid] = cand.size
            selected_caps.append(cand.capitol)
            selected.append(cand)

            ok = partial_metric_ok(
                assigned_state, state_sizes, selected_caps
            )

            if ok:
                recurse(covered_mask | mask)
            else:
                pruned_metric += 1

            # Undo.
            selected.pop()
            selected_caps.pop()
            del state_sizes[sid]
            for v in cand.cells:
                assigned_state[v] = -1

    recurse(0)

    return total, dict(sorted(by_k.items())), nodes, pruned_metric


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"Board: {R}x{C} = {N} cells")
    print(f"Grid edges: {len(EDGES)}")
    print(f"Observed clues: {len(CLUES)}")
    print(f"Observed zeros/capitols: {[rc(v) for v in ZERO_CELLS]}")
    print()

    print("Enumerating affine D4 actions...")
    ACTIONS = enumerate_actions()
    print("Actions:", len(ACTIONS))

    # Candidate generation is the combinatorial bottleneck.
    print("\nEnumerating symmetric connected candidate states...")
    uniq = {}

    t0 = time.time()

    for ai, act in enumerate(ACTIONS, 1):
        cands = enumerate_candidates_for_action(act)

        for cand in cands:
            # Deduplicate same state shape generated by different symmetries.
            uniq[cand.cells] = cand

        if PRINT_PROGRESS and (ai % 10 == 0 or ai == len(ACTIONS)):
            print(
                f"[candidates] actions={ai}/{len(ACTIONS)} "
                f"unique={len(uniq):,} "
                f"elapsed={time.time()-t0:.1f}s"
            )

    CANDIDATES = list(uniq.values())

    print("\n==============================================")
    print("CANDIDATE SUMMARY")
    print("==============================================")
    print("Unique legal symmetric connected states:", f"{len(CANDIDATES):,}")

    hist = defaultdict(int)
    for cand in CANDIDATES:
        hist[cand.size] += 1

    print("\nCandidates by size:")
    for s in sorted(hist):
        print(f"  size {s:3d}: {hist[s]:,}")

    print("\nStarting exact-cover + metric search...")
    total, by_k, nodes, metric_prunes = count_exact_tilings(CANDIDATES)

    print("\n==============================================")
    print("RESULT")
    print("==============================================")
    print("Exact verified tilings:", f"{total:,}")
    print("Search nodes:", f"{nodes:,}")
    print("Metric-pruned branches:", f"{metric_prunes:,}")

    print("\nTilings by number of states:")
    for k, n in by_k.items():
        print(f"  {k:3d} states : {n:,}")
