#!/usr/bin/env python3
"""
metric_tiling_solver_fast.py

Faster redesign of the metric-first tiling solver.

Main speedups vs v1
-------------------
1. Parameterize symmetries by (capitol cell, D4 symmetry type).
   No generic affine-action scan and no "rediscover all symmetries" call
   for every partial candidate.

2. Use 121-bit Python integers for cell sets.

3. Under a chosen symmetry, require the capitol to be the ONLY fixed
   cell of the candidate.  For reflections this means we allow the
   capitol singleton orbit but forbid all other singleton orbits.

4. Precompute short clue-path masks.  Candidate metric rejection is then
   just fast bit operations:
       if a concrete path lies wholly inside a size-s state,
       then len(path)*s >= clue_gap.

5. Generate candidates lazily by (capitol, symmetry) and cache them.
   Exact-cover search asks only for candidates covering the current MRV
   cell; candidates are not globally materialized up front unless needed.

Modeling assumption
-------------------
This follows the course-note interpretation that each state has a capitol
which is the unique common fixed square of its nontrivial symmetry.
Here we use one chosen nontrivial symmetry centered/axed through that
capitol and require that symmetry to have exactly one fixed square *inside
the candidate state*.  This is a necessary/sufficient condition for that
chosen symmetry to identify the capitol under this simplified model.

Run:
    python metric_tiling_solver_fast.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from itertools import permutations
import math
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

# Short concrete paths used for candidate-state metric pruning.
MAX_PRECOMPUTED_PATH_LEN = 6

# Set to e.g. 100 while debugging; None means count all verified tilings.
MAX_VERIFIED_TILINGS = None

# Progress reporting.
PROGRESS_EVERY_SEARCH_NODES = 10_000


# ============================================================
# GRID
# ============================================================

def vid(r, c):
    return r * C + c

def rc(v):
    return divmod(v, C)

def bit(v):
    return 1 << v

ALL_MASK = (1 << N) - 1

NEI = [[] for _ in range(N)]
EDGES = []

for r in range(R):
    for c in range(C):
        u = vid(r, c)
        if r + 1 < R:
            v = vid(r + 1, c)
            NEI[u].append(v)
            NEI[v].append(u)
            EDGES.append((u, v))
        if c + 1 < C:
            v = vid(r, c + 1)
            NEI[u].append(v)
            NEI[v].append(u)
            EDGES.append((u, v))

CLUES = {}
for r in range(R):
    for c in range(C):
        if GRID[r][c] != 'X':
            CLUES[vid(r, c)] = int(GRID[r][c])

ZERO_MASK = 0
NONZERO_CLUE_MASK = 0
for v, val in CLUES.items():
    if val == 0:
        ZERO_MASK |= bit(v)
    else:
        NONZERO_CLUE_MASK |= bit(v)


# ============================================================
# D4 ACTIONS CENTERED AT A CAPITOL
# ============================================================

SYM_TYPES = ("R90", "R180", "R270", "REF_H", "REF_V", "REF_D", "REF_A")

def transform_about_capitol(v, cap, sym):
    """
    Apply a D4 symmetry about/through the capitol cell.

    Coordinates are ordinary integer cell coordinates.  Because the
    center/axis passes through a cell center, integer cells map to integer
    cells.
    """
    r, c = rc(v)
    cr, cc = rc(cap)
    x = r - cr
    y = c - cc

    if sym == "R90":
        a, b = -y, x
    elif sym == "R180":
        a, b = -x, -y
    elif sym == "R270":
        a, b = y, -x
    elif sym == "REF_H":      # reflect across row through capitol
        a, b = -x, y
    elif sym == "REF_V":      # reflect across column through capitol
        a, b = x, -y
    elif sym == "REF_D":      # diagonal y=x
        a, b = y, x
    elif sym == "REF_A":      # anti-diagonal y=-x
        a, b = -y, -x
    else:
        raise ValueError(sym)

    rr = cr + a
    cc2 = cc + b

    if 0 <= rr < R and 0 <= cc2 < C:
        return vid(rr, cc2)
    return None


def symmetry_order(sym):
    return 4 if sym in ("R90", "R270") else 2


@dataclass(frozen=True)
class OrbitSystem:
    cap: int
    sym: str
    orbit_masks: tuple[int, ...]
    orbit_sizes: tuple[int, ...]
    orbit_adj_masks: tuple[int, ...]  # bitset over orbit indices
    root: int


@lru_cache(maxsize=None)
def build_orbit_system(cap, sym):
    """
    Build all complete board-contained orbits for a symmetry centered at cap.

    Crucial pruning:
      - root orbit is {cap}
      - all OTHER singleton orbits are removed, so cap is the only fixed
        square of this chosen symmetry inside any generated candidate.
    """
    ordg = symmetry_order(sym)
    seen_cells = set()
    orbits = []

    for v in range(N):
        if v in seen_cells:
            continue

        cur = v
        orb = []
        ok = True

        for _ in range(ordg):
            if cur is None:
                ok = False
                break
            orb.append(cur)
            cur = transform_about_capitol(cur, cap, sym)

        if not ok or cur != v:
            continue

        orb = tuple(sorted(set(orb)))

        # Closure verification
        if any(transform_about_capitol(x, cap, sym) not in orb for x in orb):
            continue

        # Mark only complete orbit cells.
        for x in orb:
            seen_cells.add(x)

        orbits.append(orb)

    # Find capitol singleton.
    root_candidates = [i for i, o in enumerate(orbits) if o == (cap,)]
    if not root_candidates:
        return None

    root_old = root_candidates[0]

    # Forbid every other singleton fixed cell.
    keep_old = [
        i for i, o in enumerate(orbits)
        if len(o) > 1 or i == root_old
    ]

    old_to_new = {old: new for new, old in enumerate(keep_old)}
    kept = [orbits[i] for i in keep_old]

    root = old_to_new[root_old]

    orbit_masks = []
    orbit_sizes = []
    cell_to_orbit = {}

    for oi, orb in enumerate(kept):
        m = 0
        for v in orb:
            m |= bit(v)
            cell_to_orbit[v] = oi
        orbit_masks.append(m)
        orbit_sizes.append(len(orb))

    O = len(kept)
    adj_sets = [set() for _ in range(O)]

    for u, v in EDGES:
        ou = cell_to_orbit.get(u)
        ov = cell_to_orbit.get(v)
        if ou is None or ov is None or ou == ov:
            continue
        adj_sets[ou].add(ov)
        adj_sets[ov].add(ou)

    orbit_adj_masks = []
    for s in adj_sets:
        m = 0
        for j in s:
            m |= 1 << j
        orbit_adj_masks.append(m)

    return OrbitSystem(
        cap=cap,
        sym=sym,
        orbit_masks=tuple(orbit_masks),
        orbit_sizes=tuple(orbit_sizes),
        orbit_adj_masks=tuple(orbit_adj_masks),
        root=root,
    )


# ============================================================
# PRECOMPUTE SHORT CONCRETE CLUE PATHS
# ============================================================

def canonical_shortest_paths(start, goal):
    r1, c1 = rc(start)
    r2, c2 = rc(goal)

    dr = r2 - r1
    dc = c2 - c1
    nv = abs(dr)
    nh = abs(dc)
    L = nv + nh

    if L == 0:
        return []

    vstep = (1 if dr > 0 else -1, 0)
    hstep = (0, 1 if dc > 0 else -1)

    seq = ["V"] * nv + ["H"] * nh
    out = []
    seen = set()

    for perm in permutations(seq):
        if perm in seen:
            continue
        seen.add(perm)

        r, c = r1, c1
        cell_mask = bit(start)

        for ch in perm:
            if ch == "V":
                r += vstep[0]
            else:
                c += hstep[1]
            cell_mask |= bit(vid(r, c))

        out.append((cell_mask, L))

    return out


PATH_CONSTRAINTS = []
clue_items = list(CLUES.items())

for i in range(len(clue_items)):
    x, fx = clue_items[i]
    for j in range(i + 1, len(clue_items)):
        y, fy = clue_items[j]
        gap = abs(fx - fy)

        r1, c1 = rc(x)
        r2, c2 = rc(y)
        L = abs(r1-r2) + abs(c1-c2)

        if gap == 0 or L == 0 or L > MAX_PRECOMPUTED_PATH_LEN:
            continue

        for path_mask, plen in canonical_shortest_paths(x, y):
            PATH_CONSTRAINTS.append((path_mask, plen, gap))

# Strongest first helps reject quickly.
PATH_CONSTRAINTS.sort(key=lambda t: t[2] / t[1], reverse=True)


# ============================================================
# FAST CANDIDATE CHECKS
# ============================================================

def candidate_metric_ok(mask, size):
    """
    If an entire concrete clue-to-clue path lies inside this state,
    all its edges have exact weight = state size.
    """
    for path_mask, plen, gap in PATH_CONSTRAINTS:
        if path_mask & mask == path_mask:
            if plen * size < gap:
                return False
    return True


def candidate_capitol_clue_ok(mask, cap, size):
    """
    Fast necessary condition using the candidate's own capitol.

    If clue x is in this state, any concrete internal x->cap path of L
    edges has cost L*size. Since f(x) is distance to the nearest capitol,
        f(x) <= L*size.
    We compute shortest unweighted distance inside the candidate only for
    clue cells actually inside it.
    """
    # Published nonzero clue cannot be the capitol.
    if cap in CLUES and CLUES[cap] != 0:
        return False

    # A candidate containing a known zero must have that zero as capitol.
    z = mask & ZERO_MASK
    if z:
        if z != bit(cap):
            return False

    # BFS once from capitol through candidate.
    dist = {cap: 0}
    q = [cap]
    head = 0

    while head < len(q):
        u = q[head]
        head += 1
        for v in NEI[u]:
            if not (mask & bit(v)):
                continue
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    for x, fx in CLUES.items():
        if mask & bit(x):
            L = dist.get(x)
            if L is None:
                return False
            if L * size < fx:
                return False

    return True


@dataclass(frozen=True)
class Candidate:
    mask: int
    size: int
    cap: int
    sym: str


# ============================================================
# LAZY CANDIDATE GENERATION
# ============================================================

# Cache by (cap,sym); generated only when exact-cover search first asks.
CAND_CACHE = {}

def generate_candidates(cap, sym):
    key = (cap, sym)
    if key in CAND_CACHE:
        return CAND_CACHE[key]

    sys = build_orbit_system(cap, sym)
    if sys is None:
        CAND_CACHE[key] = ()
        return ()

    # If cap itself is a nonzero clue, impossible.
    if cap in CLUES and CLUES[cap] != 0:
        CAND_CACHE[key] = ()
        return ()

    root = sys.root
    O = len(sys.orbit_masks)

    root_choice = 1 << root
    root_mask = sys.orbit_masks[root]

    # If cap is not an observed zero, candidate may not contain observed zero.
    forbidden_zero_mask = 0 if cap in ZERO_MASK_CELLS else ZERO_MASK

    out = []
    seen = set()

    def rec(chosen_orbits, frontier, cell_mask, size):
        # Chosen orbit bitset is canonical state.
        if chosen_orbits in seen:
            return
        seen.add(chosen_orbits)

        # Known zero pruning.
        if forbidden_zero_mask and (cell_mask & forbidden_zero_mask):
            return

        # Current union is connected by construction and is a legal complete
        # candidate at this size.
        if candidate_metric_ok(cell_mask, size):
            if candidate_capitol_clue_ok(cell_mask, cap, size):
                out.append(Candidate(cell_mask, size, cap, sym))

        # Extend connectedly.
        f = frontier
        while f:
            lsb = f & -f
            oi = lsb.bit_length() - 1
            f ^= lsb

            new_chosen = chosen_orbits | lsb
            om = sys.orbit_masks[oi]
            new_mask = cell_mask | om
            new_size = size + sys.orbit_sizes[oi]

            new_frontier = (
                frontier
                | sys.orbit_adj_masks[oi]
            ) & ~new_chosen

            rec(new_chosen, new_frontier, new_mask, new_size)

    initial_frontier = sys.orbit_adj_masks[root]
    rec(root_choice, initial_frontier, root_mask, 1)

    # Dedup same cell shape under this cap/sym path.
    best = {}
    for cand in out:
        best[cand.mask] = cand

    ans = tuple(best.values())
    CAND_CACHE[key] = ans
    return ans


ZERO_MASK_CELLS = frozenset(v for v, val in CLUES.items() if val == 0)


# Candidate index by cell, populated lazily.
CELL_CANDIDATES = [None] * N

def candidates_covering_cell(v):
    if CELL_CANDIDATES[v] is not None:
        return CELL_CANDIDATES[v]

    dedup = {}

    # Any cell that belongs to a candidate may have any possible capitol.
    # Generate per (cap,sym) lazily here.
    for cap in range(N):
        if cap in CLUES and CLUES[cap] != 0:
            continue

        for sym in SYM_TYPES:
            for cand in generate_candidates(cap, sym):
                if cand.mask & bit(v):
                    # Same shape/cap can arise through >1 symmetry.
                    dedup[(cand.mask, cand.cap)] = cand

    ans = tuple(dedup.values())
    CELL_CANDIDATES[v] = ans

    print(
        f"[lazy] cell {rc(v)} -> {len(ans):,} candidates "
        f"(cached symmetry systems: {len(CAND_CACHE):,})"
    )

    return ans


# ============================================================
# METRIC CHECKS DURING EXACT COVER
# ============================================================

def fixed_graph_dijkstra(state_of, state_sizes, source):
    INF = 10**18
    dist = [INF] * N

    if state_of[source] == -1:
        return dist

    dist[source] = 0
    pq = [(0, source)]

    while pq:
        du, u = heappop(pq)
        if du != dist[u]:
            continue

        su = state_of[u]

        for v in NEI[u]:
            sv = state_of[v]
            if sv == -1:
                continue

            w = min(state_sizes[su], state_sizes[sv])
            nd = du + w

            if nd < dist[v]:
                dist[v] = nd
                heappush(pq, (nd, v))

    return dist


def partial_metric_ok(state_of, state_sizes, selected_caps):
    assigned_clues = [v for v in CLUES if state_of[v] != -1]

    for x in assigned_clues:
        fx = CLUES[x]
        dist = fixed_graph_dijkstra(state_of, state_sizes, x)

        # Triangle lower-bound check.
        for y in assigned_clues:
            if y <= x:
                continue
            if dist[y] < abs(fx - CLUES[y]):
                return False

        # Cheap path to known capitol check.
        for cap in selected_caps:
            if dist[cap] < fx:
                return False

    return True


def full_verify(selected):
    state_of = [-1] * N
    sizes = {}
    caps = []

    for sid, cand in enumerate(selected):
        sizes[sid] = cand.size
        caps.append(cand.cap)
        m = cand.mask
        while m:
            b = m & -m
            v = b.bit_length() - 1
            m ^= b
            if state_of[v] != -1:
                return False
            state_of[v] = sid

    if any(s == -1 for s in state_of):
        return False

    INF = 10**18
    dist = [INF] * N
    pq = []

    for cap in caps:
        dist[cap] = 0
        heappush(pq, (0, cap))

    while pq:
        du, u = heappop(pq)
        if du != dist[u]:
            continue

        su = state_of[u]

        for v in NEI[u]:
            sv = state_of[v]
            w = min(sizes[su], sizes[sv])
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                heappush(pq, (nd, v))

    return all(dist[v] == val for v, val in CLUES.items())


# ============================================================
# EXACT COVER
# ============================================================

def solve():
    state_of = [-1] * N
    state_sizes = {}
    selected = []
    selected_caps = []

    total = 0
    nodes = 0
    metric_prunes = 0
    by_k = defaultdict(int)

    t0 = time.time()

    def rec_search(covered):
        nonlocal total, nodes, metric_prunes
        nodes += 1

        if (
            PROGRESS_EVERY_SEARCH_NODES
            and nodes % PROGRESS_EVERY_SEARCH_NODES == 0
        ):
            print(
                f"[search] nodes={nodes:,} exact={total:,} "
                f"metric_prunes={metric_prunes:,} "
                f"elapsed={time.time()-t0:.1f}s"
            )

        if MAX_VERIFIED_TILINGS is not None and total >= MAX_VERIFIED_TILINGS:
            return

        if covered == ALL_MASK:
            if full_verify(selected):
                total += 1
                by_k[len(selected)] += 1
                print(f"*** exact tiling #{total} ({len(selected)} states)")
            return

        # MRV, but candidate lists themselves are lazy.
        # To avoid forcing generation for all 121 cells immediately,
        # first prefer observed clue cells, especially 0/high clues.
        uncovered = [v for v in range(N) if not (covered & bit(v))]

        priority = sorted(
            uncovered,
            key=lambda v: (
                0 if v in CLUES else 1,
                -CLUES.get(v, -1)
            )
        )

        # Examine a small pool first. If all fail, expand.
        best_v = None
        best_opts = None

        for v in priority:
            opts = [
                cand for cand in candidates_covering_cell(v)
                if not (cand.mask & covered)
            ]

            if not opts:
                return

            if best_opts is None or len(opts) < len(best_opts):
                best_v = v
                best_opts = opts

            # Strong enough MRV stop.
            if len(best_opts) <= 2:
                break

            # Avoid eagerly materializing candidates for every cell.
            if best_opts is not None and priority.index(v) >= 7:
                break

        best_opts.sort(key=lambda c: -c.size)

        for cand in best_opts:
            sid = len(selected)

            # assign
            m = cand.mask
            touched = []
            while m:
                b = m & -m
                v = b.bit_length() - 1
                m ^= b
                state_of[v] = sid
                touched.append(v)

            state_sizes[sid] = cand.size
            selected.append(cand)
            selected_caps.append(cand.cap)

            if partial_metric_ok(state_of, state_sizes, selected_caps):
                rec_search(covered | cand.mask)
            else:
                metric_prunes += 1

            selected_caps.pop()
            selected.pop()
            del state_sizes[sid]

            for v in touched:
                state_of[v] = -1

    rec_search(0)

    print("\n======================================")
    print("RESULT")
    print("======================================")
    print("Exact verified tilings:", f"{total:,}")
    print("Search nodes:", f"{nodes:,}")
    print("Metric-pruned branches:", f"{metric_prunes:,}")
    print("Cached (capitol,symmetry) generators:", f"{len(CAND_CACHE):,}")

    print("\nBy number of states:")
    for k in sorted(by_k):
        print(f"  {k:3d}: {by_k[k]:,}")


if __name__ == "__main__":
    print(f"Board: {R}x{C} ({N} cells)")
    print(f"Edges: {len(EDGES)}")
    print(f"Observed clues: {len(CLUES)}")
    print(f"Known zero clues: {[rc(v) for v in ZERO_MASK_CELLS]}")
    print(f"Precomputed concrete path constraints: {len(PATH_CONSTRAINTS)}")
    print("\nStarting lazy symmetry + exact-cover search...\n")

    solve()
