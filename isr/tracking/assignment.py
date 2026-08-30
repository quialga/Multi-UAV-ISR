"""
isr/tracking/assignment.py — rectangular linear-sum assignment (Hungarian).

Self-contained so the tracker does not pull in scipy for one function; the
matrices here are tiny (n_tracks x n_detections_from_one_blue, both <= ~10),
so the classic O(n^3) Kuhn-Munkres with potentials is far more than fast
enough.  Validated against brute-force enumeration in the tests.

Forbidden pairings (outside the gate) must be passed as a LARGE FINITE cost,
never ``inf`` — the shortest-augmenting-path formulation needs finite
arithmetic.  Filter the returned pairs by the real gate afterwards; see
``solve_gated``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-cost one-to-one matching of rows to columns.

    Returns ``(row_idx, col_idx)`` with ``min(n_rows, n_cols)`` pairs, sorted
    by row.  Costs must be finite.
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("cost must be 2-D")
    if c.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if not np.all(np.isfinite(c)):
        raise ValueError("cost must be finite (use a large value, not inf)")

    transposed = False
    if c.shape[0] > c.shape[1]:
        c = c.T
        transposed = True
    n, m = c.shape

    INF = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)       # p[j] = row matched to column j
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] > 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def solve_gated(cost: np.ndarray, gate: np.ndarray,
                big: float = 1e6) -> list:
    """Assignment restricted to gated-in pairs.

    ``gate[i, j]`` True means the pairing is admissible.  Inadmissible pairs
    get a large finite cost so the solver stays well-defined, and any pair it
    returns that was actually gated out is dropped — the solver is forced to
    pick something when rows/cols outnumber the admissible pairs.

    Returns a list of ``(row, col)`` pairs.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    padded = np.where(gate, cost, big)
    r, c = linear_sum_assignment(padded)
    return [(int(i), int(j)) for i, j in zip(r, c) if gate[i, j]]
