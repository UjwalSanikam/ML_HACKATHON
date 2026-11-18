"""Derive per-observation reference labels from the reference route.

The .route file is an ordered list of arcs but carries no timestamps, so we
must decide which route arc each GPS fix belongs to. We solve a monotone
alignment by dynamic programming:

    observations i = 0..N-1 are assigned route positions j_i with j_i <= j_{i+1}
    cost = sum_i dist(obs_i, route_arc[j_i]) + lambda * (number of skipped route arcs)

Monotonicity respects the travel order (so loops / revisited arcs are handled
correctly) and the skip penalty stops the alignment from jumping ahead.
Complexity O(N*L) with a running prefix minimum.
"""
from __future__ import annotations

import numpy as np

from . import config
from .geo import point_segment


def align_route(obs_xy: np.ndarray, A: np.ndarray, B: np.ndarray,
                lam: float = config.ALIGN_SKIP_PENALTY_M,
                clip: float = config.ALIGN_COST_CLIP_M):
    N, L = len(obs_xy), len(A)
    jidx = np.arange(L, dtype=np.float64)
    back = np.zeros((N, L), np.int32)
    cost0, _, _ = point_segment(obs_xy[0], A, B)
    dp = np.minimum(cost0, clip) + lam * jidx
    ar = np.arange(L)
    for i in range(1, N):
        c, _, _ = point_segment(obs_xy[i], A, B)
        vals = dp - lam * jidx
        run = np.minimum.accumulate(vals)
        arg = np.maximum.accumulate(np.where(vals <= run, ar, 0))
        dp = np.minimum(c, clip) + run + lam * jidx
        back[i] = arg
    j = int(np.argmin(dp + lam * (L - 1 - jidx)))
    path = np.empty(N, np.int64)
    for i in range(N - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    dist, _, _ = point_segment(obs_xy, A[path], B[path])
    return path, dist


def observation_labels(net, obs_xy: np.ndarray, route: np.ndarray):
    """Return (label_arc per obs, route position per obs, distance to label arc)."""
    route = np.asarray(route, np.int64)
    pos, dist = align_route(obs_xy, net.a[route], net.b[route])
    return route[pos], pos, dist
