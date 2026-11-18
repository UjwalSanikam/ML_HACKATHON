"""Road network: projected geometry, spatial index and bounded shortest paths."""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .geo import LocalProjection, bearing_deg, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 bbox_xy: Optional[Tuple[float, float, float, float]] = None,
                 densify_step: float = config.DENSIFY_STEP_M):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        self.xy = proj.forward(nodes_lonlat)
        self.a = self.xy[self.arcs[:, 0]]
        self.b = self.xy[self.arcs[:, 1]]
        d = self.b - self.a
        self.length = np.sqrt((d ** 2).sum(1))
        self.bearing = bearing_deg(d)
        self.unit = np.where(self.length[:, None] > 0, d / np.maximum(self.length[:, None], 1e-9), 0.0)

        # --- crop to region of interest (keeps original arc ids) ---
        if bbox_xy is not None:
            x0, y0, x1, y1 = bbox_xy
            def inside(p):
                return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
            self.active = inside(self.a) | inside(self.b)
        else:
            self.active = np.ones(self.n_arcs, bool)
        self.active_ids = np.nonzero(self.active)[0]

        # --- topology ---
        fr, to = self.arcs[:, 0], self.arcs[:, 1]
        self.out_deg = np.bincount(fr, minlength=self.n_nodes)
        self.in_deg = np.bincount(to, minlength=self.n_nodes)
        act = self.active_ids
        order = act[np.argsort(fr[act], kind="stable")]
        self._out_arcs = order
        self._out_ptr = np.zeros(self.n_nodes + 1, np.int64)
        np.add.at(self._out_ptr, fr[order] + 1, 1)
        self._out_ptr = np.cumsum(self._out_ptr)
        # reverse arc lookup (two-way streets)
        key = {(int(u), int(v)): i for i, (u, v) in enumerate(self.arcs)}
        self.reverse = np.array([key.get((int(v), int(u)), -1) for u, v in self.arcs], np.int64)

        # --- densified KD-tree over active segments ---
        L = self.length[act]
        n = np.maximum(1, np.ceil(L / densify_step).astype(int)) + 1
        rep = np.repeat(act, n)
        offs = np.concatenate([np.linspace(0, 1, k) for k in n]) if len(n) else np.empty(0)
        pts = self.a[rep] + offs[:, None] * (self.b[rep] - self.a[rep])
        self._sample_arc = rep
        self._tree = cKDTree(pts) if len(pts) else None
        self._step = densify_step
        self._sp_cache: Dict[int, Tuple[float, dict, dict]] = {}

    # ------------------------------------------------------------------
    @classmethod
    def for_track(cls, nodes_lonlat, arcs, track_lonlat, margin=config.BBOX_MARGIN_M):
        proj = LocalProjection.from_points(track_lonlat)
        txy = proj.forward(track_lonlat)
        bbox = (txy[:, 0].min() - margin, txy[:, 1].min() - margin,
                txy[:, 0].max() + margin, txy[:, 1].max() + margin)
        return cls(nodes_lonlat, arcs, proj, bbox)

    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_ptr[node]:self._out_ptr[node + 1]]

    def connected(self, a1: int, a2: int) -> bool:
        return a1 == a2 or self.arcs[a1, 1] == self.arcs[a2, 0]

    # ------------------------------------------------------------------
    def candidates(self, pts: np.ndarray, radius: float, k: int, radius_max: float):
        """For each point return up to k nearest arcs (directed) within radius.

        The radius is doubled for points with no candidate until radius_max.
        Returns list of arrays (arc ids sorted by distance) and the radius used.
        """
        out: List[np.ndarray] = [None] * len(pts)
        used = np.full(len(pts), radius)
        todo = np.arange(len(pts))
        r = radius
        while len(todo) and r <= radius_max + 1e-6:
            hits = self._tree.query_ball_point(pts[todo], r + self._step / 2)
            nxt = []
            for qi, h in zip(todo, hits):
                if not h:
                    nxt.append(qi)
                    continue
                ids = np.unique(self._sample_arc[h])
                d, _, _ = point_segment(pts[qi], self.a[ids], self.b[ids])
                m = d <= r
                if not m.any():
                    nxt.append(qi)
                    continue
                ids, d = ids[m], d[m]
                o = np.argsort(d, kind="stable")[:k]
                out[qi] = ids[o]
                used[qi] = r
            todo = np.array(nxt, int)
            r *= 2
        for qi in todo:
            out[qi] = np.empty(0, np.int64)
        return out, used

    # ------------------------------------------------------------------
    def dijkstra(self, src: int, bound: float):
        """Bounded single-source shortest path over nodes (arc length weights).
        Cached per source; a cached run with a larger bound is reused."""
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= bound:
            return c[1], c[2]
        dist = {src: 0.0}
        pred = {}
        h = [(0.0, src)]
        done = set()
        while h:
            du, u = heapq.heappop(h)
            if u in done:
                continue
            done.add(u)
            if du > bound:
                break
            for e in self.out_arcs(u):
                v = int(self.arcs[e, 1])
                nd = du + self.length[e]
                if nd <= bound and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = int(e)
                    heapq.heappush(h, (nd, v))
        if len(self._sp_cache) > 200_000:
            self._sp_cache.clear()
        self._sp_cache[src] = (bound, dist, pred)
        return dist, pred

    def path_arcs(self, src: int, dst: int, pred: dict) -> List[int]:
        out = []
        v = dst
        guard = 0
        while v != src and guard < 100_000:
            e = pred.get(v)
            if e is None:
                return []
            out.append(e)
            v = int(self.arcs[e, 0])
            guard += 1
        return out[::-1]

    def shortest_path(self, src_node: int, dst_node: int, bound: float):
        dist, pred = self.dijkstra(src_node, bound)
        if dst_node not in dist:
            return None, []
        return dist[dst_node], self.path_arcs(src_node, dst_node, pred)

    def arc_lonlat(self, arc_ids) -> list:
        out = []
        for e in arc_ids:
            u, v = self.arcs[int(e)]
            out.append([self.nodes_lonlat[u, ::-1].tolist(), self.nodes_lonlat[v, ::-1].tolist()])
        return out
