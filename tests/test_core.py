"""Unit tests on synthetic data: geometry, label alignment, metrics, HMM decoding and the
no-leakage guarantee of the prediction pipeline.   Run:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap.features import build_candidates
from pathsnap.geo import LocalProjection, haversine_m, point_segment
from pathsnap.hmm import viterbi
from pathsnap.labels import align_route
from pathsnap.metrics import lcs_len, route_continuity, route_metrics
from pathsnap.network import RoadNetwork


def grid_network(n=4, spacing=100.0, lon0=10.0, lat0=50.0):
    """n x n grid of two-way streets around (lon0, lat0)."""
    proj = LocalProjection(lon0, lat0)
    xy = np.array([[i * spacing, j * spacing] for j in range(n) for i in range(n)], float)
    nodes = proj.inverse(xy)
    arcs = []
    for j in range(n):
        for i in range(n):
            u = j * n + i
            if i + 1 < n:
                arcs += [(u, u + 1), (u + 1, u)]
            if j + 1 < n:
                arcs += [(u, u + n), (u + n, u)]
    return nodes, np.array(arcs), proj


def arc_id(arcs, u, v):
    return int(np.nonzero((arcs[:, 0] == u) & (arcs[:, 1] == v))[0][0])


def test_point_segment_and_projection():
    d, t, _ = point_segment(np.array([5.0, 3.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0]))
    assert d == pytest.approx(3.0) and t == pytest.approx(0.5)
    proj = LocalProjection(10.0, 50.0)
    ll = proj.inverse(np.array([[1000.0, 0.0]]))[0]
    assert haversine_m(10.0, 50.0, ll[0], ll[1]) == pytest.approx(1000.0, rel=2e-3)


def test_lcs():
    assert lcs_len([1, 2, 3, 4], [1, 3, 4]) == 3
    assert lcs_len([], [1]) == 0
    assert lcs_len([5, 1, 5], [5, 5]) == 2


def _drive(nodes, arcs, proj, path_nodes, speed=10.0, noise=4.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = proj.forward(nodes)
    pts = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        a, b = xy[u], xy[v]
        L = np.hypot(*(b - a))
        for s in np.arange(0, L, speed):
            pts.append(a + (b - a) * s / L)
    pts = np.array(pts) + rng.normal(0, noise, (len(pts), 2))
    t = np.arange(len(pts), dtype=float)
    return pts, t


def test_alignment_and_hmm_recover_route():
    nodes, arcs, _ = grid_network()
    path = [0, 1, 2, 6, 10, 9]                 # east, east, north, north, west
    route = [arc_id(arcs, u, v) for u, v in zip(path[:-1], path[1:])]
    track_lonlat = nodes[path]
    net = RoadNetwork.for_track(nodes, arcs, track_lonlat)
    xy, t = _drive(nodes, arcs, net.proj, path)
    # monotone alignment assigns every fix to the right route arc
    pos, dist = align_route(xy, net.a[route], net.b[route])
    assert np.all(np.diff(pos) >= 0) and dist.max() < 20
    labels = np.array(route)[pos]
    # HMM with distance-only emissions: direction must come from the transition model
    cand = build_candidates(net, xy, t, radius=40, k=8, radius_max=200)
    P = np.where(cand.valid, 1.0, 0.0); P /= P.sum(1, keepdims=True)
    res = viterbi(net, xy, t, cand.C, cand.valid, cand.proj_t, P, cand.dist,
                  dict(emission_weight=0.0, dist_weight=1.0, sigma_dist=8.0, beta=5.0))
    acc = np.mean(res.arcs == labels)
    assert acc > 0.85
    m = route_metrics(net, res.route, route)
    assert m["route_f1_len"] > 0.9
    assert route_continuity(net, res.arcs, xy) > 0.95


def test_nearest_baseline_confuses_direction():
    nodes, arcs, _ = grid_network()
    path = [0, 1, 2, 3]
    net = RoadNetwork.for_track(nodes, arcs, nodes[path])
    xy, t = _drive(nodes, arcs, net.proj, path, noise=0.5)
    cand = build_candidates(net, xy, t, radius=40, k=8, radius_max=200)
    # forward and reverse arcs are at identical distance -> heading feature must separate them
    assert np.all(cand.X["head_diff"].values >= 0)
    first_two = cand.dist[:, :2]
    assert np.allclose(first_two[:, 0], first_two[:, 1], atol=1e-6)


def test_prediction_never_reads_route(monkeypatch, tmp_path):
    """The Predictor must work when the reference route file does not even exist."""
    import joblib
    from pathsnap import io as pio
    from pathsnap.pipeline import Predictor

    nodes, arcs, _ = grid_network()
    path = [0, 1, 2, 6]
    net = RoadNetwork.for_track(nodes, arcs, nodes[path])
    xy, t = _drive(nodes, arcs, net.proj, path)
    lonlat = net.proj.inverse(xy)
    d = tmp_path / "evaluator_test" / "T1"
    d.mkdir(parents=True)
    np.savetxt(d / "T1.track", np.c_[lonlat, t], fmt="%.7f")
    np.savetxt(d / "T1.nodes", nodes, fmt="%.7f")
    np.savetxt(d / "T1.arcs", arcs, fmt="%d")

    np.savetxt(d / "T1.route", np.array([0]), fmt="%d")   # present, but must never be read
    real_read = pio._read

    def guarded(path, *a, **k):
        if str(path).endswith(".route"):
            raise AssertionError("reference route accessed during prediction")
        return real_read(path, *a, **k)
    monkeypatch.setattr(pio, "_read", guarded)

    class Uniform:
        def predict_proba(self, X):
            return np.c_[np.zeros(len(X)), np.ones(len(X))]
    out = Predictor(model=Uniform(), hmm_params=None).predict("evaluator_test", "T1", root=tmp_path)
    assert len(out["route"]) > 0
