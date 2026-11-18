"""End-to-end inference: raw trip -> candidates -> classifier -> HMM/Viterbi -> route.

Used identically by the evaluation scripts and the FastAPI backend. The reference
route is never loaded inside `Predictor.predict`; `evaluate` loads it afterwards.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import config
from .features import FEATURES
from .hmm import reconstruct_route, viterbi
from .labels import observation_labels
from .io import load_route
from .metrics import RELIABLE_LABEL_M, normalise_by_obs, route_continuity, route_metrics
from .prepare import PreparedTrip, prepare

FINAL_MODEL = config.ARTIFACTS / "final_model.joblib"
HMM_CONFIG = config.ARTIFACTS / "hmm_config.json"


def load_hmm_params(path: Path = HMM_CONFIG) -> dict:
    p = dict(config.HMM_DEFAULTS)
    p["uturn_penalty"] = 2.0
    if Path(path).exists():
        p.update(json.loads(Path(path).read_text()).get("params", {}))
    return p


class Predictor:
    def __init__(self, model=None, hmm_params=None, model_path: Path = FINAL_MODEL):
        self.model = model if model is not None else joblib.load(model_path)
        self.hmm_params = hmm_params or load_hmm_params()

    def probabilities(self, pt: PreparedTrip) -> np.ndarray:
        N, K = pt.cand.C.shape
        p = self.model.predict_proba(pt.cand.X[FEATURES])[:, 1]
        pn = normalise_by_obs(pt.cand.obs_idx, p, N)
        P = np.zeros((N, K))
        flat = np.flatnonzero(pt.cand.valid.ravel())
        P.ravel()[flat] = pn
        return P

    def decode(self, pt: PreparedTrip, P: np.ndarray, params=None):
        c = pt.cand
        return viterbi(pt.net, pt.xy, pt.t, c.C, c.valid, c.proj_t, P, c.dist, params or self.hmm_params)

    def predict(self, partition: str, trip_id: str, root=None, use_hmm: bool = True) -> dict:
        t_all = time.perf_counter()
        pt = prepare(partition, trip_id, with_labels=False, root=root)
        t0 = time.perf_counter()
        P = self.probabilities(pt)
        pt.timings["classifier"] = time.perf_counter() - t0
        N = len(pt.xy)
        clf_state = np.where(pt.cand.valid.any(1), P.argmax(1), -1)
        clf_arcs = np.where(clf_state >= 0, pt.cand.C[np.arange(N), np.maximum(clf_state, 0)], -1)
        out = dict(pt=pt, P=P, clf_arcs=clf_arcs)
        if use_hmm:
            t0 = time.perf_counter()
            res = self.decode(pt, P)
            pt.timings["hmm"] = time.perf_counter() - t0
            out.update(hmm_arcs=res.arcs, route=res.route, breaks=res.breaks, route_gaps=res.route_gaps,
                       forbidden_steps=res.n_forbidden_steps)
        pt.timings["total"] = time.perf_counter() - t_all
        return out


_LABEL_CACHE: dict = {}


def evaluate(pred: dict, partition: str, trip_id: str, root=None) -> dict:
    """Server-side scoring against the reference route (loaded only here)."""
    pt: PreparedTrip = pred["pt"]
    route = load_route(partition, trip_id, root)
    if route is None:
        return {}
    key = (partition, trip_id, id(pt))
    if key not in _LABEL_CACHE:
        _LABEL_CACHE[key] = observation_labels(pt.net, pt.xy, route)
    la, _, ld = _LABEL_CACHE[key]
    ok = ld <= RELIABLE_LABEL_M
    net = pt.net
    cand_has = (pt.cand.C == la[:, None]).any(1)
    res = dict(n_obs=int(len(la)), n_scored=int(ok.sum()), label_coverage=float(ok.mean()),
               candidate_miss_rate=float(1 - cand_has[ok].mean()) if ok.any() else None)
    clf = pred["clf_arcs"]
    res["clf_top1_acc"] = float(np.mean(clf[ok] == la[ok])) if ok.any() else None
    res["clf_continuity"] = route_continuity(net, clf, pt.xy)
    clf_route, _ = reconstruct_route(net, clf)
    res.update({f"clf_{k}": v for k, v in route_metrics(net, clf_route, route).items()})
    if "hmm_arcs" in pred:
        h = pred["hmm_arcs"]
        res["hmm_top1_acc"] = float(np.mean(h[ok] == la[ok])) if ok.any() else None
        res["hmm_continuity"] = route_continuity(net, h, pt.xy)
        res.update({f"hmm_{k}": v for k, v in route_metrics(net, pred["route"], route).items()})
        res["hmm_segments"] = len(pred["breaks"])
        res["hmm_route_gaps"] = pred["route_gaps"]
    return res
