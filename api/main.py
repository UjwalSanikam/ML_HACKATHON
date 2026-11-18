"""PathSnap FastAPI backend (Laptop 1).

Hosts the trained classifier, the HMM/Viterbi decoder and the road-network data.
Ground-truth routes stay on this machine: they are read only *after* prediction,
inside `evaluate`, and only aggregate scores are returned to the UI.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Environment:
    PATHSNAP_DATA            root folder holding the partitions (default ../Student folder)
    PATHSNAP_EVAL_PARTITION  partition shown to evaluators (default evaluator_test;
                             falls back to student_test if it does not exist)
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.hmm import reconstruct_route
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import FINAL_MODEL, HMM_CONFIG, Predictor, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")

PARTITION = config.EVALUATOR_TEST
if not list_trips(PARTITION):
    log.warning("partition '%s' not found under %s - falling back to '%s'", PARTITION, config.DATA_ROOT,
                config.STUDENT_TEST)
    PARTITION = config.STUDENT_TEST
ALLOWED_PARTITIONS = {PARTITION, config.STUDENT_TEST}

app = FastAPI(title="PathSnap API", version="1.0",
              description="Map-matching of noisy GPS trajectories: candidate classifier + HMM/Viterbi.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

t0 = time.perf_counter()
PREDICTOR = Predictor()
MODEL_META = json.loads((config.ARTIFACTS / "final_model_config.json").read_text()) \
    if (config.ARTIFACTS / "final_model_config.json").exists() else {}
HMM_META = json.loads(HMM_CONFIG.read_text()) if HMM_CONFIG.exists() else {}
log.info("model loaded from %s in %.2fs (%s)", FINAL_MODEL.name, time.perf_counter() - t0,
         type(PREDICTOR.model).__name__)
log.info("serving partition '%s' with %d trips from %s", PARTITION, len(list_trips(PARTITION)), config.DATA_ROOT)
_TRIP_INFO: dict = {}


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:6]
    start = time.perf_counter()
    client = request.client.host if request.client else "?"
    log.info("[%s] --> %s %s from %s", rid, request.method, request.url.path, client)
    response = await call_next(request)
    log.info("[%s] <-- %s %s %d in %.0f ms", rid, request.method, request.url.path, response.status_code,
             1000 * (time.perf_counter() - start))
    return response


class PredictRequest(BaseModel):
    trip_id: str
    partition: Optional[str] = None
    include_classifier_only: bool = True


def _trip_info(partition, tid):
    key = (partition, tid)
    if key not in _TRIP_INFO:
        tr = load_trip(partition, tid).track  # GPS only - routes are never read here
        _TRIP_INFO[key] = dict(trip_id=tid, n_obs=int(len(tr)), duration_s=float(tr[-1, 2] - tr[0, 2]),
                               center=[float(np.median(tr[:, 1])), float(np.median(tr[:, 0]))])
    return _TRIP_INFO[key]


@app.get("/health")
def health():
    return dict(status="ok", host=socket.gethostname(), partition=PARTITION, model=type(PREDICTOR.model).__name__)


@app.get("/model")
def model_info():
    return dict(model=type(PREDICTOR.model).__name__, features=FEATURES, hyperparameters=MODEL_META.get("best_params"),
                bayes_opt=dict((k, MODEL_META.get(k)) for k in ("objective", "sampler", "n_trials", "best_value")),
                hmm=HMM_META.get("params", PREDICTOR.hmm_params))


@app.get("/trips")
def trips(partition: Optional[str] = None):
    part = partition or PARTITION
    if part not in ALLOWED_PARTITIONS:
        raise HTTPException(400, f"partition must be one of {sorted(ALLOWED_PARTITIONS)}")
    ids = list_trips(part)
    return dict(partition=part, count=len(ids), trips=[_trip_info(part, t) for t in ids])


def _poly(net, arcs):
    """Merge a continuous arc list into polylines of [lat, lon] (split where arcs do not touch)."""
    lines, cur = [], []
    for e in arcs:
        u, v = net.arcs[e]
        if cur and net.arcs[prev, 1] == u:
            cur.append(net.nodes_lonlat[v, ::-1].round(6).tolist())
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [net.nodes_lonlat[u, ::-1].round(6).tolist(), net.nodes_lonlat[v, ::-1].round(6).tolist()]
        prev = e
    if len(cur) > 1:
        lines.append(cur)
    return lines


@app.post("/predict")
def predict(req: PredictRequest):
    part = req.partition or PARTITION
    if part not in ALLOWED_PARTITIONS:
        raise HTTPException(400, f"partition must be one of {sorted(ALLOWED_PARTITIONS)}")
    if req.trip_id not in list_trips(part):
        raise HTTPException(404, f"trip {req.trip_id} not found in {part}")
    log.info("predict trip=%s partition=%s", req.trip_id, part)
    t_start = time.perf_counter()
    out = PREDICTOR.predict(part, req.trip_id)           # <- no reference data involved
    t_pred = time.perf_counter() - t_start
    scores = evaluate(out, part, req.trip_id)           # <- server-side scoring only
    pt = out["pt"]
    net = pt.net
    track = pt.trip.track
    step = max(1, len(track) // 3000)
    matched = []
    for i in range(0, len(pt.xy), step):
        a = int(out["hmm_arcs"][i])
        if a < 0:
            continue
        s = pt.cand.proj_t[i, int(np.nonzero(pt.cand.C[i] == a)[0][0])]
        p = net.a[a] + s * (net.b[a] - net.a[a])
        ll = net.proj.inverse(p)
        matched.append([round(float(ll[1]), 6), round(float(ll[0]), 6)])
    resp = dict(
        trip_id=req.trip_id, partition=part,
        n_observations=int(len(pt.xy)), n_route_arcs=len(out["route"]),
        gps=track[::step, [1, 0]].round(6).tolist(),
        matched_points=matched,
        route=_poly(net, out["route"]),
        segments=len(out["breaks"]), route_gaps=int(out["route_gaps"]),
        timings_ms={k: round(1000 * v, 1) for k, v in pt.timings.items()},
        prediction_seconds=round(t_pred, 3),
        metrics={k: (round(v, 5) if isinstance(v, float) else v) for k, v in scores.items()},
    )
    if req.include_classifier_only:
        clf_route, _ = reconstruct_route(net, out["clf_arcs"])
        resp["classifier_only_route"] = _poly(net, clf_route)
    log.info("trip=%s obs=%d route_arcs=%d HMM top-1=%.4f route F1=%.4f continuity=%.4f | %.2fs",
             req.trip_id, len(pt.xy), len(out["route"]), scores.get("hmm_top1_acc") or float("nan"),
             scores.get("hmm_route_f1_len") or float("nan"), scores.get("hmm_continuity") or float("nan"), t_pred)
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
