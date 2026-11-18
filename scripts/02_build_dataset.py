"""Build the candidate-level training / development datasets.

train        : every TRAIN_OBS_STRIDE-th observation of each of the 80 training trips
student_test : every observation of the 10 student-testing trips
Rows from observations with unreliable reference labels (> 50 m from the aligned
reference arc) are flagged with label_ok = 0 and excluded from fitting / scoring.
Output: .cache/dataset_<partition>.joblib
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config
from pathsnap.io import list_trips
from pathsnap.metrics import RELIABLE_LABEL_M
from pathsnap.prepare import prepare


def one(partition, tid, stride):
    pt = prepare(partition, tid, with_labels=True)
    N = len(pt.xy)
    obs_keep = np.zeros(N, bool)
    obs_keep[::stride] = True
    label_ok = pt.label_dist <= RELIABLE_LABEL_M
    m = obs_keep[pt.cand.obs_idx]
    X = pt.cand.X[m].reset_index(drop=True)
    # re-index observations to 0..n-1 within the trip subset
    kept_obs = np.nonzero(obs_keep)[0]
    remap = -np.ones(N, np.int64)
    remap[kept_obs] = np.arange(len(kept_obs))
    obs_local = remap[pt.cand.obs_idx[m]]
    return dict(trip_id=tid, X=X, y=pt.y[m], obs=obs_local, arc=pt.cand.arc[m],
                label_ok=label_ok[kept_obs], n_obs=len(kept_obs), timings=pt.timings)


def build(partition, stride):
    t0 = time.time()
    ids = list_trips(partition)
    parts = Parallel(n_jobs=-1)(delayed(one)(partition, t, stride) for t in ids)
    off = 0
    X, y, obs, trip, arc, ok = [], [], [], [], [], []
    for p in parts:
        X.append(p["X"]); y.append(p["y"]); obs.append(p["obs"] + off); arc.append(p["arc"])
        trip.append(np.full(len(p["y"]), p["trip_id"])); ok.append(p["label_ok"])
        off += p["n_obs"]
    obs_trip = np.concatenate([np.full(p["n_obs"], p["trip_id"]) for p in parts])
    ds = dict(X=pd.concat(X, ignore_index=True), y=np.concatenate(y), obs=np.concatenate(obs),
              trip=np.concatenate(trip), arc=np.concatenate(arc), label_ok=np.concatenate(ok),
              obs_trip=obs_trip, n_obs=off, stride=stride)
    joblib.dump(ds, config.CACHE / f"dataset_{partition}.joblib", compress=3)
    pos = ds["y"].mean()
    print(f"{partition}: {len(ids)} trips, {off} obs, {len(ds['y'])} rows, positive rate {pos:.3f}, "
          f"reliable obs {ds['label_ok'].mean():.3f}, {time.time()-t0:.1f}s")
    return ds


if __name__ == "__main__":
    build(config.TRAIN, config.TRAIN_OBS_STRIDE)
    build(config.STUDENT_TEST, 1)
