"""Train and compare all candidate classifiers.

* 5-fold GroupKFold over the 80 training trips (groups = trip id; no trip is split)
* refit on all 80 training trips, evaluate on the 10 student-testing trips
* metrics: CV macro F1, top-1 accuracy, log loss, precision, recall, candidate miss
  rate, classifier-only route continuity, training and inference time
Outputs: results/model_comparison.csv, results/cv_folds.csv, results/experiment_log.jsonl,
         artifacts/models/<name>.joblib, figures.
Usage: python scripts/03_train_compare.py [--only NAME ...] [--skip-cv]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, route_continuity, top1_mask
from pathsnap.models import subsample_obs, zoo
from pathsnap.network import RoadNetwork
from pathsnap.io import load_trip, clean_track

MODELS_DIR = config.ARTIFACTS / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load(part):
    return joblib.load(config.CACHE / f"dataset_{part}.joblib")


def rows_ok(ds):
    return ds["label_ok"][ds["obs"]]


def predict_p(model, X):
    return model.predict_proba(X[FEATURES])[:, 1]


def test_continuity(ds, p):
    """Classifier-only route continuity on the full-rate student-test sequences."""
    top = top1_mask(ds["obs"], p, ds["n_obs"])
    best = np.full(ds["n_obs"], -1)
    best[ds["obs"][top]] = ds["arc"][top]
    vals = []
    for tid in np.unique(ds["obs_trip"]):
        tr = load_trip(config.STUDENT_TEST, tid)
        track, _ = clean_track(tr.track)
        net = _net_cache.setdefault(tid, RoadNetwork.for_track(tr.nodes, tr.arcs, track[:, :2]))
        vals.append(route_continuity(net, best[ds["obs_trip"] == tid], net.proj.forward(track[:, :2])))
    return float(np.mean(vals))


_net_cache = {}


def nearest_baseline(ds):
    """Non-learned reference: score = -distance."""
    return 1.0 / (1.0 + ds["X"]["dist"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--continuity-only", action="store_true",
                    help="recompute the route_continuity column from saved student-test probabilities")
    args = ap.parse_args()

    if args.continuity_only:
        te = load(config.STUDENT_TEST)
        out = config.RESULTS / "model_comparison.csv"
        df = pd.read_csv(out)
        for i, r in df.iterrows():
            if r.model.startswith("NearestRoad"):
                p = nearest_baseline(te)
            else:
                p = np.load(config.CACHE / f"test_proba_{r.model}.npy")
            df.loc[i, "route_continuity"] = test_continuity(te, p)
        df.to_csv(out, index=False)
        print(df[["model", "route_continuity"]])
        return

    tr, te = load(config.TRAIN), load(config.STUDENT_TEST)
    ok_tr = rows_ok(tr)
    groups = tr["trip"]
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    folds = list(gkf.split(tr["X"], tr["y"], groups))
    log = open(config.RESULTS / "experiment_log.jsonl", "a")
    summary, fold_rows = [], []

    # ---- baseline (no learning) ----
    if not args.only or "NearestRoad" in args.only:
        cvm = [candidate_metrics(tr["obs"][va], tr["y"][va], nearest_baseline(tr)[va], tr["n_obs"],
                                 tr["label_ok"]) for _, va in folds]
        p = nearest_baseline(te)
        t0 = time.perf_counter(); nearest_baseline(te); inf = time.perf_counter() - t0
        tm = candidate_metrics(te["obs"], te["y"], p, te["n_obs"], te["label_ok"])
        summary.append(dict(model="NearestRoad (baseline)", family="Heuristic", config="argmin perpendicular distance",
                            cv_macro_f1=np.mean([m["macro_f1"] for m in cvm]),
                            cv_macro_f1_std=np.std([m["macro_f1"] for m in cvm]),
                            cv_top1=np.mean([m["top1_acc"] for m in cvm]), cv_log_loss=np.nan,
                            test_top1=tm["top1_acc"], test_macro_f1=tm["macro_f1"],
                            test_precision=tm["precision_macro"], test_recall=tm["recall_macro"],
                            test_log_loss=np.nan, test_miss_rate=tm["candidate_miss_rate"],
                            route_continuity=test_continuity(te, p), train_time_s=0.0,
                            infer_ms_per_1k_obs=1000 * inf / te["n_obs"] * 1000, train_top1=np.nan))

    for spec in zoo():
        if args.only and spec.name not in args.only:
            continue
        print(f"== {spec.name}", flush=True)
        cvm = []
        if not args.skip_cv:
            for k, (tri, vai) in enumerate(folds):
                m = np.zeros(len(tr["y"]), bool); m[tri] = True
                m &= ok_tr & subsample_obs(tr["obs"], spec.obs_frac, seed=config.SEED + k)
                mdl = spec.build()
                t0 = time.perf_counter()
                mdl.fit(tr["X"].loc[m, FEATURES], tr["y"][m])
                ft = time.perf_counter() - t0
                p = predict_p(mdl, tr["X"].iloc[vai])
                r = candidate_metrics(tr["obs"][vai], tr["y"][vai], p, tr["n_obs"], tr["label_ok"])
                r.update(model=spec.name, fold=k, fit_time_s=ft)
                cvm.append(r); fold_rows.append(r)
                print(f"   fold {k}: macroF1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f} ll={r['log_loss']:.4f} ({ft:.1f}s)",
                      flush=True)
        # ---- refit on all training trips, evaluate on student test ----
        m = ok_tr & subsample_obs(tr["obs"], spec.obs_frac)
        mdl = spec.build()
        t0 = time.perf_counter()
        mdl.fit(tr["X"].loc[m, FEATURES], tr["y"][m])
        fit_t = time.perf_counter() - t0
        p_tr = predict_p(mdl, tr["X"])
        trm = candidate_metrics(tr["obs"], tr["y"], p_tr, tr["n_obs"], tr["label_ok"])
        t0 = time.perf_counter()
        p = predict_p(mdl, te["X"])
        inf = time.perf_counter() - t0
        tm = candidate_metrics(te["obs"], te["y"], p, te["n_obs"], te["label_ok"])
        row = dict(model=spec.name, family=spec.family, config=spec.config,
                   cv_macro_f1=np.mean([c["macro_f1"] for c in cvm]) if cvm else np.nan,
                   cv_macro_f1_std=np.std([c["macro_f1"] for c in cvm]) if cvm else np.nan,
                   cv_top1=np.mean([c["top1_acc"] for c in cvm]) if cvm else np.nan,
                   cv_log_loss=np.mean([c["log_loss"] for c in cvm]) if cvm else np.nan,
                   test_top1=tm["top1_acc"], test_macro_f1=tm["macro_f1"], test_precision=tm["precision_macro"],
                   test_recall=tm["recall_macro"], test_log_loss=tm["log_loss"],
                   test_miss_rate=tm["candidate_miss_rate"], route_continuity=test_continuity(te, p),
                   train_time_s=fit_t, infer_ms_per_1k_obs=1000 * inf / te["n_obs"] * 1000,
                   train_top1=trm["top1_acc"], train_macro_f1=trm["macro_f1"], obs_frac=spec.obs_frac)
        print("   TEST", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        summary.append(row)
        log.write(json.dumps(dict(time=time.time(), seed=config.SEED, **row, folds=cvm), default=float) + "\n")
        log.flush()
        joblib.dump(mdl, MODELS_DIR / f"{spec.name}.joblib", compress=3)
        np.save(config.CACHE / f"test_proba_{spec.name}.npy", p)

    out = config.RESULTS / "model_comparison.csv"
    df = pd.DataFrame(summary)
    if args.only and out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.model.isin(df.model)], df], ignore_index=True)
    df.to_csv(out, index=False)
    if fold_rows:
        fo = config.RESULTS / "cv_folds.csv"
        fdf = pd.DataFrame(fold_rows)
        if args.only and fo.exists():
            old = pd.read_csv(fo)
            fdf = pd.concat([old[~old.model.isin(fdf.model)], fdf], ignore_index=True)
        fdf.to_csv(fo, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
