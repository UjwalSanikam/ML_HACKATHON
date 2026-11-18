"""HMM / Viterbi route reconstruction experiments.

1. Decoder tuning (Optuna TPE) on training trips using *out-of-fold* classifier emissions
   (each training trip is scored by the fold model that never saw it).
   Objective: mean over trips of 0.5 * (pointwise top-1 accuracy + length-weighted route F1).
2. Student-test comparison for the final model:
     classifier-only | HMM distance-only emissions (Newson-Krumm style, no classifier)
     | classifier + HMM (default params) | classifier + HMM (tuned params)
3. Every model from the comparison passed through the tuned decoder (sequential compatibility).
4. Folium maps for every student-test trip.
Outputs: artifacts/hmm_config.json, results/hmm_*.csv, results/optuna_hmm.db, results/maps/*.html
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
import optuna
import pandas as pd

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import normalise_by_obs
from pathsnap.pipeline import Predictor, evaluate, load_hmm_params
from pathsnap.prepare import prepare
from pathsnap.viz import trip_map


def proba_matrix(model, pt):
    N, K = pt.cand.C.shape
    p = model.predict_proba(pt.cand.X[FEATURES])[:, 1]
    pn = normalise_by_obs(pt.cand.obs_idx, p, N)
    P = np.zeros((N, K)); P.ravel()[np.flatnonzero(pt.cand.valid.ravel())] = pn
    return P


def run(pr: Predictor, pt, P, params, partition, tid, use_hmm=True):
    N = len(pt.xy)
    clf_state = np.where(pt.cand.valid.any(1), P.argmax(1), -1)
    clf_arcs = np.where(clf_state >= 0, pt.cand.C[np.arange(N), np.maximum(clf_state, 0)], -1)
    out = dict(pt=pt, P=P, clf_arcs=clf_arcs)
    t0 = time.perf_counter()
    if use_hmm:
        res = pr.decode(pt, P, params)
        out.update(hmm_arcs=res.arcs, route=res.route, breaks=res.breaks, route_gaps=res.route_gaps)
    dt = time.perf_counter() - t0
    r = evaluate(out, partition, tid)
    r["hmm_seconds"] = dt
    return out, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--max-obs", type=int, default=6000, help="skip very long training trips during tuning")
    ap.add_argument("--n-trips", type=int, default=32, help="number of training trips used for decoder tuning")
    args = ap.parse_args()

    final = joblib.load(config.ARTIFACTS / "final_model.joblib")
    pr = Predictor(model=final, hmm_params=dict(config.HMM_DEFAULTS, uturn_penalty=2.0))
    fold_dir = config.ARTIFACTS / "fold_models"
    fold_map = json.loads((fold_dir / "fold_map.json").read_text())
    folds = {k: joblib.load(fold_dir / f"fold{k}.joblib") for k in set(fold_map.values())}

    # ---------------- 1. tune decoder on training trips (OOF emissions) ----------------
    tune = []
    for tid in list_trips(config.TRAIN):
        if len(tune) >= args.n_trips:
            break
        pt = prepare(config.TRAIN, tid)
        if len(pt.xy) > args.max_obs:
            continue
        tune.append((tid, pt, proba_matrix(folds[fold_map[tid]], pt)))
    print(f"tuning on {len(tune)} training trips", flush=True)

    def score(params):
        vals = []
        for tid, pt, P in tune:
            _, r = run(pr, pt, P, params, config.TRAIN, tid)
            if r.get("hmm_top1_acc") is None:
                continue
            vals.append(0.5 * (r["hmm_top1_acc"] + r["hmm_route_f1_len"]))
        return float(np.mean(vals))

    study = optuna.create_study(study_name="pathsnap_hmm", direction="maximize",
                                storage=f"sqlite:///{config.RESULTS / 'optuna_hmm.db'}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=8))
    if not study.trials:
        study.enqueue_trial({k: v for k, v in pr.hmm_params.items()
                             if k in ("beta", "emission_weight", "uturn_penalty", "back_tol_m", "dist_weight",
                                      "gap_break_s")})

    def objective(trial):
        params = dict(pr.hmm_params)
        params.update(beta=trial.suggest_float("beta", 1.0, 60.0, log=True),
                      emission_weight=trial.suggest_float("emission_weight", 0.2, 3.0),
                      uturn_penalty=trial.suggest_float("uturn_penalty", 0.0, 12.0),
                      back_tol_m=trial.suggest_float("back_tol_m", 2.0, 40.0),
                      dist_weight=trial.suggest_float("dist_weight", 0.0, 1.0),
                      gap_break_s=trial.suggest_float("gap_break_s", 20.0, 900.0, log=True))
        return score(params)

    t0 = time.perf_counter()
    remaining = args.trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    tune_s = time.perf_counter() - t0
    study.trials_dataframe().to_csv(config.RESULTS / "optuna_hmm_trials.csv", index=False)
    tuned = dict(pr.hmm_params); tuned.update(study.best_params)
    default = dict(config.HMM_DEFAULTS, uturn_penalty=2.0)
    (config.ARTIFACTS / "hmm_config.json").write_text(json.dumps(dict(
        params=tuned, objective="mean_trip 0.5*(HMM top-1 acc + length-weighted route F1), OOF emissions",
        tuning_trips=[t for t, _, _ in tune], n_trials=len(study.trials), best_value=study.best_value,
        default_value=study.trials[0].value, tuning_seconds=tune_s), indent=2))
    print("HMM tuned:", study.best_value, tuned, flush=True)

    # ---------------- 2. student-test comparison ----------------
    rows, per_trip = [], []
    test_ids = list_trips(config.STUDENT_TEST)
    prepared = {tid: prepare(config.STUDENT_TEST, tid) for tid in test_ids}
    P_final = {tid: proba_matrix(final, pt) for tid, pt in prepared.items()}
    outs = {}
    settings = [
        ("Classifier only (final model)", "clf", tuned, P_final),
        ("HMM, distance-only emissions", "hmm", dict(default, emission_weight=0.0, dist_weight=1.0, sigma_dist=10.0),
         {t: np.where(prepared[t].cand.valid, 1.0, 0.0) for t in test_ids}),
        ("Classifier + HMM (default params)", "hmm", default, P_final),
        ("Classifier + HMM (tuned params)", "hmm", tuned, P_final),
    ]
    for name, kind, params, Ps in settings:
        for tid in test_ids:
            pt = prepared[tid]
            out, r = run(pr, pt, Ps[tid], params, config.STUDENT_TEST, tid, use_hmm=True)
            pre = "clf_" if kind == "clf" else "hmm_"
            per_trip.append(dict(setting=name, trip_id=tid, n_obs=r["n_obs"], top1_acc=r[f"{pre}top1_acc"],
                                 continuity=r[f"{pre}continuity"], route_f1_len=r[f"{pre}route_f1_len"],
                                 route_ordered_f1=r[f"{pre}route_ordered_f1"],
                                 route_precision_len=r[f"{pre}route_precision_len"],
                                 route_recall_len=r[f"{pre}route_recall_len"],
                                 segments=r.get("hmm_segments") if kind == "hmm" else None,
                                 decode_seconds=r["hmm_seconds"] if kind == "hmm" else 0.0))
            if name.endswith("(tuned params)"):
                outs[tid] = out
    pt_df = pd.DataFrame(per_trip)
    pt_df.to_csv(config.RESULTS / "hmm_per_trip.csv", index=False)
    agg = pt_df.groupby("setting", sort=False).agg(
        top1_acc=("top1_acc", "mean"), continuity=("continuity", "mean"), route_f1_len=("route_f1_len", "mean"),
        route_ordered_f1=("route_ordered_f1", "mean"), route_precision_len=("route_precision_len", "mean"),
        route_recall_len=("route_recall_len", "mean"), decode_seconds=("decode_seconds", "mean")).reset_index()
    # observation-weighted accuracy as well
    w = pt_df.assign(c=pt_df.top1_acc * pt_df.n_obs).groupby("setting", sort=False)
    agg["top1_acc_obs_weighted"] = (w.c.sum() / w.n_obs.sum()).values
    agg.to_csv(config.RESULTS / "hmm_comparison.csv", index=False)
    print(agg.to_string(), flush=True)

    # ---------------- 3. every model through the tuned decoder ----------------
    cmp = pd.read_csv(config.RESULTS / "model_comparison.csv")
    allm = []
    for name in cmp.model:
        if name.startswith("NearestRoad"):
            Ps = {t: np.where(prepared[t].cand.valid, 1.0 / (1.0 + prepared[t].cand.dist), 0.0) for t in test_ids}
            Ps = {t: P / np.maximum(P.sum(1, keepdims=True), 1e-12) for t, P in Ps.items()}
        else:
            mdl = joblib.load(config.ARTIFACTS / "models" / f"{name}.joblib")
            Ps = {t: proba_matrix(mdl, prepared[t]) for t in test_ids}
        res = [run(pr, prepared[t], Ps[t], tuned, config.STUDENT_TEST, t)[1] for t in test_ids]
        allm.append(dict(model=name,
                         clf_top1=np.mean([r["clf_top1_acc"] for r in res]),
                         hmm_top1=np.mean([r["hmm_top1_acc"] for r in res]),
                         clf_continuity=np.mean([r["clf_continuity"] for r in res]),
                         hmm_continuity=np.mean([r["hmm_continuity"] for r in res]),
                         clf_route_f1_len=np.mean([r["clf_route_f1_len"] for r in res]),
                         hmm_route_f1_len=np.mean([r["hmm_route_f1_len"] for r in res]),
                         hmm_route_ordered_f1=np.mean([r["hmm_route_ordered_f1"] for r in res])))
        print(allm[-1], flush=True)
    pd.DataFrame(allm).to_csv(config.RESULTS / "hmm_all_models.csv", index=False)

    # ---------------- 4. maps ----------------
    for tid, out in outs.items():
        pt = out["pt"]
        from pathsnap.hmm import reconstruct_route
        clf_route, _ = reconstruct_route(pt.net, out["clf_arcs"])
        near = np.unique(pt.cand.C[pt.cand.valid])
        ref = load_trip(config.STUDENT_TEST, tid, with_route=True).route
        m = trip_map(pt.net, pt.trip.track[:, :2], reference=ref, predicted=out["route"],
                     classifier_only=clf_route, network_arcs=near,
                     title=f"student_test/{tid}: classifier + HMM", obs_step=2)
        m.save(str(config.MAPS / f"pred_student_test_{tid}.html"))


if __name__ == "__main__":
    main()
