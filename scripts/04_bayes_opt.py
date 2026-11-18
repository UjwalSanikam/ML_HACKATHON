"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected classifier.

Objective : mean 5-fold GroupKFold (by trip) macro F1 on the 80 training trips
            (reliable-label rows only). Folds are identical to scripts/03.
Sampler   : TPESampler(seed=42, multivariate) with 10 random start-up trials;
            MedianPruner prunes trials whose running fold score falls behind.
Record    : results/optuna_<model>.db (SQLite study), results/optuna_<model>_trials.csv
Then      : refit with best params on all 80 trips -> artifacts/final_model.joblib,
            fit 5 out-of-fold models (used to give the HMM tuner honest emissions)
            and compare untuned vs tuned on CV + student-test.
Usage     : python scripts/04_bayes_opt.py --model lightgbm --trials 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics
from pathsnap.models import lgbm_default


def space(trial, model):
    if model == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 1200, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        )
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 1200, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 50, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 2.0),
    )


def build(model, params):
    if model == "lightgbm":
        return lgbm_default(**params)
    return XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=config.SEED, n_jobs=-1, **params)


UNTUNED = {"lightgbm": {}, "xgboost": dict(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                                          colsample_bytree=0.8)}


def cv_eval(model, params, tr, folds, ok, trial=None):
    scores = []
    for k, (tri, vai) in enumerate(folds):
        m = np.zeros(len(tr["y"]), bool); m[tri] = True; m &= ok
        mdl = build(model, params)
        mdl.fit(tr["X"].loc[m, FEATURES], tr["y"][m])
        p = mdl.predict_proba(tr["X"].iloc[vai][FEATURES])[:, 1]
        r = candidate_metrics(tr["obs"][vai], tr["y"][vai], p, tr["n_obs"], tr["label_ok"])
        scores.append(r)
        if trial is not None:
            trial.report(float(np.mean([s["macro_f1"] for s in scores])), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    tr = joblib.load(config.CACHE / "dataset_train.joblib")
    te = joblib.load(config.CACHE / "dataset_student_test.joblib")
    ok = tr["label_ok"][tr["obs"]]
    folds = list(GroupKFold(n_splits=config.N_FOLDS).split(tr["X"], tr["y"], tr["trip"]))

    db = config.RESULTS / f"optuna_{args.model}.db"
    study = optuna.create_study(
        study_name=f"pathsnap_{args.model}", direction="maximize", storage=f"sqlite:///{db}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=10, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))

    def objective(trial):
        params = space(trial, args.model)
        t0 = time.perf_counter()
        sc = cv_eval(args.model, params, tr, folds, ok, trial)
        trial.set_user_attr("cv_top1", float(np.mean([s["top1_acc"] for s in sc])))
        trial.set_user_attr("cv_log_loss", float(np.mean([s["log_loss"] for s in sc])))
        trial.set_user_attr("cv_macro_f1_std", float(np.std([s["macro_f1"] for s in sc])))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean([s["macro_f1"] for s in sc]))

    remaining = args.trials - len([t for t in study.trials if t.state.is_finished()])
    t_hpo = time.perf_counter()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    hpo_seconds = time.perf_counter() - t_hpo
    df = study.trials_dataframe()
    df.to_csv(config.RESULTS / f"optuna_{args.model}_trials.csv", index=False)
    best = study.best_params
    print("best", study.best_value, best)

    # ---------- untuned vs tuned ----------
    rows = []
    for tag, params in [("untuned", UNTUNED[args.model]), ("tuned", best)]:
        sc = cv_eval(args.model, params, tr, folds, ok)
        mdl = build(args.model, params)
        t0 = time.perf_counter(); mdl.fit(tr["X"].loc[ok, FEATURES], tr["y"][ok]); fit = time.perf_counter() - t0
        t0 = time.perf_counter(); p = mdl.predict_proba(te["X"][FEATURES])[:, 1]; inf = time.perf_counter() - t0
        tm = candidate_metrics(te["obs"], te["y"], p, te["n_obs"], te["label_ok"])
        rows.append(dict(model=f"{args.model}_{tag}", cv_macro_f1=np.mean([s["macro_f1"] for s in sc]),
                         cv_macro_f1_std=np.std([s["macro_f1"] for s in sc]),
                         cv_top1=np.mean([s["top1_acc"] for s in sc]), cv_log_loss=np.mean([s["log_loss"] for s in sc]),
                         test_top1=tm["top1_acc"], test_macro_f1=tm["macro_f1"], test_log_loss=tm["log_loss"],
                         test_precision=tm["precision_macro"], test_recall=tm["recall_macro"],
                         train_time_s=fit, infer_ms_per_1k_obs=1e6 * inf / te["n_obs"]))
        np.save(config.CACHE / f"test_proba_{args.model}_{tag}.npy", p)
        if tag == "tuned":
            joblib.dump(mdl, config.ARTIFACTS / "final_model.joblib", compress=3)
            # out-of-fold models for honest HMM tuning on training trips
            fold_dir = config.ARTIFACTS / "fold_models"; fold_dir.mkdir(exist_ok=True)
            fold_map = {}
            for k, (tri, vai) in enumerate(folds):
                m = np.zeros(len(tr["y"]), bool); m[tri] = True; m &= ok
                fm = build(args.model, params); fm.fit(tr["X"].loc[m, FEATURES], tr["y"][m])
                joblib.dump(fm, fold_dir / f"fold{k}.joblib", compress=3)
                for t in np.unique(tr["trip"][vai]):
                    fold_map[str(t)] = k
            (fold_dir / "fold_map.json").write_text(json.dumps(fold_map, indent=1))
    cmp = pd.DataFrame(rows)
    cmp.to_csv(config.RESULTS / "bayes_opt_tuned_vs_untuned.csv", index=False)
    print(cmp.to_string())
    meta = dict(model=args.model, objective="mean 5-fold GroupKFold macro F1 (candidate level, reliable labels)",
                sampler="TPE (multivariate, seed 42, 10 startup trials)", pruner="MedianPruner",
                n_trials=len(study.trials), n_complete=len([t for t in study.trials if t.state.name == "COMPLETE"]),
                n_pruned=len([t for t in study.trials if t.state.name == "PRUNED"]),
                best_value=study.best_value, best_params=best, hpo_seconds_this_run=hpo_seconds,
                total_trial_seconds=float(df["duration"].dt.total_seconds().sum()))
    (config.ARTIFACTS / "final_model_config.json").write_text(json.dumps(meta, indent=2, default=str))

    # ---------- plots ----------
    comp = df[df.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(comp.number, comp.value, s=18, color="#1565c0", label="trial")
    ax[0].plot(comp.number, comp.value.cummax(), color="#d81b60", label="best so far")
    ax[0].axhline(rows[0]["cv_macro_f1"], color="gray", ls="--", label="untuned")
    ax[0].set(title=f"Optuna TPE history ({args.model})", xlabel="trial", ylabel="CV macro F1"); ax[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#2e7d32")
        ax[1].set(title="Hyperparameter importance (fANOVA)")
    except Exception as e:  # pragma: no cover
        ax[1].text(0.1, 0.5, str(e))
    fig.tight_layout(); fig.savefig(config.FIGURES / f"bayes_opt_{args.model}.png"); plt.close(fig)

    params = [c for c in comp.columns if c.startswith("params_")]
    n = len(params); cols = 3; rws = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(12, 3 * rws))
    for ax, c in zip(axs.ravel(), params):
        ax.scatter(comp[c], comp.value, s=14, c=comp.number, cmap="viridis")
        ax.set(xlabel=c.replace("params_", ""), ylabel="CV macro F1")
        if c.replace("params_", "") in ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda", "reg_alpha",
                                        "min_child_weight"):
            ax.set_xscale("log")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(config.FIGURES / f"bayes_opt_{args.model}_slices.png"); plt.close(fig)


if __name__ == "__main__":
    main()
