"""Report figures generated from the recorded experiment results.

Also runs the *training-history* experiments for the iterative models (MLP epochs,
LightGBM / XGBoost boosting rounds) on a trip-grouped 80/20 split of the training
partition and stores the histories in results/training_histories.csv.
Non-iterative models (trees, KNN, NB, LR solver) get no training curves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.metrics import top1_mask
from pathsnap.models import lgbm_default

F = config.FIGURES
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
C1, C2, C3 = "#1565c0", "#d81b60", "#ef6c00"


def model_plots():
    df = pd.read_csv(config.RESULTS / "model_comparison.csv")
    folds = pd.read_csv(config.RESULTS / "cv_folds.csv")
    order = df.sort_values("cv_macro_f1").model.tolist()
    d = df.set_index("model").loc[order]

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].barh(order, d.cv_macro_f1, xerr=d.cv_macro_f1_std, color=C1, alpha=.85)
    ax[0].set(title="5-fold GroupKFold CV macro F1 (train, 80 trips)", xlim=(0.7, 1.0))
    data = [folds[folds.model == m].macro_f1.values for m in order if m in set(folds.model)]
    labels = [m for m in order if m in set(folds.model)]
    ax[1].boxplot(data, vert=False, labels=labels)
    ax[1].set(title="CV macro F1 per fold (stability)")
    fig.tight_layout(); fig.savefig(F / "models_cv_macro_f1.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(order)); w = .38
    ax.bar(x - w / 2, d.test_top1, w, label="Top-1 accuracy", color=C1)
    ax.bar(x + w / 2, d.test_macro_f1, w, label="Macro F1", color=C2)
    for i, (a, b) in enumerate(zip(d.test_top1, d.test_macro_f1)):
        ax.text(i - w / 2, a + .004, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, b + .004, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set(ylim=(0.5, 1.0), title="Student-test (10 trips): observation-level top-1 accuracy and macro F1")
    ax.legend(); fig.tight_layout(); fig.savefig(F / "models_student_test.png"); plt.close(fig)

    dd = d.dropna(subset=["test_log_loss"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].barh(dd.index, dd.test_log_loss, color=C3, label="student-test")
    ax[0].scatter(dd.cv_log_loss, dd.index, color="k", zorder=3, s=14, label="CV")
    ax[0].set(title="Candidate log loss (lower is better)", xscale="log"); ax[0].legend()
    ax[1].scatter(d.infer_ms_per_1k_obs, d.test_top1, s=40, color=C1)
    for m, r in d.iterrows():
        ax[1].annotate(m, (r.infer_ms_per_1k_obs, r.test_top1), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].set(xscale="log", title="Accuracy vs inference cost", xlabel="ms per 1000 observations",
              ylabel="student-test top-1")
    fig.tight_layout(); fig.savefig(F / "models_logloss_speed.png"); plt.close(fig)

    dt = d.dropna(subset=["train_top1"])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(dt)); w = .27
    ax.bar(x - w, dt.train_top1, w, label="train (fit data)", color="#90a4ae")
    ax.bar(x, dt.cv_top1, w, label="CV (held-out trips)", color=C1)
    ax.bar(x + w, dt.test_top1, w, label="student-test", color=C2)
    ax.set_xticks(x); ax.set_xticklabels(dt.index, rotation=30, ha="right")
    ax.set(ylim=(0.8, 1.0), title="Over/under-fitting check: top-1 accuracy on train vs CV vs student-test")
    ax.legend(); fig.tight_layout(); fig.savefig(F / "models_overfitting.png"); plt.close(fig)

    # confusion matrices (candidate level, top-1 decision) on student-test
    te = joblib.load(config.CACHE / "dataset_student_test.joblib")
    keep = te["label_ok"][te["obs"]]
    names = [m for m in order if (config.CACHE / f"test_proba_{m}.npy").exists()]
    cols = 4; rows = int(np.ceil(len(names) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.1 * rows))
    for ax, m in zip(axs.ravel(), names):
        p = np.load(config.CACHE / f"test_proba_{m}.npy")
        top = top1_mask(te["obs"], p, te["n_obs"])
        cm = confusion_matrix(te["y"][keep], top[keep].astype(int), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > .5 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["other", "reference"], yticklabels=["other", "reference"],
               title=m, xlabel="predicted", ylabel="true"); ax.grid(False)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Candidate-level confusion matrices on student-test (row-normalised)")
    fig.tight_layout(); fig.savefig(F / "models_confusion_matrices.png"); plt.close(fig)


def training_histories():
    tr = joblib.load(config.CACHE / "dataset_train.joblib")
    ok = tr["label_ok"][tr["obs"]]
    X, y, g = tr["X"].loc[ok, FEATURES].to_numpy(), tr["y"][ok], tr["trip"][ok]
    tri, vai = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED).split(X, y, g))
    rows = []
    # MLP: epoch-wise loss
    sc = StandardScaler().fit(X[tri])
    Xt, Xv = sc.transform(X[tri]), sc.transform(X[vai])
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                        random_state=config.SEED)
    rng = np.random.default_rng(config.SEED)
    for ep in range(40):
        idx = rng.permutation(len(Xt))
        mlp.partial_fit(Xt[idx], y[tri][idx], classes=[0, 1])
        rows.append(dict(model="MLP", step=ep + 1, train_loss=log_loss(y[tri], mlp.predict_proba(Xt)[:, 1]),
                         val_loss=log_loss(y[vai], mlp.predict_proba(Xv)[:, 1])))
    # LightGBM: boosting rounds
    best = json.loads((config.ARTIFACTS / "final_model_config.json").read_text())["best_params"] \
        if (config.ARTIFACTS / "final_model_config.json").exists() else {}
    for tag, params in [("LightGBM (untuned)", {}), ("LightGBM (tuned)", best)]:
        m = lgbm_default(**params)
        m.fit(X[tri], y[tri], eval_set=[(X[tri], y[tri]), (X[vai], y[vai])], eval_metric="binary_logloss")
        ev = m.evals_result_
        k = list(ev.keys())
        for i, (a, b) in enumerate(zip(ev[k[0]]["binary_logloss"], ev[k[1]]["binary_logloss"])):
            rows.append(dict(model=tag, step=i + 1, train_loss=a, val_loss=b))
    xg = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", eval_metric="logloss", random_state=config.SEED, n_jobs=-1)
    xg.fit(X[tri], y[tri], eval_set=[(X[tri], y[tri]), (X[vai], y[vai])], verbose=False)
    ev = xg.evals_result()
    for i, (a, b) in enumerate(zip(ev["validation_0"]["logloss"], ev["validation_1"]["logloss"])):
        rows.append(dict(model="XGBoost", step=i + 1, train_loss=a, val_loss=b))
    h = pd.DataFrame(rows)
    h.to_csv(config.RESULTS / "training_histories.csv", index=False)
    models = h.model.unique()
    fig, axs = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6))
    for ax, m in zip(axs, models):
        s = h[h.model == m]
        ax.plot(s.step, s.train_loss, label="train", color=C1)
        ax.plot(s.step, s.val_loss, label="validation (held-out trips)", color=C2)
        ax.set(title=m, xlabel="epoch" if m == "MLP" else "boosting round", ylabel="log loss")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "training_histories.png"); plt.close(fig)


def hmm_plots():
    agg = pd.read_csv(config.RESULTS / "hmm_comparison.csv")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    metrics = ["top1_acc", "continuity", "route_f1_len", "route_ordered_f1"]
    x = np.arange(len(metrics)); w = 0.8 / len(agg)
    cols = ["#ef6c00", "#90a4ae", "#64b5f6", "#d81b60"]
    for i, r in agg.iterrows():
        ax.bar(x + (i - (len(agg) - 1) / 2) * w, [r[m] for m in metrics], w, label=r.setting, color=cols[i % 4])
        for j, m in enumerate(metrics):
            ax.text(x[j] + (i - (len(agg) - 1) / 2) * w, r[m] + .004, f"{r[m]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(["Top-1 accuracy", "Route continuity", "Route F1 (length)", "Ordered route F1"])
    ax.set(ylim=(0.5, 1.02), title="Student-test: classifier-only vs HMM/Viterbi decoding (mean of 10 trips)")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(F / "hmm_comparison.png"); plt.close(fig)

    pt = pd.read_csv(config.RESULTS / "hmm_per_trip.csv")
    a = pt[pt.setting.str.startswith("Classifier only")].set_index("trip_id")
    b = pt[pt.setting.str.endswith("(tuned params)")].set_index("trip_id")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(a)); w = .38
    ax[0].bar(x - w / 2, a.top1_acc, w, color=C3, label="classifier only")
    ax[0].bar(x + w / 2, b.top1_acc, w, color=C2, label="classifier + HMM")
    ax[0].set_xticks(x); ax[0].set_xticklabels(a.index, rotation=45); ax[0].set(ylim=(.6, 1), title="Top-1 accuracy per trip")
    ax[0].legend()
    ax[1].bar(x - w / 2, a.route_f1_len, w, color=C3); ax[1].bar(x + w / 2, b.route_f1_len, w, color=C2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(a.index, rotation=45); ax[1].set(ylim=(.5, 1), title="Route F1 (length) per trip")
    fig.tight_layout(); fig.savefig(F / "hmm_per_trip.png"); plt.close(fig)

    am = pd.read_csv(config.RESULTS / "hmm_all_models.csv").sort_values("hmm_top1")
    fig, ax = plt.subplots(figsize=(12, 4.4))
    x = np.arange(len(am)); w = .38
    ax.bar(x - w / 2, am.clf_top1, w, color=C3, label="classifier only")
    ax.bar(x + w / 2, am.hmm_top1, w, color=C2, label="+ HMM (tuned decoder)")
    ax.set_xticks(x); ax.set_xticklabels(am.model, rotation=30, ha="right")
    ax.set(ylim=(.5, 1), title="Every model's emissions through the same HMM decoder (student-test top-1)")
    ax.legend(); fig.tight_layout(); fig.savefig(F / "hmm_all_models.png"); plt.close(fig)

    tr = pd.read_csv(config.RESULTS / "optuna_hmm_trials.csv")
    tr = tr[tr.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.scatter(tr.number, tr.value, s=16, color=C1); ax.plot(tr.number, tr.value.cummax(), color=C2)
    ax.set(title="HMM decoder tuning (TPE, OOF emissions, train trips)", xlabel="trial",
           ylabel="0.5·(top-1 + route F1)")
    fig.tight_layout(); fig.savefig(F / "hmm_tuning_history.png"); plt.close(fig)


def importance_plot():
    m = joblib.load(config.ARTIFACTS / "final_model.joblib")
    if not hasattr(m, "booster_"):
        return
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES).sort_values()
    imp = imp / imp.sum()
    imp.to_csv(config.RESULTS / "feature_importance.csv")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(imp.index, imp.values, color=C1); ax.set(title="Final model feature importance (gain share)")
    fig.tight_layout(); fig.savefig(F / "feature_importance.png"); plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["models", "histories", "hmm", "importance"]
    if "models" in what: model_plots()
    if "histories" in what: training_histories()
    if "hmm" in what: hmm_plots()
    if "importance" in what: importance_plot()
