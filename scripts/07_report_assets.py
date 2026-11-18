"""Report assets: static map exports and markdown tables built from the recorded CSVs.

* results/figures/map_<trip>.png : static export of the predicted vs reference route
  (same data as the interactive Folium maps in results/maps/)
* results/report_tables.md       : every table used in the report, generated from results/*.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.hmm import reconstruct_route
from pathsnap.io import load_route
from pathsnap.pipeline import Predictor, evaluate


def static_map(pr, tid, zoom_box=None):
    out = pr.predict(config.STUDENT_TEST, tid)
    r = evaluate(out, config.STUDENT_TEST, tid)
    pt = out["pt"]; net = pt.net
    ref = load_route(config.STUDENT_TEST, tid)
    clf_route, _ = reconstruct_route(net, out["clf_arcs"])
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    for ax, box in zip(axs, [None, zoom_box]):
        x0, y0 = pt.xy.min(0) - 150; x1, y1 = pt.xy.max(0) + 150
        if box is not None:
            x0, y0, x1, y1 = box
        act = net.active_ids
        mid = (net.a[act] + net.b[act]) / 2
        sel = act[(mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1)]
        for e in sel:
            ax.plot(*np.c_[net.a[e], net.b[e]], color="#cfd8dc", lw=.6, zorder=1)
        def draw(arcs, color, lw, label, z, ls="-"):
            first = True
            for e in arcs:
                ax.plot(*np.c_[net.a[e], net.b[e]], color=color, lw=lw, zorder=z, ls=ls,
                        label=label if first else None, solid_capstyle="round")
                first = False
        draw(ref, "#2e7d32", 6, "reference route", 2)
        draw(clf_route, "#ef6c00", 1.6, "classifier only", 3, "--")
        draw(out["route"], "#d81b60", 2.2, "classifier + HMM", 4)
        ax.scatter(*pt.xy.T, s=2, color="#1565c0", zorder=5, label="raw GPS")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.grid(False)
        ax.set(xlabel="x (m, local projection)", ylabel="y (m)")
    axs[0].legend(loc="best", fontsize=8)
    axs[0].set_title(f"student_test/{tid}: HMM top-1 {r['hmm_top1_acc']:.3f}, route F1 {r['hmm_route_f1_len']:.3f}")
    axs[1].set_title("zoomed detail" if zoom_box else "")
    fig.tight_layout(); fig.savefig(config.FIGURES / f"map_{tid}.png"); plt.close(fig)
    return out, r


def worst_box(out, half=250):
    """Zoom window centred where classifier-only and HMM disagree the most."""
    pt = out["pt"]
    diff = np.nonzero(out["clf_arcs"] != out["hmm_arcs"])[0]
    if len(diff) == 0:
        c = pt.xy[len(pt.xy) // 2]
    else:
        # densest disagreement window
        counts = [np.sum(np.abs(diff - i) < 30) for i in diff]
        c = pt.xy[diff[int(np.argmax(counts))]]
    return (c[0] - half, c[1] - half, c[0] + half, c[1] + half)


def md(df, floatfmt=4):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "N/A" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.values) + "\n"


def tables():
    R = config.RESULTS
    s = ["# Report tables (auto-generated from results/*.csv)\n"]
    mc = pd.read_csv(R / "model_comparison.csv").sort_values("cv_macro_f1", ascending=False)
    s.append("## Overall model comparison (CV = 5-fold GroupKFold on 80 train trips; test = 10 student-test trips)\n")
    s.append(md(mc[["model", "family", "config", "cv_macro_f1", "cv_macro_f1_std", "cv_top1", "test_top1",
                    "test_macro_f1", "test_precision", "test_recall", "test_log_loss", "route_continuity",
                    "train_time_s", "infer_ms_per_1k_obs"]]))
    s.append("\n## Overfitting check (top-1 accuracy)\n")
    s.append(md(mc[["model", "train_top1", "cv_top1", "test_top1"]].dropna()))
    if (R / "bayes_opt_tuned_vs_untuned.csv").exists():
        s.append("\n## Bayesian optimisation: untuned vs tuned\n")
        s.append(md(pd.read_csv(R / "bayes_opt_tuned_vs_untuned.csv")))
        meta = json.loads((config.ARTIFACTS / "final_model_config.json").read_text())
        s.append("\nBest hyperparameters:\n\n```json\n" + json.dumps(meta["best_params"], indent=2) + "\n```\n")
        s.append(f"\nTrials: {meta['n_trials']} ({meta['n_complete']} complete, {meta['n_pruned']} pruned); "
                 f"total trial time {meta['total_trial_seconds']/60:.1f} min; best CV macro F1 {meta['best_value']:.4f}\n")
    if (R / "hmm_comparison.csv").exists():
        s.append("\n## HMM / Viterbi: classifier-only vs decoded (student-test, mean of 10 trips)\n")
        s.append(md(pd.read_csv(R / "hmm_comparison.csv")))
        s.append("\n## HMM per trip\n")
        s.append(md(pd.read_csv(R / "hmm_per_trip.csv")))
        s.append("\n## All models through the tuned HMM decoder\n")
        s.append(md(pd.read_csv(R / "hmm_all_models.csv")))
        hc = json.loads((config.ARTIFACTS / "hmm_config.json").read_text())
        s.append("\nTuned HMM parameters:\n\n```json\n" + json.dumps(hc["params"], indent=2) + "\n```\n")
        s.append(f"\nDecoder tuning: {hc['n_trials']} trials, objective {hc['objective']}, "
                 f"default {hc['default_value']:.4f} -> best {hc['best_value']:.4f}\n")
    if (R / "eda_missing_values.csv").exists():
        s.append("\n## Missing values / data-quality counts (EDA)\n")
        s.append(md(pd.read_csv(R / "eda_missing_values.csv")))
    (R / "report_tables.md").write_text("\n".join(s))


LOSS = {
    "LogisticRegression": "binary cross-entropy + L2 penalty (lbfgs)",
    "GaussianNB": "maximum likelihood of class-conditional Gaussians (after quantile-normal transform)",
    "KNN": "none (lazy learner); probability = distance-weighted vote of 40 neighbours",
    "DecisionTree": "Gini impurity reduction (greedy splits)",
    "RandomForest": "Gini impurity per tree; bagging + feature subsampling",
    "ExtraTrees": "Gini impurity with randomised thresholds; averaging",
    "HistGradientBoosting": "binary log loss (gradient boosting on histograms)",
    "LightGBM": "binary log loss (leaf-wise gradient boosting)",
    "XGBoost": "binary log loss (depth-wise gradient boosting, 2nd-order)",
    "LinearSVM_RBF_Nystroem": "squared hinge loss on Nystroem RBF features; sigmoid (Platt) calibration for probabilities",
    "MLP": "binary cross-entropy, Adam, L2 alpha=1e-4, early stopping on 10% validation",
    "NearestRoad (baseline)": "none (heuristic)",
}


def fact_sheets():
    """Per-model facts for report section 3.2 (numbers only - write the analysis yourselves)."""
    R = config.RESULTS
    mc = pd.read_csv(R / "model_comparison.csv")
    hm = pd.read_csv(R / "hmm_all_models.csv").set_index("model") if (R / "hmm_all_models.csv").exists() else None
    folds = pd.read_csv(R / "cv_folds.csv")
    out = ["# Per-model fact sheets (auto-generated; all numbers from results/*.csv)\n"]
    for _, r in mc.sort_values("cv_macro_f1", ascending=False).iterrows():
        f = folds[folds.model == r.model]
        out.append(f"## {r.model}  ({r.family})\n")
        out.append(f"- Configuration: {r.config}")
        out.append(f"- Objective / loss: {LOSS.get(r.model, '')}")
        if len(f):
            out.append(f"- CV macro F1 per fold: {', '.join(f'{v:.4f}' for v in f.macro_f1)} "
                       f"(mean {r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f})")
            out.append(f"- CV top-1 per fold: {', '.join(f'{v:.4f}' for v in f.top1_acc)}")
        out.append(f"- Student-test: top-1 {r.test_top1:.4f}, macro F1 {r.test_macro_f1:.4f}, precision {r.test_precision:.4f}, "
                   f"recall {r.test_recall:.4f}, log loss {r.test_log_loss if pd.notna(r.test_log_loss) else 'N/A'}")
        if pd.notna(r.get("train_top1", np.nan)):
            out.append(f"- Over/under-fitting evidence: train top-1 {r.train_top1:.4f} vs CV {r.cv_top1:.4f} vs "
                       f"test {r.test_top1:.4f} (gap train-CV = {r.train_top1 - r.cv_top1:+.4f})")
        out.append(f"- Classifier-only route continuity: {r.route_continuity:.4f}")
        out.append(f"- Training time {r.train_time_s:.1f} s; inference {r.infer_ms_per_1k_obs:.1f} ms / 1000 obs")
        if hm is not None and r.model in hm.index:
            h = hm.loc[r.model]
            out.append(f"- Through the tuned HMM: top-1 {h.clf_top1:.4f} -> {h.hmm_top1:.4f}; route F1 "
                       f"{h.clf_route_f1_len:.4f} -> {h.hmm_route_f1_len:.4f}")
        out.append("")
    (R / "report_model_fact_sheets.md").write_text("\n".join(out))


if __name__ == "__main__":
    if sys.argv[1:] == ["tables"]:
        tables(); fact_sheets(); sys.exit()
    pr = Predictor()
    for tid in ["00000056", "00000009"]:
        out, _ = static_map(pr, tid)
        static_map(pr, tid, worst_box(out))
    tables()
    fact_sheets()
    print("written", config.RESULTS / "report_tables.md")
