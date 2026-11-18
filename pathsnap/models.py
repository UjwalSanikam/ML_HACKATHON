"""Model zoo for the candidate classifier.

Each entry returns an *unfitted* estimator with predict_proba. `obs_frac` is the
fraction of training observations used to fit models whose cost grows
super-linearly with data size (KNN, kernel SVM, MLP) - recorded in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import SEED


@dataclass
class ModelSpec:
    name: str
    family: str
    config: str
    build: Callable
    obs_frac: float = 1.0
    params: dict = field(default_factory=dict)


def lgbm_default(**kw):
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             random_state=SEED, n_jobs=-1, verbose=-1)
    p.update(kw)
    return LGBMClassifier(**p)


def zoo():
    return [
        ModelSpec("LogisticRegression", "Linear / probabilistic", "StandardScaler + L2, C=1.0",
                  lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
        ModelSpec("GaussianNB", "Linear / probabilistic", "QuantileTransformer(normal) + GaussianNB",
                  lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                        GaussianNB())),
        ModelSpec("KNN", "Instance-based", "k=40, distance-weighted, standardised, 25% obs",
                  lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=40, weights="distance",
                                                                               n_jobs=-1)), obs_frac=0.25),
        ModelSpec("DecisionTree", "Tree-based", "max_depth=12, min_samples_leaf=50",
                  lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED)),
        ModelSpec("RandomForest", "Ensemble (bagging)", "300 trees, max_depth=20, min_leaf=10, sqrt features",
                  lambda: RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=10,
                                                 max_features="sqrt", n_jobs=-1, random_state=SEED)),
        ModelSpec("ExtraTrees", "Ensemble (bagging)", "300 trees, max_depth=22, min_leaf=10",
                  lambda: ExtraTreesClassifier(n_estimators=300, max_depth=22, min_samples_leaf=10,
                                               max_features="sqrt", n_jobs=-1, random_state=SEED)),
        ModelSpec("HistGradientBoosting", "Boosting", "lr=0.1, max_iter=300, max_leaf_nodes=31",
                  lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                                         random_state=SEED)),
        ModelSpec("LightGBM", "Boosting", "400 trees, lr=0.05, 63 leaves (untuned)", lambda: lgbm_default()),
        ModelSpec("XGBoost", "Boosting", "400 trees, lr=0.05, depth=7, hist",
                  lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                                        colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                                        random_state=SEED, n_jobs=-1)),
        ModelSpec("LinearSVM_RBF_Nystroem", "Kernel-based", "Nystroem RBF(400 comps) + LinearSVC + sigmoid calib., 25% obs",
                  lambda: make_pipeline(StandardScaler(), Nystroem(kernel="rbf", gamma=0.05, n_components=400,
                                                                   random_state=SEED),
                                        CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=5000), cv=3,
                                                               method="sigmoid")), obs_frac=0.25),
        ModelSpec("MLP", "Neural network", "128-64 ReLU, Adam lr=1e-3, early stopping, 50% obs",
                  lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                                                                        batch_size=512, learning_rate_init=1e-3,
                                                                        max_iter=60, early_stopping=True,
                                                                        n_iter_no_change=6, random_state=SEED)),
                  obs_frac=0.5),
    ]


def subsample_obs(obs: np.ndarray, frac: float, seed: int = SEED) -> np.ndarray:
    """Row mask keeping a random fraction of *observations* (all their candidates)."""
    if frac >= 1:
        return np.ones(len(obs), bool)
    u = np.unique(obs)
    rng = np.random.default_rng(seed)
    keep = rng.choice(u, size=max(1, int(len(u) * frac)), replace=False)
    return np.isin(obs, keep)
