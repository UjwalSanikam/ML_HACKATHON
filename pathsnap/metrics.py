"""Evaluation metrics (candidate level, observation level, route level)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

RELIABLE_LABEL_M = 50.0  # obs farther than this from their aligned reference arc are not scored


def normalise_by_obs(obs_idx: np.ndarray, p: np.ndarray, n_obs: int) -> np.ndarray:
    s = np.bincount(obs_idx, weights=p, minlength=n_obs)
    return p / np.maximum(s[obs_idx], 1e-12)


def top1_mask(obs_idx: np.ndarray, score: np.ndarray, n_obs: int) -> np.ndarray:
    """Boolean mask of the best-scoring row for each observation (ties -> first)."""
    order = np.lexsort((-score, obs_idx))
    first = np.ones(len(order), bool)
    first[1:] = obs_idx[order][1:] != obs_idx[order][:-1]
    m = np.zeros(len(score), bool)
    m[order[first]] = True
    return m


def candidate_metrics(obs_idx, y, p, n_obs, label_ok=None):
    """Metrics on candidate rows.

    * Top-1 accuracy : fraction of scored observations whose best candidate is the reference arc
      (observations whose reference arc is not in the candidate set count as errors).
    * Macro F1 / precision / recall : binary candidate classification where the top-1 candidate of
      each observation is predicted positive, macro-averaged over {not-reference, reference}.
    * Log loss : binary cross-entropy of the raw candidate probabilities.
    """
    obs_idx = np.asarray(obs_idx); y = np.asarray(y).astype(int); p = np.asarray(p, float)
    if label_ok is not None:
        keep = label_ok[obs_idx]
        obs_ok = label_ok
    else:
        keep = np.ones(len(y), bool)
        obs_ok = np.ones(n_obs, bool)
    top = top1_mask(obs_idx, p, n_obs)
    hit = np.zeros(n_obs, bool)
    np.logical_or.at(hit, obs_idx[top], y[top] == 1)
    has = np.zeros(n_obs, bool)
    np.logical_or.at(has, obs_idx, y == 1)
    scored = obs_ok & np.isin(np.arange(n_obs), obs_idx)
    yk, pk, tk = y[keep], np.clip(p[keep], 1e-6, 1 - 1e-6), top[keep].astype(int)
    return dict(
        top1_acc=float(hit[scored].mean()),
        top1_acc_given_candidate=float(hit[scored & has].mean()) if (scored & has).any() else np.nan,
        candidate_miss_rate=float(1 - has[scored].mean()),
        macro_f1=float(f1_score(yk, tk, average="macro")),
        precision_macro=float(precision_score(yk, tk, average="macro", zero_division=0)),
        recall_macro=float(recall_score(yk, tk, average="macro", zero_division=0)),
        precision_pos=float(precision_score(yk, tk, zero_division=0)),
        recall_pos=float(recall_score(yk, tk, zero_division=0)),
        log_loss=float(log_loss(yk, pk, labels=[0, 1])),
        n_obs=int(scored.sum()), n_rows=int(keep.sum()),
    )


def route_continuity(net, arc_seq, xy=None) -> float:
    """Fraction of consecutive observation transitions (where the predicted arc changes or stays)
    that are physically continuous.

    A transition a -> b is continuous if a == b, or b is reachable from a along the directed
    network by a path no longer than max(3*g, g + 100 m), where g is the GPS displacement
    between the two fixes (so skipping a short arc between two 1 Hz fixes is allowed but
    jumping to an unconnected parallel road or the opposite carriageway is not).
    Without xy only identical / directly connected arcs count.
    """
    arc_seq = np.asarray(arc_seq, int)
    idx = np.nonzero(arc_seq >= 0)[0]
    if len(idx) < 2:
        return 1.0
    a, b = arc_seq[idx[:-1]], arc_seq[idx[1:]]
    ok = (a == b) | (net.arcs[a, 1] == net.arcs[b, 0])
    if xy is not None:
        g = np.hypot(*(xy[idx[1:]] - xy[idx[:-1]]).T)
        for k in np.nonzero(~ok)[0]:
            bound = max(3 * g[k], g[k] + 100.0)
            dist, _ = net.dijkstra(int(net.arcs[a[k], 1]), bound)
            ok[k] = int(net.arcs[b[k], 0]) in dist
    return float(ok.mean())


def dedup(seq):
    out = []
    for a in seq:
        if a >= 0 and (not out or out[-1] != a):
            out.append(int(a))
    return out


def lcs_len(a, b) -> int:
    """Longest common subsequence length (O(n*m) with numpy rows)."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return 0
    prev = np.zeros(len(b) + 1, np.int32)
    for x in a:
        eq = (b == x)
        cur = np.zeros_like(prev)
        # cur[j+1] = prev[j]+1 if eq else max(prev[j+1], cur[j])
        diag = prev[:-1] + eq
        best = np.maximum(prev[1:], diag)
        cur[1:] = np.maximum.accumulate(np.maximum(best, 0))
        # maximum.accumulate is valid because LCS rows are non-decreasing
        cur[1:] = np.maximum(cur[1:], diag)
        prev = cur
    return int(prev[-1])


def route_metrics(net, pred_route, ref_route):
    """Route-level (sequence) metrics between a predicted arc list and the reference route."""
    pr, rf = dedup(pred_route), dedup(ref_route)
    sp, sr = set(pr), set(rf)
    inter = sp & sr
    L = net.length
    lp, lr, li = L[list(sp)].sum() if sp else 0, L[list(sr)].sum() if sr else 0, L[list(inter)].sum() if inter else 0
    prec = li / lp if lp else 0.0
    rec = li / lr if lr else 0.0
    lcs = lcs_len(pr, rf)
    cont = 1.0
    if len(pr) > 1:
        a, b = np.array(pr[:-1]), np.array(pr[1:])
        cont = float(np.mean(net.arcs[a, 1] == net.arcs[b, 0]))
    return dict(
        route_precision_len=float(prec), route_recall_len=float(rec),
        route_f1_len=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        route_set_f1=float(2 * len(inter) / (len(sp) + len(sr))) if sp or sr else 0.0,
        route_ordered_f1=float(2 * lcs / (len(pr) + len(rf))) if pr or rf else 0.0,
        route_arc_continuity=cont, n_pred_arcs=len(pr), n_ref_arcs=len(rf),
    )
