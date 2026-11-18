"""Candidate generation and feature engineering.

Problem formulation
-------------------
For every GPS observation we generate a candidate set of nearby *directed*
road arcs. Each (observation, candidate arc) pair is one row. A classifier
predicts P(candidate is the reference arc | features). Probabilities are
normalised within each observation and passed to the HMM as emissions.

All features are computed from GPS + road network only; the reference route
is never used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .geo import angle_diff_deg, bearing_deg, point_segment

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "head_diff", "head_cos", "head_diff_long", "head_minus_min", "heading_valid",
    "speed", "turn_rate", "dt_prev",
    "arc_len", "proj_t", "dist_to_end", "at_endpoint",
    "n_cand", "n_within_20",
    "out_deg_to", "in_deg_from", "is_twoway",
    "ctx_dist_mean", "ctx_dist_max", "ctx_dist_fwd", "ctx_dist_bwd",
    "along_disp", "radius_used",
]

CTX_OFFSETS = (-10, -5, -2, 2, 5, 10)


@dataclass
class CandidateTable:
    obs_idx: np.ndarray      # (M,) observation index of each row
    arc: np.ndarray          # (M,) arc id
    X: pd.DataFrame          # (M, F) features
    C: np.ndarray            # (N, K) padded candidate arcs (-1 = empty)
    valid: np.ndarray        # (N, K) bool
    proj_t: np.ndarray       # (N, K) projection fraction along arc
    dist: np.ndarray         # (N, K)


def _gps_heading(xy: np.ndarray, ks=(1, 2, 3, 5, 8, 12, 20), min_disp=6.0):
    N = len(xy)
    head = np.full(N, np.nan)
    ar = np.arange(N)
    for k in ks:
        v = xy[np.minimum(ar + k, N - 1)] - xy[np.maximum(ar - k, 0)]
        ok = np.isnan(head) & (np.hypot(v[:, 0], v[:, 1]) >= min_disp)
        head[ok] = bearing_deg(v[ok])
    return head


def motion_features(xy: np.ndarray, t: np.ndarray):
    N = len(xy)
    ar = np.arange(N)
    head = _gps_heading(xy)
    i2, j2 = np.maximum(ar - 2, 0), np.minimum(ar + 2, N - 1)
    disp = np.hypot(*(xy[j2] - xy[i2]).T)
    dt = np.maximum(t[j2] - t[i2], 1e-3)
    speed = np.where(j2 > i2, disp / dt, 0.0)
    v_long = xy[np.minimum(ar + 10, N - 1)] - xy[np.maximum(ar - 10, 0)]
    head_long = np.where(np.hypot(*v_long.T) > 10, bearing_deg(v_long), np.nan)
    hp = np.r_[head[:1], head[:-1]]
    turn = angle_diff_deg(head, hp)
    turn = np.where(np.isnan(turn), 0.0, turn)
    dt_prev = np.r_[0.0, np.diff(t)]
    return dict(head=head, head_long=head_long, speed=speed, turn=turn, dt_prev=dt_prev)


def build_candidates(net, xy: np.ndarray, t: np.ndarray,
                     radius=config.CAND_RADIUS_M, k=config.CAND_K,
                     radius_max=config.CAND_RADIUS_MAX_M) -> CandidateTable:
    N = len(xy)
    cand, used = net.candidates(xy, radius, k, radius_max)
    C = np.full((N, k), -1, np.int64)
    for i, c in enumerate(cand):
        C[i, :len(c)] = c
    valid = C >= 0
    Cs = np.where(valid, C, 0)
    a, b = net.a[Cs], net.b[Cs]
    d, tt, _ = point_segment(xy[:, None, :], a, b)
    d = np.where(valid, d, np.inf)

    mf = motion_features(xy, t)
    arc_bear = net.bearing[Cs]
    hd = angle_diff_deg(mf["head"][:, None], arc_bear)
    head_valid = ~np.isnan(mf["head"])
    hd_f = np.where(np.isnan(hd), 90.0, hd)
    hdl = angle_diff_deg(mf["head_long"][:, None], arc_bear)
    hdl = np.where(np.isnan(hdl), 90.0, hdl)

    dmin = np.min(d, axis=1, keepdims=True)
    dmin = np.where(np.isfinite(dmin), dmin, 0.0)
    rank = np.argsort(np.argsort(np.where(valid, d, 1e9), axis=1), axis=1)
    near = valid & (d <= dmin + 25)
    hmin = np.min(np.where(near, hd_f, 999), axis=1, keepdims=True)

    # context: how well does this arc explain neighbouring observations?
    ar = np.arange(N)
    ctx = []
    for o in CTX_OFFSETS:
        q = xy[np.clip(ar + o, 0, N - 1)]
        dq, _, _ = point_segment(q[:, None, :], a, b)
        ctx.append(np.minimum(dq, 150.0))
    ctx = np.stack(ctx, 0)
    fwd = ctx[[i for i, o in enumerate(CTX_OFFSETS) if o > 0]].mean(0)
    bwd = ctx[[i for i, o in enumerate(CTX_OFFSETS) if o < 0]].mean(0)

    i5, j5 = np.maximum(ar - 5, 0), np.minimum(ar + 5, N - 1)
    disp = xy[j5] - xy[i5]
    along = (disp[:, None, :] * net.unit[Cs]).sum(-1)

    L = net.length[Cs]
    feats = {
        "dist": d, "dist_rank": rank, "dist_minus_min": d - dmin, "dist_ratio": d / (dmin + 1.0),
        "head_diff": hd_f, "head_cos": np.cos(np.deg2rad(hd_f)), "head_diff_long": hdl,
        "head_minus_min": hd_f - hmin, "heading_valid": np.broadcast_to(head_valid[:, None], C.shape),
        "speed": np.broadcast_to(mf["speed"][:, None], C.shape),
        "turn_rate": np.broadcast_to(mf["turn"][:, None], C.shape),
        "dt_prev": np.broadcast_to(np.minimum(mf["dt_prev"], 600)[:, None], C.shape),
        "arc_len": L, "proj_t": tt, "dist_to_end": L * (1 - tt),
        "at_endpoint": ((tt <= 1e-6) | (tt >= 1 - 1e-6)),
        "n_cand": np.broadcast_to(valid.sum(1)[:, None], C.shape),
        "n_within_20": np.broadcast_to((valid & (d <= 20)).sum(1)[:, None], C.shape),
        "out_deg_to": net.out_deg[net.arcs[Cs, 1]], "in_deg_from": net.in_deg[net.arcs[Cs, 0]],
        "is_twoway": net.reverse[Cs] >= 0,
        "ctx_dist_mean": ctx.mean(0), "ctx_dist_max": ctx.max(0),
        "ctx_dist_fwd": fwd, "ctx_dist_bwd": bwd,
        "along_disp": along,
        "radius_used": np.broadcast_to(used[:, None], C.shape),
    }
    m = valid.ravel()
    X = pd.DataFrame({f: np.asarray(feats[f], dtype=np.float32).ravel()[m] for f in FEATURES})
    obs_idx = np.repeat(ar, k)[m]
    return CandidateTable(obs_idx, C.ravel()[m], X, C, valid, tt, d)
