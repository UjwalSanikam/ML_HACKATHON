"""HMM map matching with Viterbi decoding.

Hidden states   : for observation t, the candidate directed arcs C[t, k] (with the projected
                  position of the GPS fix along that arc). The candidate <-> arc correspondence
                  from candidate generation is preserved, so the decoded state index maps
                  straight back to a road segment.
Emissions       : log e_t(k) = w * log P_clf(k | obs t)  [+ optional Gaussian distance term]
                  where P_clf is the classifier probability normalised over the candidates of t.
Transitions     : route distance d between the two projected positions along the directed
                  network (same arc -> along-arc difference; otherwise rest of arc + bounded
                  Dijkstra between nodes + offset on the next arc), compared with the straight-line
                  GPS displacement g:   log a = -|d - g| / beta  (- u-turn penalty).
                  Transitions are forbidden (-inf) when no path exists within the search bound,
                  when the implied speed exceeds max_speed, or when moving backwards on the same
                  arc by more than back_tol.
Gaps            : if dt > gap_break_s, or every transition into step t is forbidden, the chain is
                  broken and a new Viterbi segment starts at t (scored by emissions only).
                  Observations with no candidates are skipped (prediction -1) and the next valid
                  observation is linked to the last valid one.
Recurrence      : delta_t(j) = log e_t(j) + max_i [delta_{t-1}(i) + log a_ij],  psi_t(j) = argmax_i.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import config

NEG = -1e18


@dataclass
class DecodeResult:
    arcs: np.ndarray                 # (N,) decoded arc per observation (-1 if no candidate)
    state: np.ndarray                # (N,) decoded candidate index
    route: List[int]                 # reconstructed continuous arc sequence
    breaks: List[int] = field(default_factory=list)   # observation indices where a new segment starts
    route_gaps: int = 0              # junctions in the route that could not be connected
    n_forbidden_steps: int = 0


def emission_logp(P, dist, valid, p):
    le = p["emission_weight"] * np.log(np.clip(P, 1e-9, 1.0))
    if p.get("dist_weight", 0) > 0:
        le = le - p["dist_weight"] * 0.5 * (dist / p["sigma_dist"]) ** 2
    return np.where(valid, le, NEG)


def _transition(net, a_prev, s_prev, a_cur, s_cur, g, dt, p):
    """(K_prev, K_cur) log transition matrix."""
    Kp, Kc = len(a_prev), len(a_cur)
    T = np.full((Kp, Kc), NEG)
    bound = p["route_factor"] * g + p["route_slack_m"]
    vmax_d = p["max_speed_mps"] * max(dt, 1.0) + 30.0
    to_nodes = net.arcs[a_cur, 0]
    for i in range(Kp):
        a = a_prev[i]
        rest = net.length[a] - s_prev[i]
        dist, _ = net.dijkstra(int(net.arcs[a, 1]), bound)
        for j in range(Kc):
            b = a_cur[j]
            if a == b:
                d = s_cur[j] - s_prev[i]
                if d < -p["back_tol_m"]:
                    continue
                d = max(d, 0.0)
            else:
                sp = dist.get(int(to_nodes[j]))
                if sp is None:
                    continue
                d = rest + sp + s_cur[j]
            if d > vmax_d:
                continue
            lt = -abs(d - g) / p["beta"]
            if a != b and net.reverse[a] == b:
                lt -= p.get("uturn_penalty", 0.0)
            T[i, j] = lt
    return T


def viterbi(net, xy, t, C, valid, proj_t, P, dist, params=None) -> DecodeResult:
    p = dict(config.HMM_DEFAULTS)
    p.setdefault("uturn_penalty", 2.0)
    if params:
        p.update(params)
    N, K = C.shape
    LE = emission_logp(P, dist, valid, p)
    S = proj_t * net.length[np.where(valid, C, 0)]
    obs = np.nonzero(valid.any(1))[0]
    state = np.full(N, -1, np.int64)
    breaks, forbidden = [], 0
    if len(obs) == 0:
        return DecodeResult(np.full(N, -1), state, [], [], 0, 0)

    seg_obs = [obs[0]]
    k0 = np.nonzero(valid[obs[0]])[0]
    delta = LE[obs[0], k0]
    ks = [k0]
    psis = [None]
    breaks.append(int(obs[0]))
    for prev, cur in zip(obs[:-1], obs[1:]):
        kc = np.nonzero(valid[cur])[0]
        kp = ks[-1]
        dt = t[cur] - t[prev]
        g = float(np.hypot(*(xy[cur] - xy[prev])))
        T = None
        if dt <= p["gap_break_s"]:
            T = _transition(net, C[prev, kp], S[prev, kp], C[cur, kc], S[cur, kc], g, dt, p)
            cand = delta[:, None] + T
            best = cand.max(0)
            if np.all(best <= NEG / 2):
                forbidden += 1
                T = None
        if T is None:  # break the chain: finish current segment, start new one
            _finish(state, seg_obs, ks, psis, delta)
            seg_obs, ks, psis = [cur], [kc], [None]
            delta = LE[cur, kc]
            breaks.append(int(cur))
            continue
        psi = cand.argmax(0)
        delta = best + LE[cur, kc]
        delta = delta - delta.max()  # keep numbers bounded
        seg_obs.append(cur); ks.append(kc); psis.append(psi)
    _finish(state, seg_obs, ks, psis, delta)

    arcs = np.where(state >= 0, C[np.arange(N), np.maximum(state, 0)], -1)
    route, gaps = reconstruct_route(net, arcs, breaks, p)
    return DecodeResult(arcs, state, route, breaks, gaps, forbidden)


def _finish(state, seg_obs, ks, psis, delta):
    j = int(np.argmax(delta))
    for idx in range(len(seg_obs) - 1, -1, -1):
        state[seg_obs[idx]] = ks[idx][j]
        if idx > 0:
            j = int(psis[idx][j])


def reconstruct_route(net, arcs, breaks=(), p=None, bridge_bound_m=3000.0):
    """Turn the per-observation arc sequence into a continuous arc route by inserting the
    shortest connecting path between consecutive distinct arcs."""
    seq = [int(a) for a in arcs if a >= 0]
    route: List[int] = []
    gaps = 0
    for a in seq:
        if not route:
            route.append(a); continue
        last = route[-1]
        if a == last:
            continue
        if net.arcs[last, 1] == net.arcs[a, 0]:
            route.append(a); continue
        d, path = net.shortest_path(int(net.arcs[last, 1]), int(net.arcs[a, 0]), bridge_bound_m)
        if d is None:
            gaps += 1
        else:
            route.extend(path)
        route.append(a)
    return route, gaps
