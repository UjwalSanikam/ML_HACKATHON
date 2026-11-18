"""Exploratory data analysis of the raw PathSnap data.

Partitions analysed: train (80 trips) and student_test (10 trips).
Outputs:
  results/eda_inventory.csv         per-trip inventory + quality statistics
  results/eda_summary.json          partition-level summary
  results/eda_data_quality.md       data-quality issues, assumptions and decisions
  results/figures/eda_*.png         plots
  results/maps/eda_*.html           Folium maps
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import config
from pathsnap.geo import haversine_m
from pathsnap.io import list_trips, load_trip, trip_dir
from pathsnap.prepare import prepare
from pathsnap.viz import trip_map

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})


def trip_stats(partition, tid):
    raw = load_trip(partition, tid, with_route=True)
    d = trip_dir(partition, tid)
    tr, nd, ar, rt = raw.track, raw.nodes, raw.arcs, raw.route
    dt = np.diff(tr[:, 2])
    V, E = len(nd), len(ar)
    g = coo_matrix((np.ones(E), (ar[:, 0], ar[:, 1])), shape=(V, V))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(ar[:, 0], minlength=V) + np.bincount(ar[:, 1], minlength=V)
    big_strong = np.bincount(lab_s).max() / V
    route_steps = len(rt) - 1
    disc = int(np.sum(ar[rt[:-1], 1] != ar[rt[1:], 0]))
    ext_ew = haversine_m(tr[:, 0].min(), tr[:, 1].mean(), tr[:, 0].max(), tr[:, 1].mean())
    ext_ns = haversine_m(tr[:, 0].mean(), tr[:, 1].min(), tr[:, 0].mean(), tr[:, 1].max())
    step = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    pt = prepare(partition, tid, with_labels=True)
    hit = np.zeros(len(pt.xy), bool)
    np.logical_or.at(hit, pt.cand.obs_idx, pt.y == 1)
    near = pt.cand.dist[:, 0]
    reliable = pt.label_dist <= 50
    return dict(
        partition=partition, trip_id=tid,
        track_bytes=(d / f"{tid}.track").stat().st_size, nodes_bytes=(d / f"{tid}.nodes").stat().st_size,
        arcs_bytes=(d / f"{tid}.arcs").stat().st_size, route_bytes=(d / f"{tid}.route").stat().st_size,
        n_obs=len(tr), n_nodes=V, n_arcs=E, route_len=len(rt), route_unique=len(np.unique(rt)),
        nan_values=int(np.isnan(tr).sum()), duplicate_timestamps=int((dt == 0).sum()),
        non_monotonic_timestamps=int((dt < 0).sum()), t_start=float(tr[0, 2]),
        duration_s=float(tr[-1, 2] - tr[0, 2]), pct_dt_1s=float(np.mean(np.isclose(dt, 1.0))),
        n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt60s=int((dt > 60).sum()), max_gap_s=float(dt.max()),
        max_step_m=float(step.max()), median_speed_mps=float(np.median(step / np.maximum(dt, 1e-3))),
        lon_c=float(np.median(tr[:, 0])), lat_c=float(np.median(tr[:, 1])),
        extent_ew_km=float(ext_ew / 1000), extent_ns_km=float(ext_ns / 1000),
        path_length_km=float(step.sum() / 1000),
        weak_components=int(n_weak), strong_components=int(n_strong), largest_scc_frac=float(big_strong),
        isolated_nodes=int((deg == 0).sum()), arcs_two_way_frac=float(np.mean(pt.net.reverse >= 0)),
        self_loops=int((ar[:, 0] == ar[:, 1]).sum()), zero_len_arcs=int((pt.net.length < 0.01).sum()),
        mean_arc_len_m=float(pt.net.length.mean()),
        route_disconnected_steps=disc, route_steps=route_steps,
        route_invalid_ids=int(((rt < 0) | (rt >= E)).sum()),
        nearest_arc_dist_median=float(np.median(near[np.isfinite(near)])),
        nearest_arc_dist_p95=float(np.percentile(near[np.isfinite(near)], 95)),
        no_candidate_obs=int((~pt.cand.valid[:, 0]).sum()),
        mean_candidates=float(pt.cand.valid.sum(1).mean()),
        candidate_miss_rate=float(1 - hit.mean()),
        candidate_miss_rate_reliable=float(1 - hit[reliable].mean()) if reliable.any() else np.nan,
        label_dist_median=float(np.median(pt.label_dist)), label_dist_p95=float(np.percentile(pt.label_dist, 95)),
        unreliable_label_frac=float(1 - reliable.mean()),
        nearest_arc_top1=float(np.mean(pt.cand.C[:, 0] == pt.label_arc)),
        t_network_s=pt.timings["network"], t_features_s=pt.timings["features"],
    )


def main():
    rows = []
    for part in (config.TRAIN, config.STUDENT_TEST):
        ids = list_trips(part)
        rows += Parallel(n_jobs=-1)(delayed(trip_stats)(part, t) for t in ids)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "eda_inventory.csv", index=False)

    summ = {}
    for part, g in df.groupby("partition"):
        summ[part] = dict(
            trips=len(g), observations=int(g.n_obs.sum()),
            obs_per_trip=g.n_obs.describe().round(1).to_dict(),
            nodes_per_trip=g.n_nodes.describe().round(1).to_dict(),
            arcs_per_trip=g.n_arcs.describe().round(1).to_dict(),
            pct_steps_1s=float((g.pct_dt_1s * (g.n_obs - 1)).sum() / (g.n_obs - 1).sum()),
            nan_values=int(g.nan_values.sum()), duplicate_timestamps=int(g.duplicate_timestamps.sum()),
            non_monotonic=int(g.non_monotonic_timestamps.sum()),
            trips_with_gap_gt60s=int((g.n_gaps_gt60s > 0).sum()), max_gap_s=float(g.max_gap_s.max()),
            route_disconnected_steps=int(g.route_disconnected_steps.sum()),
            route_steps=int(g.route_steps.sum()),
            trips_multi_weak_component=int((g.weak_components > 1).sum()),
            isolated_nodes=int(g.isolated_nodes.sum()),
            mean_candidate_miss_rate=float(g.candidate_miss_rate.mean()),
            mean_unreliable_label_frac=float(g.unreliable_label_frac.mean()),
            nearest_arc_top1_mean=float(g.nearest_arc_top1.mean()),
        )
    (config.RESULTS / "eda_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

    # ---------------- plots ----------------
    F = config.FIGURES
    colors = {"train": "#1565c0", "student_test": "#ef6c00"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for part, g in df.groupby("partition"):
        ax[0].hist(g.n_obs, bins=30, alpha=.7, label=part, color=colors[part])
        ax[1].hist(np.log10(g.n_arcs), bins=30, alpha=.7, label=part, color=colors[part])
        ax[2].scatter(g.path_length_km, g.duration_s / 60, s=18, label=part, color=colors[part])
    ax[0].set(title="GPS observations per trip", xlabel="observations", ylabel="trips")
    ax[1].set(title="Road-network size", xlabel="log10(#arcs)", ylabel="trips")
    ax[2].set(title="Trip length vs duration", xlabel="GPS path length (km)", ylabel="duration (min)", yscale="log")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(F / "eda_trip_sizes.png"); plt.close(fig)

    # sampling interval distribution (train)
    dts = []
    for tid in list_trips(config.TRAIN):
        tr = load_trip(config.TRAIN, tid).track
        dts.append(np.diff(tr[:, 2]))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    vals, cnts = np.unique(np.round(np.clip(dts, 0, 20)), return_counts=True)
    ax[0].bar(vals, cnts, color="#1565c0"); ax[0].set_yscale("log")
    ax[0].set(title="Sampling interval (train, clipped at 20 s)", xlabel="Δt (s)", ylabel="count (log)")
    big = dts[dts > 5]
    ax[1].hist(np.log10(big), bins=40, color="#c62828")
    ax[1].set(title=f"Gaps > 5 s (n={len(big)})", xlabel="log10 Δt (s)", ylabel="count")
    fig.tight_layout(); fig.savefig(F / "eda_sampling_intervals.png"); plt.close(fig)

    # world map of trip centroids
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for part, g in df.groupby("partition"):
        ax.scatter(g.lon_c, g.lat_c, s=25, label=part, color=colors[part], alpha=.8)
    ax.set(title="Trip centroids (lon/lat)", xlabel="longitude", ylabel="latitude")
    ax.legend(); fig.tight_layout(); fig.savefig(F / "eda_trip_locations.png"); plt.close(fig)

    # data quality / candidates
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].hist(df.nearest_arc_dist_median, bins=30, color="#6a1b9a")
    ax[0].set(title="Median GPS→nearest-arc distance", xlabel="m", ylabel="trips")
    ax[1].hist(df.unreliable_label_frac * 100, bins=30, color="#c62828")
    ax[1].set(title="Obs >50 m from aligned reference arc", xlabel="% of trip", ylabel="trips")
    ax[2].hist(df.candidate_miss_rate_reliable * 100, bins=30, color="#00838f")
    ax[2].set(title="Candidate miss rate (reliable labels)", xlabel="%", ylabel="trips")
    fig.tight_layout(); fig.savefig(F / "eda_quality.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].scatter(df.n_nodes, df.n_arcs, c=df.partition.map(colors), s=18)
    ax[0].set(xscale="log", yscale="log", title="Nodes vs arcs", xlabel="nodes", ylabel="arcs")
    ax[1].hist(df.arcs_two_way_frac, bins=25, color="#2e7d32")
    ax[1].set(title="Fraction of arcs with a reverse arc (two-way)", xlabel="fraction", ylabel="trips")
    fig.tight_layout(); fig.savefig(F / "eda_network.png"); plt.close(fig)

    # missing value table figure
    mv = df.groupby("partition")[["nan_values", "duplicate_timestamps", "non_monotonic_timestamps",
                                  "n_gaps_gt5s", "n_gaps_gt60s", "no_candidate_obs", "route_disconnected_steps",
                                  "isolated_nodes"]].sum()
    mv.to_csv(config.RESULTS / "eda_missing_values.csv")

    # trajectory panel plots (local coordinates with network)
    sel = ["00000034", "00000046", "00000002", "00000005"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, tid in zip(axs, sel):
        pt = prepare(config.TRAIN, tid, with_labels=True)
        net = pt.net
        act = net.active_ids
        x0, y0 = pt.xy.min(0) - 300; x1, y1 = pt.xy.max(0) + 300
        m = act[((net.a[act] >= [x0, y0]) & (net.a[act] <= [x1, y1])).all(1)]
        for e in m:
            ax.plot([net.a[e, 0], net.b[e, 0]], [net.a[e, 1], net.b[e, 1]], color="#bdbdbd", lw=.5, zorder=1)
        r = pt.trip.route
        for e in r:
            ax.plot([net.a[e, 0], net.b[e, 0]], [net.a[e, 1], net.b[e, 1]], color="#2e7d32", lw=2.5, zorder=2)
        ax.scatter(pt.xy[:, 0], pt.xy[:, 1], s=1.2, color="#1565c0", zorder=3)
        ax.set(title=f"train/{tid}  (N={len(pt.xy)})", aspect="equal", xlabel="x (m)", ylabel="y (m)")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    fig.tight_layout(); fig.savefig(F / "eda_trajectories.png"); plt.close(fig)

    # ---------------- Folium maps ----------------
    for part, tid in [(config.TRAIN, "00000034"), (config.TRAIN, "00000002"), (config.STUDENT_TEST, "00000056")]:
        pt = prepare(part, tid, with_labels=True)
        net = pt.net
        near = np.unique(pt.cand.C[pt.cand.valid])
        m = trip_map(net, pt.trip.track[:, :2][pt.raw_index] if False else load_trip(part, tid).track[:, :2],
                     reference=pt.trip.route, network_arcs=near, title=f"{part}/{tid}: raw data", obs_step=2)
        m.save(str(config.MAPS / f"eda_{part}_{tid}.html"))

    world = folium.Map(location=[30, 10], zoom_start=2, tiles="OpenStreetMap")
    for _, r in df.iterrows():
        folium.CircleMarker([r.lat_c, r.lon_c], radius=5, color=colors[r.partition], fill=True,
                            tooltip=f"{r.partition}/{r.trip_id}: {r.n_obs} obs, {r.path_length_km:.1f} km").add_to(world)
    world.save(str(config.MAPS / "eda_trip_locations.html"))
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
