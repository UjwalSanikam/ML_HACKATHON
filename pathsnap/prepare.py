"""Per-trip preprocessing shared by training, evaluation and the API."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .features import CandidateTable, build_candidates
from .io import Trip, clean_track, load_route, load_trip
from .labels import observation_labels
from .network import RoadNetwork


@dataclass
class PreparedTrip:
    trip: Trip
    net: RoadNetwork
    xy: np.ndarray
    t: np.ndarray
    raw_index: np.ndarray
    clean_report: dict
    cand: CandidateTable
    timings: dict
    # label side (training / server-side evaluation only)
    label_arc: Optional[np.ndarray] = None
    label_pos: Optional[np.ndarray] = None
    label_dist: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None          # (M,) 1 if candidate == label arc


def prepare(partition: str, trip_id: str, with_labels: bool = False, root=None) -> PreparedTrip:
    tm = {}
    t0 = time.perf_counter()
    trip = load_trip(partition, trip_id, with_route=False, root=root)
    track, rep = clean_track(trip.track)
    tm["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    net = RoadNetwork.for_track(trip.nodes, trip.arcs, track[:, :2])
    xy = net.proj.forward(track[:, :2])
    tm["network"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cand = build_candidates(net, xy, track[:, 2])
    tm["features"] = time.perf_counter() - t0
    pt = PreparedTrip(trip, net, xy, track[:, 2], rep.pop("raw_index"), rep, cand, tm)

    if with_labels:
        attach_labels(pt, partition, trip_id, root)
    return pt


def attach_labels(pt: PreparedTrip, partition: str, trip_id: str, root=None):
    """Load the reference route *after* features are built and derive labels."""
    route = load_route(partition, trip_id, root)
    if route is None:
        return pt
    pt.trip.route = route
    t0 = time.perf_counter()
    la, lp, ld = observation_labels(pt.net, pt.xy, route)
    pt.timings["labels"] = time.perf_counter() - t0
    pt.label_arc, pt.label_pos, pt.label_dist = la, lp, ld
    pt.y = (pt.cand.arc == la[pt.cand.obs_idx]).astype(np.int8)
    return pt
