"""Raw file readers. Files are whitespace separated with no header.

.track  : lon, lat, t(seconds)
.nodes  : lon, lat               (0-based row index = node id)
.arcs   : from_node, to_node      (0-based row index = arc id, directed)
.route  : arc_id per line         (chronological reference route)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import config


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (V, 2) lon, lat
    arcs: np.ndarray             # (E, 2) int from, to
    route: Optional[np.ndarray] = None  # (L,) int arc ids, server side only
    meta: dict = field(default_factory=dict)


def _read(path: Path, dtype, ncols: int) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c")
    return df.to_numpy()


def trip_dir(partition: str, trip_id: str, root: Path = None) -> Path:
    return Path(root or config.DATA_ROOT) / partition / trip_id


def list_trips(partition: str, root: Path = None) -> List[str]:
    p = Path(root or config.DATA_ROOT) / partition
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith("."))


def load_trip(partition: str, trip_id: str, with_route: bool = False, root: Path = None) -> Trip:
    """Load one trip. The reference route is only read when explicitly asked for,
    so the prediction pipeline can never accidentally see it."""
    d = trip_dir(partition, trip_id, root)
    track = _read(d / f"{trip_id}.track", np.float64, 3)
    nodes = _read(d / f"{trip_id}.nodes", np.float64, 2)
    arcs = _read(d / f"{trip_id}.arcs", np.int64, 2)
    route = None
    if with_route:
        r = _read(d / f"{trip_id}.route", np.int64, 1)
        route = r.ravel() if r.size else None
    return Trip(trip_id, partition, track, nodes, arcs, route)


def load_route(partition: str, trip_id: str, root: Path = None) -> Optional[np.ndarray]:
    d = trip_dir(partition, trip_id, root)
    r = _read(d / f"{trip_id}.route", np.int64, 1)
    return r.ravel() if r.size else None


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Basic GPS cleaning. Returns cleaned track and a report.

    * drops rows with NaN / out-of-range coordinates
    * sorts by timestamp (stable) and drops exact duplicate timestamps
    Index mapping back to the raw rows is kept in the report so predictions can
    be returned for every raw observation.
    """
    n0 = len(track)
    keep = np.isfinite(track).all(axis=1)
    keep &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    idx = np.nonzero(keep)[0]
    t = track[idx, 2]
    order = np.argsort(t, kind="stable")
    idx = idx[order]
    t = track[idx, 2]
    dup = np.zeros(len(idx), bool)
    dup[1:] = np.diff(t) <= 0
    idx = idx[~dup]
    rep = dict(raw=n0, invalid=int(n0 - keep.sum()), unsorted=int((np.diff(track[:, 2]) < 0).sum()) if n0 > 1 else 0,
               duplicate_time=int(dup.sum()), kept=int(len(idx)))
    return track[idx], dict(rep, raw_index=idx)
