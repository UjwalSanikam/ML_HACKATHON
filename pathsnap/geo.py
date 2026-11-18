"""Geodesy helpers.

Trips come from different cities around the world, so we project every trip
into its own local tangent plane (equirectangular projection centred on the
trip). For extents of a few tens of km the distortion is < 0.1 % which is far
below GPS noise, and it lets us use fast planar geometry in metres.
Haversine is used for validation / reporting.
"""
from __future__ import annotations

import numpy as np

from .config import EARTH_RADIUS_M


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * EARTH_RADIUS_M

    @classmethod
    def from_points(cls, lonlat: np.ndarray) -> "LocalProjection":
        return cls(np.median(lonlat[:, 0]), np.median(lonlat[:, 1]))

    def forward(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, float)
        x = (lonlat[..., 0] - self.lon0) * self.kx
        y = (lonlat[..., 1] - self.lat0) * self.ky
        return np.stack([x, y], axis=-1)

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        lon = xy[..., 0] / self.kx + self.lon0
        lat = xy[..., 1] / self.ky + self.lat0
        return np.stack([lon, lat], axis=-1)


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised point-to-segment projection.

    p, a, b broadcastable (..., 2). Returns (distance, t in [0,1], projected point).
    """
    ab = b - a
    ap = p - a
    L2 = (ab ** 2).sum(-1)
    t = np.where(L2 > 0, (ap * ab).sum(-1) / np.where(L2 > 0, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab
    d = np.sqrt(((p - proj) ** 2).sum(-1))
    return d, t, proj


def bearing_deg(dxdy: np.ndarray) -> np.ndarray:
    """Compass-like bearing of a planar vector (0 = north, clockwise)."""
    return (np.degrees(np.arctan2(dxdy[..., 0], dxdy[..., 1])) + 360.0) % 360.0


def angle_diff_deg(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
