"""2D geometry helpers. All coordinates are in the right-handed ENU world frame
(x forward/east, y left/north, yaw counter-clockwise from +x, radians)."""
from __future__ import annotations

import math

import numpy as np


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def world_to_body(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Transform (N,2) world points into the vehicle frame (x forward, y left)."""
    d = np.asarray(points, dtype=float).reshape(-1, 2) - (x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]))


def body_to_world(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack((x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]))


def arclength(pts: np.ndarray, closed: bool = False) -> np.ndarray:
    p = np.vstack((pts, pts[:1])) if closed else pts
    return np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))


def resample(pts: np.ndarray, ds: float, closed: bool = False) -> np.ndarray:
    """Resample a polyline at (approximately) uniform spacing ds."""
    pts = np.asarray(pts, dtype=float)
    p = np.vstack((pts, pts[:1])) if closed else pts
    s = arclength(pts, closed)
    n = max(int(round(s[-1] / ds)), 2)
    si = np.linspace(0.0, s[-1], n + 1)
    if closed:
        si = si[:-1]
    return np.column_stack((np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])))


def headings(pts: np.ndarray, closed: bool = False) -> np.ndarray:
    if closed:
        d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    else:
        d = np.gradient(pts, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def curvature(pts: np.ndarray, closed: bool = False) -> np.ndarray:
    """Signed Menger curvature (+ = turning left); open-polyline endpoints get 0."""
    pts = np.asarray(pts, dtype=float)
    if closed:
        a, c = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
    else:
        a, c = np.vstack((pts[:1], pts[:-1])), np.vstack((pts[1:], pts[-1:]))
    b = pts
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    den = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ac.T)
    return np.where(den > 1e-12, 2.0 * cross / np.maximum(den, 1e-12), 0.0)


def smooth_closed(pts: np.ndarray, window: int) -> np.ndarray:
    k = np.ones(window) / window
    ext = np.vstack((pts[-window:], pts, pts[:window]))
    out = np.column_stack([np.convolve(ext[:, i], k, mode="same") for i in range(2)])
    return out[window:-window]


class Polyline:
    """Polyline with arc-length parameterization and windowed projection."""

    def __init__(self, pts: np.ndarray, closed: bool = False):
        self.pts = np.asarray(pts, dtype=float)
        self.closed = closed
        p = np.vstack((self.pts, self.pts[:1])) if closed else self.pts
        self._a = p[:-1]
        self._d = p[1:] - p[:-1]
        self._len = np.hypot(self._d[:, 0], self._d[:, 1])
        self._len2 = np.maximum(self._len ** 2, 1e-12)
        self.s = np.concatenate(([0.0], np.cumsum(self._len)))
        self.length = float(self.s[-1])
        self.n_seg = len(self._a)

    def _segment_ids(self, s_hint, window):
        if s_hint is None or window is None:
            return np.arange(self.n_seg)
        i0 = int(np.searchsorted(self.s, (s_hint % self.length) if self.closed else s_hint)) - 1
        mean_len = self.length / self.n_seg
        k = int(math.ceil(window / mean_len)) + 1
        ids = np.arange(i0 - k, i0 + k + 1)
        if self.closed:
            return ids % self.n_seg
        return ids[(ids >= 0) & (ids < self.n_seg)]

    def project(self, p, s_hint: float | None = None, window: float | None = None):
        """Return (s, lateral, segment_index). lateral > 0 means p is left of the line."""
        ids = self._segment_ids(s_hint, window)
        a, d = self._a[ids], self._d[ids]
        w = np.asarray(p, dtype=float) - a
        t = np.clip((w * d).sum(1) / self._len2[ids], 0.0, 1.0)
        q = a + d * t[:, None]
        dist2 = ((q - p) ** 2).sum(1)
        j = int(np.argmin(dist2))
        i = int(ids[j])
        cross = d[j, 0] * w[j, 1] - d[j, 1] * w[j, 0]
        lat = math.copysign(math.sqrt(dist2[j]), cross) if dist2[j] > 0 else 0.0
        return float(self.s[i] + t[j] * self._len[i]), lat, i

    def distance(self, p) -> float:
        w = np.asarray(p, dtype=float) - self._a
        t = np.clip((w * self._d).sum(1) / self._len2, 0.0, 1.0)
        q = self._a + self._d * t[:, None]
        return float(np.sqrt(((q - p) ** 2).sum(1).min()))

    def point_at(self, s: float) -> np.ndarray:
        if self.closed:
            s = s % self.length
        s = min(max(s, 0.0), self.length)
        i = min(int(np.searchsorted(self.s, s, side="right")) - 1, self.n_seg - 1)
        t = (s - self.s[i]) / max(self._len[i], 1e-12)
        return self._a[i] + self._d[i] * t

    def heading_at(self, s: float) -> float:
        if self.closed:
            s = s % self.length
        i = min(max(int(np.searchsorted(self.s, s, side="right")) - 1, 0), self.n_seg - 1)
        return math.atan2(self._d[i, 1], self._d[i, 0])


def polyline_min_distance(path: np.ndarray, p) -> float:
    """Distance from point p to an open polyline (N,2); N==1 degrades to point distance."""
    path = np.asarray(path, dtype=float)
    if len(path) == 1:
        return float(np.hypot(*(path[0] - p)))
    a, d = path[:-1], np.diff(path, axis=0)
    w = np.asarray(p, dtype=float) - a
    t = np.clip((w * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
    q = a + d * t[:, None]
    return float(np.sqrt(((q - p) ** 2).sum(1).min()))


def polyline_window_distance(path: np.ndarray, s_lo: float, s_hi: float, p) -> float:
    """Distance from p to the part of polyline ``path`` between arc lengths s_lo and s_hi.
    If the polyline ends before s_lo, the distance to its end point is returned."""
    path = np.asarray(path, dtype=float)
    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
    if s[-1] <= s_lo:
        return float(np.hypot(*(path[-1] - p)))
    lo, hi = max(s_lo, 0.0), min(s_hi, s[-1])
    inner = path[(s > lo) & (s < hi)]
    ends = [np.array([np.interp(v, s, path[:, 0]), np.interp(v, s, path[:, 1])]) for v in (lo, hi)]
    return polyline_min_distance(np.vstack([ends[0], inner, ends[1]]), p)


def path_is_plausible(path: np.ndarray, max_curvature: float = 0.5, min_length: float = 2.0) -> bool:
    """A drivable-looking path: long enough, no folds or zig-zags tighter than ``max_curvature``
    (1/m, sampled every 1 m)."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return False
    seg = np.hypot(*np.diff(path, axis=0).T)
    if seg.sum() < min_length:
        return False
    keep = np.concatenate(([True], seg > 1e-6))  # drop repeated points
    q = resample(path[keep], 1.0, False) if seg.sum() >= 3.0 else path
    if len(q) < 3:
        return True
    h = np.unwrap(np.arctan2(*np.diff(q, axis=0).T[::-1]))
    step = np.hypot(*np.diff(q, axis=0).T)
    return bool(np.all(np.abs(np.diff(h)) <= max_curvature * np.maximum(step[1:], 1e-6) + 1e-9))


def points_in_obb(pts: np.ndarray, x: float, y: float, yaw: float,
                  half_length: float, half_width: float, margin: float = 0.0) -> np.ndarray:
    b = world_to_body(pts, x, y, yaw)
    return (np.abs(b[:, 0]) <= half_length + margin) & (np.abs(b[:, 1]) <= half_width + margin)


def segments_intersect_any(pts: np.ndarray, min_index_gap: int = 3, closed: bool = False) -> bool:
    """True if a polyline self-intersects (ignoring near-neighbour segments)."""
    p = np.vstack((pts, pts[:1])) if closed else pts
    a, b = p[:-1], p[1:]
    n = len(a)
    for i in range(n):
        j = np.arange(i + min_index_gap, n)
        if closed:
            j = j[(n - (j - i)) >= min_index_gap]
        if len(j) == 0:
            continue
        c, d = a[j], b[j]
        d1 = _orient(c, d, a[i]); d2 = _orient(c, d, b[i])
        d3 = _orient(a[i], b[i], c); d4 = _orient(a[i], b[i], d)
        if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
            return True
    return False


def _orient(a, b, c):
    a, b, c = np.atleast_2d(a), np.atleast_2d(b), np.atleast_2d(c)
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
