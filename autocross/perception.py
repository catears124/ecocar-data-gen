"""Ground-truth-derived perception outputs. Deliberately cheap (no rendering) so the whole
harness fits on a laptop.

* ``RoadModelSensor`` - what a production perception stack hands to planning: a road/lane
  model of the course within range (centerline offset, heading error, curvature, widths,
  preview geometry, corners, obstacles). Teams do not need to do any sensor fusion.
* ``ConeSensor`` - raw cone detections, for teams that want to go lower level.

Noise is configurable to test robustness against imperfect perception."""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g
from .api import ConeDetections, Corner, ObstacleInfo, RoadModel


class RoadModelSensor:
    def __init__(self, track, cfg: dict, rng: np.random.Generator):
        self.track = track
        self.range = float(cfg.get("range_m", 40.0))
        self.step = float(cfg.get("step_m", 1.0))
        self.sig_off = float(cfg.get("offset_noise_std_m", 0.0))
        self.sig_head = math.radians(float(cfg.get("heading_noise_std_deg", 0.0)))
        self.sig_curv = float(cfg.get("curvature_noise_std", 0.0))
        self.sig_obs = float(cfg.get("obstacle_noise_std_m", 0.0))
        self.rng = rng
        self.corners = track.find_corners(float(cfg.get("corner_min_radius_m", 80.0)),
                                          float(cfg.get("corner_min_angle_deg", 15.0)))
        self.s_hint = track.spawn_s

    def _rel(self, s, s_v):
        """Signed distance along the course from station s_v to station s."""
        d = np.asarray(s, dtype=float) - s_v
        if self.track.closed:
            L = self.track.length
            d = (d + L / 2) % L - L / 2
        return d

    def measure(self, x: float, y: float, yaw: float, progress: float) -> RoadModel:
        tr = self.track
        s_v, lat, _ = tr.line.project(np.array([x, y]), self.s_hint, 25.0)
        self.s_hint = s_v
        d = np.arange(0.0, self.range + 1e-9, self.step)
        pts, hd, kap, wl, wr, valid = tr.sample(s_v + d)
        d, pts, hd, kap, wl, wr = d[valid], pts[valid], hd[valid], kap[valid], wl[valid], wr[valid]
        if len(d) < 2:
            e = np.zeros((0, 2))
            return RoadModel(False, self.range, lat, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros(0), np.zeros(0), e,
                             np.zeros(0), np.zeros(0), e, e, [], [], float("nan"))
        # coherent lane-model error: offset + heading + curvature error, growing with distance
        e0 = self.rng.normal(0.0, self.sig_off) if self.sig_off else 0.0
        eh = self.rng.normal(0.0, self.sig_head) if self.sig_head else 0.0
        ek = self.rng.normal(0.0, self.sig_curv) if self.sig_curv else 0.0
        err = e0 + eh * d + 0.5 * ek * d ** 2
        nrm = np.column_stack((-np.sin(hd), np.cos(hd)))
        cl = pts + nrm * err[:, None]
        lb = pts + nrm * (wl + err)[:, None]
        rb = pts - nrm * (wr - err)[:, None]
        kap = kap + ek
        lateral = lat - e0
        heading_error = float(g.wrap_angle(yaw - hd[0] - eh))

        corners = []
        for c in self.corners:
            d_in = float(self._rel(c["entry"], s_v))
            d_out = d_in + (c["exit"] - c["entry"])
            if d_out < 0.0 or d_in > d[-1]:
                continue
            m = (d >= max(d_in, 0.0)) & (d <= min(d_out, d[-1]))
            if not m.any():
                continue
            kk, dd = kap[m], d[m]
            angle = float(np.sum(0.5 * (kk[1:] + kk[:-1]) * np.diff(dd))) if len(dd) > 1 else float(kk[0] * self.step)
            d_apex = d_in + (c["apex"] - c["entry"])
            corners.append(Corner(
                direction=c["direction"], entry_distance=d_in,
                apex_distance=d_apex if d_apex <= d[-1] else float("nan"),
                exit_distance=d_out if d_out <= d[-1] else float("nan"),
                min_radius=float(1.0 / max(np.abs(kk).max(), 1e-9)), turn_angle=angle,
                fully_visible=d_out <= d[-1]))
        corners.sort(key=lambda c: c.entry_distance)

        obstacles = []
        if len(tr.obstacle_cones):
            od = self._rel(tr.obstacle_s, s_v)
            vis = (od >= -3.0) & (od <= d[-1])
            for gid in np.unique(tr.obstacle_gid[vis]):
                m = vis & (tr.obstacle_gid == gid)
                body = g.world_to_body(tr.obstacle_cones[m], x, y, yaw)
                if self.sig_obs:
                    body = body + self.rng.normal(0.0, self.sig_obs, body.shape)
                dist = float(od[m].mean())
                ol = tr.obstacle_lat[m] - np.interp(dist, d, err)
                obstacles.append(ObstacleInfo(dist, float(ol.min()), float(ol.max()), body.mean(0), body))
            obstacles.sort(key=lambda o: o.distance)

        fd = tr.finish_progress - progress
        body = lambda p: g.world_to_body(p, x, y, yaw)  # noqa: E731
        return RoadModel(
            valid=True, range_m=float(d[-1]), lateral_offset=float(lateral), heading_error=heading_error,
            curvature=float(kap[0]), width_left=float(wl[0]), width_right=float(wr[0]),
            dist_to_left_boundary=float(wl[0] - lateral), dist_to_right_boundary=float(wr[0] + lateral),
            preview_s=d, preview_curvature=kap, preview_centerline=body(cl), preview_width_left=wl,
            preview_width_right=wr, preview_left_boundary=body(lb), preview_right_boundary=body(rb),
            corners=corners, obstacles=obstacles,
            finish_distance=float(fd) if 0.0 <= fd <= d[-1] else float("nan"))


class ConeSensor:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.enabled = bool(cfg.get("enabled", True))
        self.range = float(cfg.get("range_m", 30.0))
        self.fov = math.radians(float(cfg.get("fov_deg", 140.0)))
        self.noise = float(cfg.get("noise_std_m", 0.03))
        self.noise_per_m = float(cfg.get("noise_std_per_m", 0.004))
        self.dropout = float(cfg.get("dropout", 0.0))
        self.misclass = float(cfg.get("misclassification", 0.0))
        self.mount_x = float(cfg.get("mount_x_m", 1.5))
        self.rng = rng

    def detect(self, cones: np.ndarray, colors: np.ndarray, x: float, y: float, yaw: float) -> ConeDetections | None:
        if not self.enabled:
            return None
        b = g.world_to_body(cones, x, y, yaw)
        rel = b - (self.mount_x, 0.0)
        r = np.hypot(rel[:, 0], rel[:, 1])
        vis = (r <= self.range) & (np.abs(np.arctan2(rel[:, 1], rel[:, 0])) <= self.fov / 2)
        if self.dropout > 0:
            vis &= self.rng.random(len(vis)) >= self.dropout
        pos, col, rr = b[vis], colors[vis].copy(), r[vis]
        if len(pos):
            std = (self.noise + self.noise_per_m * rr)[:, None]
            pos = pos + self.rng.normal(0.0, 1.0, pos.shape) * std
            if self.misclass > 0:
                flip = self.rng.random(len(col)) < self.misclass
                col[flip] = self.rng.integers(0, 3, int(flip.sum()))
        return ConeDetections(pos, col, self.range, self.fov)
