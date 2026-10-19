"""Behaviour-cloning controller for the IL experiment.

The pursuit and pedal equations below are a local copy of the expert formulas so
that a submission does not import the experiment expert at runtime.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams
from autocross.datagen.dataset import policy_features



class StudentBCController(Controller):
    """Predict lateral offset and longitudinal acceleration from canonical observations."""

    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        self._grid = np.arange(0.0, 11.0, 1.0)
        self._weights: dict[str, np.ndarray] = {}
        path = self.config.get("weights")
        if not path:
            raise ValueError("controller config requires weights path")
        with np.load(Path(path), allow_pickle=False) as z:
            metadata: dict[str, object] = {}
            if "metadata" in z.files:
                raw_metadata = np.asarray(z["metadata"])
                if raw_metadata.ndim == 0:
                    raw_metadata = raw_metadata.item()
                if isinstance(raw_metadata, bytes):
                    raw_metadata = raw_metadata.decode()
                if isinstance(raw_metadata, str):
                    parsed = json.loads(raw_metadata)
                    if isinstance(parsed, dict):
                        metadata = parsed
            self._weights = {
                k: np.asarray(z[k], dtype=float) for k in z.files if k != "metadata"
            }
        self._grid = self._weights.get("grid", self._grid)
        self._mean = self._weights.get("x_mean", self._weights.get("mean", np.zeros(0)))
        self._std = self._weights.get("x_std", self._weights.get("std", np.ones(0)))
        self._seed = self.config.get("seed", metadata.get("seed", 0))
        self._budget = self.config.get("budget", metadata.get("budget", 0))
        self._condition = self.config.get("condition", metadata.get("condition", ""))
        self._certified = False

    def _predict(self, x: np.ndarray) -> tuple[float, float]:
        """Evaluate the trainer's shared ReLU trunk and offset/accel heads."""
        if self._mean.size == x.size:
            x = (x - self._mean) / np.maximum(self._std, 1e-8)
        if "W_offset" in self._weights and "W_accel" in self._weights:
            h = x
            i = 0
            while f"W{i}" in self._weights:
                h = h @ self._weights[f"W{i}"] + self._weights.get(f"b{i}", 0.0)
                h = np.maximum(h, 0.0)
                i += 1
            off = h @ self._weights["W_offset"] + self._weights.get("b_offset", 0.0)
            acc = h @ self._weights["W_accel"] + self._weights.get("b_accel", 0.0)
            return float(np.ravel(off)[0]), float(np.ravel(acc)[0])
        wkeys = sorted((k for k in self._weights if k.lower().startswith(("w", "weight"))),
                       key=lambda k: int("".join(c for c in k if c.isdigit()) or 0))
        bkeys = sorted((k for k in self._weights if k.lower().startswith(("b", "bias"))),
                       key=lambda k: int("".join(c for c in k if c.isdigit()) or 0))
        h = x
        for i, wk in enumerate(wkeys):
            h = np.asarray(h @ self._weights[wk] + (self._weights[bkeys[i]] if i < len(bkeys) else 0.0))
            if i + 1 < len(wkeys):
                h = np.maximum(h, 0.0)
        h = np.ravel(h)
        return (float(h[0]) if len(h) else 0.0, float(h[1]) if len(h) > 1 else 0.0)

    @staticmethod
    def _pursuit(path_xy: np.ndarray, s: np.ndarray, v: float, vp: VehicleParams) -> tuple[float, float]:
        ld = float(np.clip(0.0 + 0.45 * v, 3.0, 14.0))
        tx = float(np.interp(ld, s, path_xy[:, 0]))
        ty = float(np.interp(ld, s, path_xy[:, 1]))
        alpha = math.atan2(ty, max(tx, 0.1))
        delta = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), max(math.hypot(tx, ty), 1.0))
        return float(np.clip(delta / vp.max_steer, -1.0, 1.0)), alpha

    @staticmethod
    def _pedals(a_des: float, vp: VehicleParams, ay: float) -> tuple[float, float]:
        if a_des >= 0.0:
            lat_use = min(abs(ay) / (vp.mu * 9.81), 1.0)
            return float(min(a_des / vp.max_accel, max(1.0 - lat_use ** 2, 0.15), 1.0)), 0.0
        return 0.0, float(np.clip((-a_des - 0.2) / vp.max_decel, 0.0, 1.0))

    def step(self, obs: Observation) -> Command:
        r = obs.road
        if r is None or not r.valid or len(r.preview_s) < 2:
            return Command(brake=1.0, debug=self._debug())
        pol = {"schema": "autocross.datagen.policy_observation/1",
               **{k: float(getattr(obs, k)) for k in ("speed", "vx", "vy", "yaw_rate", "ax", "ay", "steer", "throttle", "brake")}}
        scalar = ("valid", "range_m", "lateral_offset", "heading_error", "curvature", "width_left", "width_right",
                  "dist_to_left_boundary", "dist_to_right_boundary")
        vector = ("preview_s", "preview_curvature", "preview_width_left", "preview_width_right")
        points = ("preview_centerline", "preview_left_boundary", "preview_right_boundary")
        road = {k: (bool(getattr(r, k)) if k == "valid" else float(getattr(r, k))) for k in scalar}
        road.update({k: np.asarray(getattr(r, k), dtype=float).tolist() for k in vector})
        road.update({k: np.asarray(getattr(r, k), dtype=float).reshape(-1, 2).tolist() for k in points})
        road["obstacles"] = [{"distance": float(o.distance), "lateral_min": float(o.lateral_min),
                              "lateral_max": float(o.lateral_max), "center": np.asarray(o.center, dtype=float).tolist(),
                              "points": np.asarray(o.points, dtype=float).reshape(-1, 2).tolist()} for o in r.obstacles]
        pol["road"] = road
        off, accel = self._predict(policy_features(pol, self._grid))
        off = float(np.clip(off, -r.width_right, r.width_left))
        s, path = np.asarray(r.preview_s, float), np.asarray(r.preview_centerline, float).copy()
        q = np.clip(s / 5.0, 0.0, 1.0); q = q * q * (3.0 - 2.0 * q)
        path[:, 1] += (1.0 - q) * float(r.lateral_offset) + q * off
        # Startup floor: training states are on the declared teacher speed grid
        # (>= 2 m/s); from a standstill the regressor has no useful gradient, so
        # the actuation layer supplies a bounded launch acceleration.  This is
        # the same kind of declared tracking reserve as the expert's pedal map.
        if obs.speed < 2.0:
            accel = max(accel, 2.0)
        steer, alpha = self._pursuit(path, s, max(obs.speed, 0.0), self.vehicle)
        throttle, brake = self._pedals(accel, self.vehicle, obs.ay)
        return Command(steer=steer, throttle=throttle, brake=brake, path=path, target_speed=None, debug={**self._debug(), "offset": off, "accel": accel, "alpha": alpha})

    def _debug(self) -> dict[str, object]:
        return {"planner": "bc-student", "condition": self._condition, "seed": self._seed,
                "budget": self._budget, "certified": False}
