"""Team wrapper for the proof-carrying exact joint teacher."""
from __future__ import annotations


import numpy as np

from autocross.api import Command, Controller, Observation, RoadModel, TrackInfo, VehicleParams
from autocross.optimal.control import command_from_result, solve_observation


class JointExactController(Controller):
    """Run the exact finite-horizon joint teacher on every policy observation."""

    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        cfg = self.config or {}
        self.horizon = max(int(cfg.get("horizon", 8)), 1)
        self.offsets = cfg.get("offsets", None)
        self.speeds = cfg.get("speeds", None)
        self.mode = str(cfg.get("mode", "exact")).lower()
        if self.mode not in {"exact", "realtime"}:
            raise ValueError("joint exact mode must be 'exact' or 'realtime'")
        self.realtime = bool(cfg.get("realtime", False) or self.mode == "realtime")
        self.cert_dir = cfg.get("cert_dir", None)

        # Exercise the same complete model -> DP -> certificate path before the
        # first live observation.  A one-transition declaration keeps startup
        # bounded while still warming imports, numpy, and the exact solver.
        speed = max(1.0, min(float(vehicle.max_speed), 2.0))
        speed_mm = int(round(speed * 1000.0))
        warm_road = RoadModel(
            valid=True, range_m=2.0, lateral_offset=0.0, heading_error=0.0,
            curvature=0.0, width_left=3.0, width_right=3.0,
            dist_to_left_boundary=3.0, dist_to_right_boundary=3.0,
            preview_s=np.asarray([0.0, 1.0, 2.0], dtype=float),
            preview_curvature=np.asarray([0.0, 0.0, 0.0], dtype=float),
            preview_centerline=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float),
            preview_width_left=np.asarray([3.0, 3.0, 3.0], dtype=float),
            preview_width_right=np.asarray([3.0, 3.0, 3.0], dtype=float),
            preview_left_boundary=np.asarray([[0.0, 3.0], [1.0, 3.0], [2.0, 3.0]], dtype=float),
            preview_right_boundary=np.asarray([[0.0, -3.0], [1.0, -3.0], [2.0, -3.0]], dtype=float),
            corners=[], obstacles=[], finish_distance=float("nan"),
        )
        warm = Observation(
            t=0.0, step=0, x=0.0, y=0.0, yaw=0.0, vx=speed, vy=0.0,
            yaw_rate=0.0, ax=0.0, ay=0.0, speed=speed, steer=0.0,
            throttle=0.0, brake=0.0, progress=0.0, lap=0, timing_started=False,
            road=warm_road,
        )
        solve_observation(warm, vehicle, horizon=1, offsets=(0,), speeds=(speed_mm * speed_mm,),
                          realtime=False, cert_dir=None)

    def step(self, obs: Observation) -> Command:
        road = getattr(obs, "road", None)
        # Avoid entering quantisation/DP with a malformed perception frame.
        if road is None or not getattr(road, "valid", False):
            return Command(
                brake=1.0, target_speed=0.0,
                debug={"planner": "exact-joint", "certified": False,
                       "claimed": None, "certificate_ok": False, "solver_ms": 0.0,
                       "horizon": float(self.horizon), "grid_offsets": 0.0,
                       "grid_speeds": 0.0, "realtime": self.realtime,
                       "realtime_exact": False, "fallback_reason": "invalid road"},
            )
        try:
            if len(road.preview_s) < 3 or len(road.preview_centerline) < 3:
                raise ValueError("short preview")
        except (AttributeError, TypeError, ValueError):
            return Command(
                brake=1.0, target_speed=0.0,
                debug={"planner": "exact-joint", "certified": False,
                       "claimed": None, "certificate_ok": False, "solver_ms": 0.0,
                       "horizon": float(self.horizon), "grid_offsets": 0.0,
                       "grid_speeds": 0.0, "realtime": self.realtime,
                       "realtime_exact": False, "fallback_reason": "short preview"},
            )
        result = solve_observation(
            obs, self.vehicle, horizon=self.horizon, offsets=self.offsets,
            speeds=self.speeds, realtime=self.realtime, cert_dir=self.cert_dir,
        )
        return command_from_result(obs, self.vehicle, result, realtime=self.realtime)


__all__ = ["JointExactController"]
