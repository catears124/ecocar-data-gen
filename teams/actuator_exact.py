"""Direct actuator-space exact controller.

Each step builds the declared finite actuator model from the current policy
observation, solves the exact DP over every declared state/action pair, and
publishes the first action of an optimal sequence through the deterministic
pedal map.  No pursuit/tracking layer sits between the optimal action and the
command.

The formal claim is for the finite model (see docs/actuator_controller.md): the
first action is optimal for the declared actuator dynamics, grids, horizon and
objective.  Simulator closed-loop behavior is measured, not proved.
"""
from __future__ import annotations

import math
import time

import numpy as np

from autocross.api import Command, Controller, Observation, RoadModel, TrackInfo, VehicleParams
from certline import CertifiedRacingLine
from autocross.actuator.build import command_from_first, model_from_observation
try:
    from autocross.actuator.numba_solver import solve_numba
except Exception:  # pragma: no cover
    solve_numba = None
from autocross.actuator.solver import solve


class ActuatorExactController(Controller):
    """Receding-horizon exact controller in steering/acceleration space."""

    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        cfg = self.config or {}
        self.horizon = max(1, int(cfg.get("horizon", 4)))
        self.offsets = cfg.get("offsets", None)
        self.psis = cfg.get("psis", None)
        self.speeds = cfg.get("speeds", None)
        self.deltas = cfg.get("deltas", None)
        self.steer_inputs = cfg.get("steer_inputs", None)
        self.acc_inputs = cfg.get("acc_inputs", None)
        self.terminal_distance = int(cfg.get("terminal_distance", 8000))
        self.steer_tau_s = float(cfg.get("steer_tau_s", 0.06))
        self._last = None
        self._solver_name = "numpy"
        self._solver_error = ""
        self._fallback = CertifiedRacingLine({})
        self._fallback.on_start(track, vehicle)
        self._warm()

    def _warm(self) -> None:
        vp = self.vehicle
        speed = max(2.0, min(float(vp.max_speed), 4.0))
        road = RoadModel(
            valid=True, range_m=3.0, lateral_offset=0.0, heading_error=0.0,
            curvature=0.0, width_left=3.0, width_right=3.0,
            dist_to_left_boundary=3.0, dist_to_right_boundary=3.0,
            preview_s=np.arange(0.0, 21.0),
            preview_curvature=np.zeros(21), preview_centerline=np.column_stack((np.arange(21.0), np.zeros(21))),
            preview_width_left=np.full(21, 3.0), preview_width_right=np.full(21, 3.0),
            preview_left_boundary=np.column_stack((np.arange(21.0), np.full(21, 3.0))),
            preview_right_boundary=np.column_stack((np.arange(21.0), np.full(21, -3.0))),
            corners=[], obstacles=[], finish_distance=float("nan"))
        obs = Observation(t=0.0, step=0, x=0.0, y=0.0, yaw=0.0, vx=speed, vy=0.0,
                          yaw_rate=0.0, ax=0.0, ay=0.0, speed=speed, steer=0.0,
                          throttle=0.0, brake=0.0, progress=0.0, lap=0,
                          timing_started=False, road=road)
        self._plan(obs)

    def _plan(self, obs: Observation, horizon: int | None = None):
        m = model_from_observation(
            obs, self.vehicle, horizon=self.horizon if horizon is None else horizon,
            offsets=self.offsets, psis=self.psis, speeds=self.speeds,
            deltas=self.deltas, steer_inputs=self.steer_inputs,
            acc_inputs=self.acc_inputs, terminal_distance=self.terminal_distance,
            steer_tau_s=self.steer_tau_s)
        if m is None:
            return None
        t0 = time.perf_counter()
        if solve_numba is not None:
            try:
                first_t, cost, first_action, diag = solve_numba(m)
                self._solver_name = "numba"
            except Exception as exc:
                self._solver_error = f"{type(exc).__name__}: {exc}"
                first_t, cost, first_action, diag = solve(m)
                self._solver_name = "numpy-fallback"
        else:
            first_t, cost, first_action, diag = solve(m)
            self._solver_name = "numpy"
        elapsed = (time.perf_counter() - t0) * 1000.0
        return m, first_t, cost, first_action, diag, elapsed

    def step(self, obs: Observation) -> Command:
        road = getattr(obs, "road", None)
        if road is None or not getattr(road, "valid", False) or len(getattr(road, "preview_s", [])) < 3:
            return Command(brake=1.0, debug={"planner": "actuator-exact", "certified": False,
                                             "fallback_reason": "invalid/short preview"})
        # Safety governor: outside the declared actuator grid, or with a large
        # lateral/heading error, use the certified-racing-line baseline.  This
        # is an explicit fallback; it is never labelled actuator-optimal.
        lat = abs(float(getattr(road, "lateral_offset", 0.0)))
        head = abs(float(getattr(road, "heading_error", 0.0)))
        if lat > 0.25 or head > 0.025:
            cmd = self._fallback.step(obs)
            cmd.debug.update({"planner": "certline-fallback", "certified": False,
                              "fallback_reason": "outside actuator safety envelope"})
            return cmd
        result = self._plan(obs)
        if result is None or result[3] is None:
            # Uncertified centreline recovery, explicitly labelled.
            path = np.asarray(road.preview_centerline, dtype=float)
            s = np.asarray(road.preview_s, dtype=float)
            try:
                from autocross.datagen.expert import _pursuit
                steer_cmd, _ld = _pursuit(path, s, max(float(obs.speed), 0.0), self.vehicle)
            except Exception:
                steer_cmd = 0.0
            target = 3.0
            err = target - max(float(obs.speed), 0.0)
            a = max(-self.vehicle.max_decel, min(self.vehicle.max_accel, 1.5 * err))
            if a >= 0:
                throttle = min(a / self.vehicle.max_accel, 1.0); brake = 0.0
            else:
                throttle = 0.0; brake = min((-a) / self.vehicle.max_decel, 1.0)
            return Command(steer=steer_cmd, throttle=throttle, brake=brake,
                           path=path, path_frame="vehicle", target_speed=target,
                           debug={"planner": "actuator-recovery", "certified": False,
                                  "fallback_reason": "no admissible exact actuator plan"})
        m, t, cost, (u, a), diag, elapsed = result
        cmd = command_from_first(obs, self.vehicle, m, u, a)
        cmd.debug.update({"cost": float(cost) if cost is not None else -1.0,
                          "solver_ms": float(elapsed), "states": float(diag["states"]),
                          "actions": float(diag["actions"]),
                          "solver_code": 1.0 if self._solver_name == "numba" else (2.0 if self._solver_name == "numpy-fallback" else 0.0),
                          "solver_error_len": float(len(self._solver_error))})
        return cmd
