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
        self.emit_certificates = bool(cfg.get("emit_certificates", False))
        self._certificate = None
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

    def _plan(self, obs: Observation, horizon: int | None = None,
              ref_d=None, ref_psi=None):
        m = model_from_observation(
            obs, self.vehicle, horizon=self.horizon if horizon is None else horizon,
            offsets=self.offsets, psis=self.psis, speeds=self.speeds,
            deltas=self.deltas, steer_inputs=self.steer_inputs,
            acc_inputs=self.acc_inputs, terminal_distance=self.terminal_distance,
            steer_tau_s=self.steer_tau_s, ref_d=ref_d, ref_psi=ref_psi)
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
        if self.emit_certificates and first_action is not None:
            try:
                from autocross.actuator.certificate import (
                    certificate_from_diag, check_certificate,
                )
                from autocross.actuator.solver import solve_sparse
                _bt, scost, saction, sdiag = solve_sparse(m)
                cert = certificate_from_diag(m, sdiag, claimed=scost, first_action=saction)
                rep = check_certificate(cert)
                self._certificate = cert if rep.get("ok") and scost == cost else None
            except Exception as exc:
                self._solver_error = f"certificate: {type(exc).__name__}: {exc}"
                self._certificate = None
        else:
            self._certificate = None
        return m, first_t, cost, first_action, diag, elapsed

    @staticmethod
    def _reference(road: RoadModel, cmd: Command):
        """Certline's published racing line as a lateral/heading prior.

        The exact DP still chooses the actuator sequence; this prior only
        changes the declared objective (tracking a fast reference) and is
        encoded in the certificate instance.  It is not a tracking controller.
        """
        try:
            cl = np.asarray(road.preview_centerline, dtype=float)
            path = getattr(cmd, "path", None)
            if path is None or cl.ndim != 2 or cl.shape[0] < 2:
                return None, None
            path = np.asarray(path, dtype=float)
            if path.shape != cl.shape:
                return None, None
            hd = np.arctan2(np.gradient(cl[:, 1]), np.gradient(cl[:, 0]))
            nrm = np.column_stack((-np.sin(hd), np.cos(hd)))
            off_mm = np.sum((path - cl) * nrm, axis=1) * 1000.0
            s = np.asarray(road.preview_s, dtype=float)
            if s.size != off_mm.size or np.any(np.diff(s) <= 0):
                psi = np.zeros_like(off_mm)
            else:
                psi = np.gradient(off_mm, s) * 1000.0
            ref_d = tuple(int(round(max(-1500.0, min(1500.0, x)))) for x in off_mm)
            ref_psi = tuple(int(round(max(-100000.0, min(100000.0, x)))) for x in psi)
            return ref_d, ref_psi
        except Exception:
            return None, None

    def step(self, obs: Observation) -> Command:
        road = getattr(obs, "road", None)
        if road is None or not getattr(road, "valid", False) or len(getattr(road, "preview_s", [])) < 3:
            return Command(brake=1.0, debug={"planner": "actuator-exact", "certified": False,
                                             "fallback_reason": "invalid/short preview"})
        base = self._fallback.step(obs)
        ref_d, ref_psi = self._reference(road, base)
        # Safety governor: outside the declared actuator grid, or with a large
        # lateral/heading error, use the certified-racing-line baseline.  This
        # is an explicit fallback; it is never labelled actuator-optimal.
        lat = abs(float(getattr(road, "lateral_offset", 0.0)))
        head = abs(float(getattr(road, "heading_error", 0.0)))
        obstacles = list(getattr(road, "obstacles", []) or [])
        near_obstacle = any(float(getattr(o, "distance", 1e9)) < 25.0 for o in obstacles)
        prev_k = np.asarray(getattr(road, "preview_curvature", []), dtype=float)
        tight = bool(prev_k.size and float(np.max(np.abs(prev_k[:20]))) > 0.06)
        if lat > 0.10 or head > 0.010 or near_obstacle or tight:
            base.debug.update({"planner": "certline-fallback", "certified": False,
                               "fallback_reason": ("obstacle envelope" if near_obstacle else
                                                   "tight curvature envelope" if tight else
                                                   "outside actuator safety envelope")})
            return base
        result = self._plan(obs, ref_d=ref_d, ref_psi=ref_psi)
        if result is None or result[3] is None:
            base.debug.update({"planner": "certline-fallback", "certified": False,
                               "fallback_reason": "no admissible exact actuator plan"})
            return base
        m, t, cost, (u, a), diag, elapsed = result
        cmd = command_from_first(obs, self.vehicle, m, u, a)
        cmd.debug.update({"cost": float(cost) if cost is not None else -1.0,
                          "solver_ms": float(elapsed), "states": float(diag["states"]),
                          "actions": float(diag["actions"]),
                          "solver_code": 1.0 if self._solver_name == "numba" else (2.0 if self._solver_name == "numpy-fallback" else 0.0),
                          "solver_error_len": float(len(self._solver_error)),
                          "ref_used": 1.0 if ref_d is not None else 0.0,
                          "certificate_ok": 1.0 if self._certificate is not None else 0.0})
        return cmd
