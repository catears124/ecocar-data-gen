"""Direct actuator-space exact controller.

Each step builds the declared finite actuator model from the current policy
observation, solves the exact DP over every declared state/action pair, and
publishes the first action of an optimal sequence through the deterministic
pedal map.  No pursuit/tracking layer sits between the optimal action and the
command.

The formal claim is for the finite model (see docs/actuator_controller.md): the
first action is optimal for the declared actuator dynamics, grids, horizon and
objective.  Simulator closed-loop behavior is measured, not proved.

Every command carries the numeric certification ladder of
:mod:`autocross.certstatus` (``status``, ``certified``, ``fallback_reason``):
exact steps are ``STATUS_SOLVER`` unless ``emit_certificates`` produced a
certificate for exactly this instance and this executed first action that the
independent Python checker accepted (``STATUS_PY_CHECKED``).  ``certified`` is
never 1 at runtime: only an offline Lean audit may assign
``STATUS_LEAN_CHECKED``.  ``solver_code`` reports the solver that actually ran
(0 numpy, 1 numba, 2 numpy after a numba error); inside the controller sandbox
numba cannot load (``ctypes.dlopen`` is denied), so it is 0 there.
"""
from __future__ import annotations

import math
import time

import numpy as np

from autocross.api import Command, Controller, Observation, RoadModel, TrackInfo, VehicleParams
from autocross.certstatus import STATUS_FALLBACK, STATUS_PY_CHECKED, STATUS_SOLVER, status_debug
from certline import CertifiedRacingLine
from autocross.actuator.build import command_from_first, model_from_observation
from autocross.actuator.certificate import (
    certificate_for_action, check_certificate, instance_sha256, sha52,
)
from autocross.actuator.numba_solver import NUMBA_AVAILABLE, solve_numba
from autocross.actuator.solver import solve

SOLVER_NUMPY = 0
SOLVER_NUMBA = 1
SOLVER_NUMPY_AFTER_NUMBA_ERROR = 2


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
        self.performance_guard = bool(cfg.get("performance_guard", False))
        self._certificate = None
        self._solver_error = ""
        self._fallback = CertifiedRacingLine({})
        self._fallback.on_start(track, vehicle)
        try:
            self._warm()
        except Exception as exc:  # warm-up only compiles/caches; steps report errors
            self._solver_error = f"warm: {type(exc).__name__}: {exc}"

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
        """Build and solve the declared model.

        Returns ``None`` when no model can be declared, else a dict with the
        model, the solver result, the solver code that ran, and the runtime
        certificate status of the returned first action.
        """
        m = model_from_observation(
            obs, self.vehicle, horizon=self.horizon if horizon is None else horizon,
            offsets=self.offsets, psis=self.psis, speeds=self.speeds,
            deltas=self.deltas, steer_inputs=self.steer_inputs,
            acc_inputs=self.acc_inputs, terminal_distance=self.terminal_distance,
            steer_tau_s=self.steer_tau_s, ref_d=ref_d, ref_psi=ref_psi)
        if m is None:
            return None
        t0 = time.perf_counter()
        if NUMBA_AVAILABLE:
            try:
                first_t, cost, first_action, diag = solve_numba(m)
                code = SOLVER_NUMBA
            except Exception as exc:
                self._solver_error = f"numba: {type(exc).__name__}: {exc}"
                first_t, cost, first_action, diag = solve(m)
                code = SOLVER_NUMPY_AFTER_NUMBA_ERROR
        else:
            first_t, cost, first_action, diag = solve(m)
            code = SOLVER_NUMPY
        elapsed = (time.perf_counter() - t0) * 1000.0
        out = {"model": m, "first": first_t, "cost": cost, "action": first_action,
               "diag": diag, "elapsed": elapsed, "solver_code": code,
               "status": STATUS_SOLVER, "certificate_ok": False,
               "cert_sha52": -1.0, "certificate_error": 0.0}
        self._certificate = None
        if self.emit_certificates and first_action is not None:
            self._bind_certificate(m, first_action, cost, out)
        return out

    def _bind_certificate(self, m, action, cost, out: dict) -> None:
        """Certify exactly the executed first action of exactly the solved model.

        ``STATUS_PY_CHECKED`` requires a certificate built from the same
        ``ActModel`` the runtime solver solved, whose ``firstAction`` is the
        executed ``(u, a)``, whose claimed optimum equals the runtime optimal
        cost, and which the independent Python checker accepts.  A certificate
        that cannot be built for the executed action, or disagrees, marks a
        mismatch (the caller falls back); a checker exception leaves the step at
        ``STATUS_SOLVER``.
        """
        try:
            cert = certificate_for_action(m, action)
            ok = (cert is not None
                  and instance_sha256(cert["instance"]) == instance_sha256(m)
                  and cert["firstAction"] == [int(action[0]), int(action[1])]
                  and cert["claimed"] == cost
                  and bool(check_certificate(cert).get("ok")))
        except Exception as exc:
            self._solver_error = f"certificate: {type(exc).__name__}: {exc}"
            out["certificate_error"] = 1.0
            return
        if not ok:
            out["certificate_error"] = 2.0
            out["mismatch"] = True
            return
        self._certificate = cert
        out.update(status=STATUS_PY_CHECKED, certificate_ok=True,
                   cert_sha52=sha52(cert["certificate_sha256"]))

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

    def _fallback_command(self, base: Command, reason: str, **extra) -> Command:
        """Explicit certline fallback; never labelled actuator-optimal."""
        base.debug.update({"planner": "certline-fallback", **status_debug(STATUS_FALLBACK, reason),
                           "numba_available": 1.0 if NUMBA_AVAILABLE else 0.0, **extra})
        return base

    def step(self, obs: Observation) -> Command:
        road = getattr(obs, "road", None)
        if road is None or not getattr(road, "valid", False) or len(getattr(road, "preview_s", [])) < 3:
            return Command(brake=1.0, debug={"planner": "actuator-exact",
                                             **status_debug(STATUS_FALLBACK, "invalid_preview")})
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
        if near_obstacle:
            return self._fallback_command(base, "obstacle")
        if tight:
            return self._fallback_command(base, "tight_curvature")
        if lat > 0.10:
            return self._fallback_command(base, "lateral_envelope")
        if head > 0.010:
            return self._fallback_command(base, "heading_envelope")
        try:
            result = self._plan(obs, ref_d=ref_d, ref_psi=ref_psi)
        except Exception as exc:
            self._solver_error = f"solve: {type(exc).__name__}: {exc}"
            return self._fallback_command(base, "solver_error")
        if result is None or result["action"] is None:
            extra = {} if result is None else {"solver_code": float(result["solver_code"])}
            return self._fallback_command(base, "infeasible", **extra)
        if result.get("mismatch"):
            return self._fallback_command(base, "certificate_mismatch",
                                          solver_code=float(result["solver_code"]))
        m = result["model"]
        u, a = result["action"]
        base_a = (float(getattr(base, "throttle", 0.0)) * self.vehicle.max_accel
                  - float(getattr(base, "brake", 0.0)) * self.vehicle.max_decel)
        if self.performance_guard and a / 1000.0 < base_a - 0.4:
            return self._fallback_command(base, "performance_guard",
                                          solver_code=float(result["solver_code"]))
        cmd = command_from_first(obs, self.vehicle, m, u, a, status=result["status"])
        cost = result["cost"]
        cmd.debug.update({"cost": float(cost) if cost is not None else -1.0,
                          "solver_ms": float(result["elapsed"]),
                          "states": float(result["diag"]["states"]),
                          "actions": float(result["diag"]["actions"]),
                          "solver_code": float(result["solver_code"]),
                          "numba_available": 1.0 if NUMBA_AVAILABLE else 0.0,
                          "solver_error_len": float(len(self._solver_error)),
                          "ref_used": 1.0 if ref_d is not None else 0.0,
                          "certificate_ok": 1.0 if result["certificate_ok"] else 0.0,
                          "certificate_error": float(result["certificate_error"]),
                          "cert_sha52": float(result["cert_sha52"]),
                          "instance_sha52": sha52(instance_sha256(m))})
        return cmd
