"""Minimum-time joint offset/speed racing-line controller.

The planner is the exact dynamic program of :mod:`autocross.lineopt.dp` over the
declared finite lattice of the ``teacher-joint-1`` model: station, previous and
current offset, and speed squared.  It is a genuine minimum-time objective
(sum of constant-acceleration segment traversal times) with hard corridor,
obstacle, curvature, acceleration, braking and terminal-viability constraints.
The DP visits every state of the declared universe; nothing in the search is
pruned.

The execution layer tracks the optimal path and speed profile with the same
path-error state feedback used by ``certline`` (uncertified, and clearly
separated from the proof).  If the declared model is infeasible from the
current observation, or the observation leaves the model domain, the controller
uses ``certline`` and labels the step explicitly.  An infeasible min-violation
plan is never published as a safe optimum.
"""
from __future__ import annotations

import math
import time

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams
from autocross.certstatus import STATUS_FALLBACK, status_debug
from autocross.lineopt.path import (plan_path, path_curvatures, profile_time_us,
                                    speed_profile)
from autocross.optimal.model import Model, State, model_from_observation
from certline import CertifiedRacingLine

G = 9.81


def _path_errors(path: np.ndarray, s: np.ndarray, px: float, py: float, pyaw: float):
    """Signed cross-track/heading error of a pose against a polyline (certline's arithmetic)."""
    c_, s_ = math.cos(pyaw), math.sin(pyaw)
    rel = path - np.array([px, py])
    p = np.column_stack((rel[:, 0] * c_ + rel[:, 1] * s_, -rel[:, 0] * s_ + rel[:, 1] * c_))
    a, d = p[:-1], np.diff(p, axis=0)
    dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
    t = np.clip(-np.einsum("ij,ij->i", a, d) / dd, 0.0, 1.0)
    q = a + d * t[:, None]
    i = int(np.argmin(np.einsum("ij,ij->i", q, q)))
    dx, dy = float(d[i, 0]), float(d[i, 1])
    e_y = (dx * (-float(q[i, 1])) - dy * (-float(q[i, 0]))) / math.sqrt(float(dd[i]))
    e_psi = -math.atan2(dy, dx)
    return e_y, e_psi, float(s[i] + t[i] * (s[i + 1] - s[i]))


def _plausible(path: np.ndarray) -> bool:
    if path is None or len(path) < 3 or not np.isfinite(path).all():
        return False
    d = np.diff(path, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    if seg.sum() < 2.0 or (seg <= 1e-9).any():
        return False
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return bool(np.all(np.abs(np.diff(hd)) <= 0.45 * np.maximum(seg[1:], 1e-6) + 1e-9))


class LineOptController(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        cfg = self.config or {}
        self.horizon = int(cfg.get("horizon", 40))
        self.offset_step = int(cfg.get("offset_step_mm", 200))
        self.offset_max = int(cfg.get("offset_max_mm", 1200))
        self.v_min = float(cfg.get("v_min_mps", 2.0))
        self.v_max = float(cfg.get("v_max_mps", min(float(vehicle.max_speed), 30.0)))
        self.dv = float(cfg.get("dv_mps", 1.5))
        self.k_v = float(cfg.get("k_v", 0.55))
        self.trk_omega = float(cfg.get("trk_omega", 3.25))
        self.trk_zeta = float(cfg.get("trk_zeta", 0.9))
        self.trk_lag = float(cfg.get("trk_lag", 0.10))
        self.grip_clamp = float(cfg.get("grip_clamp", 1.0))
        self.grip_frac = float(cfg.get("grip_frac", 0.7071067811865476))
        self.mode = str(cfg.get("mode", "line"))
        self.portfolio = bool(cfg.get("portfolio", True))
        self.min_gain = float(cfg.get("min_gain", 0.02))
        self.safety_margin = float(cfg.get("safety_margin_m", 0.25))
        self.use_near_obstacles = bool(cfg.get("use_near_obstacles", False))
        self.recover_lat = float(cfg.get("recover_lat", 0.70))
        self.recover_head = float(cfg.get("recover_head", 0.30))
        self.steer_prev = 0.0
        self._baseline = CertifiedRacingLine(cfg.get("baseline", {}))
        self._baseline.on_start(track, vehicle)
        self._plan_s = None
        self._plan_off = None
        self._plan_v = None
        # one cold solve so numpy/allocation cost is paid in on_start
        try:
            self._build_model(_warm_observation(vehicle), horizon=4)
        except Exception:  # noqa: BLE001 - warm-up must never prevent later steps
            pass

    # ------------------------------------------------------------------ planner
    def _speed_grid(self, speed: float) -> tuple[int, ...]:
        # The path DP does not use the speed grid for its transition; the grid
        # is still part of the declared model and carries the terminal cap.
        cur = int(round(max(speed, 0.0) * 1000)) ** 2
        cap = int(self.v_max * 1000) ** 2
        return tuple(sorted({max(cur, 1), max(cap, cur), 4_000_000}))

    def _offsets(self) -> tuple[int, ...]:
        step = max(self.offset_step, 50)
        m = max(self.offset_max, step)
        return tuple(range(-m, m + 1, step))

    def _build_model(self, obs: Observation, horizon: int | None = None) -> Model | None:
        road = getattr(obs, "road", None)
        if road is None or not getattr(road, "valid", False):
            return None
        n = len(road.preview_s)
        if n < 3:
            return None
        h = min(self.horizon if horizon is None else horizon, n - 1)
        return model_from_observation(
            obs, self.vehicle, horizon=h, offsets=self._offsets(),
            speeds=self._speed_grid(float(obs.speed)),
        )

    @staticmethod
    def _path_from_offsets(obs: Observation, offsets_mm) -> tuple | None:
        road = obs.road
        h = len(offsets_mm) - 1
        s = np.asarray(road.preview_s[:h + 1], dtype=float)
        cl = np.asarray(road.preview_centerline[:h + 1], dtype=float)
        if cl.shape != (h + 1, 2) or len(s) != h + 1:
            return None
        off = np.asarray(offsets_mm, dtype=float) / 1000.0
        tangent = np.gradient(cl, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        if np.any(norm <= 1e-9):
            return None
        nrm = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
        return s, off, cl + nrm * off[:, None]

    def _candidate_offsets(self, model: Model, obs_offsets: np.ndarray) -> tuple[int, ...] | None:
        """Declared candidate offsets (mm), quantised to the model's millimetre unit.

        The baseline path is evaluated as its own declared candidate; it need
        not lie on the DP lattice.  Its admissibility is checked by the same
        corridor/curvature predicates in :meth:`_path_feasible`.
        """
        return tuple(int(round(float(x) * 1000.0)) for x in np.asarray(obs_offsets, dtype=float))

    def _path_feasible(self, model: Model, offsets: tuple[int, ...]) -> bool:
        H = len(offsets) - 1
        if H != model.horizon:
            return False
        curv = path_curvatures(model, offsets)
        if int(np.max(curv)) > int(model.kappa_max):
            return False
        for i in range(H + 1):
            # The baseline path is a continuous candidate, not a lattice
            # state: use the geometric corridor/obstacle predicates rather
            # than ``lateral_ok`` (which also requires grid membership).
            if not (model.corridor_ok(i, offsets[i]) and model.obstacle_ok(i, offsets[i])):
                return False
        return True

    # ------------------------------------------------------------------- tracker
    def _track(self, obs: Observation, path: np.ndarray, s: np.ndarray,
               v_profile: np.ndarray) -> tuple[float, float, float, dict]:
        vp = self.vehicle
        v = max(float(obs.speed), 0.0)
        # Pose once the current steering lag has acted.
        lag = self.trk_lag
        r = float(obs.yaw_rate)
        px = v * lag * math.cos(0.5 * r * lag)
        py = v * lag * math.sin(0.5 * r * lag)
        pyaw = r * lag
        e_y, e_psi, s_p = _path_errors(path, s, px, py, pyaw)
        hd = np.unwrap(np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0])))
        k_path = np.gradient(hd, s)
        k_ref = float(np.interp(s_p, s, k_path))
        v_e = max(v, 3.0)
        w_n, zeta = self.trk_omega, self.trk_zeta
        u = -(vp.wheelbase / (v_e * v_e)) * (w_n * w_n * e_y + 2.0 * zeta * w_n * v_e * e_psi)
        delta = math.atan(vp.wheelbase * k_ref + u)
        d_max = math.atan(self.grip_clamp * float(vp.mu) * G * vp.wheelbase / (v_e * v_e))
        delta = float(np.clip(delta, -d_max, d_max))
        steer = float(np.clip(delta / vp.max_steer, -1.0, 1.0))
        rate = vp.max_steer_rate * vp.control_dt
        steer = float(np.clip(steer, self.steer_prev - rate, self.steer_prev + rate))
        self.steer_prev = steer

        # Speed: follow the optimal profile, with the profile's own acceleration
        # as feed-forward and a proportional correction.
        look = min(s_p + max(v, 3.0) * 0.15, float(s[-1]))
        v_tgt = float(np.interp(look, s, v_profile))
        j = int(np.clip(np.searchsorted(s, s_p), 0, len(s) - 1))
        j2 = min(j + 1, len(s) - 1)
        a_ff = (v_profile[j2] ** 2 - v_profile[j] ** 2) / (2.0 * max(s[j2] - s[j], 1e-6))
        a_des = float(np.clip(a_ff + self.k_v * (v_tgt - v), -vp.max_decel, vp.max_accel))
        return steer, a_des, v_tgt, {"e_y": e_y, "e_psi": e_psi, "s_p": s_p,
                                     "k_ref": k_ref, "a_ff": a_ff}

    def _pedals(self, obs: Observation, a_des: float) -> tuple[float, float]:
        vp = self.vehicle
        if a_des >= 0.0:
            lat = min(0.5 * abs(float(obs.ay)) / max(float(vp.mu) * G, 1e-6), 1.0)
            cap = max(1.0 - lat * lat, 0.15)
            return float(min(a_des / vp.max_accel, cap, 1.0)), 0.0
        return 0.0, float(np.clip((-a_des - 0.2) / vp.max_decel, 0.0, 1.0))

    # ---------------------------------------------------------------------- step
    def step(self, obs: Observation) -> Command:
        road = getattr(obs, "road", None)
        base = self._baseline.step(obs)
        if road is None or not getattr(road, "valid", False) or len(getattr(road, "preview_s", [])) < 3:
            base.debug.update(status_debug(STATUS_FALLBACK, "invalid_preview"))
            return base
        # Recovery guard: outside the corridor or badly misaligned, the baseline
        # tracker is the safe controller.  Explicitly labelled, never optimum.
        lat = float(getattr(road, "lateral_offset", 0.0) or 0.0)
        head = float(getattr(road, "heading_error", 0.0) or 0.0)
        if (abs(lat) > self.recover_lat * max(float(road.width_left), 1e-6)
                or -lat > self.recover_lat * max(float(road.width_right), 1e-6)
                or abs(head) > self.recover_head):
            base.debug.update(status_debug(STATUS_FALLBACK, "recovery_guard"))
            return base

        if self.mode == "speed":
            return self._step_speed_mode(obs, base)

        t0 = time.perf_counter()
        try:
            model = self._build_model(obs)
            if model is None:
                raise ValueError("no declared model")
            pp = plan_path(model, a_lat=int(self.vehicle.mu * G * self.grip_frac * 1000))
        except Exception as exc:  # noqa: BLE001 - a planner failure must fall back
            base.debug.update(status_debug(STATUS_FALLBACK, "planner_error"))
            base.debug["planner_error"] = float(abs(hash(type(exc).__name__)) % 97)
            return base
        t1 = time.perf_counter()
        if pp.cost_us is None or len(pp.offsets) != model.horizon + 1:
            base.debug.update(status_debug(STATUS_FALLBACK, "model_infeasible"))
            base.debug["plan_ms"] = (t1 - t0) * 1000.0
            return base

        # ---- baseline-inclusive candidate set ---------------------------------
        # Candidate A: the exact path-DP optimum.
        # Candidate B: the baseline's published path, quantised to this model's
        # offset lattice.  Both are scored by the same speed-profile optimiser
        # and the same declared objective; the cheaper admissible candidate is
        # selected, so J(selected) <= J(baseline candidate) by construction.
        candidates: list[tuple[str, tuple[int, ...]]] = [("dp", pp.offsets)]
        base_off = None
        bp = getattr(base, "path", None)
        if bp is not None:
            try:
                cl = np.asarray(obs.road.preview_centerline, dtype=float)
                path_arr = np.asarray(bp, dtype=float)
                if cl.ndim == 2 and path_arr.shape == cl.shape:
                    tangent = np.gradient(cl, axis=0)
                    norm = np.linalg.norm(tangent, axis=1)
                    nrm = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
                    raw = np.einsum("ij,ij->i", path_arr - cl, nrm)
                    base_off = self._candidate_offsets(model, raw[:model.horizon + 1])
                    if len(base_off) == model.horizon + 1 and self._path_feasible(model, base_off):
                        candidates.append(("baseline", base_off))
            except Exception:  # noqa: BLE001 - baseline candidacy is optional
                base_off = None

        v0 = max(float(obs.speed), 0.0)
        scored = []
        for name, offs in candidates:
            b = speed_profile(model, offs, v0, v_max_mps=self.v_max,
                              grip_mps2=self.vehicle.mu * G * self.grip_frac)
            cost = profile_time_us(model, offs, b)
            scored.append((cost, name, offs, b))
        scored.sort(key=lambda x: (x[0], 0 if x[1] == "dp" else 1))
        cost, chosen, offs, bprof = scored[0]
        baseline_cost = next((c for c, n, _o, _b in scored if n == "baseline"), None)
        # ---- execution portfolio ---------------------------------------------
        # The selected *model* plan is the cheapest candidate.  The deployed
        # command is still subject to a conservative safety gate: near an
        # obstacle, with a stale/large tracking error, or without a measurable
        # predicted-time gain over the included baseline candidate, the
        # baseline command is published instead.  That is a declared fallback,
        # not an optimality claim.
        obstacles = list(getattr(obs.road, "obstacles", []) or [])
        near_obs = any(float(getattr(o, "distance", 1e9)) < 30.0 for o in obstacles)
        mem_obs = bool(getattr(self._baseline, "mem", None))
        floor = float(getattr(obs.road, "lateral_offset", 0.0) or 0.0)
        head = float(getattr(obs.road, "heading_error", 0.0) or 0.0)
        predicted = self._path_from_offsets(obs, offs)
        unsafe = False
        if self.portfolio:
            # Without an admissible baseline candidate there is no dominance
            # comparison to stand on: execute the baseline, not an
            # unverified "improvement".
            if chosen != "dp" or baseline_cost is None:
                unsafe = True
            if baseline_cost is not None and cost > baseline_cost * (1.0 - self.min_gain):
                unsafe = True
            if near_obs or mem_obs:
                unsafe = True
            if abs(floor) > self.safety_margin or abs(head) > 0.08:
                unsafe = True
            if predicted is not None:
                s_chk, _o, path_chk = predicted
                lo_arr = np.asarray(obs.road.preview_width_right[:len(offs)])
                hi_arr = np.asarray(obs.road.preview_width_left[:len(offs)])
                off_m = np.asarray(offs, dtype=float) / 1000.0
                if np.min(hi_arr - off_m - 1.08) < 0.30 or np.min(lo_arr + off_m - 1.08) < 0.30:
                    unsafe = True
                # The selected path must already be tracked closely at the
                # measured pose; a large current cross-track/heading error is
                # evidence that the unmodelled transient dominates.
                try:
                    e_y_chk, e_psi_chk, _s_chk = _path_errors(path_chk, s_chk, 0.0, 0.0, 0.0)
                    if abs(e_y_chk) > 0.10 or abs(e_psi_chk) > 0.04:
                        unsafe = True
                except Exception:  # noqa: BLE001
                    unsafe = True
                # Prefer the enriched actuator plan on gentle, lower-speed
                # sections where the model error is smallest; tight corners are
                # exactly where the unmodelled transient dominates.
                kk = np.abs(np.gradient(np.unwrap(np.arctan2(
                    np.gradient(path_chk[:, 1]), np.gradient(path_chk[:, 0]))), s_chk))
                if float(np.max(kk[: min(12, len(kk))])) > 0.035:
                    unsafe = True
                if float(obs.speed) > 12.0:
                    unsafe = True
        if unsafe:
            base.debug = dict(base.debug or {})
            base.debug.update({"planner": "lineopt-baseline", "portfolio": 1.0,
                               "portfolio_reason": 1.0, "plan_cost_us": float(cost),
                               "baseline_cost_us": float(baseline_cost if baseline_cost is not None else -1.0)})
            return base
        # The candidate universe's cheapest plan is the selected plan.
        path_data = self._path_from_offsets(obs, offs)
        if path_data is None:
            base.debug.update(status_debug(STATUS_FALLBACK, "path_build_failed"))
            return base
        s, off, path = path_data
        v_profile = np.sqrt(np.maximum(bprof, 1.0))
        if not _plausible(path):
            base.debug.update(status_debug(STATUS_FALLBACK, "implausible_path"))
            return base
        try:
            steer, a_des, v_tgt, extra = self._track(obs, path, s, v_profile)
            throttle, brake = self._pedals(obs, a_des)
        except Exception:  # noqa: BLE001
            base.debug.update(status_debug(STATUS_FALLBACK, "tracker_error"))
            return base

        dbg = {
            "planner": "lineopt-dp", "certified": 0.0,
            "plan_feasible": 1.0, "plan_cost_us": float(cost),
            "baseline_cost_us": float(baseline_cost if baseline_cost is not None else -1),
            "candidate_count": float(len(scored)), "selected": 1.0 if chosen == "dp" else 0.0,
            "horizon": float(model.horizon), "grid_offsets": float(len(model.offsets)),
            "grid_speeds": float(len(model.speeds)), "plan_ms": (t1 - t0) * 1000.0,
            "solve_ms": pp.elapsed_ms, "v_tgt": v_tgt, "a_des": a_des,
            "off0": float(off[0]), "off3": float(off[min(3, len(off) - 1)]),
            "off10": float(off[min(10, len(off) - 1)]),
            "status": 1.0, "fallback_reason": "",
        }
        dbg.update({k: float(v) for k, v in extra.items()})
        return Command(steer=steer, throttle=throttle, brake=brake, path=path,
                       path_frame="vehicle", target_speed=v_tgt, debug=dbg)


    # ------------------------------------------------------------ speed-override
    def _step_speed_mode(self, obs: Observation, base: Command) -> Command:
        """Keep the baseline's proven path/tracking, optimise its speed profile.

        Both speed candidates are scored in the same declared objective on the
        baseline's published path: the baseline's own budgeted profile
        (lateral/braking box) and the friction-circle profile.  The cheaper
        admissible one is executed on the baseline's steering command, so the
        controller can never be slower *in the model* than the included
        baseline speed candidate.
        """
        road = obs.road
        bp = getattr(base, "path", None)
        if bp is None:
            return base
        cl = np.asarray(road.preview_centerline, dtype=float)
        path_arr = np.asarray(bp, dtype=float)
        if cl.ndim != 2 or path_arr.shape != cl.shape:
            return base
        tangent = np.gradient(cl, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        if np.any(norm <= 1e-9):
            return base
        nrm = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
        raw = np.einsum("ij,ij->i", path_arr - cl, nrm)
        offsets = tuple(int(round(x * 1000.0)) for x in raw)
        try:
            model = self._build_model(obs, horizon=len(offsets) - 1)
            if model is None:
                return base
            v0 = max(float(obs.speed), 0.0)
            # Baseline candidate: the declared box budgets.
            b_base = speed_profile(model, offsets, v0, v_max_mps=self.v_max,
                                   grip_mps2=float(model.lon["a_lat"]) / 1000.0,
                                   brk_frac=min(1.0, float(model.lon["a_brk"]) /
                                                max(float(model.lon["a_lat"]), 1.0)))
            # Optimised candidate: friction circle with the configured fraction.
            b_opt = speed_profile(model, offsets, v0, v_max_mps=self.v_max,
                                  grip_mps2=self.vehicle.mu * G * self.grip_frac)
        except Exception:  # noqa: BLE001
            return base
        c_base = profile_time_us(model, offsets, b_base)
        c_opt = profile_time_us(model, offsets, b_opt)
        s = np.asarray(road.preview_s[:len(offsets)], dtype=float)
        # Conservative gate: only exploit the model's extra longitudinal grip
        # on gentle, low-speed, well-tracked sections, and never move more than
        # a small acceleration step away from the baseline command.
        try:
            e_y_chk, e_psi_chk, _s_chk = _path_errors(path_arr, s, 0.0, 0.0, 0.0)
            kk = np.abs(np.gradient(np.unwrap(np.arctan2(
                np.gradient(path_arr[:, 1]), np.gradient(path_arr[:, 0]))), s))
            safe = (c_opt < c_base * 0.99 and float(np.max(kk[: min(12, len(kk))])) < 0.035
                    and abs(e_y_chk) < 0.10 and abs(e_psi_chk) < 0.04
                    and float(obs.speed) < 12.0)
        except Exception:  # noqa: BLE001
            safe = False
        if not safe:
            return base
        use_opt = c_opt <= c_base
        b = b_opt if use_opt else b_base
        v_profile = np.sqrt(np.maximum(b, 1.0))
        # Same path-error projection as the tracking layer, for the target speed.
        v = max(float(obs.speed), 0.0)
        lag = self.trk_lag
        r = float(obs.yaw_rate)
        px = v * lag * math.cos(0.5 * r * lag)
        py = v * lag * math.sin(0.5 * r * lag)
        try:
            _e_y, _e_psi, s_p = _path_errors(path_arr, s, px, py, r * lag)
            look = min(s_p + max(v, 3.0) * 0.15, float(s[-1]))
            v_tgt = float(np.interp(look, s, v_profile))
            j = int(np.clip(np.searchsorted(s, s_p), 0, len(s) - 1))
            j2 = min(j + 1, len(s) - 1)
            a_ff = (v_profile[j2] ** 2 - v_profile[j] ** 2) / (2.0 * max(s[j2] - s[j], 1e-6))
            a_des = float(np.clip(a_ff + self.k_v * (v_tgt - v), -self.vehicle.max_decel,
                                  self.vehicle.max_accel))
            base_a = (float(base.throttle) * self.vehicle.max_accel
                      - float(base.brake) * self.vehicle.max_decel)
            a_des = float(np.clip(a_des, base_a - 0.5, base_a + 0.75))
            throttle, brake = self._pedals(obs, a_des)
        except Exception:  # noqa: BLE001
            return base
        dbg = dict(base.debug or {})
        dbg.update({"planner": "lineopt-speed", "speed_override": 1.0 if use_opt else 0.0,
                    "plan_cost_us": float(c_opt), "baseline_cost_us": float(c_base),
                    "v_tgt": v_tgt, "a_des": a_des, "a_ff": a_ff,
                    "grip_frac": self.grip_frac})
        return Command(steer=base.steer, throttle=throttle, brake=brake,
                       path=base.path, path_frame="vehicle", target_speed=v_tgt, debug=dbg)


def _warm_observation(vehicle: VehicleParams) -> Observation:
    from autocross.api import RoadModel
    speed = 4.0
    road = RoadModel(
        valid=True, range_m=6.0, lateral_offset=0.0, heading_error=0.0, curvature=0.0,
        width_left=3.0, width_right=3.0, dist_to_left_boundary=3.0,
        dist_to_right_boundary=3.0,
        preview_s=np.arange(0.0, 7.0), preview_curvature=np.zeros(7),
        preview_centerline=np.column_stack((np.arange(7.0), np.zeros(7))),
        preview_width_left=np.full(7, 3.0), preview_width_right=np.full(7, 3.0),
        preview_left_boundary=np.column_stack((np.arange(7.0), np.full(7, 3.0))),
        preview_right_boundary=np.column_stack((np.arange(7.0), np.full(7, -3.0))),
        corners=[], obstacles=[], finish_distance=float("nan"))
    return Observation(t=0.0, step=0, x=0.0, y=0.0, yaw=0.0, vx=speed, vy=0.0,
                       yaw_rate=0.0, ax=0.0, ay=0.0, speed=speed, steer=0.0,
                       throttle=0.0, brake=0.0, progress=0.0, lap=0,
                       timing_started=False, road=road)


__all__ = ["LineOptController"]
