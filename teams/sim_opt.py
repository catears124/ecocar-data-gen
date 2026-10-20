"""Simulator-exact receding-horizon controller.

Every control period:

1. build a declared finite candidate set of command sequences (constant
   steering/pedal pairs over the horizon, the baseline's current command held,
   and the previous selected sequence shifted);
2. roll every candidate from the measured state through the harness's own
   bicycle transition function (``autocross.backends.bicycle.substep``), with
   the harness steering slew limiter and two 25 ms ticks per control period;
3. score the simulator-exact outcomes with hard corridor/obstacle constraints,
   a terminal curvature/braking condition, and a lexicographic objective
   ``(violation, -progress, effort)``;
4. publish the first command of the first minimum.

This is optimality over the **declared finite candidate set in the simulator at
the measured time t**.  It is not optimality over all continuous controls; the
control-resolution/epsilon gap is measured by
``tools/simulation_certificate.py`` and documented in
``docs/simulation_optimality.md``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams
from autocross.backends.bicycle import DEFAULTS
from autocross.simcert.sim import Policy, SimInstance, first_command, rollout, rollout_policies
from certline import CertifiedRacingLine


def _vehicle_cfg(vp: VehicleParams) -> dict:
    b = dict(DEFAULTS)
    b.update({
        "mass": float(vp.mass), "wheelbase": float(vp.wheelbase),
        "cg_to_front": float(vp.cg_to_front), "track_width": float(vp.track_width),
        "length": float(vp.length), "width": float(vp.width), "mu": float(vp.mu),
    })
    return {"bicycle": b, "max_steer_deg": math.degrees(float(vp.max_steer))}


class SimOptController(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        cfg = self.config or {}
        self.H = max(int(cfg.get("horizon", 10)), 1)
        self.steers = tuple(float(x) for x in cfg.get("steers", (-0.4, -0.2, 0.0, 0.2, 0.4)))
        self.pedals = tuple(tuple(float(v) for v in p) for p in
                            cfg.get("pedals", ((0, 0), (0.4, 0), (0.8, 0), (1, 0),
                                               (0, 0.4), (0, 0.8), (0, 1))))
        self.margin = float(cfg.get("margin_m", 0.25))
        self.half_width = float(vehicle.width) / 2.0
        self.half_length = float(vehicle.length) / 2.0
        self.vcfg = _vehicle_cfg(vehicle)
        self._baseline = CertifiedRacingLine(cfg.get("baseline", {}))
        self._baseline.on_start(track, vehicle)
        self._prev_seq = None
        self.policies = tuple(Policy(float(ld), float(sf), float(gn))
                              for ld in (4.0, 7.0) for sf in (0.8, 1.0) for gn in (1.0, 1.4))

    # ------------------------------------------------------------------ helpers
    def _candidates(self, base: Command) -> tuple[tuple[tuple[float, float, float], ...], ...]:
        seqs = [tuple([(s, th, br)] * self.H) for s in self.steers for th, br in self.pedals]
        seqs.append(tuple([(float(base.steer), float(base.throttle), float(base.brake))] * self.H))
        if self._prev_seq is not None:
            seq = list(self._prev_seq[1:]) + [self._prev_seq[-1]]
            seqs.append(tuple(seq))
        return tuple(seqs)

    def _instance(self, obs: Observation, road, cands) -> SimInstance:
        return SimInstance(
            vehicle=self.vcfg,
            state0=(float(obs.x), float(obs.y), float(obs.yaw), float(obs.vx),
                    float(obs.vy), float(obs.yaw_rate), float(obs.steer)),
            app0=(float(obs.steer), float(obs.throttle), float(obs.brake)),
            acc0=(float(obs.ax), float(obs.ay)),
            candidates=cands,
            preview_cl=tuple(map(tuple, road.preview_centerline)),
            preview_s=tuple(map(float, road.preview_s)),
            preview_wl=tuple(map(float, road.preview_width_left)),
            preview_wr=tuple(map(float, road.preview_width_right)),
            preview_k=tuple(map(float, road.preview_curvature)),
            obstacles=tuple((float(o.distance), float(o.lateral_min), float(o.lateral_max))
                            for o in (road.obstacles or [])),
            half_width=self.half_width, half_length=self.half_length,
            margin=self.margin, baseline_index=len(cands) - 1,
            max_steer_rate=float(self.vehicle.max_steer_rate),
        )

    @staticmethod
    def _path(obs: Observation, result: dict, index: int, cl: np.ndarray) -> np.ndarray | None:
        try:
            # The rollout's final/boundary states are world coordinates; convert
            # to the observation's vehicle frame for path_frame="vehicle".
            states = result.get("boundary_world")
            if not states:
                return None
            pts = np.asarray([row[index] for row in states], dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2:
                return None
            x0, y0, yaw0 = float(obs.x), float(obs.y), float(obs.yaw)
            c, s = math.cos(yaw0), math.sin(yaw0)
            rel = pts - np.array([x0, y0])
            local = np.column_stack((rel[:, 0] * c + rel[:, 1] * s,
                                     -rel[:, 0] * s + rel[:, 1] * c))
            local[0] = (0.0, 0.0)
            # A short path is a valid metric aid; the harness rejects folds and
            # paths shorter than 2 m, so fall back to the preview centerline.
            seg = np.hypot(*np.diff(local, axis=0).T)
            if seg.sum() < 2.0 or np.any(seg <= 1e-9):
                return cl + 0.0
            return local
        except Exception:  # noqa: BLE001
            return None

    # ---------------------------------------------------------------------- step
    def step(self, obs: Observation) -> Command:
        road = getattr(obs, "road", None)
        base = self._baseline.step(obs)
        if road is None or not getattr(road, "valid", False) or len(getattr(road, "preview_s", [])) < 3:
            base.debug.update({"planner": "sim-opt-fallback", "fallback_reason": 1.0})
            return base
        t0 = time.perf_counter()
        try:
            cl = np.asarray(road.preview_centerline, dtype=float)
            if cl.ndim != 2 or cl.shape[1] != 2 or cl.shape[0] != len(road.preview_s):
                raise ValueError("preview mismatch")
            cands = self._candidates(base)
            inst = self._instance(obs, road, cands)
            result = rollout(inst)
            # Closed-loop policy candidates: pure pursuit + curvature speed cap.
            dummy = tuple(tuple((0.0, 0.0, 0.0) for _ in range(self.H))
                          for _ in self.policies)
            inst_p = self._instance(obs, road, dummy)
            result_p = rollout_policies(inst_p, self.policies)
        except Exception:  # noqa: BLE001
            base.debug.update({"planner": "sim-opt-fallback", "fallback_reason": 2.0})
            return base
        t1 = time.perf_counter()
        costs = list(result["costs"]) + list(result_p["costs"])
        n_const = len(result["costs"])
        i = min(range(len(costs)), key=lambda k: costs[k])
        viol = int(result["violation"][i]) if i < n_const else int(result_p["violation"][i - n_const])
        if viol > 0:
            # No declared candidate is feasible: the baseline is the explicit
            # recovery controller, never a min-violation plan labelled optimal.
            base.debug.update({"planner": "sim-opt-recovery", "fallback_reason": 3.0,
                               "sim_viol": float(viol), "sim_ms": (t1 - t0) * 1000.0,
                               "candidate_count": float(len(costs))})
            return base
        if i < n_const:
            self._prev_seq = cands[i]
            steer, throttle, brake = first_command(inst, i)
            path = self._path(obs, result, i, cl)
        else:
            j = i - n_const
            realized = result_p["realized"][j]
            self._prev_seq = tuple(tuple(float(x) for x in row) for row in realized)
            steer, throttle, brake = (float(realized[0, 0]), float(realized[0, 1]),
                                      float(realized[0, 2]))
            path = self._path(obs, result_p, j, cl)
        base_cost = result["costs"][-1]
        dbg = {
            "planner": "sim-opt", "certified": 0.0, "selected": float(i),
            "candidate_count": float(len(costs)), "horizon": float(self.H),
            "sim_cost_viol": float(costs[i][0]), "sim_cost_progress": float(costs[i][1]),
            "sim_cost_effort": float(costs[i][2]),
            "baseline_viol": float(base_cost[0]), "baseline_progress": float(base_cost[1]),
            "sim_ms": (t1 - t0) * 1000.0, "sim_viol": float(viol),
            "status": 1.0, "fallback_reason": "",
        }
        # Keep the baseline's internal actuator history consistent with what was
        # actually published, so a later fallback is not fed a fictitious state.
        try:
            self._baseline.steer_prev = float(steer)
            if self._baseline.cmd_hist:
                self._baseline.cmd_hist[-1] = float(steer)
        except Exception:  # noqa: BLE001
            pass
        return Command(steer=steer, throttle=throttle, brake=brake, path=path,
                       path_frame="vehicle", target_speed=float(obs.speed), debug=dbg)


__all__ = ["SimOptController"]
