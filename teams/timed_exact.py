"""Time-indexed exact actuator controller (model ``timed-1``).

Every control step:

1. encode the observation into a ``timed-1`` instance (``autocross.timed.encode``);
2. solve it exactly with the numpy solver (``autocross.timed.vector``): every plan of
   the declared move-blocked command set is evaluated by the integer model, the
   lexicographic objective (violation, -progress, effort) is minimised, ties go to
   the first plan in enumeration order;
3. publish the first block's command *as modelled*: steer = u/1000,
   throttle/brake = pedal/1000.  No tracking layer, pedal map or smoothing sits
   between the optimal command and the published one.

Status ladder (``Command.debug``; see ``autocross.certstatus``): exact steps are
``STATUS_SOLVER`` at runtime.  The offline audit (``tools/timed_audit.py``) replays
the encoder over the recorded observations, re-checks every executed command with
the Lean checker and upgrades it to ``STATUS_LEAN_CHECKED`` only on acceptance.
``feasible`` = 1 when the optimal plan has zero violation (all hard constraints hold);
0 means the model proved that no plan satisfies them and the command is the exact
minimum-violation (recovery) optimum.  Fallback commands (invalid preview, encoder
domain, int64 domain overflow) come from ``certline`` with ``STATUS_FALLBACK``.
"""
from __future__ import annotations

import time

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams
from autocross.certstatus import (FALLBACK_REASONS, STATUS_FALLBACK, STATUS_SOLVER,
                                  status_debug)
from autocross.timed.encode import EncodeError, Encoder, station_headings
from autocross.timed.model import (DEFAULT_BLOCKS, DEFAULT_CONSTS, Block, command_floats,
                                   first_command)
from autocross.timed.vector import DomainOverflow, Solver
from certline import CertifiedRacingLine


def _blocks(cfg) -> tuple[Block, ...]:
    if not cfg:
        return DEFAULT_BLOCKS
    return tuple(Block(int(b["steps"]), tuple(int(x) for x in b["steer"]),
                       tuple(int(x) for x in b["pedal"])) for b in cfg)


def _hash52(hexdigest: str) -> float:
    return float(int(hexdigest[:13], 16))   # 52 bits: exact in a float64


class TimedExactController(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        cfg = self.config or {}
        consts = dict(DEFAULT_CONSTS)
        consts["MAX_STEER"] = int(round(vehicle.max_steer * 1_000_000))
        consts["L"] = int(round(vehicle.wheelbase * 1000))
        consts["LF"] = int(round(vehicle.cg_to_front * 1000))
        consts["LR"] = int(round(vehicle.cg_to_rear * 1000))
        consts["MASS"] = int(round(vehicle.mass))
        consts["SLEW"] = int(round(vehicle.max_steer_rate * 1000 * vehicle.control_dt / consts["TICKS"]))
        consts["VMAX"] = int(round(vehicle.max_speed * 1000))
        consts.update({k: int(v) for k, v in (cfg.get("consts") or {}).items()})
        self.encoder = Encoder(consts=consts, blocks=_blocks(cfg.get("blocks")),
                               obs_pad=int(cfg.get("obs_pad_mm", 150)))
        self.warm_start = bool(cfg.get("warm_start", True))
        self.publish_path = bool(cfg.get("publish_path", True))
        self._plan = None
        self._fallback = CertifiedRacingLine(cfg.get("fallback", {}))
        self._fallback.on_start(track, vehicle)

    # ------------------------------------------------------------------ helpers
    def _fallback_cmd(self, obs: Observation, reason: str, detail: float = 0.0) -> Command:
        try:
            cmd = self._fallback.step(obs)
        except Exception:  # noqa: BLE001 - the safety fallback must answer
            cmd = Command(brake=1.0)
        cmd.debug = dict(cmd.debug or {})
        cmd.debug.update(status_debug(STATUS_FALLBACK, reason))
        cmd.debug["fallback_detail"] = float(detail)
        self._plan = None
        return cmd

    @staticmethod
    def _path(road, traj, d_end: int):
        cl = np.asarray(road.preview_centerline, dtype=float)
        s_grid = np.asarray(road.preview_s, dtype=float)
        if cl.ndim != 2 or cl.shape[0] < 2:
            return None
        hd = station_headings(cl)
        s = np.array([t[0] for t in traj], dtype=float) / 1000.0
        d = np.array([t[1] for t in traj], dtype=float) / 1000.0
        s_tail = s_grid[s_grid > (s[-1] if s.size else 0.0)]
        s_all = np.concatenate([[0.0], s, s_tail])
        d_all = np.concatenate([[d[0] if d.size else 0.0], d, np.full(s_tail.size, d_end / 1000.0)])
        x = np.interp(s_all, s_grid, cl[:, 0])
        y = np.interp(s_all, s_grid, cl[:, 1])
        h = np.interp(s_all, s_grid, hd)
        return np.column_stack((x - np.sin(h) * d_all, y + np.cos(h) * d_all))

    # --------------------------------------------------------------------- step
    def step(self, obs: Observation) -> Command:
        t0 = time.perf_counter()
        try:
            inst = self.encoder.encode(obs)
        except EncodeError as e:
            reason = "invalid_preview" if "road" in str(e) or "preview" in str(e) else "encoder_domain"
            return self._fallback_cmd(obs, reason)
        t1 = time.perf_counter()
        solver = Solver(inst)
        try:
            inc = self._plan if (self.warm_start and self._plan is not None
                                 and len(self._plan) == len(inst.blocks)
                                 and all(0 <= a < b.n for a, b in zip(self._plan, inst.blocks))) else None
            plan, J, stats = solver.solve(inc)
        except DomainOverflow:
            return self._fallback_cmd(obs, "solver_error", 1.0)
        t2 = time.perf_counter()
        self._plan = plan
        u, p = first_command(inst, plan)
        steer, throttle, brake = command_floats(u, p)
        path = None
        if self.publish_path:
            try:
                traj = solver.trajectory(plan)
                path = self._path(obs.road, traj, traj[-1][1])
            except Exception:  # noqa: BLE001 - path is a display/metric aid only
                path = None
        dbg = status_debug(STATUS_SOLVER, "none")
        dbg.update({
            "feasible": 1.0 if J[0] == 0 else 0.0, "viol": float(J[0]), "neg_progress": float(J[1]),
            "effort": float(J[2]), "u": float(u), "p": float(p),
            "plan0": float(plan[0]), "plan1": float(plan[1]) if len(plan) > 1 else -1.0,
            "plan2": float(plan[2]) if len(plan) > 2 else -1.0,
            "leaves": float(stats["leaves"][-1]), "ticks": float(stats["ticks"]),
            "encode_ms": (t1 - t0) * 1000.0, "solve_ms": (t2 - t1) * 1000.0,
            "instance_h52": _hash52(inst.sha256()), "warm": 1.0 if inc is not None else 0.0,
            "delta0": float(inst.start.delta), "sigma0": float(inst.start.sigma),
        })
        return Command(steer=steer, throttle=throttle, brake=brake, path=path, path_frame="vehicle",
                       target_speed=None, debug=dbg)


__all__ = ["TimedExactController", "FALLBACK_REASONS"]
