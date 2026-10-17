"""Closed-loop run: backend <-> actuator model <-> sandboxed controller, with the race
monitor and ground-truth telemetry logging at physics rate."""
from __future__ import annotations

import copy
import csv
import datetime as dt_
import hashlib
import json
import math
import platform
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import metrics as metrics_mod
from . import score as score_mod
from .api import Command, Observation, VehicleParams
from .config import ROOT, deep_update
from .geometry import body_to_world, path_is_plausible, polyline_window_distance
from .monitor import RaceMonitor
from .perception import ConeSensor, RoadModelSensor
from .sandbox import ControllerProcess
from .track import load as load_track

LOG_COLS = ["t", "x", "y", "yaw", "vx", "vy", "speed", "yaw_rate", "ax", "ay", "roll", "pitch", "steer_angle",
            "progress", "lateral", "boundary_margin", "path_err", "has_path", "off_course",
            "cmd_steer", "cmd_throttle", "cmd_brake", "app_steer", "app_throttle", "app_brake",
            "ctrl_step", "compute_ms", "cpu_ms", "latency_ms", "overrun", "clipped", "target_speed"]


def make_backend(name: str, cfg: dict):
    if name == "bicycle":
        from .backends.bicycle import BicycleBackend
        return BicycleBackend(cfg["sim"])
    if name == "carla":
        from .backends.carla_backend import CarlaBackend
        return CarlaBackend(cfg["sim"], cfg["carla"])
    raise ValueError(f"unknown backend '{name}'")


def apply_perturbation(cfg: dict, perturb: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    p = cfg["perception"]
    if "cone_noise_std" in perturb:
        p["noise_std_m"] = float(perturb["cone_noise_std"])
    if "cone_dropout" in perturb:
        p["dropout"] = float(perturb["cone_dropout"])
    if "misclassification" in perturb:
        p["misclassification"] = float(perturb["misclassification"])
    r = cfg.setdefault("road_model", {})
    for key, cfg_key in (("road_offset_noise", "offset_noise_std_m"), ("road_heading_noise_deg", "heading_noise_std_deg"),
                         ("road_curvature_noise", "curvature_noise_std"), ("road_range_m", "range_m")):
        if key in perturb:
            r[cfg_key] = float(perturb[key])
    if "actuation_delay_steps" in perturb:
        cfg["actuators"]["delay_steps"] = int(cfg["actuators"].get("delay_steps", 0)) + int(perturb["actuation_delay_steps"])
    return cfg


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _machine_info() -> dict:
    a = np.random.default_rng(0).random((256, 256))
    t0 = time.perf_counter()
    for _ in range(20):
        a = np.tanh(a @ a / 256.0)
    x = 0.0
    for i in range(200000):
        x += math.sin(i)
    bench = (time.perf_counter() - t0) * 1e3
    try:
        import psutil
        cores, mem = psutil.cpu_count(logical=True), psutil.virtual_memory().total / 2 ** 30
    except Exception:
        cores, mem = None, None
    return {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
            "logical_cores": cores, "ram_gb": round(mem, 1) if mem else None, "bench_ms": round(bench, 1)}


class CommittedPath:
    """Path-deviation reference. Each position is compared with the path the controller
    published ``commit_s`` earlier, near the point on it the car should have reached by now
    (distance driven since publishing, +- a slack window). Publishing a stub path just ahead of
    the car, or a zig-zag that covers the corridor, therefore earns nothing; implausible paths
    are ignored and the centerline is used instead."""

    def __init__(self, commit_s: float):
        self.commit_s = commit_s
        self.pending = deque()   # (t, path, arc length of the car's position on it, odometer)
        self.ref = None

    def publish(self, t, path, pos, odo):
        if path is None or not path_is_plausible(path):
            self.pending.append((t, None, 0.0, odo))  # an implausible path replaces the plan too
            return
        d = np.hypot(*(path - np.asarray(pos)).T)
        i = int(np.argmin(d))
        s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
        self.pending.append((t, path, float(s[i]), odo))

    def error(self, t, pos, odo):
        """(error, has_path); has_path False means: no committed path, use the centerline."""
        while self.pending and self.pending[0][0] <= t - self.commit_s + 1e-9:
            self.ref = self.pending.popleft()
        if self.ref is None or self.ref[1] is None:
            return None, False
        _, path, s0, odo0 = self.ref
        s_exp = s0 + (odo - odo0)
        slack = 3.0 + 0.25 * (odo - odo0)
        return polyline_window_distance(path, s_exp - slack, s_exp + slack, pos), True


def _check_command(cmd, pose) -> tuple[Command | None, bool]:
    """Validate a controller's return value. Returns (command or None if unusable, clipped?).
    The returned command's path is always in the world frame (vehicle-frame paths are
    transformed with ``pose``, the (x, y, yaw) of the observation being answered)."""
    if not isinstance(cmd, Command):
        return None, False
    vals = (cmd.steer, cmd.throttle, cmd.brake)
    try:
        vals = tuple(float(v) for v in vals)
    except (TypeError, ValueError):
        return None, False
    if not all(math.isfinite(v) for v in vals):
        return None, False
    clipped = abs(vals[0]) > 1.0 + 1e-6 or not (-1e-6 <= vals[1] <= 1 + 1e-6) or not (-1e-6 <= vals[2] <= 1 + 1e-6)
    path = None
    if cmd.path is not None:
        try:
            path = np.asarray(cmd.path, dtype=float).reshape(-1, 2)[:2000]
            if not np.isfinite(path).all() or len(path) == 0:
                path = None
            elif cmd.path_frame == "vehicle":
                path = body_to_world(path, *pose)
            elif cmd.path_frame != "world":
                path = None
        except (TypeError, ValueError):
            path = None
    ts = cmd.target_speed if isinstance(cmd.target_speed, (int, float)) else None
    dbg = {str(k): float(v) for k, v in (cmd.debug or {}).items() if isinstance(v, (int, float))}
    return Command(steer=float(np.clip(vals[0], -1, 1)), throttle=float(np.clip(vals[1], 0, 1)),
                   brake=float(np.clip(vals[2], 0, 1)), path=path, path_frame="world", target_speed=ts,
                   debug=dbg), clipped


def run(cfg: dict, controller: str, track_path: str, *, team: str | None = None, backend: str | None = None,
        perturb: dict | None = None, controller_class: str | None = None, controller_cfg: dict | None = None,
        out_dir: str | Path | None = None, tag: str = "", rubric: dict | None = None, viewer=None,
        backend_obj=None, verbose: bool = True, recorder=None) -> dict:
    """``recorder`` (optional) is called with ``on_control(obs, raw, cmd, stats, clipped, held)``
    at every control step and ``on_tick(state, applied, monitor, is_off)`` after every physics
    tick; the data generator uses it to log observation/action pairs."""
    perturb = dict(perturb or {})
    cfg = apply_perturbation(cfg, perturb)
    backend = backend or cfg["backend"]
    sim = cfg["sim"]
    dt = float(sim["physics_dt"])
    tpc = int(round(1.0 / (float(sim["control_hz"]) * dt)))
    if abs(tpc * dt * float(sim["control_hz"]) - 1.0) > 1e-6:
        raise ValueError("sim.control_hz must divide the physics rate 1/physics_dt")
    control_dt = tpc * dt
    seed = int(sim.get("seed", 0))
    rng = np.random.default_rng(seed)
    team = team or Path(controller).stem
    track = load_track(track_path)
    rules = cfg["rules"]

    stamp = dt_.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir) if out_dir else ROOT / cfg["output"]["runs_dir"] / f"{stamp}_{team}_{track.name}{'_' + tag if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)

    be = backend_obj or make_backend(backend, cfg)
    ctrl = None
    rows, dbg_rows = [], []
    run_info = {"controller_errors": 0, "init_ms": float("nan"), "peak_mem_bytes": 0, "realtime_max_lag_s": 0.0,
                "controller_cpu_s": float("nan")}
    first_error = None
    wall0 = time.perf_counter()
    try:
        pp = be.setup(track, cfg["vehicle"], perturb, rng)
        vcfg = cfg["vehicle"]
        vp = VehicleParams(
            mass=pp.mass, wheelbase=pp.wheelbase, track_width=pp.track_width, length=pp.length, width=pp.width,
            cg_to_front=pp.cg_to_front, cg_to_rear=pp.cg_to_rear, max_steer=pp.max_steer,
            max_steer_rate=float(vcfg["steer_rate_limit"]), mu=float(vcfg["mu"]), max_accel=float(vcfg["max_accel"]),
            max_decel=float(vcfg["max_decel"]), max_speed=float(vcfg["max_speed"]), control_dt=control_dt,
            actuation_delay_steps=int(cfg["actuators"].get("delay_steps", 0)))
        ref_time = float(track.reference_time) if track.reference_time else track.reference_time_estimate(vp)
        monitor = RaceMonitor(track, rules, vp, float(sim["max_time_s"]))
        sensor = ConeSensor(cfg["perception"], rng)
        road_sensor = RoadModelSensor(track, cfg.get("road_model", {}), rng)
        cones, colors, _ = track.all_cones()
        info = track.info(cfg.get("track_knowledge", "full") == "full", float(rules["gate_spacing_m"]))

        protected = [Path(track_path).resolve(), ROOT / cfg["output"]["runs_dir"], out.resolve()] + [
            p if p.is_absolute() else ROOT / p for p in map(Path, cfg.get("sandbox", {}).get("protected_paths", []) or [])]
        ctrl = ControllerProcess(controller, cfg["compute_budget"], controller_class, controller_cfg, protected)
        try:
            ctrl.start(info, vp)
            run_info["init_ms"] = ctrl.init_ms
        except (TimeoutError, RuntimeError) as e:
            monitor.dnf = "controller_error"
            first_error = str(e)

        x, y, yaw = track.spawn_pose()
        off = float(perturb.get("start_lateral_offset", 0.0))
        x, y = x - math.sin(yaw) * off, y + math.cos(yaw) * off
        yaw += math.radians(float(perturb.get("start_yaw_offset_deg", 0.0)))
        st = be.reset(x, y, yaw)

        delay = deque([Command(brake=1.0)] * vp.actuation_delay_steps)
        steer_bias = float(perturb.get("steer_bias", 0.0))
        target = Command(brake=1.0)
        last_cmd = Command(brake=1.0)
        app = [0.0, 0.0, 1.0]
        path = None
        committed = CommittedPath(float(rules.get("path_commit_s", 0.5)))
        odo = 0.0
        err_streak = 0
        k = 0
        rate = vp.max_steer_rate * dt
        rt_factor = float(sim.get("realtime_factor", 0.0) or 0.0)
        pace_t0 = time.perf_counter() - st.t / rt_factor if rt_factor > 0 else 0.0
        while not monitor.done:
            is_ctrl = k % tpc == 0
            stats = None
            clipped = False
            if is_ctrl:
                obs = Observation(
                    t=st.t, step=k // tpc, x=st.x, y=st.y, yaw=st.yaw, vx=st.vx, vy=st.vy, yaw_rate=st.yaw_rate,
                    ax=st.ax, ay=st.ay, speed=st.speed, steer=app[0], throttle=app[1], brake=app[2],
                    progress=monitor.progress, lap=monitor.lap, timing_started=monitor.started,
                    cones=sensor.detect(cones, colors, st.x, st.y, st.yaw),
                    road=road_sensor.measure(st.x, st.y, st.yaw, monitor.progress))
                raw, stats = ctrl.step(obs)
                if stats.error:
                    run_info["controller_errors"] += 1
                    first_error = first_error or stats.error
                cmd, clipped = _check_command(raw, (obs.x, obs.y, obs.yaw)) if raw is not None else (None, False)
                if raw is not None and cmd is None and not stats.error:
                    run_info["controller_errors"] += 1
                    first_error = first_error or f"invalid command returned: {raw!r}"[:500]
                held = cmd is None
                if cmd is None:
                    err_streak += 1 if (stats.error or raw is not None) else 0
                    cmd = last_cmd
                else:
                    err_streak = 0
                    last_cmd = cmd
                if err_streak >= int(rules.get("max_consecutive_errors", 20)) or ctrl.dead:
                    monitor.dnf = "controller_error"
                if recorder is not None:
                    recorder.on_control(obs, raw, cmd, stats, clipped, held)
                if cmd.path is not None:
                    path = cmd.path
                    be.draw_path(path, control_dt * 1.5)
                    committed.publish(st.t, path, (st.x, st.y), odo)
                delay.append(cmd)
                target = delay.popleft()
            # actuator model: steering slew-rate limit + bias; pedals direct
            app[0] = float(np.clip(app[0] + np.clip(target.steer + steer_bias - app[0], -rate, rate), -1, 1))
            app[1], app[2] = target.throttle, target.brake
            be.apply(*app)
            st = be.tick()
            if rt_factor > 0:  # pace sim time to wall time (lockstep semantics are unchanged)
                ahead = pace_t0 + st.t / rt_factor - time.perf_counter()
                if ahead > 0:
                    time.sleep(ahead)
                else:
                    run_info["realtime_max_lag_s"] = max(run_info["realtime_max_lag_s"], -ahead)
                    if -ahead > 0.25:  # too far behind: drop the backlog instead of sprinting to catch up
                        pace_t0 -= ahead
            is_off = monitor.update(st, dt)
            if recorder is not None:
                recorder.on_tick(st, (app[0], app[1], app[2]), monitor, is_off)
            odo += st.speed * dt
            pe, has_path = committed.error(st.t, (st.x, st.y), odo)
            if not has_path:
                pe = abs(monitor.lateral)
            rows.append((st.t, st.x, st.y, st.yaw, st.vx, st.vy, st.speed, st.yaw_rate, st.ax, st.ay, st.roll, st.pitch,
                         st.steer_angle, monitor.progress, monitor.lateral, monitor.boundary_margin, pe,
                         float(has_path), float(is_off), last_cmd.steer, last_cmd.throttle, last_cmd.brake,
                         app[0], app[1], app[2], float(is_ctrl),
                         stats.wall_ms if stats else math.nan, stats.cpu_ms if stats else math.nan,
                         stats.latency_ms if stats else math.nan, float(stats.overrun) if stats else 0.0,
                         float(clipped), last_cmd.target_speed if last_cmd.target_speed is not None else math.nan))
            dbg_rows.append(last_cmd.debug if is_ctrl else {})
            if viewer is not None and k % max(int(round(1 / (30 * dt))), 1) == 0:
                if not viewer.update(st, monitor, path, obs if is_ctrl else None):
                    monitor.dnf = monitor.dnf or "aborted"
            k += 1
    finally:
        if ctrl is not None:
            ctrl.close()
            run_info["peak_mem_bytes"] = ctrl.peak_rss
            run_info["controller_cpu_s"] = ctrl.drive_cpu_s
        be.close()

    wall = time.perf_counter() - wall0
    # ------------------------------------------------------------------ outputs
    arr = np.array(rows, dtype=float).reshape(-1, len(LOG_COLS))
    log = {c: arr[:, i] for i, c in enumerate(LOG_COLS)}
    dbg_keys = sorted({k for d in dbg_rows for k in d})
    with open(out / "telemetry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLS + [f"dbg_{k}" for k in dbg_keys])
        for r, d in zip(rows, dbg_rows):
            w.writerow([f"{v:.6g}" for v in r] + [f"{d[k]:.6g}" if k in d else "" for k in dbg_keys])
    events = [{"t": e.t, "kind": e.kind, "x": e.x, "y": e.y, "progress": e.progress, "detail": e.detail}
              for e in monitor.events]
    (out / "events.json").write_text(json.dumps(events, indent=1))
    span = track.finish_progress - track.start_progress
    run_state = {
        "start_time": monitor.start_time, "finish_time": monitor.finish_time, "dnf": monitor.dnf,
        "lap_times": monitor.lap_times, "off_course_time": monitor.off_time,
        "progress_fraction": (monitor.max_progress - track.start_progress) / span if span > 0 else 0.0,
        "reference_time": ref_time, "control_hz": float(sim["control_hz"]), **run_info,
    }
    vpd = vp.__dict__
    m = metrics_mod.compute(log, events, run_state, vpd, cfg["compute_budget"])
    sc = score_mod.score(m, rubric)
    summary = {
        "kind": "run", "team": team, "tag": tag,
        "controller": {"path": str(Path(controller).resolve()), "sha256": _sha(controller), "class": controller_class},
        "track": {"name": track.name, "path": str(Path(track_path).resolve()), "sha256": _sha(track_path),
                  "length_m": track.length, "closed": track.closed, "laps": track.laps},
        "backend": backend, "seed": seed, "perturbation": perturb, "vehicle": vpd,
        "created": stamp, "wall_time_s": round(wall, 2), "sim_time_s": float(log["t"][-1]) if len(rows) else 0.0,
        "realtime": {"factor": float(sim.get("realtime_factor", 0.0) or 0.0),
                     "max_lag_s": round(run_info["realtime_max_lag_s"], 3)},
        "machine": _machine_info(), "controller_error": first_error, "metrics": m, "score": sc, "config": cfg,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if cfg["output"].get("plot", True):
        try:
            from .plot import plot_run
            plot_run(out, track)
        except Exception as e:  # plotting must never lose a run
            print(f"[plot] skipped: {e}")
    if verbose:
        print_summary(summary, out)
    summary["out_dir"] = str(out)
    return summary


def print_summary(s: dict, out) -> None:
    m, sc = s["metrics"], s["score"]
    c = m["completion"]
    status = f"FINISHED in {c['time_s']:.2f} s" if c["finished"] else f"DNF ({c['dnf_reason']}) at {100 * c['progress_fraction']:.0f}%"
    p = m["penalties"]
    print(f"\n=== {s['team']} on {s['track']['name']} [{s['backend']}] -> {status}")
    print(f"    reference {c['reference_time_s']:.2f} s | cones {p['cone_hit']} | off-course {p['off_course']} | "
          f"missed gates {p['missed_gate']} | adjusted time {sc['adjusted_time_s']}")
    d, k, y = m["deviation"], m["compute"], m["dynamics"]
    print(f"    path err mean/max {d['path_err_mean_m']:.2f}/{d['path_err_max_m']:.2f} m ({d['path_source']}) | "
          f"step p99 {k['step_ms_p99']:.2f} ms / {k['deadline_ms']:.0f} ms, overruns {k['overruns']}, "
          f"mem {k['peak_memory_mb']:.0f} MB | rear slip max {y['rear_slip_max_deg']:.1f} deg, "
          f"grip-demand>110% {100 * y['grip_demand_over_frac']:.1f}%")
    print(f"    category scores {sc['category_scores']} | deductions {sum(sc['deductions'].values()):.0f} | "
          f"technical {sc['technical_points']:.1f}/{sc['technical_max']}")
    if s.get("controller_error"):
        print(f"    controller error: {s['controller_error'].strip().splitlines()[-1]}")
    rt = s.get("realtime", {})
    if rt.get("factor") and rt.get("max_lag_s", 0) > 0.05:
        print(f"    realtime x{rt['factor']:g}: could not keep up (max lag {rt['max_lag_s']:.2f} s)")
    print(f"    -> {out}")


def run_batch(cfg: dict, controller: str, batch: dict, *, team=None, backend=None, controller_class=None,
              controller_cfg=None, out_root=None, verbose=True) -> dict:
    """Robustness sweep: tracks x perturbation sets x seeds (config/robustness.yaml)."""
    stamp = dt_.datetime.now().strftime("%Y%m%d-%H%M%S")
    team = team or Path(controller).stem
    root = Path(out_root) if out_root else ROOT / cfg["output"]["runs_dir"] / f"{stamp}_{team}_batch"
    results = []
    be = None
    backend = backend or cfg["backend"]
    for tp in batch["tracks"]:
        for pname, pert in (batch.get("perturbations") or {"nominal": {}}).items():
            for seed in batch.get("seeds", [0]):
                c = deep_update(copy.deepcopy(cfg), {"sim": {"seed": int(seed)}})
                c = deep_update(c, batch.get("overrides", {}))
                name = f"{Path(tp).stem}_{pname}_s{seed}"
                s = run(c, controller, tp, team=team, backend=backend, perturb=pert or {},
                        controller_class=controller_class, controller_cfg=controller_cfg,
                        out_dir=root / name, tag=pname, backend_obj=be, verbose=verbose)
                results.append({"track": Path(tp).stem, "perturbation": pname, "seed": seed,
                                "finished": s["metrics"]["completion"]["finished"],
                                "dnf_reason": s["metrics"]["completion"]["dnf_reason"],
                                "time_s": s["metrics"]["completion"]["time_s"],
                                "progress": s["metrics"]["completion"]["progress_fraction"],
                                "cones": s["metrics"]["penalties"]["cone_hit"],
                                "off_course": s["metrics"]["penalties"]["off_course"],
                                "missed_gates": s["metrics"]["penalties"]["missed_gate"],
                                "technical": s["score"]["technical_points"], "dir": name})
    tech = np.array([r["technical"] for r in results])
    fin = np.array([r["finished"] for r in results])
    times = np.array([r["time_s"] for r in results if r["time_s"] is not None])
    agg = {"kind": "batch", "team": team, "runs": len(results), "finish_rate": float(fin.mean()),
           "technical_mean": float(tech.mean()), "technical_min": float(tech.min()), "technical_std": float(tech.std()),
           "time_mean_s": float(times.mean()) if len(times) else None,
           "time_std_s": float(times.std()) if len(times) else None,
           "by_perturbation": {}, "results": results}
    for pname in dict.fromkeys(r["perturbation"] for r in results):
        sub = [r for r in results if r["perturbation"] == pname]
        agg["by_perturbation"][pname] = {"finish_rate": float(np.mean([r["finished"] for r in sub])),
                                         "technical_mean": float(np.mean([r["technical"] for r in sub]))}
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_summary.json").write_text(json.dumps(agg, indent=2))
    if verbose:
        print(f"\n=== Robustness batch: {len(results)} runs, finish rate {100 * agg['finish_rate']:.0f}%, "
              f"technical mean {agg['technical_mean']:.1f} (min {agg['technical_min']:.1f}, std {agg['technical_std']:.1f})")
        for k, v in agg["by_perturbation"].items():
            print(f"    {k:<22} finish {100 * v['finish_rate']:5.0f}%   technical {v['technical_mean']:6.1f}")
        print(f"    -> {root}")
    return agg
