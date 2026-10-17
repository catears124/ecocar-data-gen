"""Metric extraction from a run's telemetry. Every metric is recomputable offline from
telemetry.csv + events.json, so results can be audited and re-scored."""
from __future__ import annotations

import math

import numpy as np

G = 9.81


def _pct(a, q):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _mean(a):
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def compute(log: dict, events: list[dict], run: dict, vp: dict, budget: dict) -> dict:
    t = log["t"]
    ctrl = log["ctrl_step"] > 0
    counts = {k: sum(1 for e in events if e["kind"] == k) for k in ("cone_hit", "off_course", "missed_gate")}
    obstacle_hits = sum(1 for e in events if e["kind"] == "cone_hit" and not e["detail"].get("mapped", True))

    # ---------------------------------------------------------------- completion
    start, finish = run["start_time"], run["finish_time"]
    completion = {
        "finished": finish is not None and run["dnf"] is None,
        "dnf_reason": run["dnf"],
        "time_s": (finish - start) if (finish is not None and start is not None) else None,
        "lap_times_s": run["lap_times"],
        "progress_fraction": float(np.clip(run["progress_fraction"], 0.0, 1.0)),
        "reference_time_s": run["reference_time"],
    }

    # ---------------------------------------------------------------- deviation (timed window)
    t0 = start if start is not None else math.inf
    t1 = finish if finish is not None else t[-1] if len(t) else 0.0
    win = (t >= t0) & (t <= t1)
    pe = log["path_err"][win]
    lat = np.abs(log["lateral"][win])
    has_path = log["has_path"][win]
    deviation = {
        "path_err_mean_m": _mean(pe),
        "path_err_rms_m": float(np.sqrt(np.nanmean(pe ** 2))) if len(pe) else float("nan"),
        "path_err_max_m": float(np.nanmax(pe)) if len(pe) else float("nan"),
        "path_source": "none" if not len(has_path) else ("controller" if has_path.all() else
                                                          "centerline" if not has_path.any() else "mixed"),
        "centerline_lat_mean_m": _mean(lat),
        "centerline_lat_max_m": float(lat.max()) if len(lat) else float("nan"),
        "min_boundary_margin_m": float(np.nanmin(log["boundary_margin"][win])) if win.any() else float("nan"),
        "off_course_time_s": run["off_course_time"],
    }

    # ---------------------------------------------------------------- compute
    wall = log["compute_ms"][ctrl]
    cpu = log["cpu_ms"][ctrl]
    deadline = float(budget.get("step_deadline_ms", 50.0))
    mem_mb = run["peak_mem_bytes"] / 2 ** 20
    overruns = int(log["overrun"][ctrl].sum())
    n = int(ctrl.sum())
    compute = {
        "steps": n,
        "step_ms_mean": _mean(wall), "step_ms_p50": _pct(wall, 50), "step_ms_p95": _pct(wall, 95),
        "step_ms_p99": _pct(wall, 99), "step_ms_max": float(np.nanmax(wall)) if np.isfinite(wall).any() else float("nan"),
        "cpu_ms_mean": _mean(cpu),
        # all CPU the controller process used while driving (threads included), per control step
        "cpu_ms_per_step": run.get("controller_cpu_s", float("nan")) * 1e3 / max(n, 1),
        "latency_ms_p99": _pct(log["latency_ms"][ctrl], 99),
        "deadline_ms": deadline,
        "cpu_utilisation": _mean(cpu) / deadline if n else float("nan"),
        "overruns": overruns, "overrun_fraction": overruns / max(n, 1),
        "errors": int(run["controller_errors"]),
        "init_ms": run["init_ms"],
        "peak_memory_mb": mem_mb,
        "memory_limit_mb": float(budget.get("memory_mb", 1024)),
        "over_memory": mem_mb > float(budget.get("memory_mb", 1024)),
        "cpu_cores": budget.get("cpu_cores"), "mode": budget.get("mode"),
    }

    # ---------------------------------------------------------------- dynamics compatibility
    spd = log["speed"]
    moving = spd > 3.0
    mu_g = vp["mu"] * G
    delta_cmd = np.clip(log["cmd_steer"], -1, 1) * vp["max_steer"]
    demand = spd ** 2 * np.abs(np.tan(delta_cmd)) / vp["wheelbase"] / mu_g
    vx_abs = np.maximum(np.abs(log["vx"]), 0.1)
    beta = np.degrees(np.arctan2(log["vy"], vx_abs))
    # rear-axle slip angle: ~0 while the tyres roll (even in tight low-speed turns, unlike CG
    # sideslip), large when the rear is sliding
    rear_slip = np.degrees(np.arctan2(log["vy"] - vp["cg_to_rear"] * log["yaw_rate"], vx_abs))
    a_meas = np.hypot(log["ax"], log["ay"]) / mu_g
    dt_c = 1.0 / run["control_hz"]
    cs = log["cmd_steer"][ctrl]
    steer_rate_req = np.abs(np.diff(cs)) / dt_c if len(cs) > 1 else np.zeros(0)
    ct, cb = log["cmd_throttle"][ctrl], log["cmd_brake"][ctrl]
    dynamics = {
        "grip_demand_p95": _pct(demand[moving], 95),
        "grip_demand_over_frac": float((demand[moving] > 1.1).mean()) if moving.any() else 0.0,
        "sideslip_max_deg": float(np.abs(beta[moving]).max()) if moving.any() else 0.0,
        "rear_slip_max_deg": float(np.abs(rear_slip[moving]).max()) if moving.any() else 0.0,
        "rear_slip_over_frac": float((np.abs(rear_slip[moving]) > 6.0).mean()) if moving.any() else 0.0,
        "accel_utilisation_p99": _pct(a_meas[moving], 99),
        "steer_rate_sat_frac": float((steer_rate_req > vp["max_steer_rate"] * 1.05).mean()) if len(steer_rate_req) else 0.0,
        "steer_rate_rms": float(np.sqrt(np.mean(steer_rate_req ** 2))) if len(steer_rate_req) else 0.0,
        "pedal_overlap_frac": float(((ct > 0.05) & (cb > 0.05)).mean()) if n else 0.0,
        "clipped_cmd_frac": float(log["clipped"][ctrl].mean()) if n else 0.0,
    }

    penalties = dict(counts, obstacle_hits=obstacle_hits)
    return {"completion": completion, "penalties": penalties, "deviation": deviation,
            "compute": compute, "dynamics": dynamics}
