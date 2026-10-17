"""Turns run metrics into rubric points (config/scoring.yaml) and builds leaderboards."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .config import DEFAULT_SCORING, load_yaml


def _lin(value, full_zero) -> float:
    """100 at <= full, 0 at >= zero (linear between). NaN -> 0."""
    full, zero = full_zero
    if value is None or not math.isfinite(value):
        return 0.0
    if zero == full:
        return 100.0 if value <= full else 0.0
    return float(min(max((zero - value) / (zero - full), 0.0), 1.0) * 100.0)


def score(metrics: dict, rubric: dict | None = None, judged: dict | None = None) -> dict:
    r = rubric or load_yaml(DEFAULT_SCORING)
    w = r["weights"]
    comp, dev, cpu, dyn, pen = (metrics[k] for k in ("completion", "deviation", "compute", "dynamics", "penalties"))
    finished = comp["finished"]
    prog = comp["progress_fraction"] if not finished else 1.0

    cats = {}
    ref = comp.get("reference_time_s")
    if finished and ref:
        ratio = comp["time_s"] / ref
        cats["completion_time"] = _lin(ratio, (r["time"]["full_marks_ratio"], r["time"]["zero_marks_ratio"]))
    else:
        cats["completion_time"] = float(r["dnf"]["time_points"])

    dw = r["deviation"]["weights"]
    cats["deviation"] = (dw["mean"] * _lin(dev["path_err_mean_m"], r["deviation"]["path_err_mean_m"])
                         + dw["max"] * _lin(dev["path_err_max_m"], r["deviation"]["path_err_max_m"]))

    cw = r["compute"]["weights"]
    # load = the worse of step latency (p99) and CPU used per control step, so work moved into
    # background threads between steps still counts
    load = max(cpu["step_ms_p99"], cpu.get("cpu_ms_per_step", 0.0) if math.isfinite(cpu.get("cpu_ms_per_step", 0.0)) else 0.0)
    cats["compute"] = (cw["latency"] * _lin(load / cpu["deadline_ms"], r["compute"]["p99_fraction_of_deadline"])
                       + cw["overruns"] * _lin(cpu["overrun_fraction"], r["compute"]["overrun_fraction"])
                       + cw["memory"] * _lin(cpu["peak_memory_mb"] / cpu["memory_limit_mb"], r["compute"]["memory_fraction_of_limit"]))

    yw = r["dynamics"]["weights"]
    cats["dynamics"] = (yw["grip"] * _lin(dyn["grip_demand_over_frac"], r["dynamics"]["grip_demand_over_frac"])
                        + yw["rear_slip"] * _lin(dyn["rear_slip_over_frac"], r["dynamics"]["rear_slip_over_frac"])
                        + yw["steer_rate"] * _lin(dyn["steer_rate_sat_frac"], r["dynamics"]["steer_rate_sat_frac"])
                        + yw["pedal"] * _lin(dyn["pedal_overlap_frac"], r["dynamics"]["pedal_overlap_frac"]))

    if not finished and r["dnf"].get("scale_other_by_progress", True):
        for k in ("deviation", "compute", "dynamics"):
            cats[k] *= prog

    points = {k: cats[k] * w[k] / 100.0 for k in cats}
    pp = r["penalties"]
    deductions = {
        "cone_hit": pen["cone_hit"] * pp["cone_hit"],
        "off_course": pen["off_course"] * pp["off_course"],
        "missed_gate": pen["missed_gate"] * pp["missed_gate"],
        "controller_error": min(cpu["errors"], 10) * pp["controller_error"],
        "over_memory": pp["over_memory"] if cpu["over_memory"] else 0.0,
    }
    technical = max(sum(points.values()) - sum(deductions.values()), 0.0)
    judged = judged or {}
    judged_points = {k: float(v) * w[k] / 100.0 for k, v in judged.items() if k in w}
    at = r.get("adjusted_time_s", {})
    adj = (comp["time_s"] + sum(pen[k] * at.get(k, 0) for k in ("cone_hit", "missed_gate", "off_course"))
           if comp["time_s"] is not None else None)
    return {
        "category_scores": {k: round(v, 2) for k, v in cats.items()},
        "category_points": {k: round(v, 2) for k, v in points.items()},
        "deductions": deductions,
        "technical_points": round(technical, 2),
        "technical_max": sum(v for k, v in w.items() if k in cats),
        "judged_points": judged_points,
        "total_points": round(technical + sum(judged_points.values()), 2),
        "adjusted_time_s": round(adj, 3) if adj is not None else None,
    }


def rescore(run_dir: str | Path, rubric_path=None, judged: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    rubric = load_yaml(rubric_path) if rubric_path else None
    summary["score"] = score(summary["metrics"], rubric, judged or summary.get("score", {}).get("judged_input"))
    if judged:
        summary["score"]["judged_input"] = judged
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def leaderboard(runs_root: str | Path, out_csv: str | Path | None = None, track: str | None = None) -> list[dict]:
    rows = []
    for p in sorted(Path(runs_root).glob("**/summary.json")):
        s = json.loads(p.read_text())
        if s.get("kind") == "batch" or (track and s["track"]["name"] != track):
            continue
        m, sc = s["metrics"], s["score"]
        rows.append({
            "team": s["team"], "track": s["track"]["name"], "run": p.parent.name,
            "finished": m["completion"]["finished"], "time_s": m["completion"]["time_s"],
            "adjusted_time_s": sc["adjusted_time_s"],
            "cones": m["penalties"]["cone_hit"], "off_course": m["penalties"]["off_course"],
            "missed_gates": m["penalties"]["missed_gate"],
            **{f"{k}_score": v for k, v in sc["category_scores"].items()},
            "technical": sc["technical_points"], "total": sc["total_points"],
        })
    best: dict = {}
    for r in rows:
        key = (r["team"], r["track"])
        if key not in best or r["total"] > best[key]["total"]:
            best[key] = r
    board = sorted(best.values(), key=lambda r: -r["total"])
    if out_csv and board:
        with open(out_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(board[0].keys()))
            wr.writeheader()
            wr.writerows(board)
    return board


def format_table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return "-" if v is None else str(v)
    cells = [[fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line, sep] + ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells])
