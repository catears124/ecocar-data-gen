"""Closed-loop benchmark: controllers x courses x seeds on the bicycle backend.

    python tools/bench.py --controller timed=teams/timed_exact.py \
        --controller certline=teams/certline.py --tracks "tracks/*.yaml" \
        --seeds 0 --jobs 6 --mode measure --out experiments/bench/public.json

``--controller NAME=PATH[@CONFIG.yaml]``.  ``--mode measure`` (default) waits for every
step (no held commands) and is safe to run in parallel; ``--mode enforce`` is the
official deadline mode and should be run with ``--jobs 1`` for latency claims.

Per run it records completion, raw/penalised time, cones (boundary/obstacle),
off-course, missed gates, technical points and categories, compute p50/p99/max,
peak memory, and, from telemetry ``dbg_*`` columns, the status-ladder fractions,
fallback reasons and the feasible fraction of exact steps.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _telemetry_stats(run_dir: Path) -> dict:
    p = run_dir / "telemetry.csv"
    if not p.exists():
        return {}
    rows = [r for r in csv.DictReader(open(p, newline="")) if r.get("ctrl_step") not in (None, "", "0", "0.0")]
    n = len(rows)
    out: dict = {"control_steps": n}
    if not n:
        return out
    st = Counter(r.get("dbg_status", "") for r in rows)
    out["status_counts"] = {k or "missing": v for k, v in sorted(st.items())}
    exact = [r for r in rows if r.get("dbg_status") in ("1", "2", "3")]
    out["exact_fraction"] = len(exact) / n
    out["fallback_reasons"] = dict(Counter(r.get("dbg_fallback_reason", "") for r in rows
                                           if r.get("dbg_status") == "0"))
    feas = [r for r in exact if r.get("dbg_feasible") not in (None, "")]
    if feas:
        out["feasible_fraction_of_exact"] = sum(r["dbg_feasible"] == "1" for r in feas) / len(feas)
    solve = sorted(float(r["dbg_solve_ms"]) for r in exact if r.get("dbg_solve_ms"))
    if solve:
        out["solve_ms_p50"] = solve[len(solve) // 2]
        out["solve_ms_p99"] = solve[min(len(solve) - 1, int(0.99 * len(solve)))]
        out["solve_ms_max"] = solve[-1]
    return out


def _one(job: dict) -> dict:
    from autocross.config import load_sim_config, load_yaml
    from autocross.runner import run
    cfg = load_sim_config()
    cfg["backend"] = "bicycle"
    cfg["output"]["plot"] = False
    cfg["compute_budget"]["mode"] = job["mode"]
    cfg["sim"]["seed"] = int(job["seed"])
    for k, v in (job.get("set") or {}).items():
        d = cfg
        parts = k.split(".")
        for q in parts[:-1]:
            d = d[q]
        d[parts[-1]] = v
    ctrl_cfg = load_yaml(job["config"]) if job.get("config") else None
    out_dir = Path(job["out_dir"])
    rec = None
    if job.get("record_obs"):
        class Rec:
            def __init__(self):
                self.steps = []

            def on_control(self, obs, raw, cmd, stats, clipped, held):
                self.steps.append((obs, cmd))

            def on_tick(self, st, applied, monitor, is_off):
                pass
        rec = Rec()
    s = run(copy.deepcopy(cfg), job["path"], job["track"], team=job["name"], backend="bicycle",
            controller_cfg=ctrl_cfg, out_dir=out_dir, verbose=False, perturb=job.get("perturb") or {},
            recorder=rec)
    if rec is not None:
        import pickle
        (out_dir / "observations.pkl").write_bytes(pickle.dumps(rec.steps, protocol=4))
    m, sc = s["metrics"], s["score"]
    c, pen, k = m["completion"], m["penalties"], m["compute"]
    row = {
        "controller": job["name"], "track": Path(job["track"]).stem, "seed": job["seed"],
        "perturb": job.get("perturb") or {},
        "finished": bool(c["finished"]), "dnf_reason": c.get("dnf_reason"),
        "progress": c.get("progress_fraction"), "time_s": c.get("time_s"),
        "reference_time_s": c.get("reference_time_s"),
        "adjusted_time_s": sc.get("adjusted_time_s"),
        "cones": pen["cone_hit"], "obstacle_hits": pen.get("obstacle_hit", 0), "off_course": pen["off_course"],
        "missed_gate": pen["missed_gate"], "technical": sc["technical_points"],
        "category_scores": sc.get("category_scores"),
        "step_ms_p50": k["step_ms_p50"], "step_ms_p99": k["step_ms_p99"], "step_ms_max": k["step_ms_max"],
        "overruns": k["overruns"], "peak_memory_mb": k.get("peak_memory_mb"),
        "path_err_mean_m": m["deviation"]["path_err_mean_m"], "path_err_max_m": m["deviation"]["path_err_max_m"],
        "controller_error": (s.get("controller_error") or "")[-300:], "run_dir": str(out_dir),
    }
    row.update(_telemetry_stats(out_dir))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", action="append", required=True)
    ap.add_argument("--tracks", action="append", required=True, help="glob(s)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--mode", default="measure", choices=["measure", "enforce"])
    ap.add_argument("--set", action="append", default=[], help="config override key=value (float/int/str)")
    ap.add_argument("--perturb", action="append", default=[], help="perturbation key=value")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--record-obs", action="store_true", help="save the observation stream (observations.pkl)")
    a = ap.parse_args()

    def _val(x: str):
        for f in (int, float):
            try:
                return f(x)
            except ValueError:
                pass
        return x

    sets = {k: _val(v) for k, v in (s.split("=", 1) for s in a.set)}
    pert = {k: _val(v) for k, v in (s.split("=", 1) for s in a.perturb)}
    tracks = sorted({p for g in a.tracks for p in glob.glob(g)})
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out)
    runs_root = Path(a.runs_root) if a.runs_root else out.with_suffix("")
    jobs = []
    for spec in a.controller:
        name, rest = spec.split("=", 1)
        path, _, conf = rest.partition("@")
        for t in tracks:
            for sd in seeds:
                jobs.append({"name": name, "path": path, "config": conf or None, "track": t, "seed": sd,
                             "mode": a.mode, "set": sets, "perturb": pert, "record_obs": a.record_obs,
                             "out_dir": str(runs_root / f"{name}_{Path(t).stem}_s{sd}")})
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for r in ex.map(_one, jobs):
            rows.append(r)
            print(f"{r['controller']:<14}{r['track']:<34}s{r['seed']:<3}"
                  f"{'FIN' if r['finished'] else 'DNF':<4}{(r['time_s'] or float('nan')):8.2f}s "
                  f"cones {r['cones']} off {r['off_course']} gates {r['missed_gate']} tech {r['technical']:.1f} "
                  f"p99 {r['step_ms_p99']:.1f}ms exact {r.get('exact_fraction', float('nan')):.2f}", flush=True)
    summary: dict = {"jobs": len(rows), "mode": a.mode, "seeds": seeds, "tracks": [Path(t).stem for t in tracks],
                     "set": sets, "perturb": pert, "by_controller": {}}
    for name in dict.fromkeys(r["controller"] for r in rows):
        sub = [r for r in rows if r["controller"] == name]
        fin = [r for r in sub if r["finished"]]
        summary["by_controller"][name] = {
            "runs": len(sub), "finished": len(fin),
            "technical_mean": sum(r["technical"] for r in sub) / len(sub),
            "time_mean_finished_s": (sum(r["time_s"] for r in fin) / len(fin)) if fin else None,
            "adjusted_time_mean_finished_s": (sum(r["adjusted_time_s"] for r in fin) / len(fin)) if fin else None,
            "cones": sum(r["cones"] for r in sub), "off_course": sum(r["off_course"] for r in sub),
            "missed_gate": sum(r["missed_gate"] for r in sub),
            "exact_fraction_mean": (sum(r.get("exact_fraction", 0.0) for r in sub) / len(sub)),
            "feasible_fraction_mean": (sum(r.get("feasible_fraction_of_exact", 0.0) for r in sub) / len(sub)),
            "step_ms_p50_median": sorted(r["step_ms_p50"] for r in sub)[len(sub) // 2],
            "step_ms_p99_max": max(r["step_ms_p99"] for r in sub),
            "overruns": sum(r["overruns"] for r in sub),
            "peak_memory_mb_max": max((r.get("peak_memory_mb") or 0.0) for r in sub),
            "solve_ms_p50_max": max((r.get("solve_ms_p50") or 0.0) for r in sub),
            "solve_ms_p99_max": max((r.get("solve_ms_p99") or 0.0) for r in sub),
            "categories_mean": {k: (sum((r.get("category_scores") or {}).get(k, 0.0) for r in sub) / len(sub))
                                for k in sorted({k for r in sub for k in (r.get("category_scores") or {})})},
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1, default=float))
    print(json.dumps(summary["by_controller"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
