"""Recompute reports from an existing experiment results.json without simulation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = ("optimal", "approx", "heuristic", "cert_control")


def _bootstrap(values: list[float], seed: int, n: int = 2000) -> list[float | None]:
    vals = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    if vals.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n, vals.size), replace=True).mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in result.get("runs", []):
        rows.extend(run.get("closed_loop", {}).get("results", []) or [])
    return rows


def _metrics(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for run in result.get("runs", []):
        cond, budget = str(run.get("condition")), str(int(run.get("budget", 0)))
        key = f"{cond}/b{budget}"
        out.setdefault(key, {"condition": cond, "budget": int(budget), "offline_offset_mae_mm": [], "offline_accel_mae_mm_s2": [], "closed_loop": []})
        off = run.get("offline") or {}
        if off.get("offset_mae_mm") is not None: out[key]["offline_offset_mae_mm"].append(float(off["offset_mae_mm"]))
        if off.get("accel_mae_mm_s2") is not None: out[key]["offline_accel_mae_mm_s2"].append(float(off["accel_mae_mm_s2"]))
        out[key]["closed_loop"].extend(run.get("closed_loop", {}).get("results", []) or [])
    for item in out.values():
        rows = item.pop("closed_loop")
        finished = [r for r in rows if r.get("finished") is True]
        times = [float(r["time_s"]) for r in finished if r.get("time_s") is not None]
        item.update({
            "offline_offset_mae_mm": float(np.mean(item["offline_offset_mae_mm"])) if item["offline_offset_mae_mm"] else None,
            "offline_accel_mae_mm_s2": float(np.mean(item["offline_accel_mae_mm_s2"])) if item["offline_accel_mae_mm_s2"] else None,
            "closed_loop_runs": len(rows), "finish_count": len(finished),
            "mean_time_finished_s": float(np.mean(times)) if times else None,
            "cone_hits": int(sum(int(r.get("cone_hits") or 0) for r in rows)),
            "off_course": int(sum(int(r.get("off_course") or 0) for r in rows)),
            "controller_exceptions": int(sum(bool(r.get("controller_exception")) for r in rows)),
        })
    return out


def _pairs(result: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(result)
    by: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for r in rows:
        by[(Path(str(r.get("track", ""))).stem, int(r.get("seed", 0)), str(r.get("condition", "")), int(r.get("budget", 0)))] = r
    budgets = sorted({k[3] for k in by})
    out: dict[str, Any] = {}
    for budget in budgets:
        out[str(budget)] = {}
        for other in ("approx", "heuristic", "cert_control"):
            pairs = []
            grouped: dict[str, list[float]] = {}
            courses = sorted({k[0] for k in by})
            for course in courses:
                for seed in sorted({k[1] for k in by}):
                    a, b = by.get((course, seed, "optimal", budget)), by.get((course, seed, other, budget))
                    if not a or not b or a.get("finished") is not True or b.get("finished") is not True:
                        continue
                    try: diff = float(a["time_s"]) - float(b["time_s"])
                    except (TypeError, ValueError): continue
                    pairs.append({"course": course, "seed": seed, "difference_s": diff})
                    grouped.setdefault(course, []).append(diff)
            course_values = {c: float(np.mean(v)) for c, v in grouped.items()}
            vals = list(course_values.values())
            out[str(budget)][f"optimal-vs-{other}"] = {"paired_keys": pairs, "course_differences": course_values,
                "n_courses": len(vals), "mean_difference_s": float(np.mean(vals)) if vals else None,
                "bootstrap_95_ci_over_courses": _bootstrap(vals, 9101 + budget)}
    return out


def reanalyze(results_path: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    path = Path(results_path)
    if not path.is_file(): raise FileNotFoundError(f"missing results JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict): raise ValueError("results JSON must be an object")
    output = Path(out) if out else path.parent
    output.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(path), "metrics": _metrics(data), "paired_differences": _pairs(data),
               "failures": data.get("failures", [])}
    (output / "reanalysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_reports(summary, output)
    return summary


def _write_reports(summary: dict[str, Any], out: Path) -> None:
    lines = ["# Behavioral-cloning experiment summary", "", "All values below are recomputed from the existing results JSON; no simulation was run.", ""]
    for key, item in sorted(summary["metrics"].items()):
        lines += [f"## {key}", f"- offline offset MAE (mm): {item['offline_offset_mae_mm']}", f"- offline acceleration MAE (mm/s²): {item['offline_accel_mae_mm_s2']}", f"- closed-loop finishes: {item['finish_count']} / {item['closed_loop_runs']}", f"- mean time on finished runs (s): {item['mean_time_finished_s']}", f"- cone hits: {item['cone_hits']}", f"- off-course count: {item['off_course']}", f"- controller exceptions: {item['controller_exceptions']}", ""]
    lines += ["Course (not frames) is the statistical unit for bootstrap intervals."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Analysis", "", "## Paired condition differences", "", "Comparisons are paired by (course, seed), then averaged within course before bootstrap resampling. Bootstrap confidence intervals therefore resample courses, not frames or JSONL examples."]
    for budget, comparisons in sorted(summary["paired_differences"].items()):
        lines += ["", f"### Budget {budget}"]
        for name, comp in comparisons.items():
            lines.append(f"- {name}: mean time difference (s) = {comp['mean_difference_s']}; 95% CI = {comp['bootstrap_95_ci_over_courses']}; courses = {comp['n_courses']}")
    lines += ["", "## Failure analysis", "", f"Recorded source failures: {len(summary['failures'])}. Controller exceptions, DNFs, cone hits, and off-course counts remain explicit in the per-condition metrics; no failed run is converted into a successful metric.", "", "## Are optimal labels harder to imitate?", "", "This is not assumed. Compare the reported held-out offset and acceleration MAE by condition and budget; only the observed values can support that conclusion."]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recompute IL reports from existing results.json")
    p.add_argument("--results", required=True, help="existing results.json")
    p.add_argument("--out", default=None, help="report output directory (default: results parent)")
    args = p.parse_args(argv)
    try: reanalyze(args.results, args.out)
    except (OSError, ValueError, KeyError) as exc: p.error(str(exc))
    return 0


if __name__ == "__main__": raise SystemExit(main())
