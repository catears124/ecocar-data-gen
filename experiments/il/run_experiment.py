"""Reproducible behavioral-cloning experiment orchestration."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = ("optimal", "approx", "heuristic", "cert_control")
DEFAULT_BUDGETS = (200, 800, 3200)
SEEDS = (0, 1, 2)


def _tracks(value: str | Path) -> list[Path]:
    p = Path(value)
    if p.is_dir():
        out = sorted(x for x in p.iterdir() if x.suffix.lower() in {".yaml", ".yml"})
    else:
        out = [Path(x.strip()) for x in str(value).split(",") if x.strip()]
    missing = [str(x) for x in out if not x.is_file()]
    if not out or missing:
        raise FileNotFoundError("missing track path(s): " + ", ".join(missing or [str(p)]))
    return out


def _bootstrap(values: list[float], seed: int, n: int = 2000) -> list[float | None]:
    values = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if values.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _record_failure(item: dict[str, Any], stage: str, exc: BaseException) -> None:
    item.setdefault("failures", []).append({"stage": stage, "type": type(exc).__name__, "message": str(exc)})


def _pairwise(closed_rows: list[dict[str, Any]], budgets: tuple[int, ...]) -> dict[str, Any]:
    by: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for row in closed_rows:
        key = (Path(str(row.get("track", ""))).stem, int(row.get("seed", 0)),
               str(row.get("condition", "")), int(row.get("budget", 0)))
        by[key] = row
    result: dict[str, Any] = {}
    for budget in budgets:
        budget_result: dict[str, Any] = {}
        result[str(int(budget))] = budget_result
        for other in ("approx", "heuristic", "cert_control"):
            paired_keys: list[dict[str, Any]] = []
            grouped: dict[str, list[float]] = {}
            for course in sorted({key[0] for key in by}):
                for seed in SEEDS:
                    left = by.get((course, seed, "optimal", budget))
                    right = by.get((course, seed, other, budget))
                    if not left or not right or left.get("finished") is not True or right.get("finished") is not True:
                        continue
                    try:
                        difference = float(left["time_s"]) - float(right["time_s"])
                    except (TypeError, ValueError):
                        continue
                    paired_keys.append({"course": course, "seed": seed, "difference_s": difference})
                    grouped.setdefault(course, []).append(difference)
            course_values = {course: float(np.mean(values)) for course, values in grouped.items()}
            vals = list(course_values.values())
            budget_result[f"optimal-vs-{other}"] = {
                "budget": int(budget), "paired_keys": paired_keys,
                "course_differences": course_values, "n_courses": len(vals),
                "mean_difference": float(np.mean(vals)) if vals else None,
                "bootstrap_95_ci_over_courses": _bootstrap(vals, 9101 + int(budget)),
            }
    return result


def run(corpus: str | Path, tracks: str | Path, output: str | Path,
        budgets: tuple[int, ...] = DEFAULT_BUDGETS) -> dict[str, Any]:
    from . import dataset, evaluate, train_bc
    root = dataset.validate_corpus(corpus)
    track_paths = _tracks(tracks)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for budget in budgets:
            for seed in SEEDS:
                item: dict[str, Any] = {"condition": condition, "budget": int(budget), "seed": seed}
                weight = out / "weights" / f"{condition}_b{budget}_s{seed}.npz"
                item["weights"] = str(weight)
                try:
                    train_bc.train(root, condition, int(budget), seed, weight)
                except Exception as exc:
                    _record_failure(item, "train", exc); records.append(item); continue
                try:
                    offline = evaluate.evaluate_offline(root, weight, conditions=(condition,), budgets=(budget,), seeds=(seed,))
                    item["offline"] = offline["results"][0] if offline.get("results") else None
                except Exception as exc:
                    _record_failure(item, "offline", exc)
                try:
                    closed = evaluate.evaluate_closed_loop(",".join(map(str, track_paths)), weight,
                        conditions=(condition,), budgets=(budget,), seeds=(seed,), output_dir=out / "runs")
                    item["closed_loop"] = closed
                    closed_rows.extend(closed.get("results", []))
                except Exception as exc:
                    _record_failure(item, "closed_loop", exc)
                records.append(item)
    result = {"corpus": str(root), "tracks": [str(x) for x in track_paths], "conditions": list(CONDITIONS),
              "budgets": [int(x) for x in budgets], "seeds": list(SEEDS), "runs": records,
              "paired_differences": _pairwise(closed_rows, tuple(int(x) for x in budgets)),
              "failures": [x for x in records if x.get("failures")]}
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_reports(result, out)
    return result


def _write_reports(result: dict[str, Any], out: Path) -> None:
    lines = ["# Behavioral-cloning experiment summary", "", "Metrics are reported only for successful runs; failures are listed in `results.json`.", ""]
    for condition in result["conditions"]:
        lines.append(f"## {condition}")
        for budget in result["budgets"]:
            rows = [r.get("offline") for r in result["runs"] if r["condition"] == condition and r["budget"] == budget and r.get("offline") and not r.get("failures")]
            vals = [float(r["offset_mae_mm"]) for r in rows if r.get("offset_mae_mm") is not None]
            lines.append(f"- budget {budget}: offset MAE {np.mean(vals):.6g} mm over {len(vals)} successful seed(s)" if vals else f"- budget {budget}: unavailable")
    lines += ["", "Course is the statistical unit; bootstrap intervals resample held-out courses, not frames or JSONL rows."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failures = len(result["failures"])
    text = ["# Analysis", "", "## Paired condition differences", "", "The results include optimal-vs-approx, optimal-vs-heuristic, and optimal-vs-cert_control paired by course and seed. Confidence intervals resample course-level aggregates.", "", "## Failure analysis", "", f"{failures} of {len(result['runs'])} condition/budget/seed runs recorded failures; failures are not converted into metrics.", "", "## Are optimal labels harder to imitate?", "", "This is an empirical question, not an assumption. Compare held-out offset/acceleration MAE by condition and budget; the report does not claim optimal labels are harder unless the observed results support that conclusion."]
    (out / "analysis.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train and evaluate behavioral-cloning students across corpus conditions")
    p.add_argument("--corpus", required=True, help="corpus directory containing manifest.json and examples.jsonl")
    p.add_argument("--tracks", required=True, help="comma-separated held-out track YAMLs or a directory")
    p.add_argument("--out", "--output", default="il-results", dest="output")
    p.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    args = p.parse_args(argv)
    try:
        run(args.corpus, args.tracks, args.output, tuple(args.budgets))
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
