"""Offline and closed-loop evaluation for the IL experiment.

The module deliberately records simulator failures as run records rather than
turning them into successful or fabricated metrics.  It is usable as a module
(``python -m experiments.il.evaluate``) or through the public functions below.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_CONDITIONS = ("optimal", "approx", "heuristic", "cert_control")
DEFAULT_BUDGETS = (200, 800, 3200)
DEFAULT_SEEDS = (0, 1, 2)


def _dataset(corpus: str | Path, condition: str, split: str = "test") -> Any:
    """Load the canonical dataset representation used by the trainer."""
    try:
        from .dataset import load_dataset
        return load_dataset(corpus, condition=condition, split=split)
    except ImportError:
        from experiments.il.dataset import load_dataset
        return load_dataset(corpus, condition=condition, split=split)


def _weight_path(weights: str | Path, condition: str, budget: int, seed: int) -> Path:
    text = str(weights)
    try:
        text = text.format(condition=condition, budget=budget, seed=seed)
    except (KeyError, ValueError):
        pass
    path = Path(text)
    if path.is_dir():
        path = path / f"{condition}_b{budget}_s{seed}.npz"
    return path


def _predict(path: Path, x: np.ndarray) -> np.ndarray:
    """Evaluate the exact two-layer ReLU artifact emitted by train_bc."""
    if not path.is_file():
        raise FileNotFoundError(f"missing weights file: {path}")
    with np.load(path, allow_pickle=False) as archive:
        weights = {k: np.asarray(archive[k], dtype=float) for k in archive.files if k != "metadata"}
    if {"x_mean", "x_std"}.issubset(weights):
        x = (x - weights["x_mean"]) / np.maximum(weights["x_std"], 1e-12)
    if {"W_offset", "W_accel"}.issubset(weights):
        h = x
        i = 0
        while f"W{i}" in weights:
            h = h @ weights[f"W{i}"] + weights.get(f"b{i}", 0.0)
            h = np.maximum(h, 0.0)
            i += 1
        return np.column_stack((h @ weights["W_offset"] + weights.get("b_offset", 0.0),
                                h @ weights["W_accel"] + weights.get("b_accel", 0.0)))
    raise ValueError(f"weights file is not a trainer ReLU artifact: {path}")




def evaluate_offline(corpus: str | Path, weights: str | Path, *, conditions: Iterable[str] = DEFAULT_CONDITIONS,
                     budgets: Iterable[int] = DEFAULT_BUDGETS, seeds: Iterable[int] = DEFAULT_SEEDS,
                     split: str = "test") -> dict[str, Any]:
    """Compute held-out MAE, retaining one record for every condition/budget/seed."""
    conditions, budgets, seeds = tuple(conditions), tuple(budgets), tuple(seeds)
    results: list[dict[str, Any]] = []
    for condition in conditions:
        data = _dataset(corpus, condition, split)
        x, y = np.asarray(data.x, dtype=float), np.asarray(data.y, dtype=float)
        for budget in budgets:
            for seed in seeds:
                item: dict[str, Any] = {"condition": condition, "budget": int(budget), "seed": int(seed),
                                        "split": split, "n": int(len(y)), "weights": None}
                try:
                    path = _weight_path(weights, condition, int(budget), int(seed))
                    pred = _predict(path, x)
                    if len(y):
                        err = np.abs(pred - y)
                        item.update(weights=str(path), offset_mae_mm=float(np.mean(err[:, 0]) * 1000.0),
                                    accel_mae_mm_s2=float(np.mean(err[:, 1]) * 1000.0))
                    else:
                        item.update(weights=str(path), error="no held-out rows")
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                results.append(item)
    return {"kind": "il_offline", "corpus": str(Path(corpus)), "results": results}


def load_sim_config(config: str | Path | None) -> dict[str, Any]:
    """Load the repository's simulator defaults, optionally with a YAML override."""
    try:
        from autocross.config import load_sim_config as _load
    except ImportError as exc:
        raise RuntimeError(f"simulator config support unavailable: {exc}") from exc
    path = None if config is None else Path(config)
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"missing simulator config: {path}")
    return _load(path)
def evaluate_closed_loop(tracks: str | Iterable[str], weights: str | Path, *, controller: str | Path | None = None,
                         config: str | Path | None = None, conditions: Iterable[str] = DEFAULT_CONDITIONS,
                         budgets: Iterable[int] = DEFAULT_BUDGETS, seeds: Iterable[int] = DEFAULT_SEEDS,
                         output_dir: str | Path | None = None, backend: str = "bicycle") -> dict[str, Any]:
    """Run each track/condition/budget/seed through ``autocross.runner.run``."""
    conditions, budgets, seeds = tuple(conditions), tuple(budgets), tuple(seeds)
    paths = [Path(x.strip()) for x in tracks.split(",")] if isinstance(tracks, str) else [Path(x) for x in tracks]
    if not paths or any(not p.is_file() for p in paths):
        missing = [str(p) for p in paths if not p.is_file()]
        raise FileNotFoundError(f"missing track path(s): {', '.join(missing)}")
    weight_root = Path(weights)
    if not weight_root.exists() and "{" not in str(weights):
        raise FileNotFoundError(f"missing weights path: {weight_root}")
    try:
        from autocross.runner import run
    except ImportError as exc:
        raise RuntimeError(f"closed-loop dependencies unavailable: {exc}") from exc
    cfg = load_sim_config(config)
    results: list[dict[str, Any]] = []
    controller_path = Path(controller) if controller else Path(__file__).with_name("student_controller.py")
    if not controller_path.is_file():
        raise FileNotFoundError(f"missing controller path: {controller_path}")
    for track in paths:
        for condition in conditions:
            for budget in budgets:
                for seed in seeds:
                    item = {"track": str(track), "condition": condition, "budget": int(budget), "seed": int(seed),
                            "finished": None, "time_s": None, "cone_hits": None, "off_course": None,
                            "controller_exception": None}
                    try:
                        wp = _weight_path(weights, condition, int(budget), int(seed))
                        run_cfg = json.loads(json.dumps(cfg))
                        run_cfg.setdefault("sim", {})["seed"] = int(seed)
                        summary = run(run_cfg, str(controller_path), str(track), backend=backend,
                                      controller_class="StudentBCController",
                                      controller_cfg={"weights": str(wp), "budget": budget, "seed": seed,
                                                      "condition": condition}, out_dir=(Path(output_dir) / track.stem / f"{condition}_b{budget}_s{seed}") if output_dir else None,
                                      verbose=False)
                        metrics = summary.get("metrics", {})
                        completion = metrics.get("completion", {})
                        penalties = metrics.get("penalties", {})
                        item.update(finished=completion.get("finished"), time_s=completion.get("time_s"),
                                    cone_hits=penalties.get("cone_hit"), off_course=penalties.get("off_course"),
                                    controller_exception=summary.get("controller_exception",
                                                                         summary.get("controller_error")))
                    except Exception as exc:
                        item["controller_exception"] = f"{type(exc).__name__}: {exc}"
                    results.append(item)
    return {"kind": "il_closed_loop", "tracks": [str(p) for p in paths], "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate IL models offline and in closed loop")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tracks", required=True, help="comma-separated track paths")
    parser.add_argument("--controller", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    args = parser.parse_args(argv)
    conditions = tuple(x for x in args.conditions.split(",") if x)
    budgets = tuple(int(x) for x in args.budgets.split(",") if x)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    result = evaluate_offline(args.corpus, args.weights, conditions=conditions, budgets=budgets, seeds=seeds)
    result["closed_loop"] = evaluate_closed_loop(
        args.tracks, args.weights, controller=args.controller, config=args.config,
        conditions=conditions, budgets=budgets, seeds=seeds)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
