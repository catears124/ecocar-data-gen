"""Load a released corpus for training (no CARLA, no simulator needed).

    from autocross.datagen.dataset import load_examples, to_arrays
    ex = load_examples("ecocar-imitation-full-bicycle-v1.0.1", split="train", expect_backend="bicycle")
    arrays = to_arrays(ex)

``iter_examples`` yields ``(policy_observation, training_target, certificate_status, metadata)``:

  * ``policy_observation``: the canonical policy input (whitelist; no pose, progress, lap, time,
    cones, corner list or finish distance; those live in the record's ``privileged`` block,
    which this module never reads).
  * ``training_target``: ``longitudinal`` = the CERTIFIED quantity (integer acceleration
    ``accel_mm_s2`` of model cert-2, its kind, and the certified next-station speed^2
    ``b1_mm2_s2``); ``uncertified`` = steering, path, pedals, target speed (NO guarantee).
  * ``certificate_status``: ``{"certified", "model", "kind", "scope"}``.
  * ``metadata``: sample/scenario/group ids, split, backend, step, clean flag. Use ``group_id``
    for grouping (samples of one run are a correlated time series)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from . import INDEX_SCHEMA, SAMPLE_SCHEMA, SCHEMA_VERSION, DatagenError
from .cert import CERT_MODEL
from .records import POLICY_EGO, POLICY_OBSERVATION_SCHEMA, POLICY_ROAD_KEYS, TRAINING_TARGET_SCHEMA

CERT_SCOPE = "longitudinal acceleration (mm/s^2) and per-station lateral clearance (mm) of cert-2/steer-2; lateral path, steering, pedals uncertified"


def _open_index(root: Path, expect_backend: str | None) -> dict:
    index = json.loads((root / "index.json").read_text())
    if index.get("schema") != INDEX_SCHEMA or index.get("schema_version") != SCHEMA_VERSION:
        raise DatagenError(f"{root}: index schema {index.get('schema')}/{index.get('schema_version')} is not "
                           f"{INDEX_SCHEMA}/{SCHEMA_VERSION}")
    if index.get("certificate_model") != CERT_MODEL:
        raise DatagenError(f"{root}: certificate model {index.get('certificate_model')} is not {CERT_MODEL}")
    be = index["backend"]
    if be["is_carla"] != (be["name"] == "carla"):
        raise DatagenError(f"{root}: inconsistent backend identity {be['name']}/{be['is_carla']}")
    if expect_backend is not None and be["name"] != expect_backend:
        raise DatagenError(f"{root}: backend is {be['name']!r}, expected {expect_backend!r}")
    return index


def iter_examples(corpus: str | Path, split: str | None = None, *, clean_only: bool = True,
                  expect_backend: str | None = None) -> Iterator[tuple[dict, dict, dict, dict]]:
    root = Path(corpus)
    index = _open_index(root, expect_backend)
    backend = index["backend"]["name"]
    for run in index["scenarios"]:
        if run.get("status") != "completed" or (split is not None and run["split"] != split):
            continue
        with open(root / run["files"]["samples"]["path"], encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                if s.get("schema") != SAMPLE_SCHEMA or s.get("schema_version") != SCHEMA_VERSION or s["backend"] != backend:
                    raise DatagenError(f"{s.get('sample_id')}: unexpected sample schema/backend")
                if clean_only and not s["quality"]["clean"]:
                    continue
                if s["target"] is None:
                    continue
                pol, tgt = s["policy_observation"], s["target"]
                if pol.get("schema") != POLICY_OBSERVATION_SCHEMA or tgt.get("schema") != TRAINING_TARGET_SCHEMA:
                    raise DatagenError(f"{s['sample_id']}: unexpected policy/target schema")
                lon = tgt["longitudinal"]
                lat = tgt["lateral"]
                status = {"certified": bool(lon and lon["certified"] and lat and lat["certified"]),
                          "models": {
                              "longitudinal": {"certified": bool(lon and lon["certified"]),
                                               "model": lon["model"] if lon else None,
                                               "kind": lon["kind"] if lon else None},
                              "lateral": {"certified": bool(lat and lat["certified"]),
                                          "model": lat["model"] if lat else None,
                                          "kind": lat["kind"] if lat else None}},
                          "model": lon["model"] if lon else None,
                          "kind": lon["kind"] if lon else None, "scope": CERT_SCOPE}
                meta = {"sample_id": s["sample_id"], "scenario_id": s["scenario_id"], "group_id": s["group_id"],
                        "split": s["split"], "backend": backend, "step": s["step"], "clean": s["quality"]["clean"]}
                yield pol, tgt, status, meta


def load_examples(corpus: str | Path, split: str | None = None, **kw) -> list[tuple[dict, dict, dict, dict]]:
    return list(iter_examples(corpus, split, **kw))


def _resample(s: list, v: list, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s, v = np.asarray(s, float), np.asarray(v, float)
    mask = grid <= (s[-1] if len(s) else -1.0)
    out = np.zeros_like(grid)
    if len(s):
        out[mask] = np.interp(grid[mask], s, v)
    return out, mask.astype(float)


def policy_features(pol: dict, grid: np.ndarray, max_obstacles: int = 2) -> np.ndarray:
    """Fixed-length float vector from a policy observation ONLY (raises on any other dict)."""
    if pol.get("schema") != POLICY_OBSERVATION_SCHEMA or set(pol) != {"schema", "road", *POLICY_EGO}:
        raise DatagenError("policy_features accepts only canonical policy observations")
    r = pol["road"]
    ego = [float(pol[k]) for k in POLICY_EGO]
    if r is None or set(r) != set(POLICY_ROAD_KEYS):
        return np.concatenate([ego, np.zeros(9 + 4 * len(grid) + 4 * max_obstacles)])
    sc = [float(r["valid"])] + [float(r[k]) for k in ("range_m", "lateral_offset", "heading_error", "curvature",
                                                      "width_left", "width_right", "dist_to_left_boundary",
                                                      "dist_to_right_boundary")]
    k, mask = _resample(r["preview_s"], r["preview_curvature"], grid)
    wl, _ = _resample(r["preview_s"], r["preview_width_left"], grid)
    wr, _ = _resample(r["preview_s"], r["preview_width_right"], grid)
    obs = []
    for o in sorted(r["obstacles"], key=lambda o: o["distance"])[:max_obstacles]:
        obs += [1.0, o["distance"], o["lateral_min"], o["lateral_max"]]
    obs += [0.0] * (4 * max_obstacles - len(obs))
    return np.concatenate([ego, sc, k, wl, wr, mask, obs])


def to_arrays(examples, grid: np.ndarray | None = None) -> dict:
    """NumPy arrays for a basic training script: ``x`` features, ``y_long`` certified acceleration
    (m/s^2), ``certified`` mask, ``y_clearance`` per-station lateral clearance (mm),
    ``y_steer`` / ``y_throttle`` / ``y_brake`` (uncertified), ``group``."""
    grid = np.arange(0.0, 61.0, 1.0) if grid is None else grid
    x = np.stack([policy_features(p, grid) for p, _, _, _ in examples]) if examples else np.zeros((0, 0))
    lon = [t["longitudinal"] for _, t, _, _ in examples]
    return {"x": x,
            "y_long": np.array([l["accel_mm_s2"] / 1000.0 if l else np.nan for l in lon]),
            "certified": np.array([st["certified"] for _, _, st, _ in examples], bool),
            "y_clearance": np.array([(t["lateral"] or {}).get("clearance_mm", []) for _, t, _, _ in examples],
                                    dtype=object),
            "y_steer": np.array([t["uncertified"]["steer"] for _, t, _, _ in examples]),
            "y_throttle": np.array([t["uncertified"]["throttle"] for _, t, _, _ in examples]),
            "y_brake": np.array([t["uncertified"]["brake"] for _, t, _, _ in examples]),
            "group": np.array([m["group_id"] for _, _, _, m in examples]), "grid": grid}
