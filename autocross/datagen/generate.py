"""Run every accepted manifest scenario through the harness with the expert controller and
write the corpus: per-scenario sample JSONL, harness run folders, and the corpus index.

Corpus layout (``OUT``)::

    manifest.json            byte copy of the manifest the corpus was generated from
    tracks/<id>.yaml         course built from the scenario parameters
    runs/<id>/               standard harness run output (telemetry.csv, events.json, summary.json)
    samples/<id>.jsonl       one record per control step (observation, expert action, applied
                             action, outcome, oracle diagnostics, quality flags)
    index.json               provenance, backend identity, per-scenario outcome and file hashes,
                             realised coverage (rewritten after every scenario)
    contrast_report.json     empirical effect of each contrast pair (not a proof)
"""
from __future__ import annotations

import copy
import datetime as dt_
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import (GENERATOR_NAME, GENERATOR_VERSION, INDEX_SCHEMA, SAMPLE_SCHEMA, SCHEMA_VERSION, DatagenError,
               git_state)
from . import cert
from . import expert as expert_mod
from .design import check_manifest, required_cells
from .records import command_to_dict, content_hash, obs_from_dict, obs_to_dict
from .spec import get_family, sha256_obj
from ..api import Command, VehicleParams
from ..config import ROOT, deep_update, load_sim_config, load_yaml

EXPERT_PATH = Path(expert_mod.__file__).resolve()
EXPERT_CLASS = "BrakeTurnObstacleExpert"
REPLAY_TOL = 1e-9


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expert_identity() -> dict:
    return {"name": expert_mod.EXPERT_NAME, "version": expert_mod.EXPERT_VERSION, "class": EXPERT_CLASS,
            "source": str(EXPERT_PATH.relative_to(ROOT)).replace(os.sep, "/"), "source_sha256": file_sha256(EXPERT_PATH),
            "certificate_model": cert.CERT_MODEL, "certificate_source_sha256": file_sha256(cert.__file__),
            "params": dict(expert_mod.PARAMS), "causal": True,
            "inputs": "Observation.road (+ speed, ay) and published VehicleParams; no map, world pose, progress or cones "
                      "after hand-over; scripted approach before hand-over uses scenario config (phase=approach)"}


# ------------------------------------------------------------------ backend identity
def backend_identity(name: str, cfg: dict) -> dict:
    """Fail fast and clearly when the requested backend is unavailable."""
    if name == "bicycle":
        from ..backends import bicycle
        return {"name": "bicycle", "is_carla": False,
                "simulator": "autocross dynamic single-track (bicycle) model - NOT CARLA; tooling/logic checks only",
                "model_source_sha256": file_sha256(bicycle.__file__), "vehicle_bicycle": copy.deepcopy(cfg["vehicle"]["bicycle"])}
    if name == "carla":
        try:
            import carla
        except ImportError as e:
            raise DatagenError(
                f"CARLA backend requested but the 'carla' client package is not importable ({e}). Install carla==0.9.16 "
                f"in a Python 3.10-3.12 environment (scripts/setup.ps1) and start the server (scripts/start_carla.ps1). "
                f"No data was generated. For a tooling check without CARLA use --backend bicycle "
                f"(that corpus is labelled bicycle and is not CARLA data).") from e
        host, port = cfg["carla"]["host"], int(cfg["carla"]["port"])
        try:
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            sv, cv = client.get_server_version(), client.get_client_version()
        except RuntimeError as e:
            raise DatagenError(f"No CARLA server reachable at {host}:{port} ({e}). Start it with "
                               f"scripts/start_carla.ps1 (or .sh) and check with `python -m autocross check`. "
                               f"No data was generated.") from e
        return {"name": "carla", "is_carla": True, "simulator": "CARLA", "server_version": sv, "client_version": cv,
                "host": host, "port": port, "vehicle_blueprint": cfg["vehicle"]["blueprint"]}
    raise DatagenError(f"unknown backend '{name}'")


# ------------------------------------------------------------------ recorder
class StepRecorder:
    """Collects one entry per control step from the runner hook."""

    def __init__(self):
        self.steps: list[dict] = []
        self.n_events = 0

    def on_control(self, obs, raw, cmd, stats, clipped, held):
        self.steps.append({
            "obs": obs_to_dict(obs),
            "raw": command_to_dict(raw) if isinstance(raw, Command) else None,
            "raw_error": None if isinstance(raw, Command) else (str(raw)[:200] if raw is not None else stats.error and stats.error[-200:]),
            "clipped": bool(clipped), "held": bool(held),
            "timing": {"latency_ms": _num(stats.latency_ms), "cpu_ms": _num(stats.cpu_ms), "overrun": bool(stats.overrun)},
            "ticks": 0, "first": None, "last": None, "off_course": False, "events": [],
            "min_boundary_margin": math.inf, "progress_end": None,
        })

    def on_tick(self, st, applied, monitor, is_off):
        if not self.steps:
            return
        s = self.steps[-1]
        a = {"steer": applied[0], "throttle": applied[1], "brake": applied[2]}
        s["first"] = s["first"] or a
        s["last"] = a
        s["ticks"] += 1
        s["off_course"] |= bool(is_off)
        s["min_boundary_margin"] = min(s["min_boundary_margin"], float(monitor.boundary_margin))
        s["progress_end"] = float(monitor.progress)
        for e in monitor.events[self.n_events:]:
            s["events"].append({"t": e.t, "kind": e.kind, **{k: v for k, v in e.detail.items()}})
        self.n_events = len(monitor.events)


def _num(x):
    return None if x is None or not math.isfinite(float(x)) else float(x)


# ------------------------------------------------------------------ labelling
def run_outcome(summary: dict, events: list, steps: list, layout: dict, params: dict, family) -> tuple[dict, list[str]]:
    m = summary["metrics"]
    c, p = m["completion"], m["penalties"]
    k = family.constants
    obstacle_hits = sum(1 for e in events if e["kind"] == "cone_hit" and not e["detail"].get("mapped", True))
    boundary_hits = sum(1 for e in events if e["kind"] == "cone_hit" and e["detail"].get("mapped", True))
    hand = next((s for s in steps if s["obs"]["progress"] >= layout["handover_progress"]), None)
    out = {"finished": bool(c["finished"]), "dnf_reason": c["dnf_reason"], "time_s": c["time_s"],
           "cone_hits_obstacle": obstacle_hits, "cone_hits_boundary": boundary_hits,
           "off_course_events": int(p["off_course"]), "missed_gates": int(p["missed_gate"]),
           "controller_errors": int(m["compute"]["errors"]), "controller_error_text": summary.get("controller_error"),
           "handover": None}
    fails = []
    if not c["finished"]:
        fails.append(f"dnf:{c['dnf_reason']}" if c["dnf_reason"] else "not_finished")
    if obstacle_hits:
        fails.append("obstacle_cone_hit")
    if boundary_hits:
        fails.append("boundary_cone_hit")
    if p["off_course"]:
        fails.append("off_course")
    if p["missed_gate"]:
        fails.append("missed_gate")
    if m["compute"]["errors"] or summary.get("controller_error"):
        fails.append("controller_error")
    if any(s["held"] for s in steps):
        fails.append("command_held")
    if hand is None:
        fails.append("handover_not_reached")
    else:
        r = hand["obs"]["road"] or {}
        hs = {"step": hand["obs"]["step"], "speed": hand["obs"]["speed"], "target_speed": float(params["approach_speed_mps"]),
              "lateral_offset": r.get("lateral_offset"), "heading_error": r.get("heading_error")}
        hs["speed_ok"] = abs(hs["speed"] - hs["target_speed"]) <= k["handover_speed_tol_mps"]
        hs["pose_ok"] = (r.get("lateral_offset") is not None and abs(r["lateral_offset"]) <= k["handover_lateral_tol_m"]
                         and abs(r["heading_error"]) <= k["handover_heading_tol_rad"])
        hs["nothing_visible"] = not (r.get("corners") or r.get("obstacles"))
        out["handover"] = hs
        if not hs["speed_ok"]:
            fails.append("handover_speed_off")
        if not hs["pose_ok"]:
            fails.append("handover_pose_off")
        if not hs["nothing_visible"]:
            fails.append("handover_corner_already_visible")
    if any(s["raw"] and s["raw"]["debug"].get("status") == expert_mod.STATUS_NO_GAP for s in steps):
        fails.append("expert_no_feasible_gap")
    return out, fails


def certificate_block(dbg: dict) -> dict | None:
    """Certified longitudinal decision as recorded by the expert (integers of model cert-1)."""
    if dbg.get("cert_kind") is None:
        return None
    kind = int(dbg["cert_kind"])
    out = {"model": cert.CERT_MODEL, "kind": cert.KIND_NAMES.get(kind, "unknown"),
           "certified": kind in (cert.KIND_GREATEST_SAFE, cert.KIND_MIN_VIOLATION),
           "accel_mm_s2": int(dbg["cert_a"]), "accel_mps2": int(dbg["cert_a"]) / 1000.0}
    for k in ("b1", "b0", "B0"):
        if dbg.get(f"cert_{k}") is not None:
            out[f"{k}_mm2_s2"] = int(dbg[f"cert_{k}"])
    return out


def replay_diff(obs_dict: dict, raw: dict, vp: VehicleParams, config: dict | None) -> float:
    """Max abs difference between the recorded action and the expert re-run on the recorded
    observation (``config=None`` for decision samples: proves they do not use scenario config)."""
    cmd = expert_action_dict(obs_dict, vp, config)
    diffs = [abs((cmd[k] if cmd[k] is not None else math.nan) - (raw[k] if raw[k] is not None else math.nan))
             for k in ("steer", "throttle", "brake")]
    for k in ("target_speed",):
        a, b = cmd.get(k), raw.get(k)
        if (a is None) != (b is None):
            return math.inf
        if a is not None:
            diffs.append(abs(a - b))
    for k in ("cert_kind", "cert_a", "cert_b1", "cert_b0", "cert_B0"):  # certified integers: exact
        if cmd["debug"].get(k) != raw["debug"].get(k):
            return math.inf
    return max((d if math.isfinite(d) else math.inf) for d in diffs)


def expert_action_dict(obs_dict: dict, vp: VehicleParams, config: dict | None) -> dict:
    return command_to_dict(expert_mod.expert_action(obs_from_dict(obs_dict), vp, config))


def build_samples(entry: dict, steps: list, vp: VehicleParams, backend: str, run_fails: list[str],
                  expert_cfg: dict) -> list[dict]:
    lay = entry["layout"]
    out = []
    for s in steps:
        o, raw = s["obs"], s["raw"]
        prog = o["progress"]
        dbg = (raw or {}).get("debug") or {}
        if dbg.get("phase") == expert_mod.PHASE_APPROACH:
            phase = "approach"
        elif prog <= lay["decision_end_progress"]:
            phase = "decision"
        else:
            phase = "post"
        reasons = []
        if phase == "approach":
            reasons.append("phase:approach")
        elif phase == "post":
            reasons.append("phase:post_window")
        reasons += [f"run:{f}" for f in run_fails]
        if raw is None:
            reasons.append("command_invalid")
            rdiff = None
        else:
            status = int(dbg.get("status", -1)) if dbg.get("status") is not None else -1
            if phase != "approach" and status not in (expert_mod.STATUS_OK, expert_mod.STATUS_OVER_BUDGET):
                reasons.append(f"expert_status:{expert_mod.STATUS_NAMES.get(status, status)}")
            rdiff = replay_diff(o, raw, vp, None if phase != "approach" else expert_cfg)
            if not rdiff <= REPLAY_TOL:
                reasons.append("label_replay_mismatch")
        if s["held"]:
            reasons.append("command_held")
        if s["clipped"]:
            reasons.append("command_clipped")
        if s["first"] is None:
            reasons.append("no_physics_tick")
        rec = {
            "schema": SAMPLE_SCHEMA, "schema_version": SCHEMA_VERSION,
            "sample_id": f"{entry['scenario_id']}:{o['step']:05d}",
            "scenario_id": entry["scenario_id"], "group_id": entry["group_id"], "family": entry["family"],
            "split": entry["split"], "backend": backend, "step": o["step"], "t": o["t"],
            "observation": o,
            "expert": None if raw is None else {
                "steer": raw["steer"], "throttle": raw["throttle"], "brake": raw["brake"],
                "target_speed": raw["target_speed"], "path": raw["path"], "path_frame": raw["path_frame"],
                "phase": phase if phase == "approach" else "expert",
                "status": expert_mod.STATUS_NAMES.get(int(dbg["status"]), "unknown") if dbg.get("status") is not None else "unknown",
                "certificate": certificate_block(dbg) if phase != "approach" else None,
                "debug": dbg},
            "applied": {"first_tick": s["first"], "end_of_period": s["last"], "ticks": s["ticks"],
                        "command_held": s["held"], "command_clipped": s["clipped"]},
            "outcome": {"off_course": s["off_course"], "events": s["events"],
                        "min_boundary_margin_m": _num(s["min_boundary_margin"]) if s["ticks"] else None,
                        "progress_end": s["progress_end"]},
            "oracle": {  # privileged ground truth: diagnostics only, never an input or target
                "phase": phase,
                "distance_to_corner_entry_m": lay["corner_entry_progress"] - prog,
                "distance_to_obstacle_m": lay["obstacle_progress"] - prog,
                "distance_to_handover_m": lay["handover_progress"] - prog,
                "obstacle_lateral_min_m": lay["obstacle_lateral_min_m"],
                "obstacle_lateral_max_m": lay["obstacle_lateral_max_m"]},
            "timing": s["timing"],  # wall-clock dependent: not reproducible, excluded from hashes
            "quality": {"clean": not reasons, "exclusion_reasons": reasons,
                        "label_replay_max_abs_diff": rdiff if rdiff is None or math.isfinite(rdiff) else None},
        }
        rec["content_sha256"] = content_hash(o, rec["expert"]) if raw is not None else None
        out.append(rec)
    return out


# ------------------------------------------------------------------ corpus
def generation_config(design_path: str | None, sim_config: str | None, backend: str, sets: list[str] | None) -> dict:
    cfg = load_sim_config(sim_config, sets)
    design = load_yaml(design_path or ROOT / "config" / "datagen.yaml")
    deep_update(cfg, design.get("generation", {}))
    cfg["backend"] = backend
    if cfg.get("track_knowledge") != "perception":
        raise DatagenError("data generation requires track_knowledge: perception (the expert must not get a map)")
    return cfg


def run_scenario(e: dict, cfg: dict, out: Path, backend: str) -> dict:
    """Run one accepted scenario, write its track, run folder and samples; return its index entry.
    Module-level so it can run in a worker process."""
    from ..runner import run
    from ..track import save
    family = get_family(e["family"])
    t0 = time.perf_counter()
    sid, params = e["scenario_id"], e["params"]
    track_path = out / "tracks" / f"{sid}.yaml"
    save(family.track_spec(params, sid), track_path)
    run_dir = out / "runs" / sid
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg_s = deep_update(copy.deepcopy(cfg), family.sim_overrides(params))
    cfg_s["sim"]["seed"] = int(e["sim_seed"])
    rec = StepRecorder()
    ecfg = family.expert_config(params)
    entry = {"scenario_id": sid, "split": e["split"], "group_id": e["group_id"], "sim_seed": e["sim_seed"],
             "config_overrides": {**family.sim_overrides(params), "sim": {"seed": int(e["sim_seed"])}},
             "expert_config": ecfg}
    try:
        summary = run(cfg_s, str(EXPERT_PATH), str(track_path), team=expert_mod.EXPERT_NAME, backend=backend,
                      controller_class=EXPERT_CLASS, controller_cfg=ecfg, out_dir=run_dir, verbose=False, recorder=rec)
    except Exception as ex:  # simulator failure: record it, never drop the scenario silently
        entry.update({"status": "sim_error", "error": f"{type(ex).__name__}: {ex}"[:1000], "clean_run": False,
                      "run_failures": ["sim_error"], "n_samples": 0, "n_clean": 0})
        return entry
    events = json.loads((run_dir / "events.json").read_text())
    vp = VehicleParams(**summary["vehicle"])
    outcome, fails = run_outcome(summary, events, rec.steps, e["layout"], params, family)
    samples = build_samples(e, rec.steps, vp, backend, fails, ecfg)
    spath = out / "samples" / f"{sid}.jsonl"
    tmp = spath.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for s in samples:
            f.write(json.dumps(s, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(tmp, spath)
    entry.update({
        "status": "completed", "clean_run": not fails, "run_failures": fails, "outcome": outcome,
        "vehicle": summary["vehicle"], "sim_time_s": summary["sim_time_s"],
        "wall_time_s": round(time.perf_counter() - t0, 2), "n_samples": len(samples),
        "n_decision": sum(1 for s in samples if s["oracle"]["phase"] == "decision"),
        "n_clean": sum(1 for s in samples if s["quality"]["clean"]),
        "n_certified_clean": sum(1 for s in samples if s["quality"]["clean"] and s["expert"]["certificate"]
                                 and s["expert"]["certificate"]["certified"]),
        "files": {"samples": _fref(out, spath), "track": _fref(out, track_path),
                  **{k: _fref(out, run_dir / k) for k in ("telemetry.csv", "events.json", "summary.json")}},
    })
    return entry


def generate_corpus(manifest_path: str, out_dir: str, backend: str, *, design_path: str | None = None,
                    sim_config: str | None = None, sets: list[str] | None = None, limit: int | None = None,
                    resume: bool = False, workers: int = 1, verbose: bool = True) -> dict:
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    errs = check_manifest(manifest)
    if errs:
        raise DatagenError("manifest check failed:\n  " + "\n  ".join(errs[:20]))
    family = get_family(manifest["family_spec"]["name"])
    if backend == "carla" and workers != 1:
        raise DatagenError("CARLA generation uses one server: --workers must be 1")
    cfg = generation_config(design_path, sim_config, backend, sets)
    out = Path(out_dir).resolve()
    idx_path = out / "index.json"
    previous = {}
    if idx_path.exists():
        old = json.loads(idx_path.read_text())
        if old["backend"]["name"] != backend:
            raise DatagenError(f"{out} holds a {old['backend']['name']} corpus; refusing to add {backend} runs")
        if not resume:
            raise DatagenError(f"{out} already holds a corpus; use --resume or a new --out")
        if old["manifest"]["content_sha256"] != manifest["content_sha256"]:
            raise DatagenError(f"{out} was generated from a different manifest")
        previous = {s["scenario_id"]: s for s in old["scenarios"] if s.get("status") == "completed"}
    be_info = backend_identity(backend, cfg)
    for d in ("tracks", "runs", "samples"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_bytes(manifest_bytes)
    cfg.setdefault("sandbox", {})["protected_paths"] = list(cfg["sandbox"].get("protected_paths", []) or []) + [str(out)]

    accepted = [e for e in manifest["scenarios"] if e["status"] == "accepted"]
    if limit is not None:
        accepted = accepted[:limit]
    order = {e["scenario_id"]: i for i, e in enumerate(accepted)}
    index = {
        "schema": INDEX_SCHEMA, "schema_version": SCHEMA_VERSION,
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION, "git": git_state()},
        "manifest": {"file": "manifest.json", "file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                     "content_sha256": manifest["content_sha256"], "source_path": str(Path(manifest_path).resolve())},
        "backend": be_info, "expert": expert_identity(), "certificate_model": cert.CERT_MODEL,
        "sim_config": cfg, "sim_config_sha256": sha256_obj(cfg),
        "per_scenario_overrides": "road_model.range_m = preview_m; sim.seed = scenario sim_seed; "
                                  "controller config = scripted approach (handover_progress, speed)",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "created_utc": dt_.datetime.now(dt_.timezone.utc).isoformat(timespec="seconds"),
        "limit": limit, "workers": workers, "scenarios": [],
    }
    todo = []
    for e in accepted:
        prev = previous.get(e["scenario_id"])
        if prev and _files_intact(out, prev):
            index["scenarios"].append(prev)
        else:
            todo.append(e)

    def record(entry: dict, k: int):
        index["scenarios"].append(entry)
        index["scenarios"].sort(key=lambda s: order[s["scenario_id"]])
        _write_index(out, index, family, manifest)
        if verbose:
            if entry["status"] != "completed":
                print(f"[{k}/{len(todo)}] {entry['scenario_id']}: SIM ERROR {entry['error']}")
            else:
                st = "clean" if entry["clean_run"] else "EXCLUDED (" + ", ".join(entry["run_failures"]) + ")"
                print(f"[{k}/{len(todo)}] {entry['scenario_id']} [{backend}] {entry['split']:<5} "
                      f"{entry['n_samples']:4d} samples, {entry['n_clean']:4d} clean - {st}", flush=True)
        if backend == "carla" and entry["status"] == "sim_error":
            raise DatagenError(f"CARLA failed during generation: {entry['error']}")

    if workers <= 1:
        for k, e in enumerate(todo, 1):
            record(run_scenario(e, cfg, out, backend), k)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_scenario, e, cfg, out, backend) for e in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                record(fut.result(), k)
    _write_index(out, index, family, manifest)
    report = contrast_report(out, manifest, index)
    (out / "contrast_report.json").write_text(json.dumps(report, indent=1, allow_nan=False))
    return index


def _fref(root: Path, p: Path) -> dict:
    return {"path": str(p.relative_to(root)).replace(os.sep, "/"), "sha256": file_sha256(p)}


def _files_intact(root: Path, entry: dict) -> bool:
    try:
        return all(file_sha256(root / f["path"]) == f["sha256"] for f in entry["files"].values())
    except (OSError, KeyError):
        return False


def realised_coverage(manifest: dict, index: dict) -> dict:
    family = get_family(manifest["family_spec"]["name"])
    by_id = {e["scenario_id"]: e for e in manifest["scenarios"] if e["status"] == "accepted"}
    clean = [by_id[s["scenario_id"]] for s in index["scenarios"] if s.get("clean_run") and s.get("n_clean", 0) > 0]
    from .design import cell_covered, cell_id
    cells = required_cells(family, manifest["required_coverage"]["strength"])
    res = {"definition": "cells covered by scenarios whose run is clean and has >= 1 clean sample",
           "n_required": len(cells), "missing": [], "by_split": {}}
    for c in cells:
        if not any(cell_covered(c, e["bins"]) for e in clean):
            res["missing"].append(cell_id(c))
    for sp in ("train", "val", "test"):
        sub = [e for e in clean if e["split"] == sp]
        res["by_split"][sp] = {"n_scenarios": len(sub),
                               "n_cells_covered": sum(1 for c in cells if any(cell_covered(c, e["bins"]) for e in sub))}
    res["n_covered"] = len(cells) - len(res["missing"])
    return res


def _write_index(out: Path, index: dict, family, manifest: dict) -> None:
    sc = index["scenarios"]
    index["totals"] = {
        "scenarios_run": len(sc), "clean_runs": sum(1 for s in sc if s.get("clean_run")),
        "samples": sum(s.get("n_samples", 0) for s in sc), "clean_samples": sum(s.get("n_clean", 0) for s in sc),
        "clean_certified_samples": sum(s.get("n_certified_clean", 0) for s in sc),
        "clean_samples_by_split": {sp: sum(s.get("n_clean", 0) for s in sc if s["split"] == sp)
                                   for sp in ("train", "val", "test")},
        "run_failures": _count(f for s in sc for f in s.get("run_failures", [])),
    }
    index["coverage_realised"] = realised_coverage(manifest, index)
    tmp = out / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=1, allow_nan=False))
    os.replace(tmp, out / "index.json")


def _count(items) -> dict:
    d: dict = {}
    for x in items:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


# ------------------------------------------------------------------ contrast report
def load_samples(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _series(samples: list[dict]) -> dict | None:
    dec = [s for s in samples if s["oracle"]["phase"] == "decision" and s["expert"] is not None]
    if len(dec) < 3:
        return None
    d = np.array([s["oracle"]["distance_to_corner_entry_m"] for s in dec])
    order = np.argsort(d)
    get = lambda f: np.array([f(s) for s in dec])[order]  # noqa: E731
    return {"d": d[order], "steer": get(lambda s: s["expert"]["steer"]), "brake": get(lambda s: s["expert"]["brake"]),
            "throttle": get(lambda s: s["expert"]["throttle"]), "speed": get(lambda s: s["observation"]["speed"]),
            "d_obs": get(lambda s: s["oracle"]["distance_to_obstacle_m"]),
            "side": [s["expert"]["debug"].get("side") for s in dec if s["expert"]["debug"].get("n_obstacles")]}


def contrast_report(out: Path, manifest: dict, index: dict) -> dict:
    """Empirical check of each contrast pair's declared hypothesis on clean runs. A pair whose
    actions do not differ is reported as non-discriminative; it is not removed."""
    runs = {s["scenario_id"]: s for s in index["scenarios"]}
    params = {e["scenario_id"]: e["params"] for e in manifest["scenarios"] if e["status"] == "accepted"}
    rows, summary = [], {}
    for p in manifest["contrast_pairs"]:
        a, b = runs.get(p["base_id"]), runs.get(p["variant_id"])
        row = {"pair_id": p["pair_id"], "factor": p["factor"], "status": "not_evaluated"}
        if a and b and a.get("clean_run") and b.get("clean_run"):
            sa = _series(load_samples(out / a["files"]["samples"]["path"]))
            sb = _series(load_samples(out / b["files"]["samples"]["path"]))
            row.update(_compare(p["factor"], params[p["base_id"]], params[p["variant_id"]], sa, sb))
        elif a and b:
            row["status"] = "run_not_clean"
        rows.append(row)
        agg = summary.setdefault(p["factor"], {"pairs": 0, "evaluated": 0, "discriminative": 0, "consistent": 0,
                                               "inconsistent": 0, "no_hypothesis": 0})
        agg["pairs"] += 1
        if row["status"] == "evaluated":
            agg["evaluated"] += 1
            agg["discriminative"] += int(row["discriminative"])
            key = {True: "consistent", False: "inconsistent", None: "no_hypothesis"}[row["hypothesis_holds"]]
            agg[key] += 1
    return {"note": "Empirical effects on simulated runs; hypotheses are declared in the family spec and are not "
                    "guaranteed. Alignment: distance to corner entry (oracle), decision window only.",
            "backend": index["backend"]["name"], "by_factor": summary, "pairs": rows}


def _compare(factor: str, pa: dict, pb: dict, a: dict | None, b: dict | None) -> dict:
    if a is None or b is None:
        return {"status": "too_few_samples"}
    lo, hi = max(a["d"][0], b["d"][0]), min(a["d"][-1], b["d"][-1])
    if hi - lo < 5.0:
        return {"status": "no_overlap"}
    grid = np.arange(lo, hi, 1.0)
    ia = {k: np.interp(grid, a["d"], a[k]) for k in ("steer", "brake", "throttle", "speed")}
    ib = {k: np.interp(grid, b["d"], b[k]) for k in ("steer", "brake", "throttle", "speed")}
    mad = {k: float(np.mean(np.abs(ia[k] - ib[k]))) for k in ia}
    disc = mad["steer"] + mad["brake"] + mad["throttle"] >= 0.02 or mad["speed"] >= 0.25
    holds, stat = None, None
    if factor == "approach_speed_mps":
        hi_, lo_ = (ia, ib) if pa[factor] > pb[factor] else (ib, ia)
        stat = float(np.mean(hi_["brake"]) - np.mean(lo_["brake"]))
        holds = stat >= 0.0
    elif factor == "preview_m":
        sh, lg = (ia, ib) if pa[factor] < pb[factor] else (ib, ia)
        stat = float(np.mean(sh["speed"]) - np.mean(lg["speed"]))
        holds = stat <= 0.1
    elif factor == "turn_radius_m":
        tg, op = (a, b) if pa[factor] < pb[factor] else (b, a)
        stat = float(np.min(tg["speed"]) - np.min(op["speed"]))
        holds = stat <= 0.1
    elif factor == "obstacle_clearance_m":
        tg, wd = (a, b) if pa[factor] < pb[factor] else (b, a)
        stat = float(np.interp(0.0, tg["d_obs"][::-1], tg["speed"][::-1]) - np.interp(0.0, wd["d_obs"][::-1], wd["speed"][::-1]))
        holds = stat <= 0.1
    elif factor == "turn_direction":
        stat = float(np.mean(np.abs(ia["steer"] + ib["steer"])))   # mirror residual
        holds = stat <= 0.02
    elif factor == "obstacle_side":
        sa, sb = (a["side"][0] if a["side"] else None), (b["side"][0] if b["side"] else None)
        holds = None if sa is None or sb is None else bool(sa == -sb)
        stat = None if holds is None else float(sa)
    return {"status": "evaluated", "aligned_m": [float(lo), float(hi)], "mean_abs_diff": mad, "discriminative": bool(disc),
            "hypothesis_statistic": stat, "hypothesis_holds": holds}


def main_generate(a) -> int:
    try:
        idx = generate_corpus(a.manifest, a.out, a.backend, design_path=a.design, sim_config=a.config, sets=a.set,
                              limit=a.limit, resume=a.resume, workers=a.workers, verbose=not a.quiet)
    except DatagenError as e:
        print(f"data-generate: {e}", file=sys.stderr)
        return 3
    t = idx["totals"]
    cov = idx["coverage_realised"]
    print(f"\n{t['scenarios_run']} scenarios on backend '{idx['backend']['name']}'"
          f"{' (NOT CARLA)' if not idx['backend']['is_carla'] else ''}: {t['clean_runs']} clean runs, "
          f"{t['clean_samples']}/{t['samples']} clean samples {t['clean_samples_by_split']} "
          f"({t['clean_certified_samples']} with a certified longitudinal label)")
    if t["run_failures"]:
        print(f"run failures: {t['run_failures']}")
    print(f"realised coverage {cov['n_covered']}/{cov['n_required']}"
          + (f"; missing {cov['missing']}" if cov["missing"] else ""))
    print(f"-> {Path(a.out).resolve()}  (next: python -m autocross data-validate {a.out})")
    return 0
