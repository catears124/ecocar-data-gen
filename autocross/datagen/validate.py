"""Corpus validator: recomputes everything that can be recomputed and reports disagreements.

Errors (exit 1): schema/version, manifest consistency, file hashes, track/run provenance,
sample identity and hashes, quality flags vs. recomputed exclusion rules, label replay (the
expert re-run on the recorded observation), applied-action consistency with the actuator model,
event/outcome consistency, duplicates and leakage across splits, realised coverage.
Warnings: excluded runs, partial corpora, within-split duplicates, expert source drift."""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from . import INDEX_SCHEMA, SAMPLE_SCHEMA, SCHEMA_VERSION
from .design import check_manifest, region_split
from .generate import (REPLAY_TOL, build_samples, expert_identity, file_sha256, realised_coverage, replay_diff,
                       run_outcome)
from .records import content_hash
from .spec import canonical_json, get_family
from ..api import VehicleParams

ACTION_KEYS = ("steer", "throttle", "brake")


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict = {}

    def err(self, msg: str):
        if len(self.errors) < 500:
            self.errors.append(msg)

    def warn(self, msg: str):
        if len(self.warnings) < 500:
            self.warnings.append(msg)


def _finite_tree(x) -> bool:
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(_finite_tree(v) for v in x.values())
    if isinstance(x, list):
        return all(_finite_tree(v) for v in x)
    return True


def _rebuild_steps(samples: list[dict]) -> list[dict]:
    """The recorder's per-step view, rebuilt from samples (for recomputing run outcome and flags)."""
    steps = []
    for s in samples:
        e = s["expert"]
        raw = None if e is None else {k: e[k] for k in ("steer", "throttle", "brake", "target_speed", "path",
                                                         "path_frame", "debug")}
        a = s["applied"]
        steps.append({"obs": s["observation"], "raw": raw, "held": a["command_held"], "clipped": a["command_clipped"],
                      "timing": s["timing"], "ticks": a["ticks"], "first": a["first_tick"], "last": a["end_of_period"],
                      "off_course": s["outcome"]["off_course"], "events": s["outcome"]["events"],
                      "min_boundary_margin": s["outcome"]["min_boundary_margin_m"] if a["ticks"] else math.inf,
                      "progress_end": s["outcome"]["progress_end"]})
    return steps


def validate_corpus(root: str | Path, *, replay: bool = True, allow_missing_coverage: bool = False) -> Report:
    root = Path(root)
    rep = Report()
    try:
        index = json.loads((root / "index.json").read_text())
        mbytes = (root / "manifest.json").read_bytes()
        manifest = json.loads(mbytes)
    except (OSError, ValueError) as e:
        rep.err(f"cannot read index.json / manifest.json: {e}")
        return rep
    if index.get("schema") != INDEX_SCHEMA or index.get("schema_version") != SCHEMA_VERSION:
        rep.err(f"unsupported index schema {index.get('schema')}/{index.get('schema_version')}")
        return rep
    for e in check_manifest(manifest):
        rep.err(f"manifest: {e}")
    if file_sha256(root / "manifest.json") != index["manifest"]["file_sha256"]:
        rep.err("manifest.json differs from the manifest recorded in index.json")
    if manifest.get("content_sha256") != index["manifest"]["content_sha256"]:
        rep.err("manifest content hash differs from index.json")
    family = get_family(manifest["family_spec"]["name"])
    backend = index["backend"]["name"]
    if index["backend"]["is_carla"] != (backend == "carla"):
        rep.err("backend.is_carla disagrees with backend.name")
    if backend != "carla":
        rep.warn(f"backend is '{backend}': NOT CARLA data; use only as a tooling/logic check")
    cur = expert_identity()
    if (cur["source_sha256"] != index["expert"]["source_sha256"] or cur["version"] != index["expert"]["version"]
            or cur["certificate_source_sha256"] != index["expert"].get("certificate_source_sha256")):
        rep.warn("expert source/version changed since generation; label replay below uses the current expert")

    by_id = {e["scenario_id"]: e for e in manifest["scenarios"] if e["status"] == "accepted"}
    runs = index["scenarios"]
    run_ids = [r["scenario_id"] for r in runs]
    if len(set(run_ids)) != len(run_ids):
        rep.err("index lists a scenario more than once")
    for sid in run_ids:
        if sid not in by_id:
            rep.err(f"index scenario {sid} is not an accepted manifest scenario")
    not_run = [sid for sid in by_id if sid not in set(run_ids)]
    if not_run:
        (rep.warn if index.get("limit") is not None else rep.err)(
            f"{len(not_run)} accepted manifest scenarios have no run" + (" (partial corpus: --limit)" if index.get("limit") is not None else ""))

    vcfg = index["sim_config"]
    dt = float(vcfg["sim"]["physics_dt"])
    delay = int(vcfg.get("actuators", {}).get("delay_steps", 0))
    seen_ids, content_split = set(), {}
    n_samples = n_clean = n_replayed = 0
    max_replay = 0.0
    for r in runs:
        sid = r["scenario_id"]
        e = by_id.get(sid)
        if e is None:
            continue
        where = f"{sid}"
        if r.get("split") != e["split"] or r.get("group_id") != e["group_id"] or r.get("sim_seed") != e["sim_seed"]:
            rep.err(f"{where}: split/group/seed in index differ from the manifest")
        if r.get("status") != "completed":
            rep.warn(f"{where}: run status {r.get('status')} ({r.get('error')})")
            if r.get("n_clean"):
                rep.err(f"{where}: failed run claims clean samples")
            continue
        for name, f in r["files"].items():
            p = root / f["path"]
            if not p.exists():
                rep.err(f"{where}: missing file {f['path']}")
            elif file_sha256(p) != f["sha256"]:
                rep.err(f"{where}: hash mismatch for {f['path']}")
        try:
            track = yaml.safe_load((root / r["files"]["track"]["path"]).read_text())
            summary = json.loads((root / r["files"]["summary.json"]["path"]).read_text())
            events = json.loads((root / r["files"]["events.json"]["path"]).read_text())
            samples = [json.loads(line) for line in open(root / r["files"]["samples"]["path"], encoding="utf-8") if line.strip()]
        except (OSError, ValueError, KeyError) as ex:
            rep.err(f"{where}: cannot read run artifacts ({ex})")
            continue
        # ---- provenance: course, simulator, controller
        if canonical_json(track) != canonical_json(family.track_spec(e["params"], sid)):
            rep.err(f"{where}: course YAML is not the course of its manifest parameters")
        if summary["backend"] != backend:
            rep.err(f"{where}: run backend {summary['backend']} != corpus backend {backend}")
        if summary["seed"] != e["sim_seed"]:
            rep.err(f"{where}: run seed {summary['seed']} != manifest sim_seed {e['sim_seed']}")
        if abs(float(summary["config"]["road_model"]["range_m"]) - float(e["params"]["preview_m"])) > 1e-9:
            rep.err(f"{where}: run preview range != scenario preview_m")
        if summary["controller"]["sha256"] != index["expert"]["source_sha256"][:16]:
            rep.err(f"{where}: run controller hash is not the recorded expert")
        if summary["config"].get("track_knowledge") != "perception":
            rep.err(f"{where}: run was not in perception (no-map) mode")
        vp = VehicleParams(**summary["vehicle"])
        rate = vp.max_steer_rate * dt
        # ---- run outcome and exclusion flags, recomputed
        steps = _rebuild_steps(samples)
        outcome, fails = run_outcome(summary, events, steps, e["layout"], e["params"], family)
        if fails != r["run_failures"] or outcome != r["outcome"]:
            rep.err(f"{where}: run outcome/failures differ from recomputation ({fails} vs {r['run_failures']})")
        if r["clean_run"] != (not fails):
            rep.err(f"{where}: clean_run flag inconsistent")
        ev_hits = sum(1 for x in events if x["kind"] == "cone_hit")
        if ev_hits != sum(1 for s in samples for x in s["outcome"]["events"] if x["kind"] == "cone_hit"):
            rep.err(f"{where}: cone hits in samples differ from events.json")
        expect = build_samples(e, steps, vp, backend, fails, family.expert_config(e["params"])) if replay else None
        # ---- per sample
        prev_step = -1
        for i, s in enumerate(samples):
            sw = f"{s.get('sample_id', where + f'#{i}')}"
            if s.get("schema") != SAMPLE_SCHEMA or s.get("schema_version") != SCHEMA_VERSION:
                rep.err(f"{sw}: bad sample schema")
                continue
            if s["step"] != prev_step + 1:
                rep.err(f"{sw}: control steps not contiguous")
            prev_step = s["step"]
            if s["sample_id"] != f"{sid}:{s['step']:05d}" or s["sample_id"] in seen_ids:
                rep.err(f"{sw}: bad or duplicate sample_id")
            seen_ids.add(s["sample_id"])
            for k, v in (("scenario_id", sid), ("group_id", e["group_id"]), ("split", e["split"]),
                         ("family", family.name), ("backend", backend)):
                if s.get(k) != v:
                    rep.err(f"{sw}: {k}={s.get(k)!r}, expected {v!r}")
            if s["observation"]["step"] != s["step"]:
                rep.err(f"{sw}: observation step mismatch")
            if not _finite_tree(s):
                rep.err(f"{sw}: non-finite number")
            x = s["expert"]
            if x is not None:
                if s["content_sha256"] != content_hash(s["observation"], x):
                    rep.err(f"{sw}: content hash mismatch")
                if not (abs(x["steer"]) <= 1 and 0 <= x["throttle"] <= 1 and 0 <= x["brake"] <= 1):
                    rep.err(f"{sw}: expert action outside the command range")
            a = s["applied"]
            if a["first_tick"] is not None and x is not None and not a["command_held"] and delay == 0:
                st0 = s["observation"]["steer"]
                want = min(max(st0 + min(max(x["steer"] - st0, -rate), rate), -1.0), 1.0)
                if (abs(a["first_tick"]["steer"] - want) > 1e-12 or a["first_tick"]["throttle"] != x["throttle"]
                        or a["first_tick"]["brake"] != x["brake"]):
                    rep.err(f"{sw}: applied action inconsistent with the proposed action and the actuator model")
            if expect is not None:
                ex = expect[i]
                if ex["quality"]["exclusion_reasons"] != s["quality"]["exclusion_reasons"]:
                    rep.err(f"{sw}: exclusion reasons {s['quality']['exclusion_reasons']} != recomputed "
                            f"{ex['quality']['exclusion_reasons']}")
                if x is not None and s["oracle"]["phase"] != "approach":
                    d = replay_diff(s["observation"], x, vp, None)
                    n_replayed += 1
                    max_replay = max(max_replay, d)
                    if not d <= REPLAY_TOL:
                        rep.err(f"{sw}: label replay differs by {d:.3g} (label is not a function of the observation)")
            if s["quality"]["clean"] != (not s["quality"]["exclusion_reasons"]):
                rep.err(f"{sw}: clean flag inconsistent with exclusion reasons")
            if s["quality"]["clean"]:
                n_clean += 1
                h = s["content_sha256"]
                if h in content_split:
                    if content_split[h] != e["split"]:
                        rep.err(f"{sw}: identical sample content in splits {content_split[h]} and {e['split']}")
                    else:
                        rep.warn(f"{sw}: duplicate sample content within split {e['split']}")
                content_split[h] = e["split"]
            n_samples += 1
        if len(samples) != r["n_samples"] or sum(s["quality"]["clean"] for s in samples) != r["n_clean"]:
            rep.err(f"{where}: sample counts differ from index")
        if not r["clean_run"]:
            rep.warn(f"{where}: run excluded ({', '.join(r['run_failures'])})")

    # ---- split leakage (scenario level)
    gsplit, hsplit = {}, {}
    for e in by_id.values():
        if gsplit.setdefault(e["group_id"], e["split"]) != e["split"]:
            rep.err(f"group {e['group_id']} spans several splits")
        if hsplit.setdefault(e["scenario_sha256"], e["split"]) != e["split"]:
            rep.err(f"scenario hash {e['scenario_sha256'][:16]} in several splits")
        if e["split"] != region_split(manifest["split_policy"], e["bins"]):
            rep.err(f"{e['scenario_id']}: split differs from its parameter region's split")
    for p in manifest["contrast_pairs"]:
        if by_id[p["base_id"]]["split"] != by_id[p["variant_id"]]["split"]:
            rep.err(f"contrast pair {p['pair_id']} crosses splits")
    # ---- coverage
    cov = realised_coverage(manifest, index)
    if cov != index.get("coverage_realised"):
        rep.err("realised coverage in index.json differs from recomputation")
    if cov["missing"]:
        (rep.warn if allow_missing_coverage else rep.err)(
            f"required coverage cells without a clean demonstration: {', '.join(cov['missing'])}")
    if not (root / "contrast_report.json").exists():
        rep.warn("contrast_report.json missing")
    rep.stats = {"backend": backend, "scenarios_in_manifest": len(by_id), "runs": len(runs),
                 "clean_runs": sum(1 for r in runs if r.get("clean_run")), "samples": n_samples, "clean_samples": n_clean,
                 "label_replays": n_replayed, "label_replay_max_abs_diff": max_replay,
                 "coverage": {"required": cov["n_required"], "covered": cov["n_covered"], "missing": cov["missing"]}}
    return rep


def main_validate(a) -> int:
    rep = validate_corpus(a.corpus, replay=not a.no_replay, allow_missing_coverage=a.allow_missing_coverage)
    out = {"ok": not rep.errors, "errors": rep.errors, "warnings": rep.warnings, "stats": rep.stats}
    Path(a.corpus, "validation_report.json").write_text(json.dumps(out, indent=1))
    for w in rep.warnings[:15]:
        print(f"  warning: {w}")
    if len(rep.warnings) > 15:
        print(f"  ... {len(rep.warnings) - 15} more warnings")
    for e in rep.errors[:30]:
        print(f"  ERROR: {e}")
    print(json.dumps(rep.stats, indent=1))
    print(f"{'OK' if not rep.errors else 'FAILED'}: {len(rep.errors)} errors, {len(rep.warnings)} warnings "
          f"-> {Path(a.corpus, 'validation_report.json')}")
    return 0 if not rep.errors else 1
