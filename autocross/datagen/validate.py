"""Corpus validator: recomputes everything that can be recomputed and reports disagreements.

Errors (exit 1): schema/version, manifest consistency, file hashes, unexpected or missing files,
track/run provenance, sample identity, record structure (policy-input whitelist), every sample
hash (policy observation, training target, content, whole record), quality flags vs. recomputed
exclusion rules, full label replay (the expert re-run on the canonical policy observation alone,
compared on the complete training target: certificate, path, path frame, steering, pedals, target
speed), an independent Python recomputation of every certificate, model constants, applied-action
consistency with the actuator model, event/outcome consistency, duplicates and leakage across
splits, realised coverage, index totals.
Warnings: excluded runs, partial corpora, within-split duplicates, expert source drift."""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from . import INDEX_SCHEMA, SAMPLE_SCHEMA, SCHEMA_VERSION
from . import cert, steer
from .design import check_manifest, region_split
from .expert import model_constants
from .generate import (REPLAY_TOL, build_samples, certificate_block, expert_identity, file_sha256, is_certified,
                       lateral_block, realised_coverage, replay_diff, run_outcome)
from .records import (POLICY_EGO, POLICY_OBSERVATION_SCHEMA, POLICY_ROAD_KEYS, PRIVILEGED_OBS, PRIVILEGED_ROAD,
                      TRAINING_TARGET_SCHEMA, merge_observation, sample_hashes)
from .spec import canonical_json, get_family
from ..api import VehicleParams

ACTION_KEYS = ("steer", "throttle", "brake")
RECORD_KEYS = {"schema", "schema_version", "sample_id", "scenario_id", "group_id", "family", "split", "backend", "step",
               "t", "policy_observation", "target", "expert", "privileged", "applied", "outcome", "timing", "quality",
               "hashes"}
CERT_KEYS = {"model", "kind", "certified", "accel_mm_s2", "b1_mm2_s2", "b0_mm2_s2", "B0_mm2_s2"}
LATERAL_KEYS = {"model", "kind", "certified", "side", "amp", "offset_mm", "clearance_mm", "curvature_micro", "steer_milli"}


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.n_errors = 0
        self.stats: dict = {}

    def err(self, msg: str):
        self.n_errors += 1
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


def _raw_command(s: dict) -> dict | None:
    """The expert's raw command as the recorder saw it, rebuilt from target + diagnostics."""
    if s["target"] is None:
        return None
    return {**s["target"]["uncertified"], "debug": s["expert"]["debug"]}


def _rebuild_steps(samples: list[dict]) -> list[dict]:
    """The recorder's per-step view, rebuilt from samples (for recomputing run outcome and flags)."""
    steps = []
    for s in samples:
        a = s["applied"]
        steps.append({"obs": merge_observation(s["policy_observation"], s["privileged"]["observation"]),
                      "raw": _raw_command(s), "held": a["command_held"], "clipped": a["command_clipped"],
                      "timing": s["timing"], "ticks": a["ticks"], "first": a["first_tick"], "last": a["end_of_period"],
                      "off_course": s["outcome"]["off_course"], "events": s["outcome"]["events"],
                      "min_boundary_margin": s["outcome"]["min_boundary_margin_m"] if a["ticks"] else math.inf,
                      "progress_end": s["outcome"]["progress_end"]})
    return steps


def _structure_errors(s: dict) -> list[str]:
    """Shape of a v2 record: the policy observation is exactly the whitelist, privileged data is
    only in ``privileged``, the target has the declared form."""
    errs = []
    if set(s) != RECORD_KEYS:
        errs.append(f"record keys {sorted(set(s) ^ RECORD_KEYS)} unexpected/missing")
        return errs
    p = s["policy_observation"]
    if not isinstance(p, dict) or set(p) != {"schema", "road", *POLICY_EGO} or p.get("schema") != POLICY_OBSERVATION_SCHEMA:
        errs.append("policy_observation is not the canonical whitelist")
    elif p["road"] is not None and set(p["road"]) != set(POLICY_ROAD_KEYS):
        errs.append("policy_observation.road is not the canonical whitelist")
    pr = s["privileged"]
    if (not isinstance(pr, dict) or set(pr) != {"observation", "oracle"}
            or set(pr["observation"]) != {*PRIVILEGED_OBS, "road"}
            or (pr["observation"]["road"] is not None and set(pr["observation"]["road"]) != set(PRIVILEGED_ROAD))):
        errs.append("privileged block malformed")
    t = s["target"]
    if t is not None:
        if set(t) != {"schema", "longitudinal", "lateral", "uncertified"} or t["schema"] != TRAINING_TARGET_SCHEMA:
            errs.append("target malformed")
        elif t["longitudinal"] is not None and set(t["longitudinal"]) != CERT_KEYS:
            errs.append("certificate block malformed")
        elif t["lateral"] is not None and set(t["lateral"]) != LATERAL_KEYS:
            errs.append("lateral certificate block malformed")
        if s["expert"] is None or set(s["expert"]) != {"phase", "status", "debug"}:
            errs.append("expert diagnostics malformed")
    elif s["expert"] is not None:
        errs.append("expert diagnostics without a target")
    return errs


def _expected_files(index: dict) -> set[str]:
    return {f["path"] for r in index["scenarios"] if r.get("status") == "completed" for f in r["files"].values()}


TOP_LEVEL_FILES = {"index.json", "manifest.json", "contrast_report.json", "validation_report.json", "lean_report.json",
                   "dataset_report.json", "lean", "samples", "runs", "tracks", "RELEASE_FILES.sha256"}


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
        rep.err(f"unsupported index schema {index.get('schema')}/{index.get('schema_version')} "
                f"(this validator reads {INDEX_SCHEMA}/{SCHEMA_VERSION})")
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
    if index.get("certificate_model") != cert.CERT_MODEL:
        rep.err(f"corpus certificate model {index.get('certificate_model')!r} != {cert.CERT_MODEL!r}")
    if index.get("lateral_certificate_model") != steer.CERT_MODEL:
        rep.err(f"corpus lateral certificate model {index.get('lateral_certificate_model')!r} != {steer.CERT_MODEL!r}")
    cur = expert_identity()
    if (cur["source_sha256"] != index["expert"]["source_sha256"] or cur["version"] != index["expert"]["version"]
            or cur["certificate_source_sha256"] != index["expert"].get("certificate_source_sha256")
            or cur["lateral_certificate_source_sha256"] != index["expert"].get("lateral_certificate_source_sha256")):
        rep.warn("expert source/version changed since generation; label replay below uses the current expert")

    # ---- unexpected / unreferenced artifacts (fail closed)
    expected = _expected_files(index)
    for sub in ("samples", "runs", "tracks"):
        d = root / sub
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and str(p.relative_to(root)).replace("\\", "/") not in expected:
                    rep.err(f"unexpected file {p.relative_to(root)} (not referenced by index.json)")
    for p in root.iterdir():
        if p.name not in TOP_LEVEL_FILES:
            rep.err(f"unexpected top-level entry {p.name}")

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
    n_samples = n_clean = n_replayed = n_cert_checked = n_lateral_checked = n_clean_cert = n_clean_lateral_cert = 0
    domain_viol_stations = domain_viol_samples = 0  # visible stations tighter than the declared domain
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
        # ---- provenance: course, simulator, controller, model constants
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
        if summary["vehicle"] != r.get("vehicle"):
            rep.err(f"{where}: vehicle parameters in index differ from the run summary")
        vp = VehicleParams(**summary["vehicle"])
        rate = vp.max_steer_rate * dt
        consts = steer.model_constants(vp)
        if r.get("cert_model") != cert.CERT_MODEL or r.get("cert_constants") != consts["lon"]:
            rep.err(f"{where}: certificate model/constants in index differ from model_constants(vehicle)")
        if r.get("lateral_cert_model") != steer.CERT_MODEL or r.get("lateral_cert_constants") != consts:
            rep.err(f"{where}: lateral certificate model/constants differ from model_constants(vehicle)")
        # ---- per-sample structure first (later checks assume it)
        bad = False
        for i, s in enumerate(samples):
            sw = s.get("sample_id", where + f"#{i}") if isinstance(s, dict) else where + f"#{i}"
            if not isinstance(s, dict) or s.get("schema") != SAMPLE_SCHEMA or s.get("schema_version") != SCHEMA_VERSION:
                rep.err(f"{sw}: bad sample schema")
                bad = True
                continue
            for m in _structure_errors(s):
                rep.err(f"{sw}: {m}")
                bad = True
        if bad:
            continue
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
            sw = s["sample_id"]
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
            if s["privileged"]["observation"]["step"] != s["step"] or s["privileged"]["observation"]["t"] != s["t"]:
                rep.err(f"{sw}: observation step/time mismatch")
            if not _finite_tree(s):
                rep.err(f"{sw}: non-finite number")
            try:
                hashes = sample_hashes(s)
            except (TypeError, ValueError) as ex:
                rep.err(f"{sw}: record cannot be hashed ({ex})")
                continue
            if hashes != s["hashes"]:
                bad_h = [k for k in hashes if hashes[k] != (s["hashes"] or {}).get(k)]
                rep.err(f"{sw}: sample hash mismatch ({', '.join(bad_h)})")
            t = s["target"]
            phase = s["privileged"]["oracle"]["phase"]
            if t is not None:
                x = t["uncertified"]
                if not (abs(x["steer"]) <= 1 and 0 <= x["throttle"] <= 1 and 0 <= x["brake"] <= 1):
                    rep.err(f"{sw}: expert action outside the command range")
                lon = t["longitudinal"]
                if (phase == "approach") != (lon is None) or (phase == "approach") != (s["expert"]["phase"] == "approach"):
                    rep.err(f"{sw}: certificate presence inconsistent with phase {phase}")
                if lon is not None:
                    # independent Python recomputation from the policy observation (Lean does the same)
                    n_cert_checked += 1
                    nv = cert.domain_violations(consts["lon"], cert.quantise_dict(s["policy_observation"]))
                    if nv:
                        domain_viol_stations += nv
                        domain_viol_samples += 1
                        rep.err(f"{sw}: {nv} visible stations are tighter than the declared domain "
                                f"(horizon assumption contradicted)")
                    want = certificate_block(_cert_debug(cert.decide(consts["lon"], cert.quantise_dict(s["policy_observation"]))))
                    if lon != want:
                        rep.err(f"{sw}: certificate differs from the cert-2 decision of its policy observation")
                    if lon["certified"] != (lon["kind"] in ("greatest_safe", "min_violation")):
                        rep.err(f"{sw}: certified flag inconsistent with kind")
                lat = t["lateral"]
                if (phase == "approach") != (lat is None):
                    rep.err(f"{sw}: lateral certificate presence inconsistent with phase")
                if lat is not None:
                    n_lateral_checked += 1
                    want_lat = lateral_block(s["policy_observation"], vp, lon)
                    if lat != want_lat:
                        rep.err(f"{sw}: lateral certificate differs from steer-1 decision")
                    if lat["certified"] != (lat["kind"] == "plan_ok"):
                        rep.err(f"{sw}: lateral certified flag inconsistent with kind")
                a = s["applied"]
                if a["first_tick"] is not None and not a["command_held"] and delay == 0:
                    st0 = s["policy_observation"]["steer"]
                    want_st = min(max(st0 + min(max(x["steer"] - st0, -rate), rate), -1.0), 1.0)
                    if (abs(a["first_tick"]["steer"] - want_st) > 1e-12 or a["first_tick"]["throttle"] != x["throttle"]
                            or a["first_tick"]["brake"] != x["brake"]):
                        rep.err(f"{sw}: applied action inconsistent with the proposed action and the actuator model")
            if expect is not None:
                ex = expect[i]
                if ex["quality"]["exclusion_reasons"] != s["quality"]["exclusion_reasons"]:
                    rep.err(f"{sw}: exclusion reasons {s['quality']['exclusion_reasons']} != recomputed "
                            f"{ex['quality']['exclusion_reasons']}")
                if t is not None:
                    d = replay_diff(s, vp, family.expert_config(e["params"]) if phase == "approach" else None)
                    n_replayed += 1
                    max_replay = max(max_replay, d)
                    if not d <= REPLAY_TOL:
                        rep.err(f"{sw}: label replay differs by {d:.3g} (training target is not the expert's output "
                                f"on the recorded policy observation)")
            if s["quality"]["clean"] != (not s["quality"]["exclusion_reasons"]):
                rep.err(f"{sw}: clean flag inconsistent with exclusion reasons")
            if s["quality"]["clean"]:
                n_clean += 1
                lat_ok = bool(((s.get("target") or {}).get("lateral") or {}).get("certified"))
                if not is_certified(s) or not lat_ok or phase != "decision" or not r["clean_run"]:
                    rep.err(f"{sw}: clean sample without both certified labels in a clean run's decision window")
                else:
                    n_clean_cert += 1
                    n_clean_lateral_cert += 1
                h = s["hashes"]["content"]
                if h in content_split:
                    if content_split[h] != e["split"]:
                        rep.err(f"{sw}: identical sample content in splits {content_split[h]} and {e['split']}")
                    else:
                        rep.warn(f"{sw}: duplicate sample content within split {e['split']}")
                content_split[h] = e["split"]
            n_samples += 1
        if (len(samples) != r["n_samples"] or sum(s["quality"]["clean"] for s in samples) != r["n_clean"]
                or sum(s["quality"]["clean"] and is_certified(s) for s in samples) != r["n_certified_clean"]
                or sum(s["quality"]["clean"] and bool(((s.get("target") or {}).get("lateral") or {}).get("certified"))
                       for s in samples) != r.get("n_lateral_certified_clean", 0)
                or sum(s["privileged"]["oracle"]["phase"] == "decision" for s in samples) != r["n_decision"]):
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
    # ---- totals and coverage
    tot = index.get("totals", {})
    if (tot.get("clean_samples") != n_clean or tot.get("clean_certified_samples") != n_clean_cert
            or tot.get("clean_lateral_certified_samples") != n_clean_lateral_cert or tot.get("samples") != n_samples):
        rep.err(f"index totals differ from validated samples ({n_samples}, {n_clean}, {n_clean_cert}, {n_clean_lateral_cert})")
    if n_clean != n_clean_cert or n_clean != n_clean_lateral_cert:
        rep.err("clean samples are not certified by both models")
    if any(r.get("n_clean") for r in runs) and n_clean == 0:
        rep.err("index claims clean samples but none were validated")
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
                 "clean_certified_samples": n_clean_cert, "certificates_recomputed_python": n_cert_checked,
                 "lateral_certificates_recomputed": n_lateral_checked,
                 "label_replays": n_replayed, "label_replay_max_abs_diff": max_replay, "replay_tolerance": REPLAY_TOL,
                 "domain_assumption": {"samples_with_violations": domain_viol_samples,
                                       "stations_tighter_than_domain": domain_viol_stations},
                 "coverage": {"required": cov["n_required"], "covered": cov["n_covered"], "missing": cov["missing"]},
                 "errors_total": rep.n_errors}
    return rep


def _cert_debug(dec: dict) -> dict:
    d = {"cert_kind": dec["kind"], "cert_a": dec["a"]}
    for k in ("b1", "b0", "B0"):
        if dec[k] is not None:
            d[f"cert_{k}"] = dec[k]
    return d


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
    print(f"{'OK' if not rep.errors else 'FAILED'}: {rep.n_errors} errors, {len(rep.warnings)} warnings "
          f"-> {Path(a.corpus, 'validation_report.json')}")
    return 0 if not rep.errors else 1
