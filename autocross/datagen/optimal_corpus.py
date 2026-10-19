"""Generate a small, reproducible scenario-level corpus for the optimal teacher."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from autocross.api import Observation, VehicleParams
from autocross.perception import RoadModelSensor
from autocross.track import build
from autocross import trackgen
from autocross.datagen.records import obs_to_dict, split_observation
from autocross.optimal.certificate import check_certificate, write_certificate
from autocross.optimal.teachers import (approx_label, cert_control_label,
                                        heuristic_label, optimal_label)

SCHEMA = "autocross.optimal.corpus"
VERSION = 1
CONDITIONS = ("optimal", "approx", "heuristic", "cert_control")
VP = VehicleParams(
    mass=1845.0, wheelbase=3.0, track_width=1.67, length=4.79, width=2.16,
    cg_to_front=1.45, cg_to_rear=1.55, max_steer=0.61, max_steer_rate=2.0,
    mu=0.95, max_accel=5.2, max_decel=9.0, max_speed=40.0,
    control_dt=0.05, actuation_delay_steps=0,
)


def _git_state(root: Path) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": run("rev-parse", "HEAD"),
            "status": run("status", "--porcelain=v1", "--untracked-files=no")}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _course_seed(seed: int, index: int) -> int:
    # SeedSequence gives independent streams without relying on Python's hash.
    return int(np.random.SeedSequence([int(seed), int(index), 0x4F5054]).generate_state(1, dtype=np.uint64)[0])


def _make_course(seed: int, closed: bool):
    spec = (trackgen.generate_closed(seed, target_length=280.0, width=6.0, laps=1)
            if closed else trackgen.generate_open(seed, target_length=280.0, width=6.0,
                                                  max_extent=320.0))
    return spec, build(spec)


def _make_observation(track, sensor: RoadModelSensor, progress: float, lateral: float,
                      speed: float, heading_error: float, step: int) -> Observation:
    pts, heads, _, wl, wr, _ = track.sample(np.array([track.s_of_progress(progress)]))
    p, heading = pts[0], float(heads[0])
    n = np.array([-np.sin(heading), np.cos(heading)])
    x, y = p + n * lateral
    sensor.s_hint = float(progress)
    road = sensor.measure(float(x), float(y), heading + heading_error, float(progress))
    return Observation(
        t=0.0, step=int(step), x=float(x), y=float(y), yaw=heading + heading_error,
        vx=float(speed), vy=0.0, yaw_rate=0.0, ax=0.0, ay=0.0, speed=float(speed),
        steer=0.0, throttle=0.0, brake=0.0, progress=float(progress), lap=0,
        timing_started=True, cones=None, road=road,
    )


def _cap_ok(track, progress: float, speed: float, vp: VehicleParams) -> bool:
    _, _, k, _, _, _ = track.sample(np.array([track.s_of_progress(progress)]))
    kap = abs(float(k[0]))
    if kap <= 1e-12:
        return speed <= vp.max_speed + 1e-9
    return speed <= float(np.sqrt(vp.mu * 9.81 / kap)) + 1e-9


def _candidates(track, samples: int, rng: np.random.Generator, vp: VehicleParams) -> list[dict]:
    # Leave enough preview after each state for a complete teacher horizon.
    end = max(float(track.length - 42.0), 0.0)
    progresses = np.arange(0.0, end + 1e-9, 4.0)
    rows = []
    idx = 0
    for progress in progresses:
        _, _, _, wl, wr, _ = track.sample(np.array([track.s_of_progress(float(progress))]))
        left, right = float(wl[0]), float(wr[0])
        for lateral0 in (-1.0, -0.5, 0.0, 0.5, 1.0):
            lateral = float(np.clip(lateral0, -right + vp.width / 2.0, left - vp.width / 2.0))
            for speed in (2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0):
                for heading_error in (0.0, -0.02, 0.02):
                    rows.append({"state_index": idx, "progress": float(progress),
                                 "lateral": lateral, "speed": speed,
                                 "heading_error": heading_error,
                                 "curvature_ok": _cap_ok(track, float(progress), speed, vp)})
                    idx += 1
    if len(rows) <= samples:
        return rows
    chosen = np.sort(rng.choice(len(rows), size=int(samples), replace=False))
    return [rows[int(i)] for i in chosen]


def _policy(obs: Observation) -> dict:
    full = obs_to_dict(obs)
    policy, privileged = split_observation(full)
    # This whitelist assertion is intentionally performed on every sample.
    if any(k in policy for k in ("x", "y", "yaw", "progress", "lap", "t", "step", "cones")):
        raise AssertionError("privileged observation field leaked into policy input")
    if privileged.get("road") is not None and any(k in policy["road"] for k in ("corners", "finish_distance")):
        raise AssertionError("privileged road field leaked into policy input")
    return policy


def _label(condition: str, obs: Observation, horizon: int):
    if condition == "optimal":
        return optimal_label(obs, VP, horizon=horizon)
    if condition == "approx":
        return approx_label(obs, VP, horizon=horizon, short_horizon=min(2, horizon))
    if condition == "heuristic":
        return heuristic_label(obs, VP)
    if condition == "cert_control":
        return cert_control_label(obs, VP, horizon=horizon)
    raise ValueError(condition)


def _write_cert(out: Path, condition: str, sample_id: str, label: dict) -> tuple[str, str]:
    cert = label.get("certificate")
    if cert is None:
        raise RuntimeError(f"{condition} label has no certificate for {sample_id}")
    report = check_certificate(cert)
    if not report.get("ok"):
        raise RuntimeError(f"invalid {condition} certificate for {sample_id}: {report.get('errors')}")
    path = out / "certificates" / condition / f"{sample_id}.json"
    write_certificate(cert, path)
    return str(path.relative_to(out)).replace("\\", "/"), _sha(path)


def generate(out: Path, *, train_courses: int, test_courses: int,
             samples_per_course: int, seed: int, val_courses: int = 1,
             conditions: tuple[str, ...] = CONDITIONS, smoke: bool = False) -> dict:
    if smoke:
        train_courses, val_courses, test_courses, samples_per_course = 1, 0, 1, 3
    if not conditions or any(c not in CONDITIONS for c in conditions):
        raise ValueError(f"conditions must be selected from {CONDITIONS}")
    if train_courses < 0 or val_courses < 0 or test_courses < 0 or samples_per_course < 1:
        raise ValueError("course and sample counts must be non-negative/positive")
    root = Path(__file__).resolve().parents[2]
    out.mkdir(parents=True, exist_ok=True)
    examples_path = out / "examples.jsonl"
    rng = np.random.default_rng(seed)
    course_rows = []
    counts = {c: {"n_examples": 0, "n_infeasible": 0, "n_skipped": 0} for c in conditions}
    course_total = train_courses + val_courses + test_courses
    with examples_path.open("w", encoding="utf-8") as examples_file:
        for ci in range(course_total):
            if ci < train_courses:
                split, local_index = "train", ci
            elif ci < train_courses + val_courses:
                split, local_index = "val", ci - train_courses
            else:
                split, local_index = "test", ci - train_courses - val_courses
            closed = bool(ci % 2)
            cseed = _course_seed(seed, ci)
            spec, track = _make_course(cseed, closed)
            scenario_id = f"{split}-course-{local_index:04d}"
            # A course-local sensor stream is deterministic and independent.
            sensor = RoadModelSensor(track, {"range_m": 40.0, "step_m": 1.0},
                                     np.random.default_rng(_course_seed(seed, ci) ^ 0xA11CE))
            candidates = _candidates(track, samples_per_course * 30, rng, VP)
            selected = 0
            skipped_cap = 0
            skipped_ambiguous = 0
            for cand in candidates:
                if selected >= samples_per_course:
                    break
                if not cand["curvature_ok"]:
                    skipped_cap += 1
                    continue
                try:
                    obs = _make_observation(track, sensor, cand["progress"], cand["lateral"],
                                            cand["speed"], cand["heading_error"], selected)
                    policy = _policy(obs)
                    labels = {c: _label(c, obs, 10) for c in conditions}
                except (AttributeError, TypeError, ValueError, RuntimeError, AssertionError):
                    skipped_ambiguous += 1
                    continue
                # Keep condition sets aligned: a scenario is represented only
                # when every requested policy has a usable label.
                if any(not labels[c].get("feasible", False) for c in conditions):
                    for c in conditions:
                        counts[c]["n_infeasible"] += 1
                    continue
                sample_id = f"{scenario_id}-s{selected:04d}"
                for condition, label in labels.items():
                    if condition in ("optimal", "approx"):
                        path, sha = _write_cert(out, condition, sample_id, label)
                        label["certificate_path"] = path
                        label["certificate_sha256"] = sha
                    else:
                        label["certificate_path"] = None
                        label["certificate_sha256"] = None
                        if condition == "cert_control":
                            label["certificate_ok"] = None
                    record = {"schema": SCHEMA, "version": VERSION,
                              "sample_id": sample_id, "scenario_id": scenario_id,
                              "state_index": int(cand["state_index"]), "split": split,
                              "condition": condition,
                              "course_family": "closed" if closed else "open",
                              "policy_observation": policy, "label": label}
                    examples_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    counts[condition]["n_examples"] += 1
                selected += 1
            if selected < samples_per_course:
                # The generator has searched a large deterministic pool; an
                # undersized split is explicit rather than silently fabricated.
                for c in conditions:
                    counts[c]["n_skipped"] += samples_per_course - selected
            for c in conditions:
                counts[c]["n_skipped"] += skipped_cap + skipped_ambiguous
            course_rows.append({"scenario_id": scenario_id, "split": split,
                                "family": "closed" if closed else "open", "seed": cseed,
                                "length_m": float(track.length), "spec": spec,
                                "n_requested": samples_per_course, "n_written": selected,
                                "n_skipped": skipped_cap + skipped_ambiguous})
    manifest = {
        "schema": SCHEMA, "version": VERSION, "seed": int(seed),
        "conditions": list(conditions), "course_families": {"open": int(sum(not bool(i % 2) for i in range(course_total))),
                                                              "closed": int(sum(bool(i % 2) for i in range(course_total)))},
        "splits": {"train_courses": int(train_courses), "val_courses": int(val_courses),
                   "test_courses": int(test_courses)},
        "counts": counts, "n_infeasible": int(sum(counts[c]["n_infeasible"] for c in conditions)),
        "n_skipped": int(sum(counts[c]["n_skipped"] for c in conditions)),
        "courses": course_rows, "examples": str(examples_path.relative_to(out)).replace("\\", "/"),
        "git": _git_state(root),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if smoke:
        print("CORPUS_OK")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-courses", type=int, default=1)
    parser.add_argument("--val-courses", type=int, default=1)
    parser.add_argument("--test-courses", type=int, default=1)
    parser.add_argument("--samples-per-course", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    generate(args.out, train_courses=args.train_courses, test_courses=args.test_courses,
             samples_per_course=args.samples_per_course, seed=args.seed, val_courses=args.val_courses,
             conditions=conditions, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
