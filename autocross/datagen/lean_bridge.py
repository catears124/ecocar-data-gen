"""Export the manifest / corpus as integer interchange files and run the Lean checker on them.

Files written (``<corpus>/lean/`` or ``--out``):
  interchange.json        spec, scenarios, pairs, splits, coverage claims, clean-sample hashes,
                          mirror-check samples (fixed-point integers only)
  certs/<scenario>.json   quantised observation + recorded certified decision for every
                          expert-phase sample of every completed run
  ../lean_report.json     the checker's verdicts, the axiom audit and the input hashes"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from . import INTERCHANGE_SCHEMA, SCHEMA_VERSION, DatagenError
from .cert import quantise
from .generate import file_sha256, load_samples
from .records import mirror_action, mirror_observation, obs_from_dict
from .spec import get_family, to_fixed
from ..api import VehicleParams
from ..config import ROOT

LEAN_DIR = ROOT / "lean"
FORMAL_CONSTANTS = ("obstacle_inset_m", "min_obstacle_width_m", "radius_margin_m")
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}
MIRROR_SAMPLES_PER_RUN = 3


def _spec(family) -> dict:
    return {"name": family.name, "version": family.version,
            "factors": [{"name": f.name, "levels": list(f.levels),
                         "bins": [{"label": b.label, "lo": to_fixed(b.lo), "hi": to_fixed(b.hi),
                                   "hiInclusive": b.hi_inclusive} for b in f.bins]} for f in family.factors],
            "nuisance": [{"name": n.name, "lo": to_fixed(n.lo), "hi": to_fixed(n.hi)} for n in family.nuisance],
            "constants": [[k, to_fixed(family.constants[k])] for k in FORMAL_CONSTANTS]}


def _scenario(family, e: dict) -> dict | None:
    if e.get("scenario_id") is None:  # schema-level rejection: parameters not representable
        return None
    c = family.canonical_params(e["params"])
    return {"id": e["scenario_id"], "hash": int(e["scenario_sha256"], 16), "family": e["family"],
            "familyVersion": e["family_version"],
            "nums": [[f.name, c[f.name]] for f in family.factors if f.numeric] + [[n.name, c[n.name]] for n in family.nuisance],
            "cats": [[f.name, c[f.name]] for f in family.factors if not f.numeric],
            "bins": [[k, v] for k, v in e["bins"].items() if v is not None],
            "accepted": e["status"] == "accepted",
            "formalRejected": any(r.startswith("formal:") for r in e["rejection_reasons"]),
            "split": e.get("split", ""), "group": e.get("group_id", ""), "region": e.get("region", "")}


def _fx(x) -> int:
    return int(round(float(x) * 1_000_000))


def _pts(p) -> list:
    return [[_fx(a), _fx(b)] for a, b in p]


def obs_m(o: dict) -> dict:
    r = o["road"]
    return {"lateral": _fx(r["lateral_offset"]), "heading": _fx(r["heading_error"]), "curvature": _fx(r["curvature"]),
            "widthLeft": _fx(r["width_left"]), "widthRight": _fx(r["width_right"]),
            "distLeft": _fx(r["dist_to_left_boundary"]), "distRight": _fx(r["dist_to_right_boundary"]),
            "vy": _fx(o["vy"]), "yawRate": _fx(o["yaw_rate"]), "ay": _fx(o["ay"]), "steer": _fx(o["steer"]),
            "previewCurvature": [_fx(v) for v in r["preview_curvature"]],
            "previewWidthLeft": [_fx(v) for v in r["preview_width_left"]],
            "previewWidthRight": [_fx(v) for v in r["preview_width_right"]],
            "centerline": _pts(r["preview_centerline"]), "leftBoundary": _pts(r["preview_left_boundary"]),
            "rightBoundary": _pts(r["preview_right_boundary"]),
            "obstacles": [{"dist": _fx(b["distance"]), "latMin": _fx(b["lateral_min"]), "latMax": _fx(b["lateral_max"])}
                          for b in r["obstacles"]],
            "cornerDirections": [int(c["direction"]) for c in r["corners"]],
            "cornerTurnAngles": [_fx(c["turn_angle"]) for c in r["corners"]]}


def act_m(a: dict) -> dict:
    return {"steer": _fx(a["steer"]), "throttle": _fx(a["throttle"]), "brake": _fx(a["brake"]),
            "path": _pts(a["path"] or [])}


def export(manifest: dict, out: Path, index: dict | None = None, corpus: Path | None = None) -> tuple[Path, Path | None]:
    """Write interchange.json (+ certs/*.json for a corpus); returns (interchange, cert dir)."""
    from .expert import model_constants
    family = get_family(manifest["family_spec"]["name"])
    out.mkdir(parents=True, exist_ok=True)
    pol = manifest["split_policy"]
    inter = {
        "schema": INTERCHANGE_SCHEMA, "schemaVersion": SCHEMA_VERSION, "spec": _spec(family),
        "strength": manifest["required_coverage"]["strength"],
        "splitPolicy": {"regionFactors": pol["region_factors"],
                        "regions": [{"split": sp, "cell": [[k, v] for k, v in reg.items()]}
                                    for sp in ("val", "test") for reg in pol.get("regions", {}).get(sp, []) or []]},
        "scenarios": [x for x in (_scenario(family, e) for e in manifest["scenarios"]) if x is not None],
        "pairs": [{"base": p["base_id"], "variant": p["variant_id"], "factor": p["factor"]} for p in manifest["contrast_pairs"]],
        "requiredCells": manifest["required_coverage"]["cells"], "manifestMissing": manifest["coverage"]["missing"],
        "hasCorpus": index is not None, "backend": index["backend"]["name"] if index else "",
        "realisedClean": [], "realisedMissing": [], "cleanSamples": [], "mirrorChecks": [],
    }
    cert_dir = None
    if index is not None:
        inter["realisedClean"] = [s["scenario_id"] for s in index["scenarios"] if s.get("clean_run") and s.get("n_clean", 0) > 0]
        inter["realisedMissing"] = index["coverage_realised"]["missing"]
        hashes = []
        cert_dir = out / "certs"
        if cert_dir.exists():
            shutil.rmtree(cert_dir)
        cert_dir.mkdir()
        for run in index["scenarios"]:
            if run.get("status") != "completed":
                continue
            samples = load_samples(corpus / run["files"]["samples"]["path"])
            consts = model_constants(VehicleParams(**run["vehicle"]))
            certs, mirrors = [], []
            for s in samples:
                x = s["expert"]
                if s["quality"]["clean"]:
                    hashes.append((int(s["content_sha256"], 16), SPLIT_INDEX[s["split"]]))
                    if len(mirrors) < MIRROR_SAMPLES_PER_RUN and s["observation"]["road"] and s["observation"]["road"]["obstacles"]:
                        mirrors.append({"sampleId": s["sample_id"], "obs": obs_m(s["observation"]), "act": act_m(x),
                                        "mirroredObs": obs_m(mirror_observation(s["observation"])),
                                        "mirroredAct": act_m(mirror_action(x))})
                cb = x.get("certificate") if x else None
                if cb is None:
                    continue
                q = quantise(obs_from_dict(s["observation"]))
                if q is None:
                    continue
                kind = {"greatest_safe": 0, "min_violation": 1, "no_feasible_gap": 2, "invalid_input": 3}[cb["kind"]]
                certs.append({"sampleId": s["sample_id"], "clean": s["quality"]["clean"], "q": q,
                              "decision": {"kind": kind, "a": cb["accel_mm_s2"], "b1": cb.get("b1_mm2_s2")}})
            inter["mirrorChecks"] += mirrors
            p = cert_dir / f"{run['scenario_id']}.json"
            p.write_text(json.dumps({"schema": "autocross.datagen.cert", "scenarioId": run["scenario_id"],
                                     "consts": consts, "samples": certs}, separators=(",", ":")))
        inter["cleanSamples"] = [list(h) for h in sorted(hashes)]
    ip = out / "interchange.json"
    ip.write_text(json.dumps(inter, separators=(",", ":")))
    return ip, cert_dir


def lake_path() -> str:
    exe = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / ("lake.exe" if os.name == "nt" else "lake"))
    if not Path(exe).exists():
        raise DatagenError("Lean 4 build tool 'lake' not found. Install elan (https://github.com/leanprover/elan); the "
                           "toolchain is pinned in lean/lean-toolchain.")
    return exe


def run_checker(interchange: Path, cert_dir: Path | None, report_path: Path) -> dict:
    lake = lake_path()
    b = subprocess.run([lake, "build"], cwd=LEAN_DIR, capture_output=True, text=True)
    if b.returncode != 0:
        raise DatagenError(f"lake build failed:\n{b.stdout[-3000:]}\n{b.stderr[-2000:]}")
    exe = LEAN_DIR / ".lake" / "build" / "bin" / ("corpuscheck.exe" if os.name == "nt" else "corpuscheck")
    r = subprocess.run([str(exe), str(interchange)] + ([str(cert_dir)] if cert_dir else []), capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise DatagenError(f"corpuscheck failed ({r.returncode}): {r.stderr[-2000:]}")
    report = json.loads(r.stdout)
    audit = subprocess.run([lake, "env", "lean", "Corpus/Audit.lean"], cwd=LEAN_DIR, capture_output=True, text=True)
    lines = [ln for ln in audit.stdout.splitlines() if "depends on axioms" in ln or "does not depend" in ln]
    report["axiom_audit"] = {"lines": lines, "sorry_free": audit.returncode == 0 and not any("sorryAx" in ln for ln in lines)}
    if not report["axiom_audit"]["sorry_free"]:
        report["ok"] = False
    report["inputs"] = {"interchange": str(interchange), "interchange_sha256": file_sha256(interchange),
                        "cert_dir": str(cert_dir) if cert_dir else None,
                        "cert_files_sha256": {p.name: file_sha256(p) for p in sorted(cert_dir.glob("*.json"))} if cert_dir else {}}
    report["lean_sources_sha256"] = {str(p.relative_to(LEAN_DIR)).replace(os.sep, "/"): file_sha256(p)
                                     for p in sorted(LEAN_DIR.glob("**/*.lean")) if ".lake" not in p.parts}
    report_path.write_text(json.dumps(report, indent=1))
    return report


def main_lean(a) -> int:
    target = Path(a.target)
    try:
        if target.is_dir():
            index = json.loads((target / "index.json").read_text())
            manifest = json.loads((target / "manifest.json").read_text())
            files = export(manifest, target / "lean", index, target)
            report_path = target / "lean_report.json"
        else:
            manifest = json.loads(target.read_text())
            out = Path(a.out) if a.out else target.with_suffix(".lean")
            files = export(manifest, out)
            report_path = out / "lean_report.json"
        report = run_checker(*files, report_path)
    except DatagenError as e:
        print(f"data-lean: {e}")
        return 3
    for r in report["results"]:
        print(f"  [{'ok' if r['ok'] else 'FAIL'}] {r['name']:<42} {r['basis']}" + (f"\n        {r['detail']}" if r["detail"] else ""))
    print(f"  axiom audit: {'only standard axioms, no sorry' if report['axiom_audit']['sorry_free'] else 'FAILED'}")
    print(json.dumps(report["stats"]))
    print(f"{'OK' if report['ok'] else 'FAILED'} (Lean {report['lean_version']}, backend '{report['backend'] or 'n/a'}') "
          f"-> {report_path}")
    return 0 if report["ok"] else 1
