"""Dataset report (``dataset_report.json``): what the corpus contains and what was excluded.

Nothing is dropped: every accepted manifest scenario is run, excluded runs keep their samples
(flagged with exclusion reasons), and this report breaks the failures down by reason, by
scenario factor bin and by split, with Wilson 95 % intervals on the per-run failure rates.

Statistical unit: samples of one run are a time series of one closed-loop rollout and are
strongly correlated; scenario groups (one per accepted scenario here) are the independent units.
``effective_samples_ar1`` is an indicative AR(1) effective sample size of the certified
longitudinal target within each clean run (n (1 - rho) / (1 + rho)), NOT a count of independent
observations."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .generate import is_certified, load_samples
from .spec import get_family

SPLITS = ("train", "val", "test")


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    """Wilson score interval for k successes out of n (None when n = 0)."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _rate(fail: int, n: int) -> dict:
    return {"runs": n, "excluded": fail, "clean": n - fail,
            "exclusion_rate": None if n == 0 else round(fail / n, 4), "exclusion_rate_wilson95": wilson(fail, n)}


def _ess_ar1(x: np.ndarray) -> float:
    n = len(x)
    if n < 3 or float(np.std(x)) == 0.0:
        return float(min(n, 1))
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    den = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    rho = float(np.sum(a * b) / den) if den > 0 else 0.0
    rho = min(max(rho, -0.99), 0.99)
    return float(min(max(n * (1 - rho) / (1 + rho), 1.0), n))


def dataset_report(root: str | Path) -> dict:
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    family = get_family(manifest["family_spec"]["name"])
    by_id = {e["scenario_id"]: e for e in manifest["scenarios"] if e["status"] == "accepted"}
    runs = index["scenarios"]
    excluded = [r for r in runs if not r.get("clean_run")]

    reasons_runs = Counter(f for r in runs for f in r.get("run_failures", []))
    sample_reasons, phases = Counter(), Counter()
    per_split = {sp: {"runs": 0, "clean_runs": 0, "samples": 0, "clean_samples": 0, "certified_clean_samples": 0,
                      "uncertified_clean_samples": 0, "groups": set(), "clean_groups": set(),
                      "effective_samples_ar1": 0.0} for sp in SPLITS}
    kinds = Counter()
    contents = Counter()
    n_samples = n_clean = n_cert = 0
    for r in runs:
        sp = per_split[r["split"]]
        sp["runs"] += 1
        sp["groups"].add(r["group_id"])
        if r.get("status") != "completed":
            continue
        samples = load_samples(root / r["files"]["samples"]["path"])
        acc = []
        for s in samples:
            n_samples += 1
            sp["samples"] += 1
            phases[s["privileged"]["oracle"]["phase"]] += 1
            for x in s["quality"]["exclusion_reasons"]:
                sample_reasons[x] += 1
            if s["quality"]["clean"]:
                n_clean += 1
                sp["clean_samples"] += 1
                if is_certified(s):
                    n_cert += 1
                    sp["certified_clean_samples"] += 1
                    kinds[s["target"]["longitudinal"]["kind"]] += 1
                    contents[s["hashes"]["content"]] += 1
                    acc.append(s["target"]["longitudinal"]["accel_mm_s2"])
                else:
                    sp["uncertified_clean_samples"] += 1
        if r.get("clean_run"):
            sp["clean_runs"] += 1
            if r.get("n_clean", 0) > 0:
                sp["clean_groups"].add(r["group_id"])
            sp["effective_samples_ar1"] += _ess_ar1(np.asarray(acc, float))
    for sp in per_split.values():
        sp["groups"], sp["clean_groups"] = len(sp["groups"]), len(sp["clean_groups"])
        sp["effective_samples_ar1"] = round(sp["effective_samples_ar1"], 1)

    by_factor = {}
    for f in family.factors:
        rows = {}
        for lab in f.labels:
            sub = [r for r in runs if by_id[r["scenario_id"]]["bins"].get(f.name) == lab]
            rows[lab] = _rate(sum(1 for r in sub if not r.get("clean_run")), len(sub))
            rows[lab]["reasons"] = dict(Counter(x for r in sub for x in r.get("run_failures", [])).most_common())
        by_factor[f.name] = rows
    by_split = {sp: _rate(sum(1 for r in runs if r["split"] == sp and not r.get("clean_run")),
                          sum(1 for r in runs if r["split"] == sp)) for sp in SPLITS}
    pairs = defaultdict(lambda: [0, 0])
    for r in runs:
        b = by_id[r["scenario_id"]]["bins"]
        key = f"turn_radius_m={b.get('turn_radius_m')}|obstacle_clearance_m={b.get('obstacle_clearance_m')}"
        pairs[key][0] += 1
        pairs[key][1] += 0 if r.get("clean_run") else 1
    cov = index["coverage_realised"]
    return {
        "schema": "autocross.datagen.dataset_report/1",
        "backend": index["backend"]["name"], "is_carla": index["backend"]["is_carla"],
        "statistical_unit": "scenario group (one closed-loop run); samples within a run are a correlated time series",
        "totals": {"accepted_scenarios": len(by_id), "runs": len(runs),
                   "clean_runs": sum(1 for r in runs if r.get("clean_run")), "excluded_runs": len(excluded),
                   "scenario_groups": len({r["group_id"] for r in runs}),
                   "clean_scenario_groups": sum(per_split[sp]["clean_groups"] for sp in SPLITS),
                   "samples": n_samples, "clean_samples": n_clean, "certified_clean_samples": n_cert,
                   "uncertified_clean_samples": n_clean - n_cert, "excluded_samples": n_samples - n_clean,
                   "samples_by_phase": dict(phases), "clean_certificate_kinds": dict(kinds),
                   "distinct_clean_examples": len(contents),
                   "repeated_clean_examples": sum(v - 1 for v in contents.values()),
                   "effective_samples_ar1": round(sum(per_split[sp]["effective_samples_ar1"] for sp in SPLITS), 1)},
        "by_split": {sp: {**per_split[sp], "run_exclusion": by_split[sp]} for sp in SPLITS},
        "run_exclusions_by_reason": dict(reasons_runs.most_common()),
        "sample_exclusion_reasons": dict(sample_reasons.most_common()),
        "run_exclusion_by_factor_bin": by_factor,
        "run_exclusion_by_turn_radius_x_clearance": {k: _rate(v[1], v[0]) for k, v in sorted(pairs.items())},
        "excluded_runs": [{"scenario_id": r["scenario_id"], "split": r["split"], "failures": r.get("run_failures"),
                           "bins": by_id[r["scenario_id"]]["bins"], "n_samples": r.get("n_samples", 0)}
                          for r in excluded],
        "coverage_clean": {"required": cov["n_required"], "covered": cov["n_covered"], "missing": cov["missing"],
                           "by_split": cov["by_split"]},
        "note": "Excluded runs are kept in the corpus with all samples and exclusion reasons. Failure rates are per run; "
                "intervals are Wilson 95 % on runs (binomial, runs treated as independent).",
    }


def main_report(a) -> int:
    rep = dataset_report(a.corpus)
    Path(a.corpus, "dataset_report.json").write_text(json.dumps(rep, indent=1))
    t = rep["totals"]
    print(json.dumps(t, indent=1))
    for f, rows in rep["run_exclusion_by_factor_bin"].items():
        print(f"{f}: " + ", ".join(f"{k} {v['excluded']}/{v['runs']} {v['exclusion_rate_wilson95']}" for k, v in rows.items()))
    print(f"-> {Path(a.corpus, 'dataset_report.json')}")
    return 0
