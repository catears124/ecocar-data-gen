"""Scenario manifest: finite coverage design, seeded in-cell variation, matched contrasts,
region-held-out splits, coverage accounting, and the manifest consistency check.

Everything here is deterministic: the same design config, profile and generator version give a
byte-identical manifest ``content`` (and therefore identical scenario IDs and hashes)."""
from __future__ import annotations

import copy
import hashlib
import itertools
import math

import numpy as np

from . import GENERATOR_NAME, GENERATOR_VERSION, MANIFEST_SCHEMA, SCHEMA_VERSION, git_state
from .spec import Family, canonical_json, get_family, sha256_obj, to_fixed

SPLITS = ("train", "val", "test")
SPLIT_RANK = {s: i for i, s in enumerate(SPLITS)}


# ------------------------------------------------------------------ covering designs
def covering_rows(levels: list[int], strength: int) -> list[list[int]]:
    """Rows of level indices such that every ``strength``-tuple of factor levels appears.
    strength 1: a Latin-style diagonal (row i uses level (i + j) mod n_j for factor j).
    strength 2: deterministic greedy pairwise cover (seed with the first uncovered pair, then
    fill each factor with the level covering most uncovered pairs, lowest index on ties)."""
    if strength == 1:
        return [[(i + j) % n for j, n in enumerate(levels)] for i in range(max(levels))]
    if strength != 2:
        raise ValueError("strength must be 1 or 2")
    F = len(levels)
    uncovered = {((a, la), (b, lb)) for a, b in itertools.combinations(range(F), 2)
                 for la in range(levels[a]) for lb in range(levels[b])}
    rows = []
    while uncovered:
        (a, la), (b, lb) = min(uncovered)
        row: list[int | None] = [None] * F
        row[a], row[b] = la, lb
        for f in range(F):
            if row[f] is not None:
                continue
            best, best_gain = 0, -1
            for lv in range(levels[f]):
                gain = sum(1 for g in range(F) if row[g] is not None and
                           ((min(f, g), lv if f < g else row[g]), (max(f, g), row[g] if f < g else lv)) in uncovered)
                if gain > best_gain:
                    best, best_gain = lv, gain
            row[f] = best
        rows.append(row)
        for x, y in itertools.combinations(range(F), 2):
            uncovered.discard(((x, row[x]), (y, row[y])))
    return rows


def required_cells(family: Family, strength: int) -> list[dict]:
    out = []
    for combo in itertools.combinations(family.factors, strength):
        for labels in itertools.product(*(f.labels for f in combo)):
            out.append({f.name: lab for f, lab in zip(combo, labels)})
    return out


def cell_id(cell: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in cell.items())


def cell_covered(cell: dict, bins: dict) -> bool:
    return all(bins.get(k) == v for k, v in cell.items())


# ------------------------------------------------------------------ sampling
def derive_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def _q(v: float) -> float:
    return round(v, 3)


def _in_bin(lo: float, hi: float, inclusive: bool, u: float) -> float:
    v = _q(lo + u * (hi - lo))
    if not inclusive and to_fixed(v) >= to_fixed(hi):
        v = _q(hi - 0.001)
    return v


def sample_params(family: Family, row: list[int], rng: np.random.Generator) -> dict:
    p = {}
    for f, lv in zip(family.factors, row):
        if f.numeric:
            b = f.bins[lv]
            p[f.name] = _in_bin(b.lo, b.hi, b.hi_inclusive, float(rng.random()))
        else:
            p[f.name] = f.levels[lv]
    for n in family.nuisance:
        p[n.name] = _q(n.lo + float(rng.random()) * (n.hi - n.lo))
    return p


def move_factor(family: Family, params: dict, factor: str, target_label: str) -> dict:
    """Copy of ``params`` with one factor moved to ``target_label``, keeping its relative position
    inside the bin (numeric) so the contrast changes nothing but the declared factor."""
    f = family.factor(factor)
    out = dict(params)
    if not f.numeric:
        out[factor] = target_label
        return out
    src = f.bin(f.bin_of(params[factor]))
    u = (params[factor] - src.lo) / (src.hi - src.lo)
    dst = f.bin(target_label)
    out[factor] = _in_bin(dst.lo, dst.hi, dst.hi_inclusive, u)
    return out


def contrast_targets(family: Family, factor: str, label: str) -> list[str]:
    labs = list(family.factor(factor).labels)
    i = labs.index(label)
    return [labs[j] for j in sorted((j for j in range(len(labs)) if j != i), key=lambda j: (-abs(j - i), j))]


# ------------------------------------------------------------------ splits
def region_split(split_cfg: dict, bins: dict) -> str:
    for sp in ("test", "val"):
        for reg in split_cfg.get("regions", {}).get(sp, []) or []:
            if all(bins.get(k) == v for k, v in reg.items()):
                return sp
    return "train"


def region_key(split_cfg: dict, bins: dict) -> str:
    return "|".join(f"{f}={bins[f]}" for f in split_cfg["region_factors"])


# ------------------------------------------------------------------ manifest
def _entry(family: Family, params: dict, origin: dict, split_cfg: dict) -> dict:
    reasons = family.violations(params)
    sid = family.scenario_id(params) if not any(r.startswith("schema:") for r in reasons) else None
    e = {"scenario_id": sid, "family": family.name, "family_version": family.version, "params": params,
         "status": "rejected" if reasons else "accepted", "rejection_reasons": reasons, "origin": origin}
    if sid is not None:
        bins = family.bins_of(params)
        e["scenario_sha256"] = family.scenario_hash(params)
        e["bins"] = bins
        if not reasons:
            e["region"] = region_key(split_cfg, bins)
            e["layout"] = family.layout(params)
            e["track_spec_sha256"] = sha256_obj(family.track_spec(params, sid))
            e["sim_seed"] = derive_seed("sim", sid) % (2 ** 31)
    return e


def build_manifest(design: dict, profile: str) -> dict:
    family = get_family(design["family"])
    prof = design["profiles"][profile]
    strength, reps = int(prof["strength"]), int(prof["replicates"])
    seed, max_att = int(design["seed"]), int(design.get("max_attempts", 8))
    split_cfg = design["split"]
    contrast_factors = list(design.get("contrast_factors", []))
    for f in contrast_factors + list(split_cfg["region_factors"]):
        family.factor(f)  # raises on unknown names

    levels = [len(f.labels) for f in family.factors]
    rows = covering_rows(levels, strength)
    scenarios, pairs, gaps = [], [], []
    by_id: dict[str, dict] = {}

    def add(e):
        if e["status"] == "accepted" and e["scenario_id"] in by_id:
            e["status"], e["rejection_reasons"] = "rejected", [f"duplicate_of:{e['scenario_id']}"]
            e.pop("region", None)
        scenarios.append(e)
        if e["status"] == "accepted":
            by_id[e["scenario_id"]] = e
        return e

    for r, row in enumerate(rows):
        for k in range(reps):
            base = None
            for att in range(max_att):
                s = derive_seed(seed, family.name, family.version, profile, r, k, att)
                params = sample_params(family, row, np.random.default_rng(s))
                e = add(_entry(family, params, {"kind": "design", "design_row": r, "replicate": k, "attempt": att,
                                                "seed": s, "row_levels": [family.factors[j].labels[lv]
                                                                          for j, lv in enumerate(row)]}, split_cfg))
                if e["status"] == "accepted":
                    base = e
                    break
            if base is None:
                gaps.append({"kind": "design_cell_unrealised", "design_row": r, "replicate": k,
                             "reason": f"all {max_att} attempts rejected"})
                continue
            gid = "grp-" + base["scenario_id"]
            base_split = region_split(split_cfg, base["bins"])
            base["group_id"], base["split"] = gid, base_split
            for f in contrast_factors:
                tried = []
                for tgt in contrast_targets(family, f, base["bins"][f]):
                    vparams = move_factor(family, base["params"], f, tgt)
                    vbins = family.bins_of(vparams)
                    vsplit = region_split(split_cfg, vbins) if all(vbins.values()) else None
                    if vsplit != base_split:
                        tried.append({"target": tgt, "reason": f"crosses_split:{vsplit}"})
                        continue
                    e = add(_entry(family, vparams, {"kind": "contrast", "base_id": base["scenario_id"], "factor": f,
                                                     "target_bin": tgt}, split_cfg))
                    if e["status"] != "accepted":
                        tried.append({"target": tgt, "reason": ";".join(e["rejection_reasons"])})
                        continue
                    e["group_id"], e["split"] = gid, base_split
                    fixed = [n for n in family.param_names() if n != f]
                    pairs.append({"pair_id": f"{base['scenario_id']}~{f}", "base_id": base["scenario_id"],
                                  "variant_id": e["scenario_id"], "factor": f, "fixed": fixed,
                                  "expected_effect": family.expected_effects.get(f, "")})
                    break
                else:
                    gaps.append({"kind": "contrast_unavailable", "base_id": base["scenario_id"], "factor": f,
                                 "tried": tried})

    accepted = [e for e in scenarios if e["status"] == "accepted"]
    cells = required_cells(family, strength)
    coverage = coverage_report(cells, accepted)
    content = {
        "schema": MANIFEST_SCHEMA, "schema_version": SCHEMA_VERSION,
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "design": {"profile": profile, "strength": strength, "replicates": reps, "seed": seed, "max_attempts": max_att,
                   "contrast_factors": contrast_factors, "config_sha256": sha256_obj(design)},
        "family_spec": family.spec_dict(),
        "split_policy": copy.deepcopy(split_cfg),
        "design_rows": [[family.factors[j].labels[lv] for j, lv in enumerate(row)] for row in rows],
        "scenarios": scenarios, "contrast_pairs": pairs, "gaps": gaps,
        "required_coverage": {"strength": strength, "cells": [cell_id(c) for c in cells]},
        "coverage": coverage,
    }
    return {**content, "content_sha256": manifest_content_hash(content),
            "provenance": {"git": git_state(), "design_config": copy.deepcopy(design)}}


def coverage_report(cells: list[dict], scenarios: list[dict]) -> dict:
    covered, missing = {}, []
    for c in cells:
        ids = [e["scenario_id"] for e in scenarios if cell_covered(c, e["bins"])]
        if ids:
            covered[cell_id(c)] = len(ids)
        else:
            missing.append(cell_id(c))
    return {"n_required": len(cells), "n_covered": len(covered), "covered": covered, "missing": missing}


def manifest_content_hash(m: dict) -> str:
    return sha256_obj({k: v for k, v in m.items() if k not in ("content_sha256", "provenance")})


# ------------------------------------------------------------------ check
def check_manifest(m: dict) -> list[str]:
    """Recompute every derived field of a manifest; return a list of problems (empty = OK)."""
    errs = []
    if m.get("schema") != MANIFEST_SCHEMA or m.get("schema_version") != SCHEMA_VERSION:
        return [f"unsupported manifest schema {m.get('schema')}/{m.get('schema_version')}"]
    try:
        family = get_family(m["family_spec"]["name"])
    except KeyError as e:
        return [str(e)]
    if canonical_json(m["family_spec"]) != canonical_json(family.spec_dict()):
        errs.append(f"family spec in manifest differs from code (family {family.name} v{family.version}); "
                    "regenerate the manifest or bump the family version")
    if m.get("content_sha256") != manifest_content_hash(m):
        errs.append("content_sha256 does not match the manifest content (edited after generation?)")
    split_cfg = m["split_policy"]
    ids, groups = {}, {}
    for i, e in enumerate(m["scenarios"]):
        where = f"scenario[{i}] {e.get('scenario_id')}"
        reasons = family.violations(e["params"])
        if e["status"] == "accepted":
            if reasons:
                errs.append(f"{where}: accepted but violates {reasons}")
                continue
            if e["scenario_id"] != family.scenario_id(e["params"]):
                errs.append(f"{where}: scenario_id does not match its parameters")
            if e.get("scenario_sha256") != family.scenario_hash(e["params"]):
                errs.append(f"{where}: scenario_sha256 does not match its parameters")
            if e.get("bins") != family.bins_of(e["params"]):
                errs.append(f"{where}: bins do not match its parameters")
            if e.get("region") != region_key(split_cfg, e["bins"]):
                errs.append(f"{where}: region does not match its bins")
            if e.get("split") != region_split(split_cfg, e["bins"]):
                errs.append(f"{where}: split {e.get('split')} != split of its region")
            if e.get("layout") != family.layout(e["params"]):
                errs.append(f"{where}: layout does not match its parameters")
            if e.get("track_spec_sha256") != sha256_obj(family.track_spec(e["params"], e["scenario_id"])):
                errs.append(f"{where}: track_spec_sha256 does not match its parameters")
            if e["scenario_id"] in ids:
                errs.append(f"{where}: duplicate accepted scenario")
            ids[e["scenario_id"]] = e
            groups.setdefault(e.get("group_id"), set()).add(e.get("split"))
        else:
            if not e.get("rejection_reasons"):
                errs.append(f"{where}: rejected without a reason")
            elif not reasons and not all(r.startswith("duplicate_of:") for r in e["rejection_reasons"]):
                errs.append(f"{where}: rejected for {e['rejection_reasons']} but passes every check now")
    for g, sp in groups.items():
        if len(sp) != 1:
            errs.append(f"group {g} spans splits {sorted(sp)}")
    for p in m["contrast_pairs"]:
        a, b = ids.get(p["base_id"]), ids.get(p["variant_id"])
        if a is None or b is None:
            errs.append(f"pair {p['pair_id']}: member not an accepted scenario")
            continue
        diff = [n for n in family.param_names()
                if family.canonical_params(a["params"])[n] != family.canonical_params(b["params"])[n]]
        if diff != [p["factor"]]:
            errs.append(f"pair {p['pair_id']}: differs in {diff}, declared {p['factor']}")
        if a["bins"][p["factor"]] == b["bins"][p["factor"]]:
            errs.append(f"pair {p['pair_id']}: factor {p['factor']} stays in bin {a['bins'][p['factor']]}")
        if sorted(p["fixed"]) != sorted(n for n in family.param_names() if n != p["factor"]):
            errs.append(f"pair {p['pair_id']}: fixed list incomplete")
        if a.get("group_id") != b.get("group_id") or a.get("split") != b.get("split"):
            errs.append(f"pair {p['pair_id']}: members in different groups/splits")
    cells = [dict(kv.split("=") for kv in c.split("|")) for c in m["required_coverage"]["cells"]]
    expect_cells = required_cells(family, m["required_coverage"]["strength"])
    if [cell_id(c) for c in expect_cells] != m["required_coverage"]["cells"]:
        errs.append("required coverage cells do not match the family spec and strength")
    if coverage_report(cells, list(ids.values())) != m["coverage"]:
        errs.append("coverage report does not match the accepted scenarios")
    return errs


def summarize(m: dict) -> str:
    acc = [e for e in m["scenarios"] if e["status"] == "accepted"]
    rej = [e for e in m["scenarios"] if e["status"] != "accepted"]
    by_split = {s: sum(1 for e in acc if e["split"] == s) for s in SPLITS}
    cov = m["coverage"]
    lines = [f"manifest {m['content_sha256'][:16]}  family {m['family_spec']['name']} v{m['family_spec']['version']}  "
             f"profile {m['design']['profile']} (strength {m['design']['strength']})",
             f"  scenarios: {len(acc)} accepted ({', '.join(f'{k} {v}' for k, v in by_split.items())}), "
             f"{len(rej)} rejected draws, {len(m['contrast_pairs'])} contrast pairs, {len(m['gaps'])} gaps",
             f"  coverage: {cov['n_covered']}/{cov['n_required']} required cells"
             + (f"; MISSING {', '.join(cov['missing'])}" if cov["missing"] else "")]
    reasons: dict[str, int] = {}
    for e in rej:
        for r in e["rejection_reasons"]:
            key = r.split(":", 2)[0] + ":" + r.split(":", 2)[1] if ":" in r else r
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        lines.append("  rejections: " + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())))
    for g in m["gaps"]:
        if g["kind"] == "contrast_unavailable":
            lines.append(f"  no {g['factor']} contrast for {g['base_id']}: "
                         + "; ".join(f"{t['target']}: {t['reason']}" for t in g["tried"]))
        else:
            lines.append(f"  {g['kind']}: row {g['design_row']} replicate {g['replicate']} ({g['reason']})")
    return "\n".join(lines)


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
