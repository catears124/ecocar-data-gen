"""Imitation-data pipeline: manifest design, label certificate, corpus validation, Lean checker."""
import copy
import itertools
import json
import shutil
from pathlib import Path

import pytest
import yaml

from autocross.datagen import cert
from autocross.datagen.design import build_manifest, check_manifest, coverage_report, required_cells
from autocross.datagen.generate import generate_corpus
from autocross.datagen.records import mirror_action, mirror_observation, obs_from_dict, obs_to_dict
from autocross.datagen.spec import get_family
from autocross.datagen.validate import validate_corpus

ROOT = Path(__file__).resolve().parent.parent
FAM = get_family("brake_turn_obstacle")


@pytest.fixture(scope="module")
def design():
    return yaml.safe_load(open(ROOT / "config" / "datagen.yaml"))


@pytest.fixture(scope="module")
def smoke(design):
    return build_manifest(design, "smoke")


def _base_params():
    return {"turn_radius_m": 15.0, "turn_direction": "left", "preview_m": 30.0, "approach_speed_mps": 12.0,
            "obstacle_side": "inside", "obstacle_clearance_m": 3.5, "road_width_m": 6.0, "turn_angle_deg": 90.0,
            "obstacle_arc_frac": 0.5}


# ------------------------------------------------------------------ scenario bounds and bins
def test_bins_respect_half_open_edges():
    r = FAM.factor("turn_radius_m")
    assert r.bin_of(13.999) == "tight" and r.bin_of(14.0) == "medium" and r.bin_of(35.0) == "open"
    assert r.bin_of(8.999) is None and r.bin_of(35.001) is None


@pytest.mark.parametrize("change,reason", [
    ({"turn_radius_m": 8.5}, "formal:out_of_bounds:turn_radius_m"),
    ({"preview_m": 61.0}, "formal:out_of_bounds:preview_m"),
    ({"turn_direction": "up"}, "formal:unknown_level:turn_direction"),
    ({"obstacle_arc_frac": 0.9}, "formal:out_of_bounds:obstacle_arc_frac"),
    ({"road_width_m": 5.0, "obstacle_clearance_m": 4.0}, "formal:clearance_exceeds_width"),
    ({"preview_m": 30.0001}, "schema:bad_value:preview_m"),
])
def test_invalid_scenarios_are_rejected_with_reason(change, reason):
    p = {**_base_params(), **change}
    assert reason in FAM.violations(p)


def test_valid_scenario_builds_the_declared_course():
    p = _base_params()
    assert FAM.violations(p) == []
    lay = FAM.layout(p)
    # clearance is realised exactly: obstacle inner edge to the opposite boundary
    assert lay["obstacle_lateral_min_m"] + p["road_width_m"] / 2 == pytest.approx(p["obstacle_clearance_m"])
    assert lay["corner_entry_progress"] - lay["handover_progress"] > p["preview_m"]  # turn hidden at hand-over


# ------------------------------------------------------------------ manifest determinism, coverage, contrasts, splits
def test_manifest_is_deterministic_and_seed_sensitive(design, smoke):
    again = build_manifest(design, "smoke")
    assert again["content_sha256"] == smoke["content_sha256"]
    assert [e["scenario_id"] for e in again["scenarios"]] == [e["scenario_id"] for e in smoke["scenarios"]]
    other = build_manifest({**design, "seed": design["seed"] + 1}, "smoke")
    assert {e["scenario_id"] for e in other["scenarios"]} != {e["scenario_id"] for e in smoke["scenarios"]}
    assert check_manifest(smoke) == []


def test_rejected_draws_are_recorded_not_dropped(smoke):
    rej = [e for e in smoke["scenarios"] if e["status"] == "rejected"]
    assert rej and all(e["rejection_reasons"] for e in rej)


def test_pairwise_cells_and_coverage_accounting(design):
    cells = required_cells(FAM, 2)
    levels = [len(f.labels) for f in FAM.factors]
    assert len(cells) == sum(a * b for a, b in itertools.combinations(levels, 2))
    full = build_manifest(design, "full")
    assert full["coverage"]["missing"] == []
    # drop every scenario covering one cell: the report must name exactly that cell as missing
    acc = [e for e in full["scenarios"] if e["status"] == "accepted"]
    cell = {"turn_radius_m": "tight", "preview_m": "short"}
    kept = [e for e in acc if not (e["bins"]["turn_radius_m"] == "tight" and e["bins"]["preview_m"] == "short")]
    rep = coverage_report(cells, kept)
    assert rep["missing"] == ["turn_radius_m=tight|preview_m=short"]


def test_contrast_pairs_change_exactly_one_factor(smoke):
    ids = {e["scenario_id"]: e for e in smoke["scenarios"] if e["status"] == "accepted"}
    assert smoke["contrast_pairs"]
    for p in smoke["contrast_pairs"]:
        a, b = ids[p["base_id"]]["params"], ids[p["variant_id"]]["params"]
        assert [k for k in a if a[k] != b[k]] == [p["factor"]]


def test_check_manifest_catches_a_second_changed_factor(smoke):
    m = copy.deepcopy(smoke)
    p = m["contrast_pairs"][0]
    var = next(e for e in m["scenarios"] if e["scenario_id"] == p["variant_id"])
    other = "turn_angle_deg"
    var["params"][other] = round(var["params"][other] + 1.0, 3)
    errs = check_manifest(m)
    assert any("scenario_id does not match" in e for e in errs)
    assert any(f"differs in" in e and other in e for e in errs)


def test_check_manifest_catches_split_leakage(smoke):
    m = copy.deepcopy(smoke)
    p = m["contrast_pairs"][0]
    var = next(e for e in m["scenarios"] if e["scenario_id"] == p["variant_id"])
    var["split"] = "test" if var["split"] != "test" else "train"
    errs = check_manifest(m)
    assert any("spans splits" in e for e in errs) and any("split of its region" in e for e in errs)


def test_held_out_regions_never_reach_train(design):
    full = build_manifest(design, "full")
    held = {("open", "nominal"): "val", ("medium", "short"): "test"}
    for e in full["scenarios"]:
        if e["status"] == "accepted":
            key = (e["bins"]["turn_radius_m"], e["bins"]["preview_m"])
            assert e["split"] == held.get(key, "train")


# ------------------------------------------------------------------ certified decision vs brute force
def _brute_max_b1(caps, s, b0, aA, aB):
    best = None
    for tail in itertools.product(range(0, max(caps) + 1), repeat=len(caps) - 1):
        b = (b0,) + tail
        ok = all(b[i] <= caps[i] for i in range(len(b)))
        ok = ok and all(b[i + 1] <= b[i] + 2 * aA * (s[i + 1] - s[i]) and b[i] <= b[i + 1] + 2 * aB * (s[i + 1] - s[i])
                        for i in range(len(b) - 1))
        if ok:
            best = b[1] if best is None else max(best, b[1])
    return best


@pytest.mark.parametrize("caps,b0", [([30, 25, 4, 30], 20), ([30, 30, 30, 2], 9), ([12, 30, 1, 1], 12),
                                     ([30, 8, 30, 30], 0), ([5, 30, 30, 3], 6)])
def test_certified_next_speed_is_the_greatest_feasible(caps, b0):
    c = {"a_acc": 2, "a_brk": 3}
    s = [0, 1, 2, 3]
    bw = cert.backward(c, caps, s)
    brute = _brute_max_b1(caps, s, b0, c["a_acc"], c["a_brk"])
    if b0 > bw[0]:
        assert brute is None  # over budget: no feasible profile at all -> min_violation branch
    else:
        assert cert.forward(c, bw, s, b0)[1] == brute


def test_decide_branches_on_quantised_input():
    from types import SimpleNamespace
    vp = SimpleNamespace(mu=0.95, max_accel=5.2, max_decel=9.0, width=2.16)
    c = cert.model_constants(vp, 0.7, 6.5, 25.0, 9.0, 3.0, 18.0, 15.0, 15, 5.0, 6.0, 0.25)
    n = 31
    q = {"v0": 10000, "s": [1000 * i for i in range(n)], "k": [0] * n, "wl": [3000] * n, "wr": [3000] * n, "obstacles": []}
    assert cert.decide(c, q)["kind"] == cert.KIND_GREATEST_SAFE
    fast = {**q, "v0": 25000}  # 25 m/s with 30 m of sight: cannot stop for a hairpin at the horizon
    d = cert.decide(c, fast)
    assert d["kind"] == cert.KIND_MIN_VIOLATION and d["a"] == -c["a_max"]
    blocked = {**q, "obstacles": [[15000, -2900, 2900]]}  # no gap wide enough
    assert cert.decide(c, blocked)["kind"] == cert.KIND_NO_GAP


# ------------------------------------------------------------------ generated corpus
@pytest.fixture(scope="module")
def corpus(tmp_path_factory, smoke):
    root = tmp_path_factory.mktemp("corpus")
    mp = root / "manifest.json"
    mp.write_text(json.dumps(smoke))
    out = root / "c"
    generate_corpus(str(mp), str(out), "bicycle", limit=2, verbose=False)
    return out


def _copy(corpus, tmp_path):
    dst = tmp_path / "c"
    shutil.copytree(corpus, dst)
    return dst


def _rewrite_first_sample(root, fn, pick=lambda s: True):
    idx = json.loads((root / "index.json").read_text())
    path = root / idx["scenarios"][0]["files"]["samples"]["path"]
    lines = path.read_text().splitlines()
    for i, ln in enumerate(lines):
        s = json.loads(ln)
        if pick(s):
            fn(s)
            lines[i] = json.dumps(s, separators=(",", ":"))
            break
    path.write_text("\n".join(lines) + "\n")
    # keep the file hash consistent so the deeper checks (not just the hash) must catch it
    from autocross.datagen.generate import file_sha256
    idx["scenarios"][0]["files"]["samples"]["sha256"] = file_sha256(path)
    (root / "index.json").write_text(json.dumps(idx, indent=1))


def test_corpus_validates_and_records_backend(corpus):
    strict = validate_corpus(corpus)  # 2 of 20 scenarios: required cells are missing -> error by default
    assert [e for e in strict.errors if "required coverage cells" in e] and len(strict.errors) == 1
    rep = validate_corpus(corpus, allow_missing_coverage=True)
    assert rep.errors == [], rep.errors[:5]
    assert rep.stats["backend"] == "bicycle" and rep.stats["clean_samples"] > 0
    assert any("NOT CARLA" in w for w in rep.warnings)
    idx = json.loads((corpus / "index.json").read_text())
    assert idx["backend"]["is_carla"] is False
    s = json.loads((corpus / idx["scenarios"][0]["files"]["samples"]["path"]).read_text().splitlines()[0])
    assert s["backend"] == "bicycle" and s["scenario_id"] == idx["scenarios"][0]["scenario_id"]


def test_clean_samples_are_decision_window_and_certified(corpus):
    idx = json.loads((corpus / "index.json").read_text())
    for run in idx["scenarios"]:
        for ln in (corpus / run["files"]["samples"]["path"]).read_text().splitlines():
            s = json.loads(ln)
            if s["quality"]["clean"]:
                assert s["oracle"]["phase"] == "decision" and run["clean_run"]
                assert s["expert"]["certificate"]["certified"]
            if s["oracle"]["phase"] == "approach":
                assert "phase:approach" in s["quality"]["exclusion_reasons"]


@pytest.mark.parametrize("tamper,expect", [
    (lambda s: s["expert"].__setitem__("brake", min(1.0, s["expert"]["brake"] + 0.3)), "content hash mismatch"),
    (lambda s: s.__setitem__("split", "test" if s["split"] != "test" else "train"), "split="),
    (lambda s: s["quality"].update(clean=True, exclusion_reasons=[]), "exclusion reasons"),
])
def test_validator_detects_corrupted_records(corpus, tmp_path, tamper, expect):
    root = _copy(corpus, tmp_path)
    _rewrite_first_sample(root, tamper, pick=lambda s: s["expert"] is not None and not s["quality"]["clean"])
    errs = validate_corpus(root).errors
    assert any(expect in e for e in errs), errs[:5]


def test_validator_detects_relabelled_action_with_rehash(corpus, tmp_path):
    """A wrong label with a recomputed content hash is still caught by label replay."""
    from autocross.datagen.records import content_hash

    def relabel(s):
        s["expert"]["steer"] = -s["expert"]["steer"] if s["expert"]["steer"] else 0.25
        s["content_sha256"] = content_hash(s["observation"], s["expert"])
    root = _copy(corpus, tmp_path)
    _rewrite_first_sample(root, relabel, pick=lambda s: s["quality"]["clean"])
    errs = validate_corpus(root).errors
    assert any("label replay" in e for e in errs), errs[:5]


def test_validator_detects_edited_track(corpus, tmp_path):
    root = _copy(corpus, tmp_path)
    idx = json.loads((root / "index.json").read_text())
    tp = root / idx["scenarios"][0]["files"]["track"]["path"]
    tp.write_text(tp.read_text().replace("width:", "width: 7.5 #", 1))
    assert any("hash mismatch" in e for e in validate_corpus(root).errors)


def test_generation_is_repeatable_on_bicycle(corpus, tmp_path, smoke):
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(smoke))
    out = tmp_path / "again"
    generate_corpus(str(mp), str(out), "bicycle", limit=1, verbose=False)
    idx_a = json.loads((corpus / "index.json").read_text())
    idx_b = json.loads((out / "index.json").read_text())
    a = [json.loads(x)["content_sha256"] for x in (corpus / idx_a["scenarios"][0]["files"]["samples"]["path"]).read_text().splitlines()]
    b = [json.loads(x)["content_sha256"] for x in (out / idx_b["scenarios"][0]["files"]["samples"]["path"]).read_text().splitlines()]
    assert a == b


def test_observation_roundtrip_and_mirror_involution(corpus):
    idx = json.loads((corpus / "index.json").read_text())
    s = next(json.loads(x) for x in (corpus / idx["scenarios"][0]["files"]["samples"]["path"]).read_text().splitlines()
             if json.loads(x)["observation"]["road"]["obstacles"])
    o = s["observation"]
    assert obs_to_dict(obs_from_dict(o)) == o
    assert mirror_observation(mirror_observation(o)) == o
    m = mirror_observation(o)
    assert m["road"]["width_left"] == o["road"]["width_right"]
    assert m["road"]["obstacles"][0]["lateral_min"] == -o["road"]["obstacles"][0]["lateral_max"]
    a = mirror_action(s["expert"])
    assert a["steer"] == -s["expert"]["steer"] and a["brake"] == s["expert"]["brake"]
    # the certified longitudinal decision is mirror invariant
    from autocross.datagen.expert import model_constants
    from autocross.api import VehicleParams
    c = model_constants(VehicleParams(**idx["scenarios"][0]["vehicle"]))
    assert cert.decide(c, cert.quantise(obs_from_dict(m))) == cert.decide(c, cert.quantise(obs_from_dict(o)))


def test_carla_generation_fails_clearly_without_carla(smoke, tmp_path):
    pytest.importorskip("yaml")
    try:
        import carla  # noqa: F401
        pytest.skip("carla client installed")
    except ImportError:
        pass
    from autocross.datagen import DatagenError
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(smoke))
    with pytest.raises(DatagenError, match="No data was generated"):
        generate_corpus(str(mp), str(tmp_path / "c"), "carla", verbose=False)
    assert not (tmp_path / "c" / "index.json").exists()


# ------------------------------------------------------------------ Lean checker on the actual artifacts
def _lake_or_skip():
    from autocross.datagen import DatagenError
    from autocross.datagen.lean_bridge import lake_path
    try:
        return lake_path()
    except DatagenError:
        pytest.skip("Lean (lake) not installed")


def test_lean_checker_accepts_corpus_and_rejects_tampered_certificate(corpus, tmp_path):
    _lake_or_skip()
    from autocross.datagen.lean_bridge import export, run_checker
    idx = json.loads((corpus / "index.json").read_text())
    man = json.loads((corpus / "manifest.json").read_text())
    inter, certs = export(man, tmp_path / "lean", idx, corpus)
    rep = run_checker(inter, certs, tmp_path / "r.json")
    assert rep["ok"], [r for r in rep["results"] if not r["ok"]]
    assert rep["axiom_audit"]["sorry_free"]
    # an off-by-one acceleration label must be rejected by the Lean recomputation
    cf = sorted(certs.glob("*.json"))[0]
    data = json.loads(cf.read_text())
    sample = next(x for x in data["samples"] if x["clean"])
    sample["decision"]["a"] += 1
    cf.write_text(json.dumps(data))
    rep = run_checker(inter, certs, tmp_path / "r2.json")
    bad = {r["name"] for r in rep["results"] if not r["ok"]}
    assert not rep["ok"] and "cert_decisions_recomputed" in bad


def test_lean_checker_rejects_split_and_pair_corruption(smoke, tmp_path):
    _lake_or_skip()
    from autocross.datagen.lean_bridge import export, run_checker
    inter, _ = export(smoke, tmp_path / "ok")
    assert run_checker(inter, None, tmp_path / "ok.json")["ok"]
    m = copy.deepcopy(smoke)
    p = m["contrast_pairs"][0]
    var = next(e for e in m["scenarios"] if e["scenario_id"] == p["variant_id"])
    var["split"] = "test" if var["split"] != "test" else "train"
    var["params"]["turn_angle_deg"] = round(var["params"]["turn_angle_deg"] + 1.0, 3)
    inter, _ = export(m, tmp_path / "bad")
    rep = run_checker(inter, None, tmp_path / "bad.json")
    bad = {r["name"] for r in rep["results"] if not r["ok"]}
    assert {"contrast_pairs_integrity", "split_rule", "group_split_consistent"} <= bad
