"""Certificate semantics, adversarial corpus corruption, observation-only contract, Lean checker.

Every corruption below keeps the ordinary file hashes (and, unless the test is about them, the
per-sample hashes) consistent, so it must be caught by the deeper checks: replay, independent
certificate recomputation (Python and Lean), identity and set-equality checks."""
import itertools
import json
import math
import random
import shutil
import subprocess

import numpy as np
import pytest

from autocross.api import VehicleParams
from autocross.datagen import cert
from autocross.datagen import expert as expert_mod
from autocross.datagen.generate import (expert_label, file_sha256, generate_corpus, label_blocks, replay_diff,
                                        tree_diff)
from autocross.datagen.records import (POLICY_EGO, POLICY_ROAD_KEYS, command_to_dict, merge_observation,
                                       mirror_action, mirror_observation, obs_from_dict, obs_to_dict,
                                       policy_to_observation, sample_hashes, split_observation)
from autocross.datagen.validate import validate_corpus
from conftest import lake_or_skip

# ======================================================================= certificate arithmetic


def test_qfix_is_exact_decimal_round_half_up():
    assert cert.qfix(0.0005, 1000) == 1          # exactly half: up
    assert cert.qfix(-0.0005, 1000) == 0         # -0.5 -> floor(0) = 0
    assert cert.qfix(1e-05, 1_000_000) == 10     # exponent notation
    assert cert.qfix(2.675, 100) == 268          # binary float 2.675 * 100 = 267.49999...; decimal repr is exact
    assert cert.qfix(-0.0, 1000) == 0
    for bad in (float("nan"), float("inf"), None, True):
        with pytest.raises(ValueError):
            cert.qfix(bad, 1000)


def _feasible(b, caps, s, aA, aB):
    """Lean `Feasible` on explicit lists: b >= 0, caps (last = terminal cap), budgets."""
    n = len(b)
    return (all(x >= 0 for x in b) and all(b[i] <= caps[i] for i in range(n))
            and all(b[i + 1] <= b[i] + 2 * aA * (s[i + 1] - s[i]) and b[i] <= b[i + 1] + 2 * aB * (s[i + 1] - s[i])
                    for i in range(n - 1)))


def _solve(caps, s, b0, aA, aB):
    c = {"a_acc": aA, "a_brk": aB}
    bw = cert.backward(c, caps, s)
    if b0 > bw[0]:
        return None
    f = cert.forward(c, bw, s, b0)
    return f, (f[1] - b0) // (2 * (s[1] - s[0]))


def test_longitudinal_solver_matches_exhaustive_enumeration():
    """Small instances, every integer profile enumerated: F_1 is the greatest feasible next value,
    F is feasible, the integer action is the greatest integer whose clamped end point starts a
    feasible profile and stays inside [-a_brk, a_acc]; over budget iff no feasible profile."""
    rng = random.Random(7)
    checked = 0
    for _ in range(400):
        n = rng.choice((2, 3, 4))
        s = [0]
        for _ in range(n - 1):
            s.append(s[-1] + rng.choice((1, 2)))
        caps = [rng.randint(0, 10) for _ in range(n)]
        aA, aB = rng.randint(0, 3), rng.randint(0, 3)
        b0 = rng.randint(0, 12)
        top = max(caps)
        profiles = [(b0,) + t for t in itertools.product(range(top + 1), repeat=n - 1)]
        feas = [b for b in profiles if _feasible(b, caps, s, aA, aB)]
        sol = _solve(caps, s, b0, aA, aB)
        if sol is None:
            assert feas == []
            continue
        f, a = sol
        assert feas and _feasible(f, caps, s, aA, aB)
        assert f[1] == max(b[1] for b in feas)
        assert all(all(x <= y for x, y in zip(b, f)) for b in feas)
        d = s[1] - s[0]
        starts = {b[1] for b in feas}
        good = [x for x in range(-20, 21) if max(0, b0 + 2 * x * d) in starts]
        assert max(good) == a and -aB <= a <= aA
        checked += 1
    assert checked > 100


def test_full_braking_is_pointwise_least_and_minimises_cap_excess():
    rng = random.Random(3)
    for _ in range(200):
        n = rng.choice((2, 3))
        ds = [rng.choice((1, 2)) for _ in range(n - 1)]
        aM, x = rng.randint(1, 3), rng.randint(0, 8)
        brake = [x]
        for d in ds:
            brake.append(max(0, brake[-1] - 2 * aM * d))
        caps = [rng.randint(0, 8) for _ in range(n)]
        excess = lambda b: sum(max(0, v - c) for v, c in zip(b, caps))  # noqa: E731
        for tail in itertools.product(range(13), repeat=n - 1):
            b = (x,) + tail
            if all(b[i] - 2 * aM * ds[i] <= b[i + 1] for i in range(n - 1)):
                assert all(p <= q for p, q in zip(brake, b))
                assert excess(brake) <= excess(b)


VP = VehicleParams(mass=1845.0, wheelbase=3.0, track_width=1.67, length=4.79, width=2.16, cg_to_front=1.45,
                   cg_to_rear=1.55, max_steer=0.61, max_steer_rate=2.0, mu=0.95, max_accel=5.2, max_decel=9.0,
                   max_speed=40.0, control_dt=0.05, actuation_delay_steps=0)
CONSTS = expert_mod.model_constants(VP)


def _q(v0=10000, n=31, step=1000, k=0, w=3000, obstacles=()):
    return {"v0": v0, "s": [step * i for i in range(n)], "k": [k] * n, "wl": [w] * n, "wr": [w] * n,
            "obstacles": [list(o) for o in obstacles]}


EDGE_CASES = {
    "zero_speed": (CONSTS, _q(v0=0)),
    "negative_speed": (CONSTS, _q(v0=-1)),
    "two_stations": (CONSTS, _q(v0=1000, n=2)),
    "smallest_spacing_1mm": (CONSTS, _q(v0=1000, n=5, step=1)),
    "extreme_curvature": (CONSTS, _q(k=1_000_000)),
    "over_budget": (CONSTS, _q(v0=25000)),
    "impossible_gap": (CONSTS, _q(obstacles=[(15000, -2900, 2900)])),
    "obstacle_beyond_preview": (CONSTS, _q(obstacles=[(90000, 1000, 2000)])),
    "non_increasing_stations": (CONSTS, {**_q(n=3), "s": [0, 1000, 1000]}),
    "nonzero_first_station": (CONSTS, {**_q(n=3), "s": [5, 1000, 2000]}),
    "misaligned_vectors": (CONSTS, {**_q(n=3), "k": [0, 0]}),
    "bad_obstacle_tuple": (CONSTS, _q(n=3, obstacles=[(1000, 2)])),
    "negative_brake_budget": ({**CONSTS, "a_brk": -1}, _q()),
    "negative_cap": ({**CONSTS, "a_lat": -1}, _q()),
    "braking_with_rounding": (CONSTS, _q(v0=20291, n=31)),  # floor(-6380.x) near the braking budget
}


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_decide_edge_cases(name):
    c, q = EDGE_CASES[name]
    d = cert.decide(c, q)
    expect_kind = {"negative_speed": 3, "non_increasing_stations": 3, "nonzero_first_station": 3,
                   "misaligned_vectors": 3, "bad_obstacle_tuple": 3, "negative_brake_budget": 3, "negative_cap": 3,
                   "over_budget": 1, "extreme_curvature": 1, "impossible_gap": 2}.get(name, 0)
    assert d["kind"] == expect_kind, d
    if d["kind"] == 0:
        ds = q["s"][1] - q["s"][0]
        assert -c["a_brk"] <= d["a"] <= c["a_acc"]
        assert d["b0"] + 2 * d["a"] * ds <= d["b1"] < d["b0"] + 2 * (d["a"] + 1) * ds  # floor, never above F_1
    else:
        assert d["a"] == -c["a_max"] and d["b1"] is None
    if name == "braking_with_rounding":
        assert d["a"] < 0 and (d["b1"] - d["b0"]) % (2 * (q["s"][1] - q["s"][0])) != 0  # a genuinely rounded brake


def _lean_lit(c, q, d):
    L = lambda v: "[" + ", ".join(L(x) if isinstance(x, list) else f"({x})" for x in v) + "]"  # noqa: E731
    O = lambda v: "none" if v is None else f"(some ({v}))"  # noqa: E731
    cs = "{ " + ", ".join(f"{k} := ({c[k]})" for k in cert.CONST_KEYS) + " }"
    qs = (f"{{ v0 := ({q['v0']}), s := {L(q['s'])}, k := {L(q['k'])}, wl := {L(q['wl'])}, wr := {L(q['wr'])}, "
          f"obstacles := {L(q['obstacles'])} }}")
    return cs, qs, f"⟨{d['kind']}, ({d['a']}), {O(d['b1'])}, {O(d['b0'])}, {O(d['B0'])}⟩"


def test_python_and_lean_decide_agree_on_edge_cases(tmp_path):
    """The Lean kernel evaluates Cert.decide on every edge case and must reproduce Python's decision."""
    lake = lake_or_skip()
    from autocross.datagen.lean_bridge import LEAN_DIR, build
    build()
    src = ["import Corpus", "set_option maxRecDepth 100000", "open Corpus.Cert"]
    for i, name in enumerate(sorted(EDGE_CASES)):
        c, q = EDGE_CASES[name]
        cs, qs, ds = _lean_lit(c, q, cert.decide(c, q))
        src.append(f"theorem e{i} : decide {cs} {qs} = {ds} := by decide")
    f = tmp_path / "Edge.lean"
    f.write_text("\n".join(src) + "\n", encoding="utf-8")
    r = subprocess.run([lake, "env", "lean", str(f)], cwd=LEAN_DIR, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    # and a wrong decision is refuted by the kernel
    c, q = EDGE_CASES["braking_with_rounding"]
    d = cert.decide(c, q)
    cs, qs, ds = _lean_lit(c, q, {**d, "a": d["a"] - 1})
    f.write_text("\n".join(src[:3] + [f"theorem bad : decide {cs} {qs} = {ds} := by decide"]) + "\n", encoding="utf-8")
    r = subprocess.run([lake, "env", "lean", str(f)], cwd=LEAN_DIR, capture_output=True, text=True)
    assert r.returncode != 0 and "is false" in r.stdout


# ======================================================================= corpus: honest corpus passes


def _runs(root):
    idx = json.loads((root / "index.json").read_text())
    return idx, [json.loads(x) for x in (root / idx["scenarios"][0]["files"]["samples"]["path"]).read_text().splitlines()]


def test_corpus_validates_and_records_backend(corpus):
    strict = validate_corpus(corpus)  # 2 of 20 scenarios: required cells are missing -> error by default
    assert [e for e in strict.errors if "required coverage cells" in e] and len(strict.errors) == 1
    rep = validate_corpus(corpus, allow_missing_coverage=True)
    assert rep.errors == [], rep.errors[:5]
    assert rep.stats["backend"] == "bicycle" and rep.stats["clean_samples"] > 0
    assert rep.stats["clean_samples"] == rep.stats["clean_certified_samples"]
    assert rep.stats["label_replay_max_abs_diff"] == 0.0
    assert any("NOT CARLA" in w for w in rep.warnings)


def test_records_separate_policy_input_from_privileged_data(corpus):
    idx, samples = _runs(corpus)
    for s in samples:
        p = s["policy_observation"]
        assert set(p) == {"schema", "road", *POLICY_EGO} and set(p["road"]) == set(POLICY_ROAD_KEYS)
        for k in ("x", "y", "yaw", "progress", "lap", "t", "step", "cones"):
            assert k not in p
        assert "corners" not in p["road"] and "finish_distance" not in p["road"]
        if s["quality"]["clean"]:
            assert s["privileged"]["oracle"]["phase"] == "decision"
            assert s["target"]["longitudinal"]["certified"] and s["target"]["longitudinal"]["model"] == cert.CERT_MODEL
        full = merge_observation(p, s["privileged"]["observation"])
        assert split_observation(full) == (p, s["privileged"]["observation"])
        assert obs_to_dict(obs_from_dict(full)) == full


def test_generation_is_repeatable_on_bicycle(corpus, tmp_path, smoke):
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(smoke))
    generate_corpus(str(mp), str(tmp_path / "again"), "bicycle", limit=1, verbose=False)
    _, a = _runs(corpus)
    _, b = _runs(tmp_path / "again")
    assert [s["hashes"]["record"] for s in a] == [s["hashes"]["record"] for s in b]


# ======================================================================= adversarial corruption


def _copy(corpus, tmp_path):
    dst = tmp_path / "c"
    shutil.copytree(corpus, dst)
    return dst


def _rewrite(root, fn, pick, rehash=True, run=0):
    """Apply ``fn`` to the first picked record of run ``run``; recompute its sample hashes (unless
    ``rehash`` is False) and the file hash in index.json."""
    idx = json.loads((root / "index.json").read_text())
    path = root / idx["scenarios"][run]["files"]["samples"]["path"]
    lines = path.read_text().splitlines()
    for i, ln in enumerate(lines):
        s = json.loads(ln)
        if pick(s):
            fn(s)
            if rehash:
                s["hashes"] = sample_hashes(s)
            lines[i] = json.dumps(s, separators=(",", ":"))
            break
    else:
        raise AssertionError("no record picked")
    path.write_text("\n".join(lines) + "\n")
    idx["scenarios"][run]["files"]["samples"]["sha256"] = file_sha256(path)
    (root / "index.json").write_text(json.dumps(idx, indent=1))


@pytest.fixture(scope="module")
def interchange(corpus, tmp_path_factory):
    """Manifest interchange exported from the PRISTINE corpus (the checker is then pointed at a
    corrupted copy, as an attacker would present it)."""
    lake_or_skip()
    from autocross.datagen.lean_bridge import export
    idx = json.loads((corpus / "index.json").read_text())
    man = json.loads((corpus / "manifest.json").read_text())
    return export(man, tmp_path_factory.mktemp("inter"), idx, corpus)


def _lean_failures(interchange, root, tmp_path):
    from autocross.datagen.lean_bridge import run_checker
    rep = run_checker(interchange, root, tmp_path / "lean_report.json", witness=False)
    return rep, {r["name"] for r in rep["results"] if not r["ok"]}


clean = lambda s: s["quality"]["clean"]  # noqa: E731
expert_sample = lambda s: s["target"] is not None and s["target"]["longitudinal"] is not None and not s["quality"]["clean"]  # noqa: E731


def _lon(s):
    return s["target"]["longitudinal"]


CORRUPTIONS = {
    # 1. a clean sample's certificate is deleted
    "certificate_deleted": (lambda s: s["target"].update(longitudinal=None), clean,
                            ["certificate presence", "clean sample without"],
                            {"cert_presence_matches_phase", "clean_samples_certified", "clean_ids_equal_checked_ids"}),
    # 3. certified action changed (hashes recomputed)
    "certificate_accel_changed": (lambda s: _lon(s).update(accel_mm_s2=_lon(s)["accel_mm_s2"] + 1), clean,
                                  ["certificate differs", "label replay"],
                                  {"cert_decisions_recomputed", "clean_samples_certified"}),
    "certificate_b1_changed": (lambda s: _lon(s).update(b1_mm2_s2=(_lon(s)["b1_mm2_s2"] or 0) + 1), clean,
                               ["certificate differs"], {"cert_decisions_recomputed"}),
    "certificate_on_unclean_sample_changed": (lambda s: _lon(s).update(accel_mm_s2=_lon(s)["accel_mm_s2"] - 7),
                                              expert_sample, ["certificate differs"], {"cert_decisions_recomputed"}),
    "uncertified_pedal_changed": (lambda s: s["target"]["uncertified"].update(brake=0.5), clean,
                                  ["label replay"], set()),
    # 4. path modified, steering unchanged
    "path_point_moved": (lambda s: s["target"]["uncertified"]["path"][5].__setitem__(1, s["target"]["uncertified"]["path"][5][1] + 0.01),
                         clean, ["label replay"], set()),
    "path_frame_changed": (lambda s: s["target"]["uncertified"].update(path_frame="world"), clean, ["label replay"], set()),
    # 5. a claimed clean sample becomes uncertified
    "clean_sample_uncertified": (lambda s: _lon(s).update(kind="no_feasible_gap", certified=False, b1_mm2_s2=None), clean,
                                 ["certificate differs", "clean sample without"],
                                 {"cert_decisions_recomputed", "clean_samples_certified"}),
    # 7. observation altered, certificate kept
    "observation_speed_altered": (lambda s: s["policy_observation"].update(speed=s["policy_observation"]["speed"] + 0.5),
                                  clean, ["certificate differs", "label replay"], {"cert_decisions_recomputed"}),
    "observation_preview_altered": (lambda s: s["policy_observation"]["road"]["preview_curvature"].__setitem__(
        3, s["policy_observation"]["road"]["preview_curvature"][3] + 0.05), clean, ["label replay"], set()),
    # 2. wrong sample id
    "sample_id_wrong": (lambda s: s.update(sample_id=s["sample_id"][:-5] + "00000"), clean,
                        ["bad or duplicate sample_id"], {"sample_identity"}),
    # 9. a sample moved to another split
    "sample_split_changed": (lambda s: s.update(split="test"), clean, ["split="], {"sample_identity"}),
    # model version substituted
    "certificate_model_changed": (lambda s: _lon(s).update(model="cert-1"), clean, ["certificate differs"],
                                  {"cert_model_and_constants"}),
    # privileged data smuggled into the policy input
    "privileged_field_in_policy_input": (lambda s: s["policy_observation"].update(progress=1.0), clean,
                                         ["policy_observation is not the canonical whitelist"], set()),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruption_is_rejected(name, corpus, tmp_path, interchange):
    fn, pick, py_expect, lean_expect = CORRUPTIONS[name]
    root = _copy(corpus, tmp_path)
    _rewrite(root, fn, pick)
    errs = validate_corpus(root, allow_missing_coverage=True).errors
    for e in py_expect:
        assert any(e in x for x in errs), (name, e, errs[:6])
    rep, bad = _lean_failures(interchange, root, tmp_path)
    assert not rep["ok"] or not lean_expect
    assert lean_expect <= bad, (name, bad)


def test_record_edit_without_rehash_is_rejected(corpus, tmp_path):
    root = _copy(corpus, tmp_path)
    _rewrite(root, lambda s: s["target"]["uncertified"].update(steer=s["target"]["uncertified"]["steer"] + 0.01),
             clean, rehash=False)
    assert any("sample hash mismatch" in e for e in validate_corpus(root, allow_missing_coverage=True).errors)


def test_duplicated_record_is_rejected(corpus, tmp_path, interchange):
    root = _copy(corpus, tmp_path)
    idx = json.loads((root / "index.json").read_text())
    path = root / idx["scenarios"][0]["files"]["samples"]["path"]
    lines = path.read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if json.loads(ln)["quality"]["clean"])
    lines.insert(k + 1, lines[k])  # the same certified record twice
    path.write_text("\n".join(lines) + "\n")
    idx["scenarios"][0]["files"]["samples"]["sha256"] = file_sha256(path)
    (root / "index.json").write_text(json.dumps(idx, indent=1))
    errs = validate_corpus(root, allow_missing_coverage=True).errors
    assert any("bad or duplicate sample_id" in e for e in errs) and any("not contiguous" in e for e in errs)
    rep, bad = _lean_failures(interchange, root, tmp_path)
    assert {"sample_identity", "run_counts_match_index", "totals_match_index"} <= bad


def test_missing_and_unexpected_sample_files_are_rejected(corpus, tmp_path, interchange):
    root = _copy(corpus, tmp_path)
    idx = json.loads((root / "index.json").read_text())
    (root / idx["scenarios"][1]["files"]["samples"]["path"]).unlink()
    (root / "samples" / "extra.jsonl").write_text("")
    errs = validate_corpus(root, allow_missing_coverage=True).errors
    assert any("missing file" in e for e in errs) and any("unexpected file" in e for e in errs)
    with pytest.raises(Exception):  # Lean refuses to read a referenced file that is not there
        _lean_failures(interchange, root, tmp_path)
    (root / idx["scenarios"][1]["files"]["samples"]["path"]).write_text("")
    rep, bad = _lean_failures(interchange, root, tmp_path)
    assert {"sample_files_exact", "run_counts_match_index", "clean_ids_equal_checked_ids"} <= bad or \
        {"sample_files_exact", "run_counts_match_index", "totals_match_index"} <= bad


def test_index_claiming_clean_samples_with_none_checked_fails(corpus, tmp_path, interchange):
    """Zero checked certificates never pass when clean certified samples are claimed."""
    root = _copy(corpus, tmp_path)
    idx = json.loads((root / "index.json").read_text())
    for r in idx["scenarios"]:
        p = root / r["files"]["samples"]["path"]
        p.write_text("")
        r["files"]["samples"]["sha256"] = file_sha256(p)
    (root / "index.json").write_text(json.dumps(idx, indent=1))
    assert validate_corpus(root, allow_missing_coverage=True).errors
    rep, bad = _lean_failures(interchange, root, tmp_path)
    assert not rep["ok"] and {"clean_ids_equal_checked_ids", "totals_match_index"} <= bad


def test_lean_checker_accepts_corpus_with_kernel_witnesses(corpus, tmp_path, interchange):
    from autocross.datagen.lean_bridge import run_checker
    rep = run_checker(interchange, corpus, tmp_path / "r.json", witness=True, witness_dir=tmp_path / "w")
    assert rep["ok"], [r for r in rep["results"] if not r["ok"]]
    assert rep["axiom_audit"]["ok"] and rep["axiom_audit"]["theorems"] >= 20
    kw = rep["kernel_witnesses"]
    n_clean = rep["stats"]["corpus"]["clean_samples"]
    assert kw["ok"] and kw["kernel_proved"] == n_clean == kw["expected_clean_samples"] > 0
    # 6. an omitted witness (proof) file is detected
    from autocross.datagen.lean_bridge import _clean_ids, build, kernel_witnesses
    files = sorted((tmp_path / "w").glob("Witness_*.lean"))
    for f in files[1:]:
        f.unlink()
    idx = json.loads((corpus / "index.json").read_text())
    kw2 = kernel_witnesses(build(), tmp_path / "w", _clean_ids(corpus, idx))
    assert not kw2["ok"] and kw2["kernel_proved"] < n_clean


def test_axiom_audit_rejects_sorry(tmp_path, monkeypatch):
    lake = lake_or_skip()
    from autocross.datagen import lean_bridge
    lean_bridge.build()
    fake = tmp_path / "lean"
    (fake / "Corpus").mkdir(parents=True)
    (fake / "Corpus" / "Audit.lean").write_text("theorem bad : 1 = 2 := sorry\n#print axioms bad\n")
    monkeypatch.setattr(lean_bridge, "LEAN_DIR", fake)
    assert not lean_bridge.axiom_audit(lake)["ok"]


# ======================================================================= observation-only contract


def _vp(corpus):
    idx = json.loads((corpus / "index.json").read_text())
    return VehicleParams(**idx["scenarios"][0]["vehicle"])


def test_privileged_fields_do_not_affect_the_label(corpus):
    """Changing every privileged field of a recorded step leaves the expert label unchanged."""
    vp = _vp(corpus)
    _, samples = _runs(corpus)
    for s in [x for x in samples if x["quality"]["clean"]][::25]:
        priv = json.loads(json.dumps(s["privileged"]["observation"]))
        priv.update(x=priv["x"] + 123.0, y=-7.0, yaw=1.0, progress=priv["progress"] + 500.0, lap=3, t=999.0, cones=None)
        priv["road"].update(corners=[], finish_distance=1.0)
        full = obs_from_dict(merge_observation(s["policy_observation"], priv))
        t, e = label_blocks(command_to_dict(expert_mod.expert_action(full, vp, None)), approach=False)
        assert tree_diff(t, s["target"]) == 0.0 and tree_diff(e, s["expert"]) == 0.0
        assert replay_diff(s, vp, None) == 0.0


def _road_obs(t, s0=5.0, rng_m=30.0):
    from autocross.perception import RoadModelSensor
    rs = RoadModelSensor(t, {"range_m": rng_m}, np.random.default_rng(0))
    p, h = t.line.point_at(s0), t.line.heading_at(s0)
    rs.s_hint = s0
    road = rs.measure(float(p[0]), float(p[1]) + 0.2, float(h) + 0.01, progress=s0)
    obs = obs_from_dict({"t": 1.0, "step": 20, "x": float(p[0]), "y": float(p[1]) + 0.2, "yaw": float(h) + 0.01,
                         "vx": 12.0, "vy": 0.0, "yaw_rate": 0.0, "ax": 0.0, "ay": 0.0, "speed": 12.0, "steer": 0.0,
                         "throttle": 0.1, "brake": 0.0, "progress": s0, "lap": 0, "timing_started": True,
                         "cones": None, "road": None})
    obs.road = road
    return obs


def test_hidden_geometry_beyond_the_preview_does_not_change_policy_input_or_label():
    """Two courses identical up to 35 m past the end of the preview and mirror images beyond it
    (same total length, exactly): the policy observation and the label are identical."""
    from autocross.track import _integrate_segments, build
    raw, _ = _integrate_segments([{"type": "straight", "length": 70}, {"type": "arc", "radius": 12, "angle": 90},
                                  {"type": "straight", "length": 30}], [0, 0, 0], 6.0)
    assert np.all(raw[raw[:, 0] <= 70.0 - 1e-9, 1] == 0.0)
    mirrored = raw * np.array([1.0, -1.0])  # identical while y == 0, the other turn direction afterwards
    outs = []
    for pts in (raw, mirrored):
        obs = _road_obs(build({"width": 6.0, "centerline": pts.tolist()}))
        pol, priv = split_observation(obs_to_dict(obs))
        outs.append((pol, command_to_dict(expert_mod.expert_action(obs, VP, None)),
                     command_to_dict(expert_mod.expert_action(policy_to_observation(pol), VP, None))))
    (pa, la, ra), (pb, lb, rb) = outs
    assert pa == pb                               # identical canonical policy input
    assert la == lb == ra == rb                   # identical label, from the full or the policy-only observation
    assert la["debug"]["cert_kind"] == cert.KIND_GREATEST_SAFE


def test_corner_list_depends_on_geometry_beyond_the_preview():
    """Why ``road.corners`` is privileged: two courses that agree over the preview report different
    corners (the apex is located on the WHOLE corner), so the corner list leaks hidden geometry."""
    from autocross.track import build
    a = build({"width": 6.0, "segments": [{"type": "straight", "length": 20},
                                          {"type": "arc", "radius": 20, "angle": 90}, {"type": "straight", "length": 60}]})
    b = build({"width": 6.0, "segments": [{"type": "straight", "length": 20}, {"type": "arc", "radius": 20, "angle": 50},
                                          {"type": "arc", "radius": 9, "angle": -120}, {"type": "straight", "length": 60}]})
    ca, cb = _road_obs(a).road.corners, _road_obs(b).road.corners
    assert len(ca) == len(cb) == 1
    assert math.isnan(ca[0].apex_distance) != math.isnan(cb[0].apex_distance)


def test_label_that_depends_on_privileged_data_is_rejected(corpus, tmp_path, monkeypatch):
    """Relabel a run with an expert that peeks at course progress: replay from the policy input
    alone cannot reproduce it, so validation fails."""
    vp = _vp(corpus)
    honest = expert_mod.expert_action

    def leaky(obs, vehicle, config=None):
        cmd = honest(obs, vehicle, config)
        if not math.isnan(obs.progress):
            cmd.steer = float(np.clip(cmd.steer + 1e-3 * math.sin(obs.progress), -1, 1))
        return cmd

    root = _copy(corpus, tmp_path)

    def relabel(s):
        full = obs_from_dict(merge_observation(s["policy_observation"], s["privileged"]["observation"]))
        s["target"], s["expert"] = label_blocks(command_to_dict(leaky(full, vp, None)), approach=False)
    _rewrite(root, relabel, clean)
    errs = validate_corpus(root, allow_missing_coverage=True).errors
    assert any("label replay" in e for e in errs), errs[:5]


def test_expert_relabels_from_policy_input_alone(corpus):
    vp = _vp(corpus)
    _, samples = _runs(corpus)
    for s in samples:
        if s["target"] is not None and s["expert"]["phase"] == "expert":
            t, e = expert_label(s["policy_observation"], None, vp, None)
            assert tree_diff(t, s["target"]) == 0.0


def test_policy_to_observation_rejects_non_canonical_input(corpus):
    _, samples = _runs(corpus)
    p = dict(samples[0]["policy_observation"])
    p["x"] = 1.0
    with pytest.raises(ValueError):
        policy_to_observation(p)


# ======================================================================= mirror symmetry


def test_mirror_transform_and_expert_symmetry(corpus):
    """Formal: the mirror is an involution and swaps sides. Behaviour: the certified longitudinal
    label is mirror invariant; steering and path flip sign except where the gap-side tie-break
    (equal gaps) applies, which is asymmetric by design."""
    vp = _vp(corpus)
    _, samples = _runs(corpus)
    n_sym = n_tie = 0
    for s in samples:
        if not s["quality"]["clean"]:
            continue
        p = s["policy_observation"]
        m = mirror_observation(p)
        assert mirror_observation(m) == p
        if p["road"]["obstacles"]:
            assert m["road"]["obstacles"][0]["lateral_min"] == -p["road"]["obstacles"][0]["lateral_max"]
        tm, _ = expert_label(m, None, vp, None)
        assert tm["longitudinal"] == s["target"]["longitudinal"]
        q = cert.quantise_dict(p)
        ties = any(cert.obstacle_plan(None, q, ob)[0] == 0 for ob in q["obstacles"])
        if ties:
            n_tie += 1
            continue
        mu = mirror_action(s["target"]["uncertified"])
        assert abs(tm["uncertified"]["steer"] - mu["steer"]) <= 1e-9
        assert tree_diff(tm["uncertified"]["path"], mu["path"]) <= 1e-9
        n_sym += 1
    assert n_sym > 0


# ======================================================================= loader and report


def test_loader_returns_only_policy_input_and_marks_certified_target(corpus):
    from autocross.datagen import DatagenError
    from autocross.datagen.dataset import iter_examples, load_examples, to_arrays
    ex = load_examples(corpus, split="train", expect_backend="bicycle")
    assert ex
    for pol, tgt, st, meta in ex:
        assert set(pol) == {"schema", "road", *POLICY_EGO}
        assert st["certified"] and st["model"] == cert.CERT_MODEL and "longitudinal" in st["scope"]
        assert set(tgt) == {"schema", "longitudinal", "uncertified"} and meta["clean"]
    arr = to_arrays(ex)
    assert arr["x"].shape[0] == len(ex) and np.isfinite(arr["x"]).all() and arr["certified"].all()
    with pytest.raises(DatagenError):
        next(iter_examples(corpus, expect_backend="carla"))


def test_dataset_report_accounts_for_every_sample(corpus):
    from autocross.datagen.report import dataset_report, wilson
    rep = dataset_report(corpus)
    t = rep["totals"]
    idx = json.loads((corpus / "index.json").read_text())
    assert t["samples"] == idx["totals"]["samples"] and t["clean_samples"] == idx["totals"]["clean_samples"]
    assert t["certified_clean_samples"] == t["clean_samples"] and t["uncertified_clean_samples"] == 0
    assert t["effective_samples_ar1"] < t["clean_samples"]
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


def test_carla_generation_fails_clearly_without_server(smoke, tmp_path):
    from autocross.datagen import DatagenError
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(smoke))
    try:
        import carla  # noqa: F401
        pytest.skip("carla client installed: covered by the CARLA smoke run")
    except ImportError:
        pass
    with pytest.raises(DatagenError, match="No data was generated"):
        generate_corpus(str(mp), str(tmp_path / "c"), "carla", verbose=False)
    assert not (tmp_path / "c" / "index.json").exists()


def test_lean_checker_rejects_split_and_pair_corruption(smoke, tmp_path):
    lake_or_skip()
    import copy
    from autocross.datagen.lean_bridge import export, run_checker
    assert run_checker(export(smoke, tmp_path / "ok"), None, tmp_path / "ok.json")["ok"]
    m = copy.deepcopy(smoke)
    p = m["contrast_pairs"][0]
    var = next(e for e in m["scenarios"] if e["scenario_id"] == p["variant_id"])
    var["split"] = "test" if var["split"] != "test" else "train"
    var["params"]["turn_angle_deg"] = round(var["params"]["turn_angle_deg"] + 1.0, 3)
    rep = run_checker(export(m, tmp_path / "bad"), None, tmp_path / "bad.json")
    bad = {r["name"] for r in rep["results"] if not r["ok"]}
    assert {"contrast_pairs_integrity", "split_rule", "group_split_consistent"} <= bad
