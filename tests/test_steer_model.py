"""Unit and Python/Lean parity tests for the integer ``steer-2`` mirror."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from autocross.api import VehicleParams
from autocross.datagen import cert, steer


VP = VehicleParams(1845.0, 3.0, 1.67, 4.79, 2.16, 1.45, 1.55, 0.61, 2.0, 0.95, 5.2, 9.0, 40.0, 0.05, 0)
C = steer.model_constants(VP)


def q(*, n=41, step=1000, lat=0, width=3000, k=0, obstacles=()):
    return {"base": {"v0": 0, "s": [i * step for i in range(n)], "k": [k] * n,
                      "wl": [width] * n, "wr": [width] * n,
                      "obstacles": [list(ob) for ob in obstacles]}, "lat": lat}


def test_ramp_endpoints_monotonic_and_bounds():
    vals = [steer.ramp(1750, x, 18000) for x in range(0, 18001, 1000)]
    assert vals[0] == 0 and vals[-1] == 1750
    assert vals == sorted(vals)
    assert all(0 <= x <= 1750 for x in vals)



def test_ramp_out_cap_regression_selects_later_hold_supplier():
    # At s=24000 the old ramp-out formula rounded A up to 1000, tying B's
    # hold.  The first-strict-largest fold then incorrectly retained A.
    d_a, d_b = 20_900, 27_000
    old_a = 1000 - steer.ramp(1000, 24_000 - d_a - C["lon"]["hold"], C["lon"]["ramp_out"])
    assert old_a == 1000
    assert steer.wOb(C, d_a, 24_000) == 999
    qq = q(obstacles=[(d_a, 500, 1500), (d_b, -1500, -500)])
    assert steer.rawPlanAt(C, qq, 24_000)[0] == d_b


def test_old_ramp_out_weight_overlaps_opposing_obstacle_while_clearance_reports_ok(monkeypatch):
    """Python mirror of `Corpus.SteerClear.old_selector_counterexample`: with the pre-fix
    weight the plan inside B's hold window follows A (offset -1250, car body overlapping B's
    extent [-1500, -500]) while every recorded clearance is nonnegative; the fixed model
    passes B on its left with nonnegative clearance."""
    lon = {**C["lon"], "hold": 3000, "ramp_in": 18000, "ramp_out": 15000, "veh_width": 2000}
    c = {**C, "lon": lon, "margin": 200}
    qq = q(obstacles=[(20_900, 500, 1500), (27_000, -1500, -500)])
    hw = steer.halfw(c)
    new_off = steer.offAt(c, qq, 24_000, 24)
    assert new_off == 1250 and new_off - hw >= -500 + c["margin"]
    assert all(x >= 0 for x in steer.clears(c, qq))

    def old_wob(cc, d, si):
        h, ri, ro = cc["lon"]["hold"], cc["lon"]["ramp_in"], cc["lon"]["ramp_out"]
        if si < d - h - ri:
            return 0
        if si < d - h:
            return steer.ramp(1000, si - d + h + ri, ri)
        if si <= d + h:
            return 1000
        if si <= d + h + ro:
            return 1000 - steer.ramp(1000, si - d - h, ro)
        return 0

    monkeypatch.setattr(steer, "wOb", old_wob)
    old_off = steer.offAt(c, qq, 24_000, 24)
    assert old_off == -1250
    assert old_off - hw < -500 and -1500 < old_off + hw          # body overlaps obstacle B
    assert all(x >= 0 for x in steer.clears(c, qq))               # ... yet the certificate said clear

def test_side_rule_tie_and_amplitude_bound():
    left = q(lat=0, obstacles=[(20_000, -500, 500)])
    right = q(lat=-1, obstacles=[(20_000, -500, 500)])
    # ``lat < 0`` is deliberately outside the certified domain, but the helper
    # itself still exposes the declared asymmetric tie rule for construction tests.
    assert steer.choose_side(C, left, left["base"]["obstacles"][0])[0] == 1
    assert steer.choose_side(C, right, right["base"]["obstacles"][0])[0] == -1
    for ob in left["base"]["obstacles"]:
        side, amp = steer.choose_side(C, left, ob)
        _, target, _ = cert.obstacle_plan(C["lon"], left["base"], ob)
        assert side in (-1, 1) and 0 <= amp <= target



def test_side_selection_strict_and_tie_rule():
    strict_left = q(obstacles=[(20_000, -2000, 0)])
    strict_right = q(obstacles=[(20_000, 0, 2000)])
    tie_left = q(lat=0, obstacles=[(20_000, -500, 500)])
    tie_right = q(lat=-1, obstacles=[(20_000, -500, 500)])
    assert steer.sideAmpOb(strict_left, strict_left["base"]["obstacles"][0])[0] == 1
    assert steer.sideAmpOb(strict_right, strict_right["base"]["obstacles"][0])[0] == -1
    assert steer.sideAmpOb(tie_left, tie_left["base"]["obstacles"][0])[0] == 1
    assert steer.sideAmpOb(tie_right, tie_right["base"]["obstacles"][0])[0] == -1


def test_side_amp_ob_mirrors_with_widths_and_obstacle_extents():
    original = q(lat=100, obstacles=[(20_000, -700, 1000)])
    mirrored = q(lat=-100, obstacles=[(20_000, -1000, 700)])
    original["base"]["wl"] = [5000] * len(original["base"]["s"])
    original["base"]["wr"] = [3000] * len(original["base"]["s"])
    mirrored["base"]["wl"] = [3000] * len(mirrored["base"]["s"])
    mirrored["base"]["wr"] = [5000] * len(mirrored["base"]["s"])
    side, amp = steer.sideAmpOb(original, original["base"]["obstacles"][0])
    mirror_side, mirror_amp = steer.sideAmpOb(mirrored, mirrored["base"]["obstacles"][0])
    assert mirror_side == -side
    assert mirror_amp == amp

def test_offsets_zero_outside_and_equal_hold():
    d = 20_000
    qq = q(n=61, obstacles=[(d, -500, 0)])
    dec = steer.decide(C, qq)
    side, amp = dec["side"][0], dec["amp"][0]
    hold, rin, rout = C["lon"]["hold"], C["lon"]["ramp_in"], C["lon"]["ramp_out"]
    for si, off in zip(qq["base"]["s"], dec["off"]):
        if si < d - hold - rin or si > d + hold + rout:
            assert off == 0
        if d - hold <= si <= d + hold:
            assert off == side * amp
    assert all(abs(dec["off"][i + 1] - dec["off"][i]) <= amp for i in range(len(dec["off"]) - 1))


def test_second_difference_curvature_and_steering_clip():
    qq = q(n=8, k=9_999_999, obstacles=[])
    dec = steer.decide(C, qq)
    assert dec["kind"] == steer.KIND_PLAN_OK
    assert all(x == 9_999_999 for x in dec["curv"])
    assert all(x == 1000 for x in dec["steer"])
    qq = q(n=8, obstacles=[(4000, -500, 500)])
    dec = steer.decide(C, qq)

    assert dec["curv"][0] == 0 and dec["curv"][-1] == 0
    assert all(-1000 <= x <= 1000 for x in dec["steer"])

def test_ramp_first_and_second_difference_bounds():
    qq = q(n=61, obstacles=[(20_000, -500, 0)])
    dec = steer.decide(C, qq)
    amp = dec["amp"][0]
    d1 = [dec["off"][i + 1] - dec["off"][i] for i in range(len(dec["off"]) - 1)]
    d2 = [dec["off"][i + 1] - 2 * dec["off"][i] + dec["off"][i - 1]
          for i in range(1, len(dec["off"]) - 1)]
    assert max(map(abs, d1)) <= amp
    assert max(map(abs, d2)) <= 2 * amp

def test_corridor_clamp_narrow_corridor_and_negative_midpoint():
    qq = q(width=2000, obstacles=[(20_000, -500, 0)])
    dec = steer.decide(C, qq)
    i_hold = qq["base"]["s"].index(20_000)
    assert dec["off"][i_hold] == steer.hi(C, qq, i_hold)
    assert abs(dec["off"][i_hold]) < abs(steer.rawOffAt(C, qq, 20_000))

    narrow = q(obstacles=[(20_000, -500, 0)])
    narrow["base"]["wl"] = [500] * len(narrow["base"]["s"])
    narrow["base"]["wr"] = [1000] * len(narrow["base"]["s"])
    assert steer.lo(C, narrow, 0) > steer.hi(C, narrow, 0)
    assert steer.clampAt(C, narrow, 0, 1234) == (steer.lo(C, narrow, 0) + steer.hi(C, narrow, 0)) // 2


def test_corridor_clamp_noop_on_fitting_hold_and_never_increases_abs():
    qq = q(obstacles=[(20_000, -500, 0)])
    raw = [steer.rawOffAt(C, qq, si) for si in qq["base"]["s"]]
    dec = steer.decide(C, qq)
    hold = C["lon"]["hold"]
    for i, (si, value) in enumerate(zip(qq["base"]["s"], dec["off"])):
        assert abs(value) <= abs(raw[i])
        if 20_000 - hold <= si <= 20_000 + hold:
            assert value == raw[i]


def test_clamp_never_increases_abs_and_reversed_interval_uses_midpoint():
    qq = q(obstacles=[(20_000, -500, 0)])
    raw = [steer.rawOffAt(C, qq, si) for si in qq["base"]["s"]]
    for i, value in enumerate(raw):
        assert abs(steer.clampAt(C, qq, i, value)) <= abs(value)

    reversed_corridor = q()
    reversed_corridor["base"]["wl"] = [500] * len(reversed_corridor["base"]["s"])
    reversed_corridor["base"]["wr"] = [1000] * len(reversed_corridor["base"]["s"])
    low, high = steer.lo(C, reversed_corridor, 0), steer.hi(C, reversed_corridor, 0)
    assert low > high
    midpoint = (low + high) // 2
    for value in (-5000, -1, 0, 1, 5000):
        assert steer.clampAt(C, reversed_corridor, 0, value) == midpoint


def test_plan_arrays_match_station_count():
    for qq in (q(), q(obstacles=[(20_000, -500, 0)])):
        decision = steer.decide(C, qq)
        n_stations = len(qq["base"]["s"])
        assert len(decision["off"]) == n_stations
        assert len(decision["clear"]) == n_stations
        assert len(decision["curv"]) == n_stations
        assert len(decision["steer"]) == n_stations


def test_well_formed_rejections():
    assert steer.well_formed(C, q())
    assert not steer.well_formed(C, {**q(), "base": {**q()["base"], "s": [0, 1000, 2500]}})
    assert not steer.well_formed(C, {**q(lat=-1)})
    overlapping = q(obstacles=[(10_000, -100, 100), (14_000, -100, 100)])
    assert not steer.well_formed(C, overlapping)
    malformed = q(obstacles=[(10_000, -100)])
    assert not steer.well_formed(C, malformed)
    badc = {**C, "step": 0}
    assert not steer.well_formed(badc, q())


def test_decision_branches_and_determinism():
    clean = steer.decide(C, q(obstacles=[(20_000, -500, 0)]))
    assert clean["kind"] == steer.KIND_PLAN_OK
    narrow = q(width=1000, obstacles=[(20_000, -900, 900)])
    assert steer.decide(C, narrow)["kind"] == steer.KIND_NO_GAP
    invalid = {**q(), "base": {**q()["base"], "s": [0, 1000, 1000]}}
    assert steer.decide(C, invalid)["kind"] == steer.KIND_INVALID
    assert steer.decide(C, q(obstacles=[(20_000, -500, 0)])) == clean


def test_zero_amplitude_and_obstacles_outside_preview():
    zero = q(obstacles=[(20_000, -3000, -3000)])
    d0 = steer.decide(C, zero)
    assert d0["amp"] == [0] and set(d0["off"]) == {0}
    before = steer.decide(C, q(obstacles=[(-5000, 0, 100)]))
    after = steer.decide(C, q(obstacles=[(100_000, 0, 100)]))
    assert len(before["off"]) == len(after["off"]) == 41
    assert before["side"] == after["side"] == [-1]


def test_quantise_dict_includes_only_policy_lateral_offset():
    obs = {"speed": 3.0, "road": {"valid": True, "lateral_offset": -0.25,
        "preview_s": [0.0, 1.0], "preview_curvature": [0.0, 0.0],
        "preview_width_left": [3.0, 3.0], "preview_width_right": [3.0, 3.0], "obstacles": []}}
    got = steer.quantise_dict(obs)
    assert got["lat"] == 250 and got["base"]["s"] == [0, 1000]
    absolute = {**obs, "road": {**obs["road"], "lateral_offset": 0.25}}
    assert steer.decide(C, got) == steer.decide(C, steer.quantise_dict(absolute))
    assert steer.quantise_dict({"road": {"valid": False}}) is None


def _lean_lit(v):
    if isinstance(v, list):
        return "[" + ", ".join(_lean_lit(x) for x in v) + "]"
    return f"({v})"


def _lean_consts(c):
    lon = c["lon"]
    fields = ", ".join(f"{k} := ({lon[k]})" for k in steer.CONST_KEYS)
    return ("{ lon := { " + fields + " }, margin := (" + str(c["margin"]) + "), step := ("
            + str(c["step"]) + "), kappa_max := (" + str(c["kappa_max"]) + ") }")


def _lean_q(qv):
    b = qv["base"]
    fields = ", ".join(f"{k} := {_lean_lit(b[k])}" if isinstance(b[k], list) else f"{k} := ({b[k]})"
                        for k in ("v0", "s", "k", "wl", "wr", "obstacles"))
    return "{ base := { " + fields + " }, lat := (" + str(qv["lat"]) + ") }"


def test_python_and_lean_decide_agree():
    """Kernel-evaluate the same finite cases as Python (skip before Steer.lean lands)."""
    root = Path(__file__).resolve().parents[1]
    module = root / "lean" / "Corpus" / "Steer.lean"
    if not module.exists():
        pytest.skip("lean/Corpus/Steer.lean is not present yet")
    lake = shutil.which("lake")
    if lake is None:
        pytest.skip("Lean lake executable is unavailable")
    built = subprocess.run([lake, "build", "Corpus.Steer"], cwd=root / "lean", capture_output=True, text=True)
    if built.returncode != 0:
        pytest.skip("Lean Corpus.Steer does not compile yet: " + (built.stdout + built.stderr)[-500:])
    cases = [q(), q(obstacles=[(20_000, -500, 0)]), q(obstacles=[(20_000, -900, 900)]),
             {**q(), "base": {**q()["base"], "s": [0, 1000, 1000]}}]
    src = ["import Corpus.Steer", "set_option maxRecDepth 100000", "open Corpus.Steer"]
    for i, qv in enumerate(cases):
        got = steer.decide(C, qv)
        cs, qs = _lean_consts(C), _lean_q(qv)
        d = "⟨%d, %s, %s, %s, %s, %s, %s⟩" % (got["kind"], _lean_lit(got["side"]), _lean_lit(got["amp"]),
                                              _lean_lit(got["off"]), _lean_lit(got["clear"]),
                                              _lean_lit(got["curv"]), _lean_lit(got["steer"]))
        src.append(f"theorem e{i} : decide {cs} {qs} = {d} := by decide")
    f = root / "SteerParity.scratch.lean"
    try:
        f.write_text("\n".join(src) + "\n", encoding="utf-8")
        r = subprocess.run([lake, "env", "lean", str(f)], cwd=root / "lean", capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    finally:
        f.unlink(missing_ok=True)
