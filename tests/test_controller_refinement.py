"""Executable refinement checks: the submitted controller's integer functions against their
Lean specification (`Corpus.Controller`, `Corpus.Steer`), evaluated by the Lean KERNEL.

For randomly generated in-domain inputs the Python functions of `teams/certline.py` are run
and their exact integer outputs are written as `by decide` theorems about the Lean
definitions.  A passing test is a kernel proof that, ON THESE INPUTS, the implementation and
the specification agree; it is evidence, not a proof about arbitrary Python executions.

Covered: `decide_long` (curvature caps, gap-speed rule, sidestep caps, terminal cap,
backward/forward passes, integer acceleration), `decide_lateral` (offsets, curvature demand,
steering feed-forward, signed-lateral tie rule), `_spare_boxes` (certified clearance
interval over the obstacle memory), `_project_plan` (rate limit + clamp)."""
from __future__ import annotations

import importlib.util
import random
import subprocess

import numpy as np
import pytest

from conftest import lake_or_skip

N_CASES = 12


def _load(root):
    spec = importlib.util.spec_from_file_location("certline_refinement", root / "teams" / "certline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _L(v):
    return "[" + ", ".join(_L(x) if isinstance(x, list) else f"({int(x)})" for x in v) + "]"


def _opt(v):
    return "none" if v is None else f"(some ({int(v)}))"


def _cert_consts(lon):
    keys = ("a_lat", "a_acc", "a_brk", "a_max", "v_max", "r_h", "hold", "ramp_in", "ramp_out",
            "r_sw_m", "veh_width")
    return "{ " + ", ".join(f"{k} := ({int(lon[k])})" for k in keys) + " }"


def _long_consts(lon):
    return (f"{{ cert := {_cert_consts(lon)}, gap_speed := {'true' if lon['gap_speed'] else 'false'}, "
            f"gap_base := ({lon['gap_base']}), gap_per := ({lon['gap_per_m']}), "
            f"gap_margin := ({lon['gap_margin']}) }}")


def _cert_q(q):
    return (f"{{ v0 := ({q['v0']}), s := {_L(q['s'])}, k := {_L(q['k'])}, wl := {_L(q['wl'])}, "
            f"wr := {_L(q['wr'])}, obstacles := {_L(q['obstacles'])} }}")


def _steer_consts(sc, lon):
    lon2 = {**lon, "hold": sc["hold"], "ramp_in": sc["ramp_in"], "ramp_out": sc["ramp_out"],
            "veh_width": sc["veh_width"]}
    return (f"{{ lon := {_cert_consts(lon2)}, margin := ({sc['margin']}), step := ({sc['step']}), "
            f"kappa_max := ({sc['kappa_max']}) }}")


def _random_q(rng: random.Random) -> dict:
    n = rng.randint(3, 7)
    s = [0]
    for _ in range(n - 1):
        s.append(s[-1] + rng.choice((500, 1000, 1500)))
    obs = []
    for _ in range(rng.randint(0, 2)):
        d = rng.randint(0, s[-1] + 2000)
        lmin = rng.randint(-2500, 1500)
        obs.append([d, lmin, lmin + rng.randint(100, 1200)])
    return {"v0": rng.randint(0, 15000), "s": s, "k": [rng.randint(-150000, 150000) for _ in s],
            "wl": [rng.randint(2500, 4500) for _ in s], "wr": [rng.randint(2500, 4500) for _ in s],
            "lat": rng.randint(-1500, 1500), "obstacles": obs}


@pytest.fixture
def ctl(root):
    mod = _load(root)
    vp = mod.VehicleParams(mass=1845.0, wheelbase=3.0, track_width=1.67, length=4.79, width=2.16,
                           cg_to_front=1.45, cg_to_rear=1.55, max_steer=0.61, max_steer_rate=2.0,
                           mu=0.95, max_accel=5.2, max_decel=9.0, max_speed=40.0, control_dt=0.05,
                           actuation_delay_steps=0)
    cfg = dict(mod.DEFAULTS)
    lon = mod.lon_constants(vp, cfg)
    return mod, lon, mod.lateral_constants(vp, cfg, lon)


def _kernel(src: str):
    lake = lake_or_skip()
    from autocross.datagen.lean_bridge import LEAN_DIR, build
    build()
    f = LEAN_DIR / "RefinementCheck.lean"
    f.write_text(src, encoding="utf-8")
    try:
        return subprocess.run([lake, "env", "lean", f.name], cwd=LEAN_DIR, capture_output=True, text=True)
    finally:
        f.unlink(missing_ok=True)


def test_controller_functions_refine_their_lean_specification(ctl):
    mod, lon, sc = ctl
    rng = random.Random(20261017)
    thms = ["import Corpus", "set_option maxRecDepth 200000", "open Corpus"]
    kinds = set()
    k = 0
    while k < N_CASES:
        q = _random_q(rng)
        if not mod.well_formed(q):
            continue
        ks = [rng.choice((0, 0, rng.randint(1, 400000))) for _ in q["s"]]
        extra = None if rng.random() < 0.5 else [rng.randint(0, 300_000_000) for _ in q["s"]]
        d = mod.decide_long(lon, q, ks, 1.0, extra)
        kinds.add(d["kind"])
        lean_extra = _L(extra) if extra is not None else "[]"
        thms.append(
            f"theorem long{k} : Controller.decideLong {_long_consts(lon)} {_cert_q(q)} {_L(ks)} {lean_extra} = "
            f"⟨{d['kind']}, ({d['a']}), {_opt(d['b1'])}, {_opt(d['b0'])}, {_opt(d['B0'])}⟩ := by decide")
        lat = mod.decide_lateral(sc, lon, q)
        sq = f"({{ base := {_cert_q(q)}, lat := ({q['lat']}) }} : Steer.Q)"
        scl = f"({_steer_consts(sc, lon)} : Steer.Consts)"
        thms.append(f"theorem off{k} : Steer.offs {scl} {sq} = {_L(lat['off'])} ∧ "
                    f"Steer.curvs {sq} (Steer.offs {scl} {sq}) = {_L(lat['curv'])} ∧ "
                    f"Steer.steers {scl} (Steer.curvs {sq} (Steer.offs {scl} {sq})) = {_L(lat['steer'])} := by decide")
        # clearance interval over a random obstacle memory
        mem = [{"d": ob[0], "lmin": ob[1], "lmax": ob[2], "side": rng.choice((1, -1))} for ob in q["obstacles"]]
        mem_lit = "[" + ", ".join(f"⟨({o['d']}), ({o['lmin']}), ({o['lmax']}), ({o['side']})⟩" for o in mem) + "]"
        for i in range(len(q["s"])):
            lo_i, hi_i = mod._spare_boxes(sc, q, mem, i)
            thms.append(f"theorem box{k}_{i} : Controller.spareBox {scl} {sq} {mem_lit} {i} = (({lo_i}), ({hi_i})) := by decide")
        # projection on integer data (exact in binary64)
        lo = [rng.randint(-2000, 500) for _ in q["s"]]
        hi = [x + rng.randint(0, 2000) for x in lo]
        raw = [rng.randint(-4000, 4000) for _ in q["s"]]
        r = rng.choice((200, 350, 700))
        it = rng.randint(1, 3)
        out = mod._project_plan(np.array(raw, float), np.array(lo, float), np.array(hi, float), float(r), iters=it)
        assert all(float(x).is_integer() for x in out)
        thms.append(f"theorem proj{k} : Controller.projectPlan {_L(raw)} {_L(lo)} {_L(hi)} ({r}) {it} = "
                    f"{_L([int(x) for x in out])} := by decide")
        k += 1
    assert len(kinds) >= 2, kinds          # the sample exercises more than one decision branch
    r = _kernel("\n".join(thms) + "\n")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


def test_kernel_rejects_a_perturbed_controller_output(ctl):
    mod, lon, sc = ctl
    q = {"v0": 8000, "s": [0, 1000, 2000, 3000], "k": [0, 20000, 40000, 0], "wl": [3000] * 4,
         "wr": [3000] * 4, "lat": 0, "obstacles": []}
    d = mod.decide_long(lon, q, [0, 20000, 40000, 0], 1.0, None)
    src = ("import Corpus\nopen Corpus\n"
           f"theorem bad : Controller.decideLong {_long_consts(lon)} {_cert_q(q)} [0, 20000, 40000, 0] [] = "
           f"⟨{d['kind']}, ({d['a'] + 1}), {_opt(d['b1'])}, {_opt(d['b0'])}, {_opt(d['B0'])}⟩ := by decide\n")
    assert _kernel(src).returncode != 0
