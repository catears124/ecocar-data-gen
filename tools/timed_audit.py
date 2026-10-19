"""Offline audit: bind recorded controller decisions to certificates and Lean.

    python tools/timed_audit.py experiments/bench/dev/tech2/d_base_practice_technical_s0 \
        --controller teams/timed_exact.py --controller-config experiments/configs/d_base.yaml \
        --lean-checker lean/.lake/build/bin/timedcheck.exe --limit 40 --out audit.json

The audit replays the observation stream the controller saw (``observations.pkl``,
written by ``tools/bench.py --record-obs``) through the *same* encoder, and for every
control step:

1. rebuilds the ``timed-1`` instance and compares its hash with ``dbg_instance_h52``
   (binding: the certificate is about the instance the controller actually solved);
2. re-solves it and compares the optimal plan with ``dbg_plan0..2``;
3. compares the first command derived from that plan with the *published* command
   (steer/throttle/brake, exact float equality with ``command_floats``);
4. for a sample of steps, builds a ``timed-1`` certificate, checks it with the
   independent Python checker, then with the compiled Lean ``timedcheck``.

A step that passes (3) is ``solver``-status evidence; a step that also passes (4)
with Lean is *Lean-checked for the executed action*.  The report states both
fractions explicitly and never upgrades a step that did not pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocross.api import TrackInfo, VehicleParams  # noqa: E402
from autocross.timed import certificate as certmod  # noqa: E402
from autocross.timed.encode import EncodeError, Encoder  # noqa: E402
from autocross.timed.model import command_floats, first_command  # noqa: E402
from autocross.timed.vector import Solver  # noqa: E402


def load_controller(path: Path, config: dict | None, vehicle: dict):
    spec = importlib.util.spec_from_file_location("audit_controller", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    cls = next(c for _, c in vars(mod).items()
               if isinstance(c, type) and c.__name__ == "TimedExactController")
    ctrl = cls(config or {})
    vp = VehicleParams(**vehicle)
    track = TrackInfo(name="course", closed=False, laps=1, centerline=np.zeros((0, 2)),
                      width_left=np.zeros(0), width_right=np.zeros(0), cones=np.zeros((0, 2)),
                      cone_colors=np.zeros(0, dtype=int), start_line=np.zeros((2, 2)),
                      finish_line=np.zeros((2, 2)), gates=np.zeros((0, 2, 2)), full_knowledge=False)
    ctrl.on_start(track, vp)
    return ctrl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--controller", default="teams/timed_exact.py")
    ap.add_argument("--controller-config", default=None)
    ap.add_argument("--lean-checker", default="lean/.lake/build/bin/timedcheck.exe")
    ap.add_argument("--lean-checker-alt", default="lean/.lake/build/bin/timedcheck")
    ap.add_argument("--limit", type=int, default=0, help="max Lean-checked steps (0 = all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = Path(a.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    steps = pickle.loads((run / "observations.pkl").read_bytes())
    summary = json.loads((run / "summary.json").read_text())
    cfg = None
    if a.controller_config:
        import yaml
        cfg = yaml.safe_load((Path(a.controller_config) if Path(a.controller_config).is_absolute()
                              else ROOT / a.controller_config).read_text())
    ctrl = load_controller(ROOT / a.controller, cfg, summary["vehicle"])
    enc: Encoder = ctrl.encoder
    lean = None
    for p in (ROOT / a.lean_checker, ROOT / a.lean_checker_alt):
        if p.exists():
            lean = p
            break

    rep = {"run": str(run), "steps": len(steps), "lean_checker": None if lean is None else str(lean),
           "exact_steps": 0, "hash_bound": 0, "plan_match": 0, "command_match": 0,
           "python_checked": 0, "lean_checked": 0, "lean_attempted": 0,
           "failures": [], "samples": []}
    pending: list[tuple[int, object, tuple, tuple]] = []
    for i, (obs, cmd) in enumerate(steps):
        try:
            inst = enc.encode(obs)
        except EncodeError:
            continue
        rep["exact_steps"] += 1
        if _hash52(inst.sha256()) == float(getattr(cmd, "debug", {}).get("instance_h52", -1.0)):
            rep["hash_bound"] += 1
        else:
            rep["failures"].append({"step": i, "kind": "instance_hash"})
        solver = Solver(inst)
        plan, J, _ = solver.solve()
        dbg = getattr(cmd, "debug", {})
        dplan = tuple(int(dbg.get(f"plan{k}", -1)) for k in range(len(inst.blocks)))
        if dplan == plan:
            rep["plan_match"] += 1
        else:
            rep["failures"].append({"step": i, "kind": "plan", "audit": list(plan), "runtime": list(dplan)})
        u, p = first_command(inst, plan)
        want = command_floats(u, p)
        got = (float(cmd.steer), float(cmd.throttle), float(cmd.brake))
        if want == got:
            rep["command_match"] += 1
        else:
            rep["failures"].append({"step": i, "kind": "command", "audit": want, "published": got})
        pending.append((i, inst, plan))
    if a.limit:
        idx = np.linspace(0, len(pending) - 1, min(a.limit, len(pending))).round().astype(int)
        pending = [pending[j] for j in sorted(set(idx.tolist()))]
    if lean is not None or pending:
        tmp = run / "audit_certificates"
        tmp.mkdir(exist_ok=True)
        for i, inst, plan in pending:
            cert = certmod.certificate_for_instance(inst, plan)
            ok_py = certmod.check_certificate(cert)["ok"]
            if ok_py:
                rep["python_checked"] += 1
            else:
                rep["failures"].append({"step": i, "kind": "python_certificate"})
            if lean is None:
                continue
            path = tmp / f"step_{i:05d}.json"
            path.write_text(json.dumps(cert, sort_keys=True, separators=(",", ":")))
            rep["lean_attempted"] += 1
            r = subprocess.run([str(lean), str(path)], capture_output=True, text=True)
            accepted = r.returncode == 0 and '"ok":true' in r.stdout.replace(" ", "")
            if accepted:
                rep["lean_checked"] += 1
                rep["samples"].append(i)
            else:
                rep["failures"].append({"step": i, "kind": "lean_reject",
                                        "stdout": r.stdout[-400:], "stderr": r.stderr[-400:]})
    n = max(rep["exact_steps"], 1)
    rep["fractions"] = {
        "hash_bound": rep["hash_bound"] / n, "plan_match": rep["plan_match"] / n,
        "command_match": rep["command_match"] / n,
        "python_checked_of_sampled": rep["python_checked"] / max(rep["python_checked"] + len(
            [f for f in rep["failures"] if f["kind"] == "python_certificate"]), 1),
        "lean_checked_of_attempted": (rep["lean_checked"] / rep["lean_attempted"]
                                      if rep["lean_attempted"] else 0.0),
        "lean_checked_of_all": rep["lean_checked"] / n,
    }
    out = Path(a.out) if a.out else run / "timed_audit.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("steps", "exact_steps", "hash_bound", "plan_match",
                                          "command_match", "python_checked", "lean_checked",
                                          "lean_attempted")}, indent=1))
    print(json.dumps(rep["fractions"], indent=1))
    print(f"failures: {len(rep['failures'])}  -> {out}")
    return 0 if not rep["failures"] else 1


def _hash52(hexdigest: str) -> float:
    return float(int(hexdigest[:13], 16))


if __name__ == "__main__":
    raise SystemExit(main())
