"""Offline audit: certify the executed first action of the actuator controller.

For each requested station the tool builds the declared model from a real
policy observation, solves it with the runtime solver (numba when available,
else numpy), builds the certificate bound to exactly that instance and executed
first action, and runs the independent Python checker and the compiled Lean
checker ``actuatorcheck``.  Only here, offline, can a step reach
``STATUS_LEAN_CHECKED``.  With ``--corrupt`` every adversarial mutation of
``autocross.actuator.certificate.CORRUPTIONS`` must be rejected (Lean does not
read the transport hash, so ``sha`` is required of Python only).

Exit status 0 iff every certificate is accepted (and every corruption rejected).

Example:
    python tools/actuator_certificate.py --track tracks/practice_basic.yaml \
        --station 20 --station 35 --speed 8 --out tmp/actuator_cert --require-lean --corrupt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocross.api import Observation, VehicleParams  # noqa: E402
from autocross.actuator.build import model_from_observation  # noqa: E402
from autocross.actuator.certificate import (  # noqa: E402
    CORRUPTIONS, certificate_for_action, check_certificate, corrupt, instance_sha256, sha52,
    write_certificate,
)
from autocross.actuator.numba_solver import NUMBA_AVAILABLE, solve_numba  # noqa: E402
from autocross.actuator.solver import solve  # noqa: E402
from autocross.certstatus import (  # noqa: E402
    STATUS_LEAN_CHECKED, STATUS_PY_CHECKED, STATUS_SOLVER, status_debug,
)

VEHICLE = VehicleParams(mass=1845, wheelbase=3.0, track_width=1.67, length=4.79,
                        width=2.16, cg_to_front=1.45, cg_to_rear=1.55,
                        max_steer=0.6108652381980153, max_steer_rate=2.0, mu=0.95,
                        max_accel=5.2, max_decel=9.0, max_speed=40.0,
                        control_dt=0.05, actuation_delay_steps=0)


def observation(track_path: str | Path, station: float, speed: float) -> Observation:
    """A noiseless policy observation on the centerline at ``station``."""
    from autocross.perception import RoadModelSensor
    from autocross.track import load
    track = load(track_path)
    sensor = RoadModelSensor(track, {"range_m": 40.0, "step_m": 1.0}, np.random.default_rng(0))
    sensor.s_hint = station
    p = track.line.point_at(station)
    yaw = track.line.heading_at(station)
    road = sensor.measure(p[0], p[1], yaw, station)
    return Observation(t=0.0, step=0, x=p[0], y=p[1], yaw=yaw, vx=speed, vy=0.0,
                       yaw_rate=0.0, ax=0.0, ay=0.0, speed=speed, steer=0.0,
                       throttle=0.0, brake=0.0, progress=station, lap=0,
                       timing_started=True, road=road)


def find_checker(name: str = "actuatorcheck") -> Path | None:
    bindir = ROOT / "lean" / ".lake" / "build" / "bin"
    for cand in (bindir / f"{name}.exe", bindir / name):
        if cand.is_file():
            return cand
    return None


def lean_verdict(exe: Path, path: Path) -> tuple[bool, str]:
    r = subprocess.run([str(exe), str(path)], capture_output=True, text=True, timeout=600)
    ok = r.returncode == 0 and '"ok":true' in r.stdout
    return ok, r.stdout.strip()


def runtime_solve(model):
    """The solver the controller would run outside the sandbox."""
    if NUMBA_AVAILABLE:
        return solve_numba(model), "numba"
    return solve(model), "numpy"


def audit_station(track: str, station: float, speed: float, out: Path, exe: Path | None,
                  do_corrupt: bool) -> dict:
    rec: dict = {"station": station, "speed": speed, "ok": False}
    m = model_from_observation(observation(track, station, speed), VEHICLE)
    if m is None:
        rec["error"] = "no model"
        return rec
    (first, cost, action, _diag), solver = runtime_solve(m)
    rec.update(solver=solver, cost=cost, action=action,
               instance_sha52=sha52(instance_sha256(m)), **status_debug(STATUS_SOLVER))
    if action is None:
        rec["error"] = "infeasible instance: no executed action to certify"
        return rec
    cert = certificate_for_action(m, action)
    if cert is None or cert["firstAction"] != list(action) or cert["claimed"] != cost:
        rec["error"] = "runtime action is not optimal per the certificate builder"
        return rec
    path = write_certificate(cert, out / f"station_{station:g}.json")
    rec.update(certificate=str(path), cert_sha52=sha52(cert["certificate_sha256"]))
    py = check_certificate(cert)
    rec["python"] = py
    if not py["ok"]:
        return rec
    rec.update(status_debug(STATUS_PY_CHECKED))
    if exe is None:
        rec["lean"] = "not built"
        rec["ok"] = True
        return rec
    lean_ok, lean_out = lean_verdict(exe, path)
    rec["lean"] = lean_out
    if not lean_ok:
        return rec
    rec.update(status_debug(STATUS_LEAN_CHECKED))
    rec["ok"] = True
    if do_corrupt:
        rejected = {}
        for kind in CORRUPTIONS:
            bad = corrupt(cert, kind)
            bpath = write_certificate(bad, out / f"station_{station:g}.corrupt_{kind}.json")
            py_rej = not check_certificate(bad)["ok"]
            lean_rej = True if kind == "sha" else not lean_verdict(exe, bpath)[0]
            rejected[kind] = {"python": py_rej, "lean": lean_rej}
            rec["ok"] &= py_rej and lean_rej
        rec["corruptions"] = rejected
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="tracks/practice_basic.yaml")
    ap.add_argument("--station", type=float, action="append",
                    help="arc-length station(s) in metres (repeatable; default 20)")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--out", default="tmp/actuator_certificates")
    ap.add_argument("--require-lean", action="store_true",
                    help="fail when the compiled Lean checker is missing")
    ap.add_argument("--corrupt", action="store_true",
                    help="also require rejection of every adversarial corruption")
    a = ap.parse_args(argv)
    exe = find_checker()
    if exe is None and a.require_lean:
        print("actuatorcheck is not built (cd lean && lake build actuatorcheck)", file=sys.stderr)
        return 2
    if exe is None and a.corrupt:
        print("--corrupt needs the compiled Lean checker", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = [audit_station(a.track, s, a.speed, out, exe, a.corrupt) for s in (a.station or [20.0])]
    print(json.dumps(recs, indent=2, default=str))
    return 0 if recs and all(r["ok"] for r in recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
