"""Generate and check a proof-carrying actuator certificate at one station.

Example:
    python tools/actuator_certificate.py --track tracks/practice_basic.yaml \
        --station 20 --speed 8 --out tmp/actuator_cert.json
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
from autocross.actuator.certificate import (  # noqa: E402
    certificate_from_diag, check_certificate, write_certificate,
)
from autocross.actuator.build import model_from_observation  # noqa: E402
from autocross.actuator.solver import solve_sparse  # noqa: E402
from autocross.perception import RoadModelSensor  # noqa: E402
from autocross.track import load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="tracks/practice_basic.yaml")
    ap.add_argument("--station", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--out", default="tmp/actuator_certificate.json")
    ap.add_argument("--lean-checker", default="lean/.lake/build/bin/actuatorcheck.exe")
    a = ap.parse_args()
    track = load(a.track)
    vp = VehicleParams(mass=1845, wheelbase=3.0, track_width=1.67, length=4.79,
                       width=2.16, cg_to_front=1.45, cg_to_rear=1.55,
                       max_steer=0.6108652381980153, max_steer_rate=2.0, mu=0.95,
                       max_accel=5.2, max_decel=9.0, max_speed=40.0,
                       control_dt=0.05, actuation_delay_steps=0)
    sensor = RoadModelSensor(track, {"range_m": 40.0, "step_m": 1.0},
                             np.random.default_rng(0))
    sensor.s_hint = a.station
    p = track.line.point_at(a.station)
    yaw = track.line.heading_at(a.station)
    road = sensor.measure(p[0], p[1], yaw, a.station)
    obs = Observation(t=0.0, step=0, x=p[0], y=p[1], yaw=yaw, vx=a.speed, vy=0.0,
                      yaw_rate=0.0, ax=0.0, ay=0.0, speed=a.speed, steer=0.0,
                      throttle=0.0, brake=0.0, progress=a.station, lap=0,
                      timing_started=True, road=road)
    m = model_from_observation(obs, vp)
    if m is None:
        print("no model", file=sys.stderr)
        return 2
    first, cost, action, diag = solve_sparse(m)
    cert = certificate_from_diag(m, diag, claimed=cost, first_action=action)
    out = Path(a.out)
    write_certificate(cert, out)
    rep = check_certificate(cert)
    print(json.dumps({"python": rep, "first": None if first is None else first.as_dict(),
                      "action": action, "certificate": str(out)}, indent=2))
    if not rep["ok"]:
        return 1
    exe = ROOT / a.lean_checker
    if exe.exists():
        r = subprocess.run([str(exe), str(out)], capture_output=True, text=True)
        print(r.stdout.strip())
        return 0 if r.returncode == 0 else 1
    print("Lean checker not built; Python checker accepted", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
