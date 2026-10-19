"""Offline audit: certify the executed first command of the ``timed-1`` controller.

Builds a noiseless policy observation on a real track (same construction as
``tools/actuator_certificate.py``), encodes it with
:class:`autocross.timed.encode.Encoder`, solves the declared model exactly with
:class:`autocross.timed.vector.Solver`, writes the proof-carrying certificate for
the optimum, and requires acceptance by the independent Python checker and (when
built) by the compiled Lean checker ``timedcheck``.

Exit status 0 iff every certificate is accepted.

Example:
    python tools/timed_certificate.py --track tracks/practice_basic.yaml \
        --station 20 --speed 8 --out tmp/timed_cert.json
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

from autocross.api import Observation  # noqa: E402
from autocross.timed.certificate import (  # noqa: E402
    certificate_for_instance, check_certificate, write_certificate,
)
from autocross.timed.encode import EncodeError, Encoder  # noqa: E402
from autocross.timed.vector import DomainOverflow, Solver  # noqa: E402


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


def find_checker(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    bindir = ROOT / "lean" / ".lake" / "build" / "bin"
    for cand in (bindir / "timedcheck.exe", bindir / "timedcheck"):
        if cand.is_file():
            return cand
    return None


def lean_verdict(exe: Path, path: Path) -> tuple[bool, str]:
    r = subprocess.run([str(exe), str(path)], capture_output=True, text=True, timeout=900)
    ok = r.returncode == 0 and '"ok":true' in r.stdout
    return ok, r.stdout.strip()


def audit_station(track: str, station: float, speed: float, out: Path,
                  exe: Path | None) -> dict:
    rec: dict = {"station": station, "speed": speed, "ok": False}
    try:
        inst = Encoder().encode(observation(track, station, speed))
    except EncodeError as e:
        rec["error"] = f"encoder: {e}"
        return rec
    try:
        plan, claimed, stats = Solver(inst).solve()
    except DomainOverflow as e:
        rec["error"] = f"solver domain: {e}"
        return rec
    cert = certificate_for_instance(inst, plan)
    path = out
    write_certificate(cert, path)
    rec.update(plan=list(plan), claimed=list(claimed), first=cert["first"],
               instance_sha256=cert["instance_sha256"], certificate=str(path),
               plans=stats["leaves"][-1] if stats.get("leaves") else None)
    py = check_certificate(cert)
    rec["python"] = py
    if not py["ok"]:
        return rec
    if exe is None:
        rec["lean"] = "not built"
        rec["ok"] = True
        return rec
    lean_ok, lean_out = lean_verdict(exe, path)
    rec["lean"] = lean_out
    if not lean_ok:
        return rec
    rec["ok"] = True
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="tracks/practice_basic.yaml")
    ap.add_argument("--station", type=float, action="append",
                    help="arc-length station(s) in metres (repeatable; default 20)")
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--out", default="tmp/timed_cert.json",
                    help="certificate path (one station) or directory prefix (several)")
    ap.add_argument("--lean-checker", default=None,
                    help="compiled timedcheck executable (default: lean/.lake/build/bin/timedcheck)")
    a = ap.parse_args(argv)
    exe = find_checker(a.lean_checker)
    out = Path(a.out)
    stations = a.station or [20.0]
    if len(stations) == 1:
        paths = [out]
    else:
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / f"station_{s:g}.json" for s in stations]
    recs = [audit_station(a.track, s, a.speed, p, exe) for s, p in zip(stations, paths)]
    print(json.dumps(recs, indent=2, default=str))
    return 0 if recs and all(r["ok"] for r in recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
