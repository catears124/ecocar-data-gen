"""Post-run report figure (report.png) and course preview."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .api import CONE_LEFT, CONE_ORANGE, CONE_RIGHT  # noqa: E402

CONE_STYLE = {CONE_LEFT: ("#1f5fd6", "left"), CONE_RIGHT: ("#e8b400", "right"), CONE_ORANGE: ("#ff7a00", "marker")}


def draw_track(ax, track, gate_spacing=20.0, show_obstacles=True):
    ax.plot(*track.center.T, color="0.75", lw=0.8, ls="--", zorder=1)
    for boundary in (track.left, track.right):
        b = np.vstack((boundary, boundary[:1])) if track.closed else boundary
        ax.plot(*b.T, color="0.55", lw=0.6, zorder=1)
    for c, (col, _) in CONE_STYLE.items():
        m = track.cone_colors == c
        ax.scatter(*track.cones[m].T, s=9, c=col, zorder=3, linewidths=0)
    if show_obstacles and len(track.obstacle_cones):
        ax.scatter(*track.obstacle_cones.T, s=22, c="#ff3b00", marker="^", zorder=4, label="unmapped obstacle")
    for p in track.gate_progress(gate_spacing):
        ln = track.cross_line(track.s_of_progress(p))
        ax.plot(*ln.T, color="#6a6a6a", lw=0.4, alpha=0.5, zorder=1)
    marks = [(track.start_progress, "START/FINISH")] if track.closed else         [(track.start_progress, "START"), (track.finish_progress, "FINISH")]
    for p, lab in marks:
        ln = track.cross_line(track.s_of_progress(p))
        ax.plot(*ln.T, color="k", lw=2, zorder=2)
        ax.annotate(lab, ln.mean(0), fontsize=7, ha="center", va="bottom", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def preview_track(track, out_path, gate_spacing=20.0):
    fig, ax = plt.subplots(figsize=(9, 8))
    draw_track(ax, track, gate_spacing)
    x, y, yaw = track.spawn_pose()
    ax.arrow(x, y, 6 * np.cos(yaw), 6 * np.sin(yaw), width=0.6, color="crimson", zorder=6)
    ttl = f"{track.name}: {track.length:.0f} m, {'closed x%d laps' % track.laps if track.closed else 'open'}, {len(track.cones)} cones"
    ax.set_title(ttl)
    if len(track.obstacle_cones):
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _read_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        rows = [[float(v) if v != "" else np.nan for v in row] for row in r]
    a = np.array(rows, dtype=float).reshape(-1, len(head))
    return {h: a[:, i] for i, h in enumerate(head)}


def plot_run(run_dir, track=None):
    run_dir = Path(run_dir)
    s = json.loads((run_dir / "summary.json").read_text())
    if track is None:
        from .track import load
        track = load(s["track"]["path"])
    log = _read_csv(run_dir / "telemetry.csv")
    events = json.loads((run_dir / "events.json").read_text())
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[:, 0])
    draw_track(ax, track, s["config"]["rules"]["gate_spacing_m"])
    if len(log["t"]):
        sc = ax.scatter(log["x"], log["y"], c=log["speed"], s=2, cmap="viridis", zorder=4)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="speed [m/s]")
    marks = {"cone_hit": ("x", "red", "cone hit"), "off_course": ("o", "magenta", "off course"),
             "missed_gate": ("s", "black", "missed gate"), "dnf": ("*", "red", "DNF")}
    for kind, (mk, col, lab) in marks.items():
        ev = [e for e in events if e["kind"] == kind]
        if ev:
            style = {"facecolors": "none", "edgecolors": col} if mk in "os" else {"c": col}
            ax.scatter([e["x"] for e in ev], [e["y"] for e in ev], marker=mk, s=50, zorder=6,
                       label=f"{lab} ({len(ev)})", **style)
    if any(e["kind"] in marks for e in events):
        ax.legend(loc="best", fontsize=8)
    m, scr = s["metrics"], s["score"]
    c = m["completion"]
    status = f"{c['time_s']:.2f} s" if c["finished"] else f"DNF: {c['dnf_reason']}"
    ax.set_title(f"{s['team']} | {s['track']['name']} | {s['backend']} | {status} | technical {scr['technical_points']:.1f}")

    t = log["t"]
    ctrl = log["ctrl_step"] > 0
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(t, log["speed"], lw=1, label="speed")
    if np.isfinite(log["target_speed"]).any():
        a1.plot(t, log["target_speed"], lw=0.8, ls="--", label="target")
    a1.set_ylabel("m/s")
    a1.legend(fontsize=7, loc="upper right")
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(t, log["path_err"], lw=1, label="path error")
    a2.plot(t, log["lateral"], lw=0.8, alpha=0.7, label="centerline offset")
    a2.plot(t, log["boundary_margin"], lw=0.8, alpha=0.7, label="boundary margin")
    a2.axhline(0, color="k", lw=0.5)
    a2.set_ylabel("m")
    a2.legend(fontsize=7, loc="upper right")
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    a3.plot(t, log["cmd_steer"], lw=0.8, label="steer cmd")
    a3.plot(t, log["app_steer"], lw=0.8, label="steer applied")
    a3.plot(t, log["cmd_throttle"], lw=0.6, alpha=0.7, label="throttle")
    a3.plot(t, -log["cmd_brake"], lw=0.6, alpha=0.7, label="-brake")
    a3.set_ylabel("norm.")
    a3.legend(fontsize=7, loc="upper right", ncol=2)
    a4 = fig.add_subplot(gs[3, 1], sharex=a1)
    a4.plot(t[ctrl], log["compute_ms"][ctrl], ".", ms=2, label="step wall ms")
    a4.plot(t[ctrl], log["cpu_ms"][ctrl], ".", ms=2, alpha=0.6, label="step cpu ms")
    a4.axhline(m["compute"]["deadline_ms"], color="r", lw=0.8, ls="--", label="deadline")
    a4.set_ylabel("ms")
    a4.set_xlabel("t [s]")
    a4.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "report.png", dpi=110)
    plt.close(fig)
