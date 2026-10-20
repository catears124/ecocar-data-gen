"""Offline global oracle for the declared joint offset/speed lattice.

For each sampled recorded observation the tool:

1. builds a *high-resolution* declaration of the ``teacher-joint-1`` model
   (offset grid, speed grid, station grid, terminal cap);
2. solves it exactly with :func:`autocross.lineopt.dp.solve_fast`, which visits
   every declared state and successor;
3. constructs the baseline candidate: ``certline``'s published path, projected
   onto the same declared state lattice with a minimum-time speed profile;
4. checks that the baseline candidate is admissible under the *same* dynamics
   and constraints, and records ``J* <= J_baseline`` (the baseline-inclusion
   guarantee) whenever it is;
5. emits an independently checkable Bellman certificate for ``J*``.

The oracle is offline: it may use seconds per decision state.  It does not run
inside the controller.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "teams") not in sys.path:
    sys.path.insert(0, str(ROOT / "teams"))

from autocross.api import VehicleParams  # noqa: E402
from autocross.config import load_sim_config  # noqa: E402
from autocross.lineopt.dp import _speed_indices, solve_fast  # noqa: E402
from autocross.lineopt.path import speed_profile  # noqa: E402
from autocross.optimal.certificate import (certificate_from_solution,  # noqa: E402
                                           check_certificate)
from autocross.optimal.model import Model, State, isqrt, model_from_observation, time_cost  # noqa: E402
from autocross.optimal.solver import Solution  # noqa: E402


def vehicle_params() -> VehicleParams:
    cfg = load_sim_config()
    b = dict(cfg["vehicle"]["bicycle"])
    vc = cfg["vehicle"]
    return VehicleParams(
        float(b["mass"]), float(b["wheelbase"]), float(b["track_width"]),
        float(b["length"]), float(b["width"]), float(b["cg_to_front"]),
        float(b["wheelbase"] - b["cg_to_front"]),
        float(np.radians(vc["max_steer_deg"])), float(vc["steer_rate_limit"]),
        float(vc["mu"]), float(vc["max_accel"]), float(vc["max_decel"]),
        float(vc["max_speed"]), 0.05, 0)


def speed_grid(step_b: int = 10_400_000, v_min: float = 1.0, v_max: float = 30.0) -> tuple[int, ...]:
    """Speed-squared lattice aligned with the declared acceleration budget.

    A uniform speed grid is too coarse at high speed for the 1 m station step:
    ``2*a_acc*ds`` is the largest speed-squared change one station can produce.
    A b-uniform grid keeps every adjacent transition physically reachable.
    """
    b0 = int(round(v_min * 1000)) ** 2
    bmax = int(round(v_max * 1000)) ** 2
    return tuple(int(x) for x in range(b0, bmax + 1, int(step_b)))


def offsets_grid(step_mm: int = 150, half_mm: int = 1200) -> tuple[int, ...]:
    return tuple(range(-half_mm, half_mm + 1, step_mm))


def baseline_candidate(obs, vp, model: Model) -> tuple[list[State] | None, str]:
    """Certline's current path, projected to the oracle lattice, with the exact
    minimum-time feasible speed profile on that path.

    The candidate is constructed and checked entirely inside the declared
    model: corridor/obstacle/curvature at every station, the speed transition
    rule, and the terminal condition.  It is not a replay of an open-loop
    feedback sequence.
    """
    try:
        from certline import CertifiedRacingLine
        ctrl = CertifiedRacingLine({})
        ctrl.on_start(None, vp)  # type: ignore[arg-type] - on_start only uses vehicle limits
        cmd = ctrl.step(obs)
        path = np.asarray(getattr(cmd, "path", None), dtype=float)
        cl = np.asarray(obs.road.preview_centerline, dtype=float)
        if path.shape != cl.shape or cl.shape[0] < model.horizon + 1:
            return None, "baseline path shape does not match the oracle preview"
        path = path[: model.horizon + 1]
        cl = cl[: model.horizon + 1]
        tangent = np.gradient(cl, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        if np.any(norm <= 1e-9):
            return None, "baseline preview has a degenerate tangent"
        nrm = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
        raw = np.einsum("ij,ij->i", path - cl, nrm)
        O = np.asarray(model.offsets, dtype=np.int64)
        off = [int(O[int(np.argmin(np.abs(O - int(round(x * 1000.0)))))] ) for x in raw]

        # The offset path itself must satisfy the declared lateral rules.
        for i in range(model.horizon + 1):
            if not (model.corridor_ok(i, off[i]) and model.obstacle_ok(i, off[i])):
                return None, f"baseline path leaves the corridor at station {i}"
        for i in range(model.horizon):
            curv = model.path_curv(model.start.previous if i == 0 else off[i - 1],
                                   off[i], off[i + 1]) if False else None
            # ``path_curv`` needs (i, p, x, n); use the scalar declaration.
            curv = model.path_curv(i, off[max(i - 1, 0)], off[i], off[i + 1])
            if curv > model.kappa_max:
                return None, f"baseline path curvature exceeds kappa_max at station {i}"

        # Exact minimum-time speed DP on the fixed offset path.
        B = sorted({int(v) for v in model.speeds} | {int(model.start.b)})
        nb = len(B)
        INF = 2 ** 62
        values = [[INF] * nb for _ in range(model.horizon + 1)]
        choice = [[-1] * nb for _ in range(model.horizon)]
        for j, w in enumerate(B):
            if 0 <= w <= int(model.terminal_cap):
                values[model.horizon][j] = 0
        for i in range(model.horizon - 1, -1, -1):
            for j, b in enumerate(B):
                best, best_k = INF, -1
                for k, w in enumerate(B):
                    if not model.speed_ok(i, b, w):
                        continue
                    nxt = values[i + 1][k]
                    if nxt >= INF:
                        continue
                    cand = time_cost(model.gap(i), b, w) + nxt
                    if cand < best:
                        best, best_k = cand, k
                values[i][j] = best
                choice[i][j] = best_k
        try:
            j0 = B.index(int(model.start.b))
        except ValueError:
            return None, "start speed is not on the declared speed lattice"
        if values[0][j0] >= INF:
            return None, "no feasible speed profile on the baseline path"
        plan = [model.start]
        j = j0
        for i in range(model.horizon):
            k = choice[i][j]
            if k < 0:
                return None, f"baseline speed profile dead end at station {i}"
            plan.append(State(i + 1, plan[-1].current, off[i + 1], B[k]))
            j = k
        if model.terminal_value(plan[-1]) is None:
            return None, "baseline candidate terminal state is not viable"
        return plan, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"baseline candidate construction failed: {type(exc).__name__}: {exc}"


def plan_cost(model: Model, plan: list[State]) -> int | None:
    total = 0
    for a, b in zip(plan, plan[1:]):
        if not model.step_ok(a, b):
            return None
        total += time_cost(model.gap(a.station), a.b, b.b)
    return total if model.terminal_value(plan[-1]) is not None else None


def fast_to_solution(model: Model, fast) -> Solution:
    # The certificate universe is the forward-reachable set, not the full
    # Cartesian product.  Closure is checked by the certificate checker and the
    # DP value of every reachable state is already computed.
    univ: list[list[State]] = [[model.start]]
    for i in range(model.horizon):
        nxt = sorted({t for s in univ[i] for t, _ in model.actions(s)})
        univ.append(nxt)
    values: list[dict[State, int | None]] = []
    choices: list[dict[State, tuple[State, int]]] = []
    off_idx = {int(v): i for i, v in enumerate(model.offsets)}
    speeds, _ = _speed_indices(model, model.start.b)
    sp_idx = {int(v): i for i, v in enumerate(speeds)}
    for i in range(model.horizon + 1):
        table: dict[State, int | None] = {}
        for s in univ[i]:
            # State 0 is the start; later states are on the declared grids.
            if i == 0:
                pi = off_idx[int(s.previous)]
                xi = off_idx[int(s.current)]
                bi = sp_idx[int(s.b)]
            else:
                pi = off_idx[int(s.previous)]
                xi = off_idx[int(s.current)]
                bi = sp_idx[int(s.b)]
            v = int(fast.values[i][pi, xi, bi])
            table[s] = None if v >= 2 ** 61 else v
        values.append(table)
    for i in range(model.horizon):
        table: dict[State, tuple[State, int]] = {}
        for s in univ[i]:
            pi = off_idx[int(s.previous)]
            xi = off_idx[int(s.current)]
            bi = sp_idx[int(s.b)]
            a = int(fast.choices[i][pi, xi, bi])
            if a < 0:
                continue
            nb = len(speeds)
            ni, wi = divmod(a, nb)
            t = State(i + 1, int(model.offsets[xi]), int(model.offsets[ni]), int(speeds[wi]))
            table[s] = (t, time_cost(model.gap(i), s.b, t.b))
        choices.append(table)
    return Solution(model=model, values=values, choices=choices, plan=list(fast.plan),
                    cost=fast.value, univ=univ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run folder containing observations.pkl")
    ap.add_argument("--steps", default="10,30,50", help="comma-separated observation indices, or 'all'")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--offset-step", type=int, default=150)
    ap.add_argument("--offset-half", type=int, default=1200)
    ap.add_argument("--speed-step", type=int, default=10_400_000,
                    help="speed-squared lattice step (mm^2/s^2)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--certificates", action="store_true", help="write one certificate per sampled step")
    a = ap.parse_args()
    run = Path(a.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    steps = pickle.loads((run / "observations.pkl").read_bytes())
    idxs = range(len(steps)) if a.steps == "all" else [int(x) for x in a.steps.split(",") if x.strip()]
    vp = vehicle_params()
    offs = offsets_grid(a.offset_step, a.offset_half)
    speeds = speed_grid(a.speed_step)
    a_speed_step = int(a.speed_step)
    out_dir = Path(a.out) if a.out else run / "oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"run": str(run), "horizon": a.horizon, "offsets": len(offs),
              "speeds": len(speeds), "steps": []}
    for idx in idxs:
        if idx >= len(steps):
            continue
        obs = steps[idx][0]
        t0 = time.perf_counter()
        sb = int(round(max(float(obs.speed), 0.0) * 1000.0)) ** 2
        nb = int(a_speed_step)
        speeds_i = tuple(sorted(set(speeds)
                               | {max(sb + k * nb, 1) for k in range(-8, 9)}))
        model = model_from_observation(obs, vp, horizon=a.horizon, offsets=offs, speeds=speeds_i)
        if model is None:
            report["steps"].append({"step": idx, "error": "observation outside model domain"})
            continue
        fast = solve_fast(model)
        solve_ms = (time.perf_counter() - t0) * 1000.0
        row: dict = {"step": int(idx), "t": float(obs.t), "speed": float(obs.speed),
                     "feasible": bool(fast.feasible), "value": fast.value,
                     "solve_ms": solve_ms, "states": int(len(model.offsets) ** 2 * len(model.speeds))}
        base_plan, why = baseline_candidate(obs, vp, model)
        if base_plan is not None:
            base_cost = plan_cost(model, base_plan)
            row["baseline_cost"] = base_cost
            row["baseline_dominance"] = bool(
                fast.feasible and base_cost is not None and int(fast.value) <= int(base_cost))
        else:
            row["baseline_cost"] = None
            row["baseline_dominance"] = None
            row["baseline_reason"] = why
        if fast.feasible:
            try:
                sol = fast_to_solution(model, fast)
                cert = certificate_from_solution(sol)
                cert["baseline"] = {
                    "candidate": "certline-published-path/lattice-projection",
                    "admissible": base_plan is not None,
                    "cost": row.get("baseline_cost"),
                    "dominance": row.get("baseline_dominance"),
                    "reason": why,
                }
                chk = check_certificate(cert)
                row["certificate_ok"] = bool(chk.get("ok"))
                row["certificate_errors"] = chk.get("errors", [])[:3]
                if a.certificates:
                    p = out_dir / f"oracle_{idx:06d}.json"
                    p.write_text(json.dumps(cert, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                    row["certificate"] = str(p)
            except Exception as exc:  # noqa: BLE001
                row["certificate_ok"] = False
                row["certificate_error"] = f"{type(exc).__name__}: {exc}"
        report["steps"].append(row)
        print(f"step {idx:4d} t={obs.t:6.2f} v={obs.speed:5.2f} "
              f"value={fast.value} base={row.get('baseline_cost')} "
              f"dom={row.get('baseline_dominance')} cert={row.get('certificate_ok')} "
              f"{solve_ms:.0f}ms", flush=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"-> {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
