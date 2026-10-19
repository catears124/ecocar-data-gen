"""Emit reference instances and their exact plan costs for the Lean/compiled checkers.

    python tools/timed_vectors.py --out lean/timed_vectors.json

The file contains, for each instance: the instance JSON, the ordered list of plans
(``model.all_plans`` order), every plan's ``(V, negP, E)`` from the scalar reference
rollout, the optimum (first minimiser) and its first command.  ``tests/test_timed_parity.py``
and the Lean test suite compare against it, so an implementation that disagrees on any
plan cost is caught immediately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocross.timed.model import (DEFAULT_BLOCKS, DEFAULT_CONSTS, Block, Instance, State,  # noqa: E402
                                   all_plans, evaluate, first_command, solve_exhaustive, well_formed)


def inst(k, wl, wr, obstacles, start, p_prev, blocks, consts=None):
    c = dict(DEFAULT_CONSTS)
    c.update(consts or {})
    return Instance(c, tuple(k), tuple(wl), tuple(wr), tuple(obstacles), start, p_prev, blocks)


def tiny_blocks():
    return (Block(1, (-100, 0, 100), (-1000, 0, 1000)),
            Block(2, (-200, 0, 200), (-1000, 0, 1000)))


def cases() -> list[tuple[str, Instance]]:
    out = []
    n = 6
    straight = [0] * 12
    out.append(("tiny_straight", inst(straight, [3000] * 12, [3000] * 12, [],
                                      State(0, 0, 0, 9000, 0, 0, 0, 0), 0, tiny_blocks())))
    out.append(("tiny_left_turn", inst([0, 0, 40000, 60000, 60000, 40000] + [0] * 6,
                                       [2500] * 12, [3500] * 12, [],
                                       State(0, 300, -20000, 12000, 0, 0, 0, 0), 0, tiny_blocks())))
    out.append(("tiny_right_turn_asym", inst([0, 0, -35000, -55000, -55000, -30000] + [0] * 6,
                                             [3500] * 12, [2500] * 12, [],
                                             State(0, -400, 25000, 11000, 0, 0, 0, 0), 300, tiny_blocks())))
    out.append(("tiny_obstacle_left", inst(straight, [3000] * 12, [3000] * 12, [(3000, 5000, 1200, 4000)],
                                           State(0, 0, 0, 10000, 0, 0, 0, 0), 0, tiny_blocks())))
    out.append(("tiny_obstacle_right", inst(straight, [3000] * 12, [3000] * 12, [(3000, 5000, -4000, -1200)],
                                            State(0, 0, 0, 10000, 0, 0, 0, 0), 0, tiny_blocks())))
    out.append(("tiny_negative_steer_optimum", inst([0, 0, 0, 0, 0, 0] + [0] * 6,
                                                    [3000] * 12, [3000] * 12, [],
                                                    State(0, 400, 0, 10000, 0, 0, 300, 180000),
                                                    0, tiny_blocks())))
    out.append(("tiny_slow", inst(straight, [2500] * 12, [2500] * 12, [],
                                  State(0, 0, 0, 1500, 0, 0, 0, 0), 0, tiny_blocks())))
    out.append(("tiny_three_blocks", inst([0] * 20, [3000] * 20, [3000] * 20, [],
                                          State(0, 0, 0, 8000, 0, 0, 0, 0), 0,
                                          (Block(1, (-100, 0, 100), (-1000, 0, 1000)),
                                           Block(2, (-200, 0, 200), (-1000, 0, 1000)),
                                           Block(3, (-300, 0, 300), (-1000, 0, 1000))))))
    out.append(("full_default", inst([0] * 8 + [30000] * 8 + [-40000] * 8 + [50000] * 17,
                                     [3200] * 41, [3200] * 41, [(12000, 14000, -800, 900)],
                                     State(0, 250, -15000, 14000, 0, 0, 0, 0), 200, DEFAULT_BLOCKS)))
    # varying corridor widths close to the start: a checker that indexes widths at
    # the wrong station (e.g. always station 0) accepts every plan here only if the
    # widths are constant, so this case pins the body-point station arithmetic.
    out.append(("tiny_varying_widths", inst([0] * 12, [2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500],
                                            [3500, 3400, 3300, 3200, 3100, 3000, 2900, 2800, 2700, 2600, 2500, 2400],
                                            [], State(0, 300, 20000, 10000, 0, 0, 0, 0), 0, tiny_blocks())))
    out.append(("full_obstacle", inst([0] * 12 + [45000] * 12 + [0] * 17,
                                      [3000] * 41, [3300] * 41, [(9000, 11000, 500, 2400)],
                                      State(0, -300, 10000, 13000, 0, 0, 0, 0), -400, DEFAULT_BLOCKS)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lean/timed_vectors.json")
    a = ap.parse_args()
    doc = {"schema": "autocross.timed.vectors", "schema_version": 1, "cases": []}
    for name, m in cases():
        err = well_formed(m)
        if err:
            raise SystemExit(f"{name}: {err}")
        plans = list(all_plans(m))
        costs = [list(evaluate(m, p)) for p in plans]
        best_plan, best = solve_exhaustive(m)
        u, p = first_command(m, best_plan)
        doc["cases"].append({
            "name": name, "instance": m.to_json(), "plans": [list(x) for x in plans], "costs": costs,
            "best": list(best_plan), "best_cost": list(best), "first": [u, p],
        })
        print(f"{name}: {len(plans)} plans, best {best_plan} {best} first {(u, p)}")
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
