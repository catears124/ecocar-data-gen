"""Exact backward dynamic program for :mod:`autocross.optimal.model`.

The solver is deliberately ordinary: every state, every admissible successor and
every integer cost is enumerated.  No heuristic pruning participates in the
returned optimum.  The only optimisation is algebraic factoring of the speed
successor loop (the cost and the speed constraints depend on ``(b,w)`` only, and
the cap depends on ``(p,x,n)``), which changes runtime, not the state space.

Deterministic tie-breaking: actions are visited in the declared grid order
(offsets outer, speeds inner) and the first minimum is kept.  The plan is
therefore the lexicographically first optimal plan in that order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .model import Model, State, time_cost


@dataclass
class Solution:
    model: Model
    # value[i][state] is the optimal remaining declared cost, or None if
    # no admissible continuation exists.
    values: list[dict[State, int | None]]
    # choice[i][state] = (successor, cost) for the first optimal action.
    choices: list[dict[State, tuple[State, int]]]
    plan: list[State]
    cost: int | None
    # Forward-reachable certificate universe: univ[i] is the set of states at
    # station i reachable from the start by admissible transitions.  The
    # certificate contains rows exactly for these states; closure is checked.
    univ: list[list[State]]

    @property
    def feasible(self) -> bool:
        return self.cost is not None


def forward_universe(model: Model) -> list[list[State]]:
    """States reachable from the start by admissible transitions, per station.

    Soundness of the sparse certificate rests on closure: the next set is
    exactly the union of the recomputed successor sets of the current set.
    """
    univ: list[list[State]] = [[model.start]]
    for i in range(model.horizon):
        nxt = sorted({t for s in univ[i] for t, _ in model.actions(s)})
        univ.append(nxt)
    return univ


def solve(model: Model) -> Solution:
    """Exact sparse backward induction over the forward-reachable universe.

    Any admissible plan from the start stays in ``univ`` by construction, so
    restricting the DP to it changes neither the optimum nor feasibility.
    """
    m = model.horizon
    univ = forward_universe(model)
    values: list[dict[State, int | None]] = [dict() for _ in range(m + 1)]
    choices: list[dict[State, tuple[State, int]]] = [dict() for _ in range(m + 1)]

    for s in univ[m]:
        values[m][s] = model.terminal_value(s)

    for i in range(m - 1, -1, -1):
        for s in univ[i]:
            best: int | None = None
            best_act: tuple[State, int] | None = None
            # model.actions returns successors in the declared deterministic
            # order; strict "<" keeps the first optimal action.
            for t, c in model.actions(s):
                nv = values[i + 1].get(t)
                if nv is None:
                    continue
                cand = c + nv
                if best is None or cand < best:
                    best = cand
                    best_act = (t, c)
            values[i][s] = best
            if best_act is not None:
                choices[i][s] = best_act

    plan: list[State] = [model.start]
    total: int | None = values[0].get(model.start)
    s = model.start
    if total is not None:
        while s.station < m:
            act = choices[s.station].get(s)
            if act is None:
                total = None
                break
            t, _c = act
            plan.append(t)
            s = t
        if total is not None and model.terminal_value(plan[-1]) is None:
            total = None
    if total is None:
        plan = []
    return Solution(model=model, values=values, choices=choices, plan=plan, cost=total, univ=univ)


def enumerate_plans(model: Model, limit: int | None = None) -> Iterator[tuple[list[State], int]]:
    """Independent exhaustive enumeration (small instances only).

    Yields every admissible plan from the start to the horizon with its declared
    cost.  Used by tests to check the DP against a separately written recursion.
    """
    count = 0

    def rec(s: State, path: list[State], cost: int) -> Iterator[tuple[list[State], int]]:
        nonlocal count
        if s.station == model.horizon:
            if model.terminal_value(s) is not None:
                yield list(path), cost
            return
        for t, c in model.actions(s):
            yield from rec(t, path + [t], cost + c)

    yield from rec(model.start, [model.start], 0)
