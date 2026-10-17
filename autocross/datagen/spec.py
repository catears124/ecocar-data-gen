"""Scenario-family specifications for the imitation-data generator.

A *family* declares a parameterised decision situation: its factors (coverage dimensions) with
bounds and bins, nuisance parameters that vary inside a cell, the formal validity constraints,
how a parameter set becomes a course YAML + simulator overrides, and the decision window in
which expert labels count as demonstrations.

Numbers are quantised to ``1 / SCALE`` of their unit. The canonical form of a scenario (used
for its ID, its hash and the Lean interchange) stores every numeric parameter as the integer
``round(value * SCALE)``, so Python and Lean compare exactly the same integers.

Only ``brake_turn_obstacle`` exists today. A new family subclasses ``Family``, fills in the
same fields and is registered in ``FAMILIES``; the manifest, generator, validator and Lean
checker are family-agnostic except for ``formal_violations`` (mirrored in ``lean/Corpus``)."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

SCALE = 1000  # parameters are stored with 1e-3 resolution (mm, mm/s, millidegrees, ...)


def to_fixed(value: float) -> int:
    return int(round(float(value) * SCALE))


def is_quantised(value: float) -> bool:
    return abs(float(value) * SCALE - round(float(value) * SCALE)) < 1e-6


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_text(canonical_json(obj))


@dataclass(frozen=True)
class Bin:
    label: str
    lo: float
    hi: float
    hi_inclusive: bool = False

    def contains_fixed(self, v: int) -> bool:
        lo, hi = to_fixed(self.lo), to_fixed(self.hi)
        return lo <= v <= hi if self.hi_inclusive else lo <= v < hi


@dataclass(frozen=True)
class Factor:
    """A coverage dimension: numeric with ordered bins, or categorical with levels."""
    name: str
    unit: str
    description: str
    bins: tuple[Bin, ...] = ()
    levels: tuple[str, ...] = ()

    @property
    def numeric(self) -> bool:
        return bool(self.bins)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(b.label for b in self.bins) if self.numeric else self.levels

    @property
    def lo(self) -> float:
        return self.bins[0].lo

    @property
    def hi(self) -> float:
        return self.bins[-1].hi

    def bin_of(self, value) -> str | None:
        if not self.numeric:
            return value if value in self.levels else None
        v = to_fixed(value)
        for b in self.bins:
            if b.contains_fixed(v):
                return b.label
        return None

    def bin(self, label: str) -> Bin:
        return next(b for b in self.bins if b.label == label)

    def to_dict(self) -> dict:
        d = {"name": self.name, "unit": self.unit, "description": self.description,
             "kind": "numeric" if self.numeric else "categorical"}
        if self.numeric:
            d["bins"] = [{"label": b.label, "lo": b.lo, "hi": b.hi, "hi_inclusive": b.hi_inclusive,
                          "lo_fixed": to_fixed(b.lo), "hi_fixed": to_fixed(b.hi)} for b in self.bins]
        else:
            d["levels"] = list(self.levels)
        return d


@dataclass(frozen=True)
class Nuisance:
    """Varied (seeded) inside a coverage cell; bounded, but not a coverage dimension."""
    name: str
    unit: str
    lo: float
    hi: float
    description: str

    def to_dict(self) -> dict:
        return {"name": self.name, "unit": self.unit, "lo": self.lo, "hi": self.hi,
                "lo_fixed": to_fixed(self.lo), "hi_fixed": to_fixed(self.hi), "description": self.description}


class Family:
    name: str = ""
    version: int = 0
    factors: tuple[Factor, ...] = ()
    nuisance: tuple[Nuisance, ...] = ()
    constants: dict = {}
    expected_effects: dict = {}

    # ---------------------------------------------------------------- canonical form
    def factor(self, name: str) -> Factor:
        return next(f for f in self.factors if f.name == name)

    def param_names(self) -> list[str]:
        return [f.name for f in self.factors] + [n.name for n in self.nuisance]

    def canonical_params(self, params: dict) -> dict:
        out = {}
        for f in self.factors:
            out[f.name] = to_fixed(params[f.name]) if f.numeric else str(params[f.name])
        for n in self.nuisance:
            out[n.name] = to_fixed(params[n.name])
        return out

    def scenario_hash(self, params: dict) -> str:
        return sha256_obj({"family": self.name, "family_version": self.version,
                           "params": self.canonical_params(params)})

    def scenario_id(self, params: dict) -> str:
        return f"{self.short}-v{self.version}-{self.scenario_hash(params)[:16]}"

    def bins_of(self, params: dict) -> dict:
        return {f.name: f.bin_of(params[f.name]) for f in self.factors}

    def spec_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "short": self.short, "scale": SCALE,
                "factors": [f.to_dict() for f in self.factors], "nuisance": [n.to_dict() for n in self.nuisance],
                "constants": dict(self.constants), "constants_fixed": {k: to_fixed(v) for k, v in self.constants.items()},
                "expected_effects": dict(self.expected_effects)}

    # ---------------------------------------------------------------- to implement
    short: str = ""

    def schema_violations(self, params: dict) -> list[str]:
        """Shape errors: missing/extra keys, wrong types, values not on the 1e-3 grid."""
        problems = []
        names = set(self.param_names())
        for k in sorted(set(params) - names):
            problems.append(f"schema:unknown_param:{k}")
        for f in self.factors:
            if f.name not in params:
                problems.append(f"schema:missing_param:{f.name}")
            elif f.numeric and (not isinstance(params[f.name], (int, float)) or isinstance(params[f.name], bool)
                                or not math.isfinite(params[f.name]) or not is_quantised(params[f.name])):
                problems.append(f"schema:bad_value:{f.name}")
            elif not f.numeric and not isinstance(params[f.name], str):
                problems.append(f"schema:bad_value:{f.name}")
        for n in self.nuisance:
            v = params.get(n.name)
            if v is None:
                problems.append(f"schema:missing_param:{n.name}")
            elif not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not is_quantised(v):
                problems.append(f"schema:bad_value:{n.name}")
        return problems

    def formal_violations(self, params: dict) -> list[str]:
        raise NotImplementedError

    def simulator_violations(self, params: dict) -> list[str]:
        raise NotImplementedError

    def violations(self, params: dict) -> list[str]:
        """All rejection reasons, in check order: schema, formal (Lean-modelled), simulator."""
        problems = self.schema_violations(params)
        if problems:
            return problems
        problems = self.formal_violations(params)
        if problems:
            return problems
        return self.simulator_violations(params)


# ===================================================================== brake_turn_obstacle
class BrakeTurnObstacle(Family):
    """Straight approach at a declared speed with limited sight distance, then a constant-radius
    turn with an unmapped obstacle cone group inside the turn. The turn and the obstacle are
    beyond the road model's preview range at the hand-over point, so both appear while the
    expert is driving: it has to decide when to brake, how fast to take the turn and which side
    of the obstacle to use, from what the road model shows at that moment."""
    name = "brake_turn_obstacle"
    short = "bto"
    version = 1
    factors = (
        Factor("turn_radius_m", "m", "centerline radius of the turn",
               bins=(Bin("tight", 9.0, 14.0), Bin("medium", 14.0, 22.0), Bin("open", 22.0, 35.0, True))),
        Factor("turn_direction", "-", "turn direction (left = positive curvature)", levels=("left", "right")),
        Factor("preview_m", "m", "road-model preview range / sight distance (road_model.range_m); event = 40 m, "
               "shorter = the harness robustness sweep's reduced sight distance",
               bins=(Bin("short", 15.0, 25.0), Bin("nominal", 25.0, 40.0, True))),
        Factor("approach_speed_mps", "m/s", "speed at the hand-over point (start of the decision window)",
               bins=(Bin("low", 8.0, 12.0), Bin("mid", 12.0, 16.0), Bin("high", 16.0, 20.0, True))),
        Factor("obstacle_side", "-", "obstacle against the inside or the outside boundary of the turn",
               levels=("inside", "outside")),
        Factor("obstacle_clearance_m", "m", "free width between the obstacle's inner edge and the opposite boundary",
               bins=(Bin("tight", 2.8, 3.3), Bin("moderate", 3.3, 3.9), Bin("wide", 3.9, 4.6, True))),
        Factor("road_width_m", "m", "corridor width (cone row to cone row)",
               bins=(Bin("narrow", 5.0, 5.6), Bin("standard", 5.6, 6.8), Bin("wide", 6.8, 8.0, True))),
    )
    nuisance = (
        Nuisance("turn_angle_deg", "deg", 70.0, 110.0, "heading change through the turn"),
        Nuisance("obstacle_arc_frac", "-", 0.30, 0.60, "obstacle station as a fraction of the turn's arc length"),
    )
    constants = {
        "obstacle_inset_m": 0.6,        # outer obstacle cone this far inside its boundary (> 0.5 m cone gap rule)
        "min_obstacle_width_m": 0.6,    # at least two cones at 0.6 m spacing
        "obstacle_cone_spacing_m": 0.6,
        "radius_margin_m": 1.5,         # Track.validate: radius >= half width + 1.5 m
        "start_offset_m": 8.0,
        "layout_accel_mps2": 2.5,       # conservative acceleration used to size the approach straight
        "settle_m": 25.0,               # extra straight for the approach speed to settle
        "handover_margin_m": 15.0,      # corner entry is preview + this far ahead at hand-over
        "exit_straight_m": 60.0,
        "finish_runoff_m": 35.0,
        "decision_exit_margin_m": 20.0,  # decision window ends this far past the turn exit
        "handover_speed_tol_mps": 0.5,
        "handover_lateral_tol_m": 0.25,
        "handover_heading_tol_rad": 0.035,
    }
    # Declared hypotheses for contrast pairs; checked empirically (contrast_report.json), never assumed.
    expected_effects = {
        "turn_radius_m": "tighter turn -> lower minimum speed in the decision window",
        "turn_direction": "mirror image -> steering label negated, pedals unchanged (exact only for a mirror-symmetric simulator)",
        "preview_m": "shorter preview -> braking starts further from the corner (or speed is already lower)",
        "approach_speed_mps": "higher approach speed -> more braking in the decision window",
        "obstacle_side": "other side -> the planned pass uses the opposite side of the corridor",
        "obstacle_clearance_m": "tighter gap -> lower speed when passing the obstacle",
        "road_width_m": "no directional hypothesis (changes corridor margins)",
    }

    # ---------------------------------------------------------------- formal constraints
    def formal_violations(self, params: dict) -> list[str]:
        """Bounds and structural constraints on the integer (fixed-point) form. Mirrored
        one-to-one by ``Corpus.Spec.violations`` in Lean; keep the two in sync (the Lean checker
        recomputes these and fails on any disagreement)."""
        c = self.canonical_params(params)
        k = {kk: to_fixed(v) for kk, v in self.constants.items()}
        problems = []
        for f in self.factors:
            if f.numeric:
                if not (to_fixed(f.lo) <= c[f.name] <= to_fixed(f.hi)):
                    problems.append(f"formal:out_of_bounds:{f.name}")
            elif c[f.name] not in f.levels:
                problems.append(f"formal:unknown_level:{f.name}")
        for n in self.nuisance:
            if not (to_fixed(n.lo) <= c[n.name] <= to_fixed(n.hi)):
                problems.append(f"formal:out_of_bounds:{n.name}")
        if c["obstacle_clearance_m"] > c["road_width_m"] - k["obstacle_inset_m"] - k["min_obstacle_width_m"]:
            problems.append("formal:clearance_exceeds_width")  # obstacle would be narrower than two cones
        if 2 * c["turn_radius_m"] < c["road_width_m"] + 2 * k["radius_margin_m"]:
            problems.append("formal:radius_too_tight_for_width")
        return problems

    # ---------------------------------------------------------------- layout
    def layout(self, params: dict) -> dict:
        """Course geometry and decision-window stations (progress == station on these open courses)."""
        k = self.constants
        v0, preview, r = float(params["approach_speed_mps"]), float(params["preview_m"]), float(params["turn_radius_m"])
        w, c = float(params["road_width_m"]), float(params["obstacle_clearance_m"])
        sgn_turn = 1.0 if params["turn_direction"] == "left" else -1.0
        sgn_side = sgn_turn if params["obstacle_side"] == "inside" else -sgn_turn
        angle = float(params["turn_angle_deg"])
        accel_len = v0 * v0 / (2.0 * k["layout_accel_mps2"]) + k["settle_m"]
        handover = k["start_offset_m"] + accel_len
        entry = handover + preview + k["handover_margin_m"]
        arc = r * math.radians(angle)
        s_obs = entry + float(params["obstacle_arc_frac"]) * arc
        ob_w = w - k["obstacle_inset_m"] - c
        edge = w / 2.0 - k["obstacle_inset_m"]           # outer obstacle cone, measured from the centerline
        ob_off = sgn_side * (edge - ob_w / 2.0)
        lat_outer, lat_inner = sgn_side * edge, sgn_side * (edge - ob_w)
        return {
            "handover_progress": round(handover, 6), "corner_entry_progress": round(entry, 6),
            "corner_exit_progress": round(entry + arc, 6), "obstacle_progress": round(s_obs, 6),
            "decision_end_progress": round(entry + arc + k["decision_exit_margin_m"], 6),
            "obstacle_offset_m": round(ob_off, 6), "obstacle_width_m": round(ob_w, 6),
            "obstacle_lateral_min_m": round(min(lat_outer, lat_inner), 6),
            "obstacle_lateral_max_m": round(max(lat_outer, lat_inner), 6),
            "turn_sign": int(sgn_turn), "obstacle_side_sign": int(sgn_side), "arc_length_m": round(arc, 6),
        }

    def track_spec(self, params: dict, name: str) -> dict:
        k = self.constants
        lay = self.layout(params)
        sgn = 1.0 if params["turn_direction"] == "left" else -1.0
        return {
            "name": name, "closed": False, "width": float(params["road_width_m"]), "cone_spacing": 5.0,
            "start_offset": k["start_offset_m"], "finish_runoff": k["finish_runoff_m"], "origin": [0.0, 0.0, 0.0],
            "segments": [
                {"type": "straight", "length": lay["corner_entry_progress"]},
                {"type": "arc", "radius": float(params["turn_radius_m"]), "angle": sgn * float(params["turn_angle_deg"])},
                {"type": "straight", "length": k["exit_straight_m"] + k["finish_runoff_m"]},
            ],
            "obstacles": [{"s": lay["obstacle_progress"], "offset": lay["obstacle_offset_m"],
                           "width": lay["obstacle_width_m"], "spacing": k["obstacle_cone_spacing_m"]}],
        }

    def sim_overrides(self, params: dict) -> dict:
        return {"road_model": {"range_m": float(params["preview_m"])}}

    def expert_config(self, params: dict) -> dict:
        """Controller config for the run. Only the scripted approach uses it; expert decisions
        after hand-over are a function of the observation alone (checked by label replay)."""
        lay = self.layout(params)
        return {"approach": {"handover_progress": lay["handover_progress"], "speed": float(params["approach_speed_mps"])}}

    # ---------------------------------------------------------------- simulator-level checks
    def simulator_violations(self, params: dict) -> list[str]:
        """Checks that need the harness's course builder (not modelled in Lean)."""
        import numpy as np

        from ..track import build
        lay = self.layout(params)
        try:
            t = build(self.track_spec(params, "probe"))
        except Exception as e:  # pragma: no cover - build errors are reported, not raised
            return [f"sim:track_build_error:{type(e).__name__}"]
        problems = [f"sim:track_invalid:{p}" for p in t.validate()]
        k = self.constants
        n_expected = max(int(round(lay["obstacle_width_m"] / k["obstacle_cone_spacing_m"])), 0) + 1
        if len(t.obstacle_cones) != n_expected:
            problems.append("sim:obstacle_cone_conflict")
        elif (abs(float(t.obstacle_lat.min()) - lay["obstacle_lateral_min_m"]) > 1e-6
              or abs(float(t.obstacle_lat.max()) - lay["obstacle_lateral_max_m"]) > 1e-6):
            problems.append("sim:obstacle_extent_mismatch")
        corners = t.find_corners(80.0, 15.0)
        if len(corners) != 1 or corners[0]["direction"] != lay["turn_sign"]:
            problems.append("sim:corner_detection_mismatch")
        else:
            c0 = corners[0]
            if abs(c0["entry"] - lay["corner_entry_progress"]) > 4.0:
                problems.append("sim:corner_entry_mismatch")
            if not (c0["entry"] <= lay["obstacle_progress"] <= c0["exit"]):
                problems.append("sim:obstacle_outside_corner")
        if not (t.finish_progress >= lay["decision_end_progress"]):
            problems.append("sim:finish_before_decision_end")
        if not np.isfinite(t.length):
            problems.append("sim:bad_length")
        return problems

    def region_key(self, params_bins: dict, region_factors: list[str]) -> str:
        return "|".join(f"{f}={params_bins[f]}" for f in region_factors)


FAMILIES: dict[str, Family] = {f.name: f for f in (BrakeTurnObstacle(),)}


def get_family(name: str) -> Family:
    if name not in FAMILIES:
        raise KeyError(f"unknown scenario family '{name}' (known: {', '.join(FAMILIES)})")
    return FAMILIES[name]
