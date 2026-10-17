"""Lossless JSON (de)serialisation of controller observations and expert commands, the canonical
policy-input / training-target split of a sample, content hashing, and the mirror (y -> -y)
transform of a policy observation.

Floats are written with Python's shortest round-trip repr, so ``obs_from_dict(obs_to_dict(o))``
reproduces every float bit for bit. JSON has no NaN: a NaN float is written as ``null`` and read
back as NaN (e.g. ``finish_distance`` beyond range, a corner apex not yet visible). Infinite
values are rejected.

Policy observation (``split_observation``): an explicit WHITELIST of what a policy may consume:
ego motion and actuator state (``POLICY_EGO``) and the road model of the visible preview
(``POLICY_ROAD_*``, obstacles inside the preview). Everything else a controller receives is
``privileged`` and is stored in a separate block of the sample: simulation time and step, world
pose, course progress and lap bookkeeping, raw cone detections (their 30 m range can exceed the
scenario's preview), the corner list (its existence, apex and fully-visible fields are computed
from the WHOLE course, including geometry beyond the preview) and ``finish_distance``."""
from __future__ import annotations

import math

import numpy as np

from ..api import Command, ConeDetections, Corner, ObstacleInfo, Observation, RoadModel
from .spec import canonical_json, sha256_text

OBS_SCALARS = ("t", "step", "x", "y", "yaw", "vx", "vy", "yaw_rate", "ax", "ay", "speed", "steer", "throttle", "brake",
               "progress", "lap", "timing_started")
ROAD_SCALARS = ("valid", "range_m", "lateral_offset", "heading_error", "curvature", "width_left", "width_right",
                "dist_to_left_boundary", "dist_to_right_boundary", "finish_distance")
ROAD_VECTORS = ("preview_s", "preview_curvature", "preview_width_left", "preview_width_right")
ROAD_POINTS = ("preview_centerline", "preview_left_boundary", "preview_right_boundary")
CORNER_FIELDS = ("direction", "entry_distance", "apex_distance", "exit_distance", "min_radius", "turn_angle",
                 "fully_visible")

# ---- the canonical policy input (whitelist) and what is privileged (everything else)
POLICY_EGO = ("speed", "vx", "vy", "yaw_rate", "ax", "ay", "steer", "throttle", "brake")
POLICY_ROAD_SCALARS = ("valid", "range_m", "lateral_offset", "heading_error", "curvature", "width_left", "width_right",
                       "dist_to_left_boundary", "dist_to_right_boundary")
POLICY_ROAD_KEYS = POLICY_ROAD_SCALARS + ROAD_VECTORS + ROAD_POINTS + ("obstacles",)
PRIVILEGED_OBS = tuple(k for k in OBS_SCALARS if k not in POLICY_EGO) + ("cones",)
PRIVILEGED_ROAD = ("corners", "finish_distance")
POLICY_OBSERVATION_SCHEMA = "autocross.datagen.policy_observation/1"
TRAINING_TARGET_SCHEMA = "autocross.datagen.training_target/1"


def _f(x):
    x = float(x)
    if math.isnan(x):
        return None
    if math.isinf(x):
        raise ValueError("infinite value cannot be serialised")
    return x


def _r(x):
    return float("nan") if x is None else float(x)


def _vec(a) -> list:
    return [_f(v) for v in np.asarray(a, float).reshape(-1)]


def _pts(a) -> list:
    return [[_f(p[0]), _f(p[1])] for p in np.asarray(a, float).reshape(-1, 2)]


def _scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return _f(v)


def obs_to_dict(obs: Observation) -> dict:
    d = {k: _scalar(getattr(obs, k)) for k in OBS_SCALARS}
    c = obs.cones
    d["cones"] = None if c is None else {"positions": _pts(c.positions), "colors": [int(v) for v in c.colors],
                                         "range_m": _f(c.range_m), "fov_rad": _f(c.fov_rad)}
    r = obs.road
    if r is None:
        d["road"] = None
        return d
    rd = {k: _scalar(getattr(r, k)) for k in ROAD_SCALARS}
    rd.update({k: _vec(getattr(r, k)) for k in ROAD_VECTORS})
    rd.update({k: _pts(getattr(r, k)) for k in ROAD_POINTS})
    rd["corners"] = [{k: _scalar(getattr(cn, k)) for k in CORNER_FIELDS} for cn in r.corners]
    rd["obstacles"] = [{"distance": _f(o.distance), "lateral_min": _f(o.lateral_min), "lateral_max": _f(o.lateral_max),
                        "center": [_f(o.center[0]), _f(o.center[1])], "points": _pts(o.points)} for o in r.obstacles]
    d["road"] = rd
    return d


def _arr(v) -> np.ndarray:
    return np.array([_r(x) for x in v], float)


def _parr(v) -> np.ndarray:
    return np.array([[_r(p[0]), _r(p[1])] for p in v], float).reshape(-1, 2)


def obs_from_dict(d: dict) -> Observation:
    kw = {k: (d[k] if k in ("step", "lap", "timing_started") else _r(d[k])) for k in OBS_SCALARS}
    c = d.get("cones")
    kw["cones"] = None if c is None else ConeDetections(_parr(c["positions"]), np.array(c["colors"], int),
                                                        _r(c["range_m"]), _r(c["fov_rad"]))
    r = d.get("road")
    if r is not None:
        rk = {k: (r[k] if k == "valid" else _r(r[k])) for k in ROAD_SCALARS}
        rk.update({k: _arr(r[k]) for k in ROAD_VECTORS})
        rk.update({k: _parr(r[k]) for k in ROAD_POINTS})
        rk["corners"] = [Corner(**{k: (cn[k] if k in ("direction", "fully_visible") else _r(cn[k])) for k in CORNER_FIELDS})
                         for cn in r["corners"]]
        rk["obstacles"] = [ObstacleInfo(_r(o["distance"]), _r(o["lateral_min"]), _r(o["lateral_max"]),
                                        np.array([_r(o["center"][0]), _r(o["center"][1])]), _parr(o["points"]))
                           for o in r["obstacles"]]
        kw["road"] = RoadModel(**rk)
    else:
        kw["road"] = None
    return Observation(**kw)


# ------------------------------------------------------------------ policy input / privileged split
def split_observation(d: dict) -> tuple[dict, dict]:
    """(policy_observation, privileged_observation) of a serialised observation. Lossless:
    ``merge_observation(*split_observation(d)) == d``."""
    pol = {"schema": POLICY_OBSERVATION_SCHEMA, **{k: d[k] for k in POLICY_EGO}}
    priv = {k: d[k] for k in PRIVILEGED_OBS}
    r = d.get("road")
    if r is None:
        pol["road"] = None
        priv["road"] = None
    else:
        pol["road"] = {k: r[k] for k in POLICY_ROAD_KEYS}
        priv["road"] = {k: r[k] for k in PRIVILEGED_ROAD}
    return pol, priv


def merge_observation(policy: dict, privileged: dict) -> dict:
    """Inverse of ``split_observation`` (the full controller observation)."""
    d = {k: policy[k] for k in POLICY_EGO}
    d.update({k: privileged[k] for k in PRIVILEGED_OBS})
    d = {k: d[k] for k in OBS_SCALARS + ("cones",)}
    if policy["road"] is None:
        d["road"] = None
    else:
        r = {**policy["road"], **privileged["road"]}
        d["road"] = {k: r[k] for k in ROAD_SCALARS + ROAD_VECTORS + ROAD_POINTS + ("corners", "obstacles")}
    return d


def policy_to_observation(policy: dict) -> Observation:
    """An ``Observation`` carrying ONLY the policy input: every privileged field is replaced by an
    uninformative sentinel (NaN pose/time/progress, no cones, no corners, NaN finish distance)."""
    if set(policy) != {"schema", "road", *POLICY_EGO} or policy["schema"] != POLICY_OBSERVATION_SCHEMA:
        raise ValueError("not a canonical policy observation")
    sentinel = {"t": None, "step": -1, "x": None, "y": None, "yaw": None, "progress": None, "lap": -1,
                "timing_started": False, "cones": None}
    road = policy["road"]
    if road is not None:
        if set(road) != set(POLICY_ROAD_KEYS):
            raise ValueError("policy road model has non-canonical fields")
        road = {**road, "corners": [], "finish_distance": None}
    return obs_from_dict({**sentinel, **{k: policy[k] for k in POLICY_EGO}, "road": road})


# ------------------------------------------------------------------ expert command / training target
def command_to_dict(cmd: Command | None) -> dict | None:
    """Raw controller output as proposed (path in the controller's own frame)."""
    if cmd is None:
        return None
    path = None if cmd.path is None else _pts(cmd.path)
    return {"steer": _f(cmd.steer), "throttle": _f(cmd.throttle), "brake": _f(cmd.brake),
            "target_speed": None if cmd.target_speed is None else _f(cmd.target_speed),
            "path": path, "path_frame": str(cmd.path_frame),
            "debug": {str(k): _f(v) for k, v in sorted((cmd.debug or {}).items())}}


def training_target(raw: dict, certificate: dict | None) -> dict:
    """The imitation target of one step. ``longitudinal`` is the certified quantity (integer
    acceleration in mm/s^2 of model cert-2, with its kind); ``uncertified`` holds the lateral path,
    steering, the pedal mapping and the target speed, which carry NO formal guarantee."""
    return {"schema": TRAINING_TARGET_SCHEMA, "longitudinal": certificate,
            "uncertified": {k: raw[k] for k in ("steer", "throttle", "brake", "target_speed", "path", "path_frame")}}


def content_hash(policy_observation: dict, target: dict) -> str:
    """Identity of an imitation example: the policy observation and the full training target."""
    return sha256_text(canonical_json({"policy_observation": policy_observation, "target": target}))


def record_hash(rec: dict) -> str:
    """Hash of a whole sample record except its own ``hashes`` block and the wall-clock ``timing``."""
    return sha256_text(canonical_json({k: v for k, v in rec.items() if k not in ("hashes", "timing")}))


def sample_hashes(rec: dict) -> dict:
    out = {"policy_observation": sha256_text(canonical_json(rec["policy_observation"])),
           "target": None if rec["target"] is None else sha256_text(canonical_json(rec["target"])),
           "content": None if rec["target"] is None else content_hash(rec["policy_observation"], rec["target"])}
    out["record"] = record_hash(rec)
    return out


# ------------------------------------------------------------------ mirror transform
def _neg(x):
    return None if x is None else -x


def _mpts(pts):
    return [[p[0], _neg(p[1])] for p in pts]


def mirror_observation(d: dict) -> dict:
    """Reflect a policy observation across the vehicle's x axis. Left and right swap: boundaries,
    widths, obstacle extents; lateral quantities change sign."""
    m = dict(d)
    for k in ("vy", "yaw_rate", "ay", "steer"):
        m[k] = _neg(d[k])
    r = d.get("road")
    if r is not None:
        mr = dict(r)
        for k in ("lateral_offset", "heading_error", "curvature"):
            mr[k] = _neg(r[k])
        mr["width_left"], mr["width_right"] = r["width_right"], r["width_left"]
        mr["dist_to_left_boundary"], mr["dist_to_right_boundary"] = r["dist_to_right_boundary"], r["dist_to_left_boundary"]
        mr["preview_curvature"] = [_neg(v) for v in r["preview_curvature"]]
        mr["preview_width_left"], mr["preview_width_right"] = r["preview_width_right"], r["preview_width_left"]
        mr["preview_centerline"] = _mpts(r["preview_centerline"])
        mr["preview_left_boundary"] = _mpts(r["preview_right_boundary"])
        mr["preview_right_boundary"] = _mpts(r["preview_left_boundary"])
        mr["obstacles"] = [{**o, "lateral_min": _neg(o["lateral_max"]), "lateral_max": _neg(o["lateral_min"]),
                            "center": [o["center"][0], _neg(o["center"][1])], "points": _mpts(o["points"])}
                           for o in r["obstacles"]]
        m["road"] = mr
    return m


def mirror_action(a: dict) -> dict:
    """Mirror of the uncertified action block: steering and the path change sign; pedals and speeds do not."""
    m = dict(a)
    m["steer"] = _neg(a["steer"])
    if a.get("path") is not None:
        m["path"] = _mpts(a["path"])
    return m
