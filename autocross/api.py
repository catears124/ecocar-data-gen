"""Team-facing interface. This is the only module a controller needs to import.

Conventions (identical for every backend):
  * World frame: right-handed ENU, metres. x east, y north, yaw CCW from +x in radians.
  * Vehicle/body frame: x forward, y left. Positive steer / yaw rate = turning left.
  * Command.steer in [-1, 1] maps linearly to [-max_steer, +max_steer] road-wheel angle.

A controller is a class deriving from ``Controller``. The harness runs it in its own
process under the configured compute budget, calls ``on_start`` once, then ``step`` at
``control_hz`` until the run ends.

What a controller knows about the course (EIC autocross rules, ``track_knowledge: perception``):
  * No map. ``on_start`` receives vehicle parameters and an empty ``TrackInfo``.
  * Every step, ``Observation.road`` is the processed output of the perception stack: a
    ``RoadModel`` describing the course *within sensor range* (centerline deviation, heading
    error, curvature, corridor widths, a centerline/boundary preview, upcoming corners,
    obstacles). Teams plan and control from this; no sensor fusion is required.
  * ``Observation.cones`` additionally gives the raw cone detections for teams that want them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CONE_LEFT, CONE_RIGHT, CONE_ORANGE = 0, 1, 2
"""Cone colour classes: left boundary (blue), right boundary (yellow), other/obstacle (orange)."""


@dataclass
class VehicleParams:
    mass: float                 # kg
    wheelbase: float            # m
    track_width: float          # m
    length: float               # m, bounding box
    width: float                # m, bounding box
    cg_to_front: float          # m, CG to front axle
    cg_to_rear: float           # m, CG to rear axle
    max_steer: float            # rad, road-wheel angle at |steer| = 1
    max_steer_rate: float       # normalised steer units per second (actuator slew limit)
    mu: float                   # nominal peak tyre-road friction (lateral accel limit ~ mu*g)
    max_accel: float            # m/s^2, approximate full-throttle accel at low speed
    max_decel: float            # m/s^2, approximate full-brake decel (positive number)
    max_speed: float            # m/s
    control_dt: float           # s, period between step() calls
    actuation_delay_steps: int  # control steps between a command and its application


@dataclass
class TrackInfo:
    """Prior map. Only populated with ``track_knowledge: full`` (practice / debugging). Under
    the event setting ``perception`` every array is empty and ``name`` is just "course": use
    ``Observation.road``."""
    name: str
    closed: bool
    laps: int
    centerline: np.ndarray      # (N,2) world, ~1 m spacing, ordered in driving direction
    width_left: np.ndarray      # (N,) distance from centerline to left boundary
    width_right: np.ndarray     # (N,) distance from centerline to right boundary
    cones: np.ndarray           # (M,2) world positions of mapped cones
    cone_colors: np.ndarray     # (M,) CONE_LEFT / CONE_RIGHT / CONE_ORANGE
    start_line: np.ndarray      # (2,2) endpoints (left, right)
    finish_line: np.ndarray     # (2,2) endpoints (left, right)
    gates: np.ndarray           # (G,2,2) scored gates (left point, right point), in order
    full_knowledge: bool = True


@dataclass
class ConeDetections:
    """Simulated perception output, in the vehicle frame (x forward, y left)."""
    positions: np.ndarray       # (K,2)
    colors: np.ndarray          # (K,) int
    range_m: float
    fov_rad: float


@dataclass
class Corner:
    """A corner (constant-sign curvature section) at least partly within perception range.
    Distances are measured along the centerline from the vehicle's current station; a negative
    ``entry_distance`` means the vehicle is already in the corner. Parts beyond range are
    unknown: ``apex_distance`` / ``exit_distance`` are NaN until they come into view."""
    direction: int              # +1 left, -1 right
    entry_distance: float       # m
    apex_distance: float        # m (point of maximum curvature seen so far), NaN if not yet visible
    exit_distance: float        # m, NaN if not yet visible
    min_radius: float           # m, over the visible part
    turn_angle: float           # rad, heading change over the visible part (signed, + left)
    fully_visible: bool


@dataclass
class ObstacleInfo:
    """An obstacle in or near the corridor (e.g. a cone group not part of the course boundary)."""
    distance: float             # m along the centerline ahead of the vehicle
    lateral_min: float          # m, lowest lateral extent relative to the centerline (+ left)
    lateral_max: float          # m, highest lateral extent relative to the centerline
    center: np.ndarray          # (2,) vehicle frame
    points: np.ndarray          # (K,2) vehicle frame, detected obstacle cones


@dataclass
class RoadModel:
    """Processed perception output describing the course ahead (vehicle frame, x fwd / y left).
    All quantities are referenced to the course centerline (midline between the cone rows)."""
    valid: bool
    range_m: float              # preview horizon along the centerline
    lateral_offset: float       # m, vehicle position relative to the centerline (+ = vehicle is left of it)
    heading_error: float        # rad, vehicle yaw minus centerline heading (+ = pointing left of the road)
    curvature: float            # 1/m at the vehicle's station (+ = road turns left)
    width_left: float           # m, centerline to left boundary at the vehicle's station
    width_right: float          # m, centerline to right boundary
    dist_to_left_boundary: float   # m, vehicle reference point to left boundary (negative = outside)
    dist_to_right_boundary: float  # m, vehicle reference point to right boundary (negative = outside)
    preview_s: np.ndarray       # (K,) distance ahead along the centerline, 0 .. range_m
    preview_curvature: np.ndarray    # (K,) 1/m
    preview_centerline: np.ndarray   # (K,2) vehicle frame
    preview_width_left: np.ndarray   # (K,)
    preview_width_right: np.ndarray  # (K,)
    preview_left_boundary: np.ndarray   # (K,2) vehicle frame
    preview_right_boundary: np.ndarray  # (K,2) vehicle frame
    corners: list[Corner]
    obstacles: list[ObstacleInfo]
    finish_distance: float      # m along the centerline to the finish line, NaN if beyond range


@dataclass
class Observation:
    t: float                    # s since run start (sim time)
    step: int                   # control step index
    # ground-truth vehicle state
    x: float
    y: float
    yaw: float
    vx: float                   # body-frame longitudinal velocity, m/s
    vy: float                   # body-frame lateral velocity, m/s (+ left)
    yaw_rate: float             # rad/s, + CCW
    ax: float                   # body-frame longitudinal accel, m/s^2
    ay: float                   # body-frame lateral accel, m/s^2
    speed: float
    # actuator state actually applied last physics tick (after rate limit / delay)
    steer: float
    throttle: float
    brake: float
    # race state (ground truth)
    progress: float             # m along centerline since spawn (unwrapped over laps)
    lap: int
    timing_started: bool
    cones: ConeDetections | None = None
    road: RoadModel | None = None


@dataclass
class Command:
    steer: float = 0.0          # [-1, 1], + left
    throttle: float = 0.0       # [0, 1]
    brake: float = 0.0          # [0, 1]
    path: np.ndarray | None = None
    """Optional (N,2) path the controller intends to follow, in ``path_frame``. Used for the
    path-deviation metric and drawn in the simulator. If omitted, deviation is measured
    against the course centerline."""
    path_frame: str = "vehicle"
    """'vehicle' (x fwd / y left, relative to the pose in the Observation this command answers)
    or 'world'."""
    target_speed: float | None = None
    debug: dict[str, float] = field(default_factory=dict)
    """Scalar values to log alongside telemetry (column prefix ``dbg_``)."""


class Controller(ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        """Called once before the first step. Heavy offline work is allowed here."""
        self.track = track
        self.vehicle = vehicle

    @abstractmethod
    def step(self, obs: Observation) -> Command:
        """Called every control period. Must return within the compute budget deadline."""

    def on_finish(self) -> None:
        """Called once after the run ends (not timed)."""
