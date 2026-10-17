"""Starter template. Copy this file, rename the class, and implement step().

Run it:   python -m autocross run --controller controllers/my_team.py --track tracks/practice_basic.yaml
Dev loop without CARLA:   add  --backend bicycle --view

There is no map (EIC rules). Everything about the course comes from obs.road, the processed
output of the perception stack, limited to what is visible ahead (road_model.range_m):

  obs.road.lateral_offset         m, you vs. the centerline (+ = you are left of it)
  obs.road.heading_error          rad, your heading vs. the road direction (+ = pointing left)
  obs.road.curvature              1/m at your position (+ = road bends left)
  obs.road.dist_to_left_boundary / dist_to_right_boundary   m to each cone line
  obs.road.preview_s              distances ahead along the course (0 .. range)
  obs.road.preview_curvature      curvature at each of those distances
  obs.road.preview_centerline     centerline points ahead, vehicle frame (x fwd, y left)
  obs.road.preview_left_boundary / preview_right_boundary / preview_width_left/right
  obs.road.corners                Corner(direction, entry/apex/exit_distance, min_radius, turn_angle, ...)
  obs.road.obstacles              ObstacleInfo(distance, lateral_min/max vs centerline, points)
  obs.road.finish_distance        m to the finish line once it is in range, else NaN
Plus ground-truth vehicle state (obs.x/y/yaw, vx, vy, yaw_rate, ax, ay) and raw cone
detections (obs.cones) if you want them.

Rules of the road (enforced by the harness):
  * step() is called every ``vehicle.control_dt`` seconds and must return within the compute
    budget deadline, otherwise your previous command is held and an overrun is recorded.
  * Your process gets ``compute_budget.cpu_cores`` CPU core(s), no GPU, and a memory ceiling.
  * Steering is slew-rate limited to ``vehicle.max_steer_rate`` (normalised units / s).
  * Publish Command.path (the path you intend to drive, vehicle frame by default) - tracking
    error against it is what the deviation metric scores, and it is drawn live in the simulator.
"""
from __future__ import annotations

import math

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams


class MyController(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        # vehicle: mass, wheelbase, max_steer, mu, max_accel, max_decel, control_dt, ...

    def step(self, obs: Observation) -> Command:
        road = obs.road
        if road is None or not road.valid:
            return Command(brake=1.0)

        # 1. plan a path from the road model (here: just the centerline, shifted away from obstacles)
        path = road.preview_centerline.copy()
        for ob in road.obstacles:
            _ = (ob.distance, ob.lateral_min, ob.lateral_max)  # e.g. shift the path to the open side

        # 2. pick a speed from what is coming (e.g. the next corner)
        corner = road.corners[0] if road.corners else None
        target_speed = 8.0 if corner and corner.entry_distance < 20.0 else 12.0

        # 3. track the path (replace with your controller)
        steer = float(np.clip(-0.5 * road.lateral_offset - 1.0 * road.heading_error, -1.0, 1.0))
        throttle = 0.3 if obs.speed < target_speed else 0.0
        return Command(steer=steer, throttle=throttle, brake=0.0, path=path, path_frame="vehicle",
                       target_speed=target_speed, debug={"heading_error_deg": math.degrees(road.heading_error)})
