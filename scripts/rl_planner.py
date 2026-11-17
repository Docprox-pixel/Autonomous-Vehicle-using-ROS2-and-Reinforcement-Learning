#!/usr/bin/env python3
"""RL planner node: the trained SAC driving policy in place of the hand-written local planner.

Same inputs and outputs as scripts/local_planner.py (it IS a LocalPlanner with a different
decide()): obstacle points + camera road boundary + route -> /cmd_drive_raw -> safety_shield.
Traffic rules (speed limit, stop distance at red lights / stop signs / pedestrians) are applied
on top exactly as for the planner, and the safety shield still has the last word.

The observation is built with rl/kin_env.obs_from, the same function the policy was trained on.
Without a route (no lane map / no goal) the policy has nothing to follow: the car holds still.

  ros2 launch auto_car sim.launch.py world:=city.wbt planner:=rl
  parameters: model (path to the SAC .zip; default rl/models/sac_driver.zip in the source tree)
"""
import os
import sys

import numpy as np
import rclpy
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.realpath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SRC, 'rl'))
from local_planner import LocalPlanner  # noqa: E402
from kin_env import action_to_command, obs_from  # noqa: E402


class RLPlanner(LocalPlanner):
    def __init__(self):
        super().__init__('rl_planner')
        path = self.declare_parameter('model', os.path.join(SRC, 'rl', 'models', 'sac_driver.zip')).value
        from stable_baselines3 import SAC
        self.model = SAC.load(path, device='cpu')
        self.prev_action = np.zeros(2, np.float32)
        self.route_corridor = self.declare_parameter('route_corridor', 4.0).value   # m
        self.get_logger().info(f'rl_planner: policy {path}')

    def decide(self, pts, goal, now):
        route = self.route_in_base()
        if route is None or len(route) < 2:
            return self.prev_steer, 0.0, {'reason': 'no_route', 'arcs': [], 'best': None}
        band = pts[(pts[:, 2] > self.ps.min_h) & (pts[:, 2] < self.ps.max_h)] if len(pts) else pts
        # Show the policy what it was trained on: in the training env the only obstacles are lane
        # edges and objects on/beside the route. Buildings, poles and kerbs far from the route were
        # out-of-distribution and froze the policy at the first junction. (The safety shield still
        # sees EVERYTHING.)
        if len(band):
            seg = np.diff(route, axis=0)
            n = np.maximum(np.ceil(np.hypot(*seg.T) / 0.5).astype(int), 1)
            dense = np.vstack([route[i] + seg[i] * np.linspace(0, 1, k, endpoint=False)[:, None]
                               for i, k in enumerate(n)] + [route[-1:]])
            d, _ = cKDTree(dense).query(band[:, :2])
            band = band[d < self.route_corridor]
        obs = obs_from(route, band[:, :2], self.v, self.prev_steer, self.prev_action)
        action, _ = self.model.predict(obs, deterministic=True)
        self.prev_action = np.asarray(action, np.float32)
        steer, v = action_to_command(action)
        return steer, v, {'reason': 'ok', 'arcs': [], 'best': {'free': 0.0, 'steer': steer},
                          'policy': [round(float(a), 3) for a in action]}


def main():
    rclpy.init()
    node = RLPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
