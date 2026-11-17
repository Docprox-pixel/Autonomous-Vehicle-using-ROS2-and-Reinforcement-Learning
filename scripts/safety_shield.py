#!/usr/bin/env python3
"""Safety shield: the last layer before the car. Nothing (planner, RL) can bypass it.

  /cmd_drive_raw (Ackermann, from planner/RL) --> [shield] --> /cmd_drive (to the vehicle)

Every cycle the requested command is checked against:
  1. OBSTACLES IN THE PATH (/perception/obstacle_points = ALL obstacles, static and moving):
     points inside the swept corridor of the predicted path (bicycle model, current steering).
     Using all points means a tracker mistake (static object labelled 'moving') can never hide
     an obstacle from the shield. If the nearest one belongs to a tracked car moving AWAY from
     us, it gets the room that car needs to stop itself (v_lead^2 / 2 a_lead_max, a safe-
     following-distance rule); anything else counts as standing still. Speed is capped so the
     car can stop before it, including the reaction time:
         v * latency + v^2 / (2 * a_brake) = d - margin   ->   solve for v_safe.
     a_brake/latency are MEASURED on the vehicle (test/braking_test.py): planning with more
     braking than the car has makes the shield brake too late.
  2. DYNAMIC obstacles (/perception/tracks, moving=True): ego (along its path) and every moving
     object (constant velocity) are simulated `horizon` s ahead. The first time they overlap is
     the time-to-collision (TTC). TTC < ttc_stop -> stop; TTC < ttc_slow -> slow down linearly.
  3. TIMEOUTS: perception older than `sensor_timeout` -> stop. The shield only publishes while
     raw commands arrive, so tests/teleop that publish /cmd_drive directly are not overridden.

Output speed = min(requested, static limit, dynamic limit). Steering passes through (clamped).
The decision logic lives in auto_car_core.safety (pure functions, unit-tested in
test/test_shield_logic.py), shared with the local planner.
"""
import json
import math

import numpy as np

from auto_car_core.safety import CAR_REAR, HALF_WIDTH, Params, evaluate

# ----------------------------------------------------------------------------- ROS node
def main():
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from auto_car.msg import TrackArray
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import String

    class SafetyShield(Node):
        def __init__(self):
            super().__init__('safety_shield')
            self.p = Params(**{k: self.declare_parameter(k, v).value for k, v in Params().__dict__.items()})
            self.max_steer = self.declare_parameter('max_steer', 0.5).value
            self.sensor_timeout = self.declare_parameter('sensor_timeout', 0.5).value
            self.cmd_timeout = self.declare_parameter('cmd_timeout', 0.5).value
            self.cmd = self.cmd_t = None
            self.points = self.points_t = None
            self.tracks, self.tracks_t = [], None
            self.ego_v = 0.0
            self.sent_stop = True
            self.pub = self.create_publisher(AckermannDriveStamped, '/cmd_drive', 10)
            self.pub_status = self.create_publisher(String, '/safety/status', 10)
            self.create_subscription(AckermannDriveStamped, '/cmd_drive_raw', self.on_cmd, 10)
            self.create_subscription(PointCloud2, '/perception/obstacle_points', self.on_points, 5)
            # Traffic rules (red light / stop sign) are HARD constraints: the stop line is a wall
            self.rule_walls = np.zeros((0, 3))
            self.create_subscription(PointCloud2, '/behaviour/stop_points', lambda m: setattr(
                self, 'rule_walls', point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))), 5)
            from std_msgs.msg import Float32
            self.stop_distance = 1e9
            self.create_subscription(Float32, '/behaviour/stop_distance', lambda m: setattr(self, 'stop_distance', m.data), 5)
            self.create_subscription(TrackArray, '/perception/tracks', self.on_tracks, 10)
            self.create_subscription(Odometry, '/odometry/filtered',
                                     lambda m: setattr(self, 'ego_v', m.twist.twist.linear.x), 20)
            # the planner's planned path (base_link): the static check follows it (fallback: arc)
            from nav_msgs.msg import Path as _Path
            self.plan_path, self.plan_path_t = None, None
            self.create_subscription(_Path, '/planner/path', self.on_plan_path, 5)
            self.create_timer(0.05, self.loop)
            self.get_logger().info(f'safety_shield active: brake {self.p.a_brake} m/s^2, '
                                   f'TTC stop {self.p.ttc_stop} s / slow {self.p.ttc_slow} s')

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_cmd(self, m):
            self.cmd, self.cmd_t = m, self.now()

        def on_points(self, m):
            self.points = point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))
            self.points_t = self.now()

        def on_plan_path(self, m):
            self.plan_path = np.array([[q.pose.position.x, q.pose.position.y] for q in m.poses]) if m.poses else None
            self.plan_path_t = self.now()

        def on_tracks(self, m):
            self.tracks = [{'pos': (t.position.x, t.position.y), 'vel': (t.velocity.x, t.velocity.y),
                            'size': (t.size.x, t.size.y)} for t in m.tracks if t.moving]
            self.tracks_t = self.now()

        def loop(self):
            t = self.now()
            if self.cmd is None or t - self.cmd_t > self.cmd_timeout:
                if not self.sent_stop:       # raw commands stopped: one stop, then stay silent
                    self.publish(0.0, 0.0, {'reason': 'no_command'})
                    self.sent_stop = True
                return
            self.sent_stop = False
            steer = float(np.clip(self.cmd.drive.steering_angle, -self.max_steer, self.max_steer))
            cmd_v = float(self.cmd.drive.speed)
            stale = (self.points_t is None or t - self.points_t > self.sensor_timeout or
                     self.tracks_t is None or t - self.tracks_t > self.sensor_timeout)
            if stale:
                self.publish(0.0, steer, {'reason': 'perception_timeout'})
                return
            pts = np.r_[self.points, self.rule_walls] if len(self.rule_walls) else self.points
            if cmd_v < -0.01:
                # REVERSING (manoeuvre): only if the space behind the car is clear; slow, max 1 m/s
                P = self.points if self.points is not None else np.zeros((0, 3))
                band = P[(P[:, 2] > self.p.min_h) & (P[:, 2] < self.p.max_h)] if len(P) else P
                behind = band[(band[:, 0] < CAR_REAR + 0.3) & (band[:, 0] > CAR_REAR - 4.0) &
                              (np.abs(band[:, 1]) < HALF_WIDTH + 0.35)] if len(band) else band
                rear_free = float((CAR_REAR - behind[:, 0]).min()) if len(behind) else 4.0
                v = -min(-cmd_v, 1.0, max(0.0, rear_free - 0.5))
                self.publish(v, steer, {'reason': 'reversing' if v < 0 else 'rear_blocked',
                                        'static_distance': rear_free, 'ttc': None})
                return
            fresh = self.plan_path is not None and self.plan_path_t is not None and t - self.plan_path_t < 0.3
            v, info = evaluate(cmd_v, steer, self.ego_v, pts, self.tracks, self.p,
                               path=self.plan_path if fresh else None)
            if self.stop_distance < 1e8:                   # hard rule, independent of steering
                room = self.stop_distance - 0.5
                v_rule = 0.0 if room <= 0 else self.p.a_brake * (-self.p.latency + math.sqrt(
                    self.p.latency ** 2 + 2 * room / self.p.a_brake))
                if v_rule < v:
                    v, info = v_rule, dict(info, reason='stop_line')
            self.publish(v, steer, info)
            if info['reason'] != 'clear':
                self.get_logger().warn(
                    f"SHIELD {info['reason']}: {cmd_v:.1f} -> {v:.1f} m/s "
                    f"(obstacle {info['static_distance']:.1f} m, TTC {info['ttc']:.1f} s)",
                    throttle_duration_sec=1.0)

        def publish(self, v, steer, info):
            m = AckermannDriveStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            m.drive.speed, m.drive.steering_angle = float(v), float(steer)
            self.pub.publish(m)
            self.pub_status.publish(String(data=json.dumps(
                {k: (round(x, 2) if isinstance(x, float) and math.isfinite(x) else (None if isinstance(x, float) else x))
                 for k, x in info.items()})))

    rclpy.init()
    node = SafetyShield()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
