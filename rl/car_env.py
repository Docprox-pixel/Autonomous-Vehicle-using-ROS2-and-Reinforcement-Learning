import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, String
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
import json
from cv_bridge import CvBridge
import cv2


class CarEnv(gym.Env):

    def __init__(self):
        super().__init__()

        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("rl_driver")

        # 🔥 UPDATED OBSERVATION (36 + 3 directional + 3 state)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(42,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.latest_scan = np.ones(36)
        self.red_detected = 0.0
        self.green_detected = 0.0
        self.speed = 0.0

        self.last_action = 0.0

        self.bridge = CvBridge()

        self.node.create_subscription(LaserScan, "/lidar/scan", self.lidar_callback, 10)
        self.node.create_subscription(Image, "/front_camera/image_raw", self.camera_callback, 10)
        self.node.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.node.create_subscription(String, "/world_model", self.world_cb, 10)

        self.ml_pub = self.node.create_publisher(Float32, "/ml/steering", 10)

        self.step_count = 0
        self.max_steps = 500
        
        self.lane_error = 0.0
        self.objects_detected = "None"

    # ---------------- LIDAR ----------------
    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=msg.range_max)

        indices = np.linspace(0, len(ranges) - 1, 36, dtype=int)
        self.latest_scan = np.clip(ranges[indices] / msg.range_max, 0, 1)

    # ---------------- CAMERA ----------------
    def camera_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            h, w = hsv.shape[:2]
            roi = hsv[0:int(h*0.4), int(w*0.3):int(w*0.7)]

            # Red
            m1 = cv2.inRange(roi, (0,120,120), (10,255,255))
            m2 = cv2.inRange(roi, (170,120,120), (180,255,255))
            red_mask = cv2.bitwise_or(m1, m2)
            self.red_detected = 1.0 if cv2.countNonZero(red_mask) > 100 else 0.0

            # Green
            green_mask = cv2.inRange(roi, (40,100,100), (80,255,255))
            self.green_detected = 1.0 if cv2.countNonZero(green_mask) > 100 else 0.0

        except:
            pass

    # ---------------- ODOM ----------------
    def odom_callback(self, msg):
        self.speed = msg.twist.twist.linear.x / 15.0

    # ---------------- WORLD ----------------
    def world_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.lane_error = data.get("lane_error", 0.0)
            obs = data.get("obstacles", [])
            if len(obs) > 0:
                self.objects_detected = ", ".join([f"{o['class']}" for o in obs])
            else:
                self.objects_detected = "None"
        except:
            pass

    # ---------------- OBS ----------------
    def get_observation(self):

        # 🔥 Directional awareness
        left  = np.mean(self.latest_scan[24:30])
        right = np.mean(self.latest_scan[6:12])
        front = np.mean(self.latest_scan[16:20])

        obs = np.concatenate([
            self.latest_scan,
            [left, right, front],
            [self.red_detected, self.green_detected, self.speed]
        ]).astype(np.float32)

        return obs

    # ---------------- STEP ----------------
    def step(self, action):

        steer_msg = Float32()
        steer_msg.data = float(np.clip(action[1] * 0.4, -0.4, 0.4))
        self.ml_pub.publish(steer_msg)

        rclpy.spin_once(self.node, timeout_sec=0.1)

        obs = self.get_observation()

        min_dist = np.min(self.latest_scan)
        left  = np.mean(self.latest_scan[24:30])
        right = np.mean(self.latest_scan[6:12])
        front = np.mean(self.latest_scan[16:20])

        terminated = False
        truncated = False

        # ===== 🔥 NEW REWARD =====
        reward = 0.0

        expected_speed = float(action[0] * 12.0)
        # 1. Forward progress
        reward += expected_speed * 1.0

        # 2. Stay safe (important)
        reward += front * 5.0

        # 3. Keep centered (balance left/right)
        reward -= abs(left - right) * 3.0

        # 4. Smooth driving
        reward -= abs(action[1] - self.last_action) * 2.0
        self.last_action = action[1]

        # 5. Penalize sharp turns
        reward -= abs(action[1]) * 1.0

        # 6. Crash penalty
        if min_dist < 0.1:
            reward -= 300.0
            terminated = True

        # 7. Traffic light
        if self.red_detected > 0.5 and expected_speed > 0.5:
            reward -= 50.0

        # 8. Timeout
        self.step_count += 1
        if self.step_count > self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0
        self.last_action = 0.0

        steer_msg = Float32()
        steer_msg.data = 0.0
        self.ml_pub.publish(steer_msg)

        rclpy.spin_once(self.node, timeout_sec=0.2)

        self.latest_scan = np.ones(36)
        self.red_detected = 0.0
        self.green_detected = 0.0
        self.speed = 0.0

        rclpy.spin_once(self.node, timeout_sec=0.5)

        return self.get_observation(), {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()