#!/usr/bin/env python3
"""Simulated driver vitals sensor (wearable / seat sensor) -> /driver/vitals for driver_monitor.

  python3 tools/driver_vitals.py normal  [SECONDS]   # healthy: HR ~72 bpm, SpO2 ~98 %
  python3 tools/driver_vitals.py collapse [SECONDS]  # healthy 20 s, then cardiac event: HR 32, SpO2 82
Also sends 'steering_touch' on /driver/input while the driver is healthy (hands on the wheel).
"""
import json
import random
import sys
import time

import rclpy
from std_msgs.msg import String

mode = sys.argv[1] if len(sys.argv) > 1 else 'normal'
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
rclpy.init()
n = rclpy.create_node('driver_vitals_sim')
pub_v = n.create_publisher(String, '/driver/vitals', 5)
pub_i = n.create_publisher(String, '/driver/input', 5)
t0 = time.time()
while time.time() - t0 < dur:
    t = time.time() - t0
    collapsed = mode == 'collapse' and t > 20.0
    hr = random.gauss(32, 2) if collapsed else random.gauss(72, 3)
    sp = random.gauss(82, 1) if collapsed else random.gauss(98, 0.5)
    pub_v.publish(String(data=json.dumps({'heart_rate': round(hr, 1), 'spo2': round(sp, 1)})))
    if not collapsed:
        pub_i.publish(String(data='steering_touch'))
    rclpy.spin_once(n, timeout_sec=0.0)
    time.sleep(1.0)
