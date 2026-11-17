#!/usr/bin/env python3
"""Steering system identification: how does the REAL (Webots) car respond to a steering command?

  python3 tools/steer_id.py            (sim running with stack:=false, car on its start road)

Drives /cmd_drive directly (no planner, no shield - run it only on the open start road):
  1. 3.5 s straight, accelerating to SPEED
  2. 7.5 s gentle steering wave: 0.08 rad, period 2.5 s
  3. 2.0 s straight, then a 0.10 rad step held 1.5 s, then straight and stop
Records the command, the bridge's reported steering (/vehicle/steering), the IMU yaw rate and the
wheel speed, and reports:
  * delay  : command -> yaw-rate response (cross-correlation over the wave)
  * gain   : measured yaw rate / kinematic yaw rate v*tan(cmd)/wheelbase
  * step   : 63 % rise time of the yaw rate after the step (first-order time constant)
These are the numbers the planner and tools/dry_run.py must use for the car model.
"""
import math
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

SPEED = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
WB = 2.995
rclpy.init()
n = rclpy.create_node('steer_id')
pub = n.create_publisher(AckermannDriveStamped, '/cmd_drive', 10)
st = {'v': 0.0, 'w': 0.0, 'steer': 0.0}
n.create_subscription(Odometry, '/wheel/odom', lambda m: st.__setitem__('v', m.twist.twist.linear.x), 20)
n.create_subscription(Imu, '/imu/data', lambda m: st.__setitem__('w', m.angular_velocity.z), 50)
n.create_subscription(Float32, '/vehicle/steering', lambda m: st.__setitem__('steer', m.data), 20)


def command(t):
    if t < 3.5:
        return SPEED, 0.0
    if t < 11.0:
        return SPEED, 0.08 * math.sin(2 * math.pi * (t - 3.5) / 2.5)
    if t < 13.0:
        return SPEED, 0.0
    if t < 14.5:
        return SPEED, 0.10
    if t < 16.0:
        return SPEED, 0.0
    return 0.0, 0.0


rows, t0, last = [], time.time(), 0.0
while time.time() - t0 < 18.0:
    rclpy.spin_once(n, timeout_sec=0.005)
    t = time.time() - t0
    if t - last < 0.05:
        continue
    last = t
    v_c, s_c = command(t)
    m = AckermannDriveStamped()
    m.header.stamp = n.get_clock().now().to_msg()
    m.header.frame_id = 'base_link'
    m.drive.speed, m.drive.steering_angle = float(v_c), float(s_c)
    pub.publish(m)
    rows.append((t, s_c, st['steer'], st['w'], st['v']))
for _ in range(10):
    pub.publish(AckermannDriveStamped())
    time.sleep(0.05)
R = np.array(rows)
t, cmd, rep, w, v = R.T
# wave segment
sel = (t > 4.5) & (t < 11.0)
w_kin = v * np.tan(cmd) / WB
lags = np.arange(0, 40)                                    # 0 .. 2 s in 0.05 s steps
cc = [np.corrcoef(w_kin[sel][: sel.sum() - L], w[sel][L:])[0, 1] if sel.sum() - L > 20 else -1 for L in lags]
L = int(np.argmax(cc))
delay = L * 0.05
a = w_kin[sel][: sel.sum() - L]
b = w[sel][L:]
gain = float(np.dot(a, b) / max(np.dot(a, a), 1e-9))
rep_cc = [np.corrcoef(cmd[sel][: sel.sum() - k], rep[sel][k:])[0, 1] if sel.sum() - k > 20 else -1 for k in lags]
rep_delay = int(np.argmax(rep_cc)) * 0.05
# step
ss = (t > 13.0) & (t < 14.5)
w_ss = np.median(w[(t > 14.0) & (t < 14.5)])
rise = None
if abs(w_ss) > 1e-3:
    idx = np.flatnonzero(ss & (np.abs(w) >= 0.63 * abs(w_ss)))
    rise = (t[idx[0]] - 13.0) if len(idx) else None
print(f'speed during test {np.median(v[sel]):.2f} m/s')
print(f'bridge-reported steering follows the command after {rep_delay:.2f} s')
print(f'YAW-RATE DELAY (command -> car turning): {delay:.2f} s (correlation {cc[L]:.2f})')
print(f'gain: measured / kinematic yaw rate = {gain:.2f}  (1.0 = car turns exactly as the bicycle model)')
print(f'step 0.10 rad: steady yaw rate {w_ss:.3f} rad/s (kinematic {np.median(v[ss]) * math.tan(0.1) / WB:.3f}), '
      f'63% rise after {rise if rise is None else round(rise, 2)} s')
np.save('tools/sim/logs/steer_id.npy', R)
