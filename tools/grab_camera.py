#!/usr/bin/env python3
"""Save one camera frame:  python3 tools/grab_camera.py OUT.png [topic]"""
import sys
import numpy as np
import rclpy
from PIL import Image as PI
from rclpy.node import Node
from sensor_msgs.msg import Image
rclpy.init()
n = Node('grab_camera')
got = []
n.create_subscription(Image, sys.argv[2] if len(sys.argv) > 2 else '/camera/image_raw', lambda m: got.append(m), 2)
while not got:
    rclpy.spin_once(n, timeout_sec=0.1)
m = got[-1]
a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
PI.fromarray(a[:, :, [2, 1, 0]] if m.encoding.startswith('bgr') else a[:, :, :3]).save(sys.argv[1])
print(sys.argv[1], m.width, m.height)
