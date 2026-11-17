#!/usr/bin/env python3
"""Check the camera detector and camera/lidar fusion in the traffic world (car parked).

  python3 test/detector_check.py --seconds 40 --save /tmp/det.png

Reports detector rate, classes seen, and how many MOVING lidar tracks inside the camera view
received a vehicle label. PASS if detections arrive at >= 8 Hz and >= 50% of moving tracks in
view are labelled as a vehicle at least once.
"""
import argparse
import collections
import math
import sys
import time

import numpy as np
import rclpy
from auto_car.msg import TrackArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

VEHICLES = {'car', 'truck', 'bus', 'motorcycle'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=40)
    ap.add_argument('--save')
    a = ap.parse_args()
    rclpy.init()
    n = Node('detector_check')
    st = {'det_stamps': [], 'classes': collections.Counter(), 'in_view': set(), 'labelled': set(), 'img': None}

    def on_det(m):
        st['det_stamps'].append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        for d in m.detections:
            st['classes'][d.results[0].hypothesis.class_id] += 1

    def on_tracks(m):
        for t in m.tracks:
            ang = math.degrees(math.atan2(t.position.y, t.position.x - 1.0))
            if t.moving and abs(ang) < 30 and 3 < t.position.x < 40:   # inside the 69 deg camera view
                st['in_view'].add(t.id)
                if t.label in VEHICLES:
                    st['labelled'].add(t.id)

    def on_img(m):
        if st['classes'] and st['img'] is None or (st['img'] is not None and sum(st['classes'].values()) % 7 == 0):
            st['img'] = m
    n.create_subscription(Detection2DArray, '/perception/detections', on_det, 10)
    n.create_subscription(TrackArray, '/perception/tracks_labeled', on_tracks, 10)
    n.create_subscription(Image, '/perception/detections_image', on_img, 2)
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.05)
    s = st['det_stamps']
    rate = (len(s) - 1) / (s[-1] - s[0]) if len(s) > 1 and s[-1] > s[0] else 0.0
    frac = len(st['labelled']) / len(st['in_view']) if st['in_view'] else float('nan')
    print(f'detection rate {rate:.1f} Hz (sim time), classes seen: {dict(st["classes"])}')
    print(f'moving lidar tracks in camera view: {len(st["in_view"])}, labelled as vehicle: {len(st["labelled"])} '
          f'({100 * frac:.0f}%)')
    if a.save and st['img'] is not None:
        from PIL import Image as PI
        m = st['img']
        PI.fromarray(np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)[:, :, ::-1]).save(a.save)
        print(f'annotated frame saved: {a.save}')
    ok = rate >= 8.0 and st['in_view'] and frac >= 0.5
    print(f"  [{'PASS' if rate >= 8 else 'FAIL'}] detector rate")
    print(f"  [{'PASS' if st['in_view'] and frac >= 0.5 else 'FAIL'}] moving lidar objects get a camera label")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
