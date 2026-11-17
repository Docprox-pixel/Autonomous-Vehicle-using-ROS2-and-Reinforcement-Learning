#!/usr/bin/env python3
"""Camera object detection (RT-DETR-L) + camera/lidar fusion.

Input : /camera/image_raw (bgra8), /camera/camera_info, /perception/tracks (lidar objects)
Output: /perception/detections        vision_msgs/Detection2DArray (pixel boxes, class, score)
        /perception/detections_image  annotated image (debug, ~5 Hz)
        /perception/tracks_labeled    auto_car/TrackArray: the lidar tracks with a camera label
                                      ('person', 'car', ...) when the track projects into a box

Model: RT-DETR-L (transformer detector, NMS-free). Chosen over YOLO26s after it found distant
traffic lights in our Webots frames that YOLO26s missed (18.6 vs 9.1 ms/frame, both well inside
the 66 ms camera period). The model is a parameter: models/yolo26s.pt works the same way.

Only classes the car acts on are kept; everything else (e.g. 'bench' on barriers) is dropped.
The lidar remains responsible for WHERE obstacles are; the camera says WHAT they are.
Processing always uses the newest frame, so latency never accumulates.
"""
import math
import os
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from auto_car.msg import TrackArray
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

KEEP = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign'}


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        share = get_package_share_directory('auto_car')
        p = self.declare_parameter
        model_path = p('model', os.path.join(share, 'models', 'rtdetr-l.pt')).value
        self.conf = p('confidence', 0.30).value
        self.rate = p('rate', 15.0).value
        self.debug_rate = p('debug_image_rate', 5.0).value
        layout = yaml.safe_load(open(p('sensor_layout', os.path.join(share, 'config', 'sensor_layout.yaml')).value))
        roof = next(l for l in layout['lidars'] if l['name'] == 'lidar')
        self.cam = np.asarray(roof['xyz'], float)       # camera shares the roof mount (base_link)
        self.ground_z = layout['ground_z']

        import torch
        from ultralytics import RTDETR, YOLO
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls = RTDETR if 'rtdetr' in os.path.basename(model_path).lower() else YOLO
        self.model = cls(model_path)
        self.model(np.zeros((480, 640, 3), np.uint8), verbose=False, device=self.device)   # warm-up
        self.names = self.model.names

        self.image = self.info = None
        self.tracks = None
        self.last_debug = 0.0
        self.timing = []
        self.pub = self.create_publisher(Detection2DArray, '/perception/detections', 10)
        self.pub_img = self.create_publisher(Image, '/perception/detections_image', 2)
        self.pub_tracks = self.create_publisher(TrackArray, '/perception/tracks_labeled', 10)
        self.memory = {}                    # track id -> (label, score, time last seen by the camera)
        self.label_memory = self.declare_parameter('label_memory', 10.0).value   # s
        self.create_subscription(Image, '/camera/image_raw', lambda m: setattr(self, 'image', m), 2)
        self.create_subscription(CameraInfo, '/camera/camera_info', lambda m: setattr(self, 'info', m), 2)
        self.create_subscription(TrackArray, '/perception/tracks', lambda m: setattr(self, 'tracks', m), 5)
        # Surround cameras (rear/left/right): one of them is processed per tick (round robin), so
        # each runs at rate/3 Hz on top of the front camera. They label lidar tracks all around the
        # car (a person beside or behind it); traffic lights/signs still come from the front camera.
        self.surround = []
        for c in layout.get('surround_cameras', []):
            cam = {'spec': c, 'image': None, 'info': None,
                   'pos': np.asarray(roof['xyz'], float) + np.asarray(c['xyz'], float), 'yaw': float(c['yaw'])}
            self.create_subscription(Image, c['topic'] + '/image_raw', lambda m, cam=cam: cam.__setitem__('image', m), 2)
            self.create_subscription(CameraInfo, c['topic'] + '/camera_info', lambda m, cam=cam: cam.__setitem__('info', m), 2)
            self.surround.append(cam)
        self.rr = 0
        self.pub_surround = self.create_publisher(Detection2DArray, '/perception/detections_surround', 10)
        self.create_timer(1.0 / self.rate, self.loop)
        self.get_logger().info(f'object_detector: {os.path.basename(model_path)} on {self.device}, '
                               f'conf >= {self.conf}, classes {sorted(KEEP)}')

    # ----------------------------------------------------------------- loop
    def loop(self):
        msg, self.image = self.image, None          # newest frame only
        if msg is None:
            return
        t0 = time.perf_counter()
        img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        res = self.model(np.ascontiguousarray(img), verbose=False, conf=self.conf, device=self.device)[0]
        dets = []
        out = Detection2DArray(header=msg.header)
        for b in res.boxes:
            name = self.names[int(b.cls)]
            if name not in KEEP:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            score = float(b.conf)
            dets.append((name, score, x1, y1, x2, y2))
            d = Detection2D(header=msg.header)
            d.bbox.center.position.x, d.bbox.center.position.y = (x1 + x2) / 2, (y1 + y2) / 2
            d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = name, score
            d.results.append(h)
            out.detections.append(d)
        self.pub.publish(out)
        self.update_labels(dets, lambda x, y, z: self.project(x, y, z))
        self.surround_step()
        self.publish_labels()
        self.timing.append((time.perf_counter() - t0) * 1000)
        if len(self.timing) >= 100:
            self.get_logger().info(f'detector: {np.mean(self.timing):.0f} ms/frame mean, {np.max(self.timing):.0f} max')
            self.timing = []
        now = time.time()
        if now - self.last_debug > 1.0 / self.debug_rate:
            self.last_debug = now
            self.publish_debug(img, dets, msg)

    # --------------------------------------------------------------- fusion
    def project(self, x, y, z):
        """base_link point -> pixel (u, v), or None if behind the camera."""
        if self.info is None:
            return None
        fx, cx, cy = self.info.k[0], self.info.k[2], self.info.k[5]
        dx, dy, dz = x - self.cam[0], y - self.cam[1], z - self.cam[2]
        if dx < 0.5:
            return None
        return cx - fx * dy / dx, cy - fx * dz / dx

    def project_surround(self, cam, x, y, z):
        """base_link point -> pixel in a surround camera (yawed pinhole), or None if behind it."""
        if cam['info'] is None:
            return None
        fx, cx, cy = cam['info'].k[0], cam['info'].k[2], cam['info'].k[5]
        dx, dy, dz = x - cam['pos'][0], y - cam['pos'][1], z - cam['pos'][2]
        c, s_ = math.cos(cam['yaw']), math.sin(cam['yaw'])
        fx_, fy_ = c * dx + s_ * dy, -s_ * dx + c * dy           # into the camera's body frame
        if fx_ < 0.5:
            return None
        return cx - fx * fy_ / fx_, cy - fx * dz / fx_

    def surround_step(self):
        if not self.surround:
            return
        cam = self.surround[self.rr % len(self.surround)]
        self.rr += 1
        msg, cam['image'] = cam['image'], None
        if msg is None:
            return
        img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        res = self.model(np.ascontiguousarray(img), verbose=False, conf=self.conf, device=self.device)[0]
        dets, out = [], Detection2DArray(header=msg.header)
        for b in res.boxes:
            name = self.names[int(b.cls)]
            if name not in KEEP or name in ('traffic light', 'stop sign'):
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            dets.append((name, float(b.conf), x1, y1, x2, y2))
            d = Detection2D(header=msg.header)
            d.bbox.center.position.x, d.bbox.center.position.y = (x1 + x2) / 2, (y1 + y2) / 2
            d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = name, float(b.conf)
            d.results.append(h)
            out.detections.append(d)
        self.pub_surround.publish(out)
        self.update_labels(dets, lambda x, y, z: self.project_surround(cam, x, y, z))

    def update_labels(self, dets, project):
        """Match detections of one camera to lidar tracks (track centre projected into the image)."""
        if self.tracks is None:
            return
        now = time.time()
        for tr in self.tracks.tracks:
            uv = project(tr.position.x, tr.position.y, self.ground_z + tr.size.z / 2)
            if uv is None:
                continue
            u, v = uv
            best = None
            for name, score, x1, y1, x2, y2 in dets:
                if name in ('traffic light', 'stop sign'):
                    continue                    # signals are not lidar tracks on the road
                if x1 <= u <= x2 and y1 <= v <= y2 and (best is None or score > best[1]):
                    best = (name, score)
            if best:
                self.memory[tr.id] = (best[0], best[1], now)

    def publish_labels(self):
        """Tracks with the label any camera gave them recently. Sticky for label_memory seconds: a
        person who walks out of every camera's view is still a person (the pedestrian speed rule
        used to lapse right next to them)."""
        if self.tracks is None:
            return
        now = time.time()
        out = TrackArray(header=self.tracks.header)
        for tr in self.tracks.tracks:
            m = self.memory.get(tr.id)
            tr.label, tr.label_score = (m[0], m[1]) if m and now - m[2] < self.label_memory else ('', 0.0)
            out.tracks.append(tr)
        live = {t.id for t in self.tracks.tracks}
        self.memory = {k: v for k, v in self.memory.items() if k in live}
        self.pub_tracks.publish(out)

    def publish_debug(self, img, dets, msg):
        vis = img.copy()
        colours = {'person': (0, 0, 255), 'bicycle': (255, 0, 255), 'traffic light': (0, 255, 255),
                   'stop sign': (0, 128, 255)}
        for name, score, x1, y1, x2, y2 in dets:
            c = np.array(colours.get(name, (0, 255, 0)), np.uint8)
            x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
            x1, x2 = max(x1, 0), min(x2, vis.shape[1] - 1)
            y1, y2 = max(y1, 0), min(y2, vis.shape[0] - 1)
            for t in range(2):
                vis[y1 + t, x1:x2] = c
                vis[y2 - t, x1:x2] = c
                vis[y1:y2, x1 + t] = c
                vis[y1:y2, x2 - t] = c
        out = Image(header=msg.header, height=vis.shape[0], width=vis.shape[1], encoding='bgr8',
                    step=vis.shape[1] * 3, data=np.ascontiguousarray(vis).tobytes())
        self.pub_img.publish(out)


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
