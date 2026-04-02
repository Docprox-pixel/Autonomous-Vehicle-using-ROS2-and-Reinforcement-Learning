#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# 👁️ YOLOv8 PERCEPTION NODE
# Detects: Cars, Pedestrians, Crosswalks.
# Filter: 5-frame persistence (Stability focus).
# Target: 10 Hz inference.

class PerceptionNodeLegacy(Node):
    def __init__(self):
        super().__init__('perception_node_v8')
        
        # Load high-performance YOLOv8 Nano
        self.model = YOLO('yolov8n.pt') 
        self.bridge = CvBridge()
        
        # ROS Topics
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        # Clean Output Flags
        self.inter_pub = self.create_publisher(Bool, '/intersection_flag', 10)
        self.obst_pub = self.create_publisher(Bool, '/obstacle_flag', 10)
        
        # Persistence Filters (Last 5 frames)
        self.inter_history = deque(maxlen=5)
        self.obst_history = deque(maxlen=5)
        
        self.get_logger().info('YOLOv8 Perception Node Active (10Hz Target)')

    def image_callback(self, msg):
        try:
            # 1. Image Preprocessing (Resize for 10Hz performance)
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_img_small = cv2.resize(cv_img, (320, 160)) 
            
            # 2. YOLO Real-time Inference
            # Classes: 0: person, 2: car, 7: truck, 9: traffic light, 11: stop sign
            results = self.model(cv_img_small, stream=True, verbose=False)
            
            frame_intersection = False
            frame_obstacle = False
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > 0.45:
                        # Obstacle Detection
                        if cls in [0, 2, 7]: # Pedestrian, Car, Truck
                            frame_obstacle = True
                        
                        # Intersection / Scene Detection
                        # Proxies for crosswalk zones: stop signs or traffic lights
                        if cls in [9, 11]:
                            frame_intersection = True

            # 3. HEURISTIC CROSSWALK DETECTION (Stripes)
            # COCO doesn't natively have "crosswalk," so we look for patterns in the road ROI
            h, w, _ = cv_img_small.shape
            road_roi = cv_img_small[int(h*0.7):h, :]
            gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(thresh) > (road_roi.size * 0.15): # 15% white fill on road
                frame_intersection = True

            # 4. TEMPORAL PERSISTENCE (FIFO Consensus)
            self.inter_history.append(frame_intersection)
            self.obst_history.append(frame_obstacle)
            
            # Consensus: Detections are ONLY true if 100% of buffer is filled (5 frames)
            stable_inter = all(self.inter_history) if len(self.inter_history) == 5 else False
            stable_obst = all(self.obst_history) if len(self.obst_history) == 5 else False
            
            # 5. PUBLISH
            msg_inter = Bool(); msg_inter.data = stable_inter
            msg_obst = Bool(); msg_obst.data = stable_obst
            
            self.inter_pub.publish(msg_inter)
            self.obst_pub.publish(msg_obst)
            
            # Debug Log (Every 10 frames)
            # if rclpy.clock.Clock().now().nanoseconds % 10 == 0:
            #     self.get_logger().info(f"[YOLOv8] OBST:{int(stable_obst)} INT:{int(stable_inter)}")

        except Exception as e:
            self.get_logger().error(f'Perception Fault: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNodeLegacy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
