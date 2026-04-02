#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from ultralytics import YOLO

# 👁️ TESLA-STYLE UNIFIED PERCEPTION
# - YOLOv8: Object Detection
# - Segmentation: Lane tracking for Stanley Control
# - Performance: 10Hz @ 320x160

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        # 1. Models & Tools
        self.model = YOLO('yolov8n.pt') 
        self.bridge = CvBridge()
        
        # 2. Subscribers
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        
        # 3. Unified Output Publishers
        self.world_pub = self.create_publisher(String, '/world_model', 10)
        self.lane_pub = self.create_publisher(Float32, '/lane_error', 10)
        self.inter_pub = self.create_publisher(Bool, '/intersection_flag', 10)
        
        # State
        self.get_logger().info('Tesla Vision Engine: Operational')

    def image_callback(self, msg):
        try:
            # --- Preprocessing ---
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_img_small = cv2.resize(cv_img, (320, 160)) 
            
            # --- 1. YOLO DETECTION ---
            results = self.model(cv_img_small, stream=True, verbose=False)
            
            world_data = {
                "obstacles": [],
                "intersection": False
            }
            
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.45:
                        # Map classes to simple types
                        label = "unknown"
                        if cls == 0: label = "pedestrian"
                        elif cls in [2, 7]: label = "vehicle"
                        elif cls == 11: label = "stop_sign"
                        elif cls == 9: label = "traffic_light"
                        
                        if label != "unknown":
                            world_data["obstacles"].append({"type": label, "conf": conf})
                            if label in ["stop_sign", "traffic_light"]:
                                world_data["intersection"] = True

            # --- 2. LANE SEGMENTATION (Cross-track Error) ---
            # Define ROI for lower road
            h, w, _ = cv_img_small.shape
            roi = cv_img_small[int(h*0.6):h, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Simple threshold for white/yellow lanes
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # Find centroid of lane markers
            M = cv2.moments(mask)
            lane_error = 0.0
            if M["m00"] > 100:
                cx = int(M["m10"] / M["m00"])
                # Normalized error from center (160 is half of 320)
                lane_error = (cx - 160) / 160.0 
            else:
                lane_error = 0.0 # No lane found
                
            # Heuristic: Check for many white pixels across horizontal (intersection stripes)
            if cv2.countNonZero(mask) > (roi.size * 0.2):
                world_data["intersection"] = True

            # --- 3. PUBLISH ---
            # World Model JSON
            msg_world = String()
            msg_world.data = json.dumps(world_data)
            self.world_pub.publish(msg_world)
            
            # Lane Error (Stanley Input)
            msg_lane = Float32()
            msg_lane.data = lane_error
            self.lane_pub.publish(msg_lane)
            
            # Dual-legacy support for Intersection Flag
            msg_inter = Bool(); msg_inter.data = world_data["intersection"]
            self.inter_pub.publish(msg_inter)

        except Exception as e:
            self.get_logger().error(f'Vision Fault: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
