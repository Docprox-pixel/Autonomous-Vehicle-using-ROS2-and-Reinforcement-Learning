#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node_v8')
        
        # Load YOLOv8 Nano (Fastest for 10Hz)
        self.model = YOLO('yolov8n.pt') 
        self.bridge = CvBridge()
        
        # ROS 2 Topics
        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, 10)
        self.inter_pub = self.create_publisher(Bool, '/intersection_flag', 10)
        self.obst_pub = self.create_publisher(Bool, '/obstacle_flag', 10)
        
        # Persistence Filters (5 frames)
        self.inter_history = []
        self.obst_history = []
        self.persistence_threshold = 5
        
        self.get_logger().info('YOLOv8 Perception Node Active (~10Hz Target)')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 1. YOLO Inference
            # Classes: 0 (person), 2 (car), 5 (bus), 7 (truck)
            results = self.model(cv_image, stream=True, verbose=False)
            
            current_obstacle = False
            current_intersection = False
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > 0.4:
                        # Obstacle Detection (Pedestrians/Cars)
                        if cls in [0, 2, 5, 7]:
                            current_obstacle = True
                        
                        # Intersection Awareness (Stop signs/Traffic lights/Crosswalk heuristics)
                        # Standard COCO doesn't have "crosswalk", 
                        # so we use 9 (traffic light) and 11 (stop sign) as proxies
                        if cls in [9, 11]:
                            current_intersection = True

            # 2. ADDITIONAL: Heuristic Crosswalk Detection (White/Yellow stripes in ROI)
            # This complements YOLO for "Human-like" vision
            h, w, _ = cv_image.shape
            roi = cv_image[int(h*0.7):h, :] # Look at the road surface
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(thresh) > (roi.size * 0.1): # 10% white coverage
                current_intersection = True

            # 3. APPLY PERSISTENCE (FIFO Buffer)
            self.inter_history.append(current_intersection)
            self.obst_history.append(current_obstacle)
            
            if len(self.inter_history) > self.persistence_threshold:
                self.inter_history.pop(0)
            if len(self.obst_history) > self.persistence_threshold:
                self.obst_history.pop(0)
                
            # Decisions require ALL 5 frames to match (Conservative safety)
            final_inter = all(self.inter_history) if len(self.inter_history) >= 5 else False
            final_obst = all(self.obst_history) if len(self.obst_history) >= 5 else False
            
            # 4. PUBLISH
            msg_inter = Bool(); msg_inter.data = final_inter
            msg_obst = Bool(); msg_obst.data = final_obst
            
            self.inter_pub.publish(msg_inter)
            self.obst_pub.publish(msg_obst)
            
        except Exception as e:
            self.get_logger().error(f'Perception Error: {str(e)}')

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
