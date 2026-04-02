#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import csv
import os
import time

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        
        # Subscriptions
        self.image_sub = self.create_subscription(Image, '/front_camera/image_raw', self.image_cb, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        
        # Dataset storage
        self.dataset_dir = 'dataset'
        self.img_dir = os.path.join(self.dataset_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(self.dataset_dir, 'labels.csv')
        
        self.bridge = CvBridge()
        self.last_steering = 0.0
        self.last_speed = 0.0
        self.frame_count = 0
        self.save_interval = 5 # Save every 5th frame to avoid redundancy
        
        self.get_logger().info('Data Collection Node Active: Recording to dataset/')

    def cmd_cb(self, msg):
        self.last_steering = msg.angular.z
        self.last_speed = msg.linear.x

    def image_cb(self, msg):
        self.frame_count += 1
        if self.frame_count % self.save_interval != 0:
            return
            
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # NVIDIA-style preprocessing: resize 320x160
            cv_img_resized = cv2.resize(cv_img, (320, 160))
            
            timestamp = int(time.time() * 1000)
            img_filename = f'img_{timestamp}.jpg'
            img_path = os.path.join(self.img_dir, img_filename)
            cv2.imwrite(img_path, cv_img_resized)
            
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([img_filename, self.last_steering, self.last_speed])
                
        except Exception as e:
            self.get_logger().error(f'Data Save Failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
