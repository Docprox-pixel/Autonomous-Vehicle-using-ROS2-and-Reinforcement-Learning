#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrafficMonitor(Node):
    def __init__(self):
        super().__init__('traffic_monitor')
        self.bridge = CvBridge()
        
        # Subscribe to unified perception debug image
        self.create_subscription(Image, '/perception/debug_image', self.debug_callback, 10)
        
        self.debug_frame = None
        self.get_logger().info('Traffic Monitor Started. Windows will open when data arrives.')

    def debug_callback(self, msg):
        self.debug_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.show()

    def show(self):
        if self.debug_frame is not None:
            # Scale up for easy viewing
            display = cv2.resize(self.debug_frame, (640, 320))
            cv2.imshow("Perception Vision Output", display)
            cv2.waitKey(1)

def main():
    rclpy.init()
    node = TrafficMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
