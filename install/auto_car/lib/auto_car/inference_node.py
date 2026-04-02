#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import numpy as np

# Re-defining model for loading (Same architecture)
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 34, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.image_sub = self.create_subscription(Image, '/front_camera/image_raw', self.image_cb, 10)
        self.steer_pub = self.create_publisher(Float32, '/ml/steering', 10)
        self.bridge = CvBridge()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PilotNet().to(self.device)
        try:
            self.model.load_state_dict(torch.load('pilotnet.pth', map_location=self.device))
            self.model.eval()
            self.get_logger().info('ML Brain Loaded: pilotnet.pth active.')
        except:
            self.get_logger().warn('Warning: pilotnet.pth not found. System waiting for training.')

    def image_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv_img = cv2.resize(cv_img, (320, 160)) / 255.0
            image = np.transpose(cv_img, (2, 0, 1)).astype(np.float32)
            image = torch.from_numpy(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_steer = self.model(image).item()
                
            steer_m = Float32(); steer_m.data = pred_steer
            self.steer_pub.publish(steer_m)
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
