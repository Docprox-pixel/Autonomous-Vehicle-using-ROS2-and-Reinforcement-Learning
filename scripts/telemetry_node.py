#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import json
import time

from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist

class TelemetryNode(Node):
    def __init__(self):
        super().__init__('telemetry_node')

        # State storage
        self.cmd_speed = 0.0
        self.cmd_steer = 0.0
        self.lane_error = 0.0
        self.objects = "None"
        
        # Subscribers
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.create_subscription(String, '/world_model', self.model_cb, 10)
        
        # Timer for throttled display (10 Hz refresh)
        self.create_timer(0.1, self.print_dashboard)
        
    def cmd_cb(self, msg):
        self.cmd_speed = msg.linear.x
        self.cmd_steer = msg.angular.z
        
    def model_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.lane_error = data.get("lane_error", 0.0)
            
            obs = data.get("obstacles", [])
            if len(obs) > 0:
                self.objects = ", ".join([f"{o['class']} ({o['conf']:.2f})" for o in obs])
            else:
                self.objects = "None"
        except Exception:
            pass

    def print_dashboard(self):
        # ANSI escape sequence to clear line and return to beginning
        # We print a structured row to the terminal 
        output = (f"\r🏎️  TELEMETRY | Speed: {self.cmd_speed:4.1f} m/s | "
                  f"Steer: {self.cmd_steer:5.2f} rad | "
                  f"Lane Err: {self.lane_error:5.2f} | "
                  f"Objects: {self.objects:<20}")
        print(output, end="", flush=True)

def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nTelemetry Stopped.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
