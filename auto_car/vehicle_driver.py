import math
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


class VehicleDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        # ROS interface - check if already initialized by webots_ros2_driver
        if not rclpy.ok():
            rclpy.init(args=None)

        self.__node = rclpy.create_node('vehicle_driver_plugin')

        # Internal state for Odometry Fusion
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time = self.__node.get_clock().now()

        self.__node.get_logger().info('VehicleDriver: Initializing...')

        # Set gear to forward
        try:
            self.__robot.setGear(1)
            self.__node.get_logger().info('VehicleDriver: Gear set to 1')
        except Exception as e:
            self.__node.get_logger().error(f'VehicleDriver: Error setting gear: {e}')

        # Subscribe to velocity commands
        self.__node.create_subscription(Twist, '/cmd_vel', self.__cmd_vel_callback, 10)

        # Enable sensors
        self.camera = self.__robot.getDevice("front_camera")
        if self.camera:
            self.camera.enable(32)

        self.lidar = self.__robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(32)

        self.gps = self.__robot.getDevice("gps")
        if self.gps:
            self.gps.enable(32)

        self.imu = self.__robot.getDevice("imu")
        if self.imu:
            self.imu.enable(32)

        # ROS Publishers
        self.camera_pub = self.__node.create_publisher(Image, "/front_camera/image_raw", 10)
        self.lidar_pub  = self.__node.create_publisher(LaserScan, "/lidar/scan", 10)
        self.gps_pub    = self.__node.create_publisher(Float64MultiArray, "/gps", 10)
        self.imu_pub    = self.__node.create_publisher(Imu, "/imu", 10)
        self.odom_pub   = self.__node.create_publisher(Odometry, "/odom", 10)

        self.__node.get_logger().info('VehicleDriver: init complete')

    def __cmd_vel_callback(self, msg):
        self.__robot.setCruisingSpeed(msg.linear.x)
        self.__robot.setSteeringAngle(msg.angular.z)

    def step(self):
        # Process ROS callbacks
        if rclpy.ok():
            try:
                rclpy.spin_once(self.__node, timeout_sec=0)
            except Exception:
                pass

        # -------- CAMERA --------
        if self.camera:
            try:
                img = Image()
                img.header.stamp = self.__node.get_clock().now().to_msg()
                img.header.frame_id = "camera_link"
                img.height = self.camera.getHeight()
                img.width = self.camera.getWidth()
                img.encoding = "bgra8"
                img.step = img.width * 4
                img.data = bytearray(self.camera.getImage())
                self.camera_pub.publish(img)
            except Exception:
                pass

        # -------- LIDAR --------
        if self.lidar:
            try:
                scan = LaserScan()
                scan.header.stamp = self.__node.get_clock().now().to_msg()
                scan.header.frame_id = "lidar_link"
                ranges = list(self.lidar.getRangeImage())
                scan.angle_min = -3.14159
                scan.angle_max = 3.14159
                scan.angle_increment = 6.28318 / len(ranges) if ranges else 0.0
                scan.range_min = 0.1
                scan.range_max = 30.0
                scan.ranges = ranges
                self.lidar_pub.publish(scan)
            except Exception:
                pass

        # -------- GPS --------
        if self.gps:
            try:
                gps_msg = Float64MultiArray()
                gps_msg.data = list(self.gps.getValues())
                self.gps_pub.publish(gps_msg)
            except Exception:
                pass

        # -------- IMU --------
        if self.imu:
            try:
                imu_msg = Imu()
                imu_msg.header.stamp = self.__node.get_clock().now().to_msg()
                imu_msg.header.frame_id = "imu_link"
                roll, pitch, yaw = self.imu.getRollPitchYaw()
                imu_msg.orientation.x = roll
                imu_msg.orientation.y = pitch
                imu_msg.orientation.z = yaw
                self.imu_pub.publish(imu_msg)
            except Exception:
                pass

        # -------- ODOMETRY FUSION (Wheel Encoders + Steering) --------
        try:
            curr_time = self.__node.get_clock().now()
            dt = (curr_time - self.last_time).nanoseconds / 1e9
            self.last_time = curr_time

            speed    = self.__robot.getCurrentSpeed()   # m/s
            steering = self.__robot.getSteeringAngle()  # rad

            # Kinematic Bicycle Model (wheelbase L ≈ 2.9m for BmwX5)
            L = 2.9
            self.x   += speed * dt * math.cos(self.yaw)
            self.y   += speed * dt * math.sin(self.yaw)
            self.yaw += (speed / L) * math.tan(steering) * dt

            odom = Odometry()
            odom.header.stamp       = curr_time.to_msg()
            odom.header.frame_id    = "odom"
            odom.child_frame_id     = "base_link"
            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
            odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
            odom.twist.twist.linear.x  = speed
            odom.twist.twist.angular.z = (speed / L) * math.tan(steering)
            self.odom_pub.publish(odom)
        except Exception:
            pass

        # NOTE: self.__robot.step() should NOT be called here if used as a webots_ros2_driver plugin,
        # as the driver node handles the simulation stepping.
