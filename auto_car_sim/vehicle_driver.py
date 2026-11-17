"""Webots bridge for the auto_car stack (webots_ros2_driver Python plugin).

This is the ONLY file that knows about Webots. It converts the simulator into
the simulator-agnostic "sensor contract" that the rest of the stack uses, so the
same autonomy code can later run on Isaac Sim with a different bridge.

Contract (SI units, ROS REP-103: x forward, y left, z up, +steering = left):
  out  /camera/image_raw      sensor_msgs/Image          frame camera_link
  out  /camera/camera_info    sensor_msgs/CameraInfo     frame camera_link
  out  /scan                  sensor_msgs/LaserScan      frame lidar_link  (roof lidar, one horizontal layer)
  out  /lidar/points          sensor_msgs/PointCloud2    frame base_link   (ALL lidars merged, all layers)
  out  /imu/data              sensor_msgs/Imu            frame imu_link
  out  /wheel/odom            nav_msgs/Odometry          odom -> base_link
  out  /gps/position          geometry_msgs/PointStamped (optional, sim-local metres)
  out  /vehicle/speed         std_msgs/Float32           m/s
  out  /vehicle/steering      std_msgs/Float32           rad, +left
  out  TF base_link -> {camera_link, imu_link, <every lidar frame>}  (static)
Lidar names, poses and frames come from config/sensor_layout.yaml (shared with
tools/coverage_check.py and tools/install_sensors.py).
  out  TF odom -> base_link   (only if publish_odom_tf, until the EKF takes over)
  in   /cmd_drive             ackermann_msgs/AckermannDriveStamped
  in   /cmd_vel               geometry_msgs/Twist  (debug teleop: linear.x m/s, angular.z = steering rad)

Webots quirks handled here (verified against the R2025a driver.c source):
  * wbu_driver_set_cruising_speed / get_current_speed use km/h  -> converted to m/s.
  * Webots positive steering angle turns RIGHT                  -> sign flipped.
  * Lidar range image runs clockwise (left to right)             -> reversed for ROS (CCW).
"""

import math

import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

KMH_PER_MS = 3.6
WEBOTS_STEER_SIGN = -1.0  # ROS +left  <->  Webots +right


def _prop(properties, name, default, cast):
    """Read an optional <plugin> child tag from the URDF."""
    value = properties.get(name)
    if value is None:
        return default
    if cast is bool:
        return str(value).strip().lower() in ('1', 'true', 'yes')
    return cast(value)


def _yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class VehicleDriver:
    # ------------------------------------------------------------------ init
    def init(self, webots_node, properties):
        self.robot = webots_node.robot
        self.timestep = int(self.robot.getBasicTimeStep())

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node(
            'vehicle_bridge',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        from rclpy.executors import SingleThreadedExecutor
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        log = self.node.get_logger()

        # ---- configuration (URDF <plugin> children can override) ----------
        self.wheelbase = _prop(properties, 'wheelbase', 2.995, float)          # BmwX5 PROTO
        self.max_steer = _prop(properties, 'maxSteeringAngle', 0.5, float)     # rad
        self.max_speed = _prop(properties, 'maxSpeed', 25.0, float)            # m/s
        self.brake_full_at = _prop(properties, 'brakeFullAt', 2.0, float)      # m/s overspeed -> 100% brake
        self.publish_odom_tf = _prop(properties, 'publishOdomTf', True, bool)  # off once EKF runs
        self.camera_period = _prop(properties, 'cameraPeriodMs', 64, int)      # ~15 Hz
        self.lidar_period = _prop(properties, 'lidarPeriodMs', 96, int)        # ~10 Hz
        self.imu_period = _prop(properties, 'imuPeriodMs', 16, int)            # ~60 Hz
        self.scan_layer = _prop(properties, 'scanLayer', -1, int)              # -1 = middle layer
        layout_path = _prop(properties, 'sensorLayout', '', str) or os.path.join(
            get_package_share_directory('auto_car'), 'config', 'sensor_layout.yaml')
        self.layout = yaml.safe_load(open(layout_path))
        roof = next(l for l in self.layout['lidars'] if l['name'] == 'lidar')
        mount = tuple(roof['xyz'])  # camera, IMU and GPS share the roof mount

        # ---- vehicle ------------------------------------------------------
        try:
            self.robot.setGear(1)
        except Exception as exc:  # gearbox-less models
            log.warn(f'setGear failed: {exc}')
        self.robot.setCruisingSpeed(0.0)
        self.robot.setSteeringAngle(0.0)
        self.cmd_speed = 0.0     # m/s
        self.cmd_steer = 0.0     # rad, ROS convention (+left)

        # Wheel encoders. webots_ros2 steps with wb_robot_step(), not wbu_driver_step(),
        # so Driver.getCurrentSpeed() is never updated and always returns 0. We therefore
        # measure speed ourselves from the (non-steered) rear wheel position sensors.
        self.wheel_radius = _prop(properties, 'wheelRadius', 0.374, float)   # BmwX5Wheel tireRadius
        self.wheel_sensors = []
        for name in ('left_rear_sensor', 'right_rear_sensor'):
            ps = self.robot.getDevice(name)
            if ps is not None:
                ps.enable(self.timestep)
                self.wheel_sensors.append(ps)
        if not self.wheel_sensors:
            log.warn('no rear wheel position sensors found: odometry speed falls back to getCurrentSpeed()')
        self.prev_wheel_pos = None
        self.wheel_speed = 0.0

        # ---- sensors (all optional so other worlds still load) -----------
        self.camera = self._enable('front_camera', self.camera_period)
        # Surround cameras (rear/left/right), ~10 Hz: see config/sensor_layout.yaml
        self.surround = []
        for c in self.layout.get('surround_cameras', []):
            dev = self._enable(c['name'], max(self.camera_period, 200))
            if dev is not None:
                self.surround.append({'spec': c, 'dev': dev, 'next': 0.0})
        self.imu = self._enable('imu', self.imu_period)
        self.gyro = self._enable('gyro', self.imu_period)
        self.accel = self._enable('accelerometer', self.imu_period)
        self.gps = self._enable('gps', self.lidar_period)
        self.lidars = []
        for spec in self.layout['lidars']:
            dev = self._enable(spec['name'], self.lidar_period)
            if dev is not None:
                self.lidars.append(self._prepare_lidar(dev, spec))
        self.roof = next((u for u in self.lidars if u['name'] == 'lidar'), None)

        # ---- publishers -------------------------------------------------
        n = self.node
        self.pub_image = n.create_publisher(Image, '/camera/image_raw', 5)
        self.pub_info = n.create_publisher(CameraInfo, '/camera/camera_info', 5)
        for c in self.surround:
            c['pub_img'] = n.create_publisher(Image, c['spec']['topic'] + '/image_raw', 2)
            c['pub_info'] = n.create_publisher(CameraInfo, c['spec']['topic'] + '/camera_info', 2)
        self.pub_scan = n.create_publisher(LaserScan, '/scan', 5)
        self.pub_points = n.create_publisher(point_cloud2.PointCloud2, '/lidar/points', 5)
        self.pub_imu = n.create_publisher(Imu, '/imu/data', 20)
        self.pub_odom = n.create_publisher(Odometry, '/wheel/odom', 20)
        self.pub_gps = n.create_publisher(PointStamped, '/gps/position', 5)
        self.pub_speed = n.create_publisher(Float32, '/vehicle/speed', 20)
        self.pub_steer = n.create_publisher(Float32, '/vehicle/steering', 20)
        self.tf = TransformBroadcaster(n)
        self.static_tf = StaticTransformBroadcaster(n)
        self._publish_static_tf(mount)

        # ---- command inputs ---------------------------------------------
        n.create_subscription(AckermannDriveStamped, '/cmd_drive', self._on_cmd_drive, 10)
        n.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        # Lights: /vehicle/headlights (dipped beams), /vehicle/hazard (hazard flashers),
        # /vehicle/indicator ('left' | 'right' | 'off'). Brake lights follow braking automatically.
        from std_msgs.msg import Bool, String as Str
        n.create_subscription(Bool, '/vehicle/headlights', lambda m: self._light('setDippedBeams', m.data), 5)
        n.create_subscription(Bool, '/vehicle/hazard', lambda m: self._light('setHazardFlashers', m.data), 5)
        n.create_subscription(Str, '/vehicle/indicator', lambda m: self._light(
            'setIndicator', {'left': self.robot.INDICATOR_LEFT, 'right': self.robot.INDICATOR_RIGHT}.get(
                m.data, self.robot.INDICATOR_OFF)), 5)
        self.last_cmd_time = -1.0
        self.cmd_timeout = 0.5  # s of sim time without commands -> stop

        # ---- odometry state ---------------------------------------------
        self.x = self.y = self.yaw = 0.0
        self.last_odom_t = None
        self.next_pub = {'camera': 0.0, 'lidar': 0.0, 'imu': 0.0}

        log.info(f'vehicle_bridge ready: timestep={self.timestep} ms, camera={bool(self.camera)}, surround={[c["spec"]["name"] for c in self.surround]}, '
                 f'lidars={[u["name"] for u in self.lidars]}, imu={bool(self.imu)}, gyro={bool(self.gyro)}, '
                 f'accel={bool(self.accel)}, gps={bool(self.gps)}')

    def _enable(self, name, period_ms):
        dev = self.robot.getDevice(name)
        if dev is None:
            self.node.get_logger().warn(f"device '{name}' not found in world")
            return None
        dev.enable(max(self.timestep, period_ms))
        return dev

    def _prepare_lidar(self, dev, spec):
        """Precompute ray directions (sensor frame) and the mount transform to base_link."""
        res, layers = dev.getHorizontalResolution(), dev.getNumberOfLayers()
        fov, vfov = dev.getFov(), dev.getVerticalFov()
        # After reversing, column i points at azimuth  -fov/2 + i * inc  (CCW, ROS).
        full_circle = abs(fov - 2 * math.pi) < 1e-3
        inc = fov / res if full_circle else fov / max(res - 1, 1)
        az = -fov / 2.0 + np.arange(res) * inc
        # Webots layer 0 is the top layer.
        el = vfov / 2.0 - np.arange(layers) * vfov / (layers - 1) if layers > 1 else np.zeros(1)
        cos_el = np.cos(el)[:, None]
        dirs = np.stack([cos_el * np.cos(az)[None, :], cos_el * np.sin(az)[None, :],
                         np.repeat(np.sin(el)[:, None], res, axis=1)], axis=-1)
        c, s_ = math.cos(spec['yaw']), math.sin(spec['yaw'])
        rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        layer = self.scan_layer if self.scan_layer >= 0 else layers // 2
        return {'name': spec['name'], 'frame': spec['frame'], 'dev': dev, 'res': res, 'layers': layers,
                'fov': fov, 'inc': inc, 'min': max(dev.getMinRange(), float(spec.get('near_blank', 0.0))), 'max': dev.getMaxRange(),
                'scan_layer': min(layer, layers - 1),
                'dirs_base': dirs @ rot.T,                      # ray directions rotated into base_link
                'origin': np.asarray(spec['xyz'], dtype=float),
                'yaw': spec['yaw']}

    def _publish_static_tf(self, mount):
        stamp = self._stamp()
        tfs = []
        tfs.append(self._make_tf('base_link', 'imu_link', mount, (0.0, 0.0, 0.0, 1.0), stamp))
        for u in self.lidars:
            q = (0.0, 0.0, math.sin(u['yaw'] / 2.0), math.cos(u['yaw'] / 2.0))
            tfs.append(self._make_tf('base_link', u['frame'], tuple(u['origin']), q, stamp))
        # camera_link = body-style (x fwd); camera_optical_frame = z fwd, x right, y down.
        tfs.append(self._make_tf('base_link', 'camera_link', mount, (0.0, 0.0, 0.0, 1.0), stamp))
        tfs.append(self._make_tf('camera_link', 'camera_optical_frame', (0.0, 0.0, 0.0),
                                 (-0.5, 0.5, -0.5, 0.5), stamp))
        for c in self.surround:
            sp = c['spec']
            xyz = tuple(m + d for m, d in zip(mount, sp['xyz']))
            q = (0.0, 0.0, math.sin(sp['yaw'] / 2.0), math.cos(sp['yaw'] / 2.0))
            tfs.append(self._make_tf('base_link', sp['name'] + '_link', xyz, q, stamp))
            tfs.append(self._make_tf(sp['name'] + '_link', sp['name'] + '_optical_frame', (0.0, 0.0, 0.0),
                                     (-0.5, 0.5, -0.5, 0.5), stamp))
        self.static_tf.sendTransform(tfs)

    @staticmethod
    def _make_tf(parent, child, xyz, quat, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = quat
        return t

    # -------------------------------------------------------------- commands
    def _on_cmd_drive(self, msg):
        self._set_command(msg.drive.speed, msg.drive.steering_angle)

    def _on_cmd_vel(self, msg):
        self._set_command(msg.linear.x, msg.angular.z)

    def _light(self, method, value):
        try:
            getattr(self.robot, method)(value)
            self.node.get_logger().info(f'{method}({value})')
        except Exception as exc:        # car models without that light
            self.node.get_logger().warn(f'{method} failed: {exc}')

    def _set_command(self, speed, steer):
        if not (math.isfinite(speed) and math.isfinite(steer)):
            return
        # reverse allowed, slowly (manoeuvring: backing away from a wall to turn), max 1.5 m/s
        self.cmd_speed = float(np.clip(speed, -1.5, self.max_speed))
        self.cmd_steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        self.last_cmd_time = self.robot.getTime()

    def _apply_command(self, now):
        if self.last_cmd_time < 0 or now - self.last_cmd_time > self.cmd_timeout:
            speed, steer = 0.0, self.cmd_steer   # watchdog: no fresh command -> stop
        else:
            speed, steer = self.cmd_speed, self.cmd_steer
        self.robot.setCruisingSpeed(speed * KMH_PER_MS)
        self.robot.setSteeringAngle(WEBOTS_STEER_SIGN * steer)
        # Webots cruise control only lowers the target speed and coasts down (~1-2 m/s^2); it never
        # brakes. A real speed controller does, and the safety shield relies on it. Brake in
        # proportion to the overspeed, full brake when asked to stop.
        excess = self.wheel_speed - speed
        if speed < -0.05:                               # reversing
            if self.wheel_speed > 0.05:
                brake = 1.0                             # still rolling forward: stop first
            elif self.wheel_speed < speed - 0.3:
                brake = min(1.0, (speed - self.wheel_speed) / self.brake_full_at)
            else:
                brake = 0.0
        elif speed < 0.05 and abs(self.wheel_speed) > 0.05:
            brake = 1.0
        elif excess > 0.3:
            brake = min(1.0, excess / self.brake_full_at)
        else:
            brake = 0.0
        self.robot.setBrakeIntensity(brake)
        try:
            self.robot.setBrakeLights(brake > 0.05)
        except Exception:
            pass

    # ------------------------------------------------------------------ step
    def _stamp(self):
        return Time(nanoseconds=int(round(self.robot.getTime() * 1e9))).to_msg()

    def step(self):
        # handle EVERY waiting message, not one per simulation step: with one per step the drive
        # commands queued up behind other messages and the car executed commands ~0.65 s old
        # (measured with tools/steer_id.py) - the main cause of the steering zig-zag
        for _ in range(20):
            self.executor.spin_once(timeout_sec=0)
        now = self.robot.getTime()
        self._apply_command(now)
        stamp = self._stamp()

        self._publish_odometry(now, stamp)
        if self.imu and now >= self.next_pub['imu']:
            self.next_pub['imu'] = now + self.imu_period / 1000.0
            self._publish_imu(stamp)
        if self.camera and now >= self.next_pub['camera']:
            self.next_pub['camera'] = now + self.camera_period / 1000.0
            self._publish_camera(stamp)
        for c in self.surround:
            if now >= c['next']:
                c['next'] = now + max(self.camera_period, 200) / 1000.0
                self._publish_camera(stamp, c['dev'], c['spec']['name'] + '_optical_frame', c['pub_img'], c['pub_info'])
        if self.lidars and now >= self.next_pub['lidar']:
            self.next_pub['lidar'] = now + self.lidar_period / 1000.0
            self._publish_lidars(stamp)
            self._publish_gps(stamp)

    # --------------------------------------------------------------- sensors
    def _publish_camera(self, stamp, cam=None, frame='camera_optical_frame', pub_img=None, pub_info=None):
        cam = cam or self.camera
        pub_img, pub_info = pub_img or self.pub_image, pub_info or self.pub_info
        raw = cam.getImage()
        if not raw:
            return
        w, h = cam.getWidth(), cam.getHeight()
        img = Image()
        img.header = Header(stamp=stamp, frame_id=frame)
        img.height, img.width = h, w
        img.encoding = 'bgra8'
        img.step = w * 4
        img.data = raw
        pub_img.publish(img)

        f = (w / 2.0) / math.tan(cam.getFov() / 2.0)
        info = CameraInfo()
        info.header = img.header
        info.height, info.width = h, w
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [f, 0.0, w / 2.0, 0.0, f, h / 2.0, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, w / 2.0, 0.0, 0.0, f, h / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        pub_info.publish(info)

    def _publish_lidars(self, stamp):
        clouds = []
        for u in self.lidars:
            data = u['dev'].getRangeImage()
            if not data:
                continue
            ranges = np.asarray(data, dtype=np.float32).reshape(u['layers'], u['res'])[:, ::-1]
            if u is self.roof:
                self._publish_scan(u, ranges, stamp)
            valid = np.isfinite(ranges) & (ranges > u['min']) & (ranges < u['max'])
            # mask first (no inf*0), then mount transform into base_link
            clouds.append(u['dirs_base'][valid] * ranges[valid][:, None] + u['origin'])
        if clouds:
            pts = np.concatenate(clouds).astype(np.float32)
            self.pub_points.publish(point_cloud2.create_cloud_xyz32(
                Header(stamp=stamp, frame_id='base_link'), pts))

    def _publish_scan(self, u, ranges, stamp):
        scan = LaserScan()
        scan.header = Header(stamp=stamp, frame_id=u['frame'])
        scan.angle_min = -u['fov'] / 2.0
        scan.angle_increment = u['inc']
        scan.angle_max = scan.angle_min + u['inc'] * (u['res'] - 1)
        scan.time_increment = 0.0
        scan.scan_time = self.lidar_period / 1000.0
        scan.range_min = float(u['min'])
        scan.range_max = float(u['max'])
        row = ranges[u['scan_layer']].copy()
        row[~np.isfinite(row) | (row >= u['max'])] = np.inf  # REP-117: no return = +inf
        scan.ranges = row.tolist()
        self.pub_scan.publish(scan)

    def _publish_imu(self, stamp):
        qx, qy, qz, qw = self.imu.getQuaternion()
        msg = Imu()
        msg.header = Header(stamp=stamp, frame_id='imu_link')
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = qx, qy, qz, qw
        msg.orientation_covariance = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4]
        if self.gyro:
            gx, gy, gz = self.gyro.getValues()
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gx, gy, gz
            msg.angular_velocity_covariance = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4]
        else:
            msg.angular_velocity_covariance[0] = -1.0  # REP-145: not provided
        if self.accel:
            ax, ay, az = self.accel.getValues()
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = ax, ay, az
            msg.linear_acceleration_covariance = [1e-2, 0.0, 0.0, 0.0, 1e-2, 0.0, 0.0, 0.0, 1e-2]
        else:
            msg.linear_acceleration_covariance[0] = -1.0
        self.pub_imu.publish(msg)

    def _publish_gps(self, stamp):
        if not self.gps:
            return
        x, y, z = self.gps.getValues()
        if not all(math.isfinite(v) for v in (x, y, z)):
            return
        msg = PointStamped()
        msg.header = Header(stamp=stamp, frame_id='world')
        msg.point.x, msg.point.y, msg.point.z = x, y, z
        self.pub_gps.publish(msg)

    def _measure_speed(self, now):
        """Longitudinal speed in m/s from rear-wheel encoders (rad -> m)."""
        if not self.wheel_sensors:
            v = self.robot.getCurrentSpeed()
            return v / KMH_PER_MS if math.isfinite(v) else 0.0
        pos = [ps.getValue() for ps in self.wheel_sensors]
        if any(not math.isfinite(p) for p in pos):
            return self.wheel_speed
        if self.prev_wheel_pos is not None and self.last_odom_t is not None:
            dt = now - self.last_odom_t
            if dt > 0:
                rate = sum(p - q for p, q in zip(pos, self.prev_wheel_pos)) / len(pos) / dt
                self.wheel_speed = 0.6 * self.wheel_speed + 0.4 * rate * self.wheel_radius
        self.prev_wheel_pos = pos
        return self.wheel_speed

    def _publish_odometry(self, now, stamp):
        speed = self._measure_speed(now)
        steer = WEBOTS_STEER_SIGN * self.robot.getSteeringAngle()
        yaw_rate = speed * math.tan(steer) / self.wheelbase

        if self.last_odom_t is not None:
            dt = now - self.last_odom_t
            mid_yaw = self.yaw + 0.5 * yaw_rate * dt
            self.x += speed * math.cos(mid_yaw) * dt
            self.y += speed * math.sin(mid_yaw) * dt
            self.yaw = math.atan2(math.sin(self.yaw + yaw_rate * dt), math.cos(self.yaw + yaw_rate * dt))
        self.last_odom_t = now
        q = _yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header = Header(stamp=stamp, frame_id='odom')
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = q
        odom.twist.twist.linear.x = speed
        odom.twist.twist.angular.z = yaw_rate
        pc = [0.0] * 36
        pc[0] = pc[7] = 0.05
        pc[14] = pc[21] = pc[28] = 1e3   # z, roll, pitch not measured
        pc[35] = 0.03
        odom.pose.covariance = pc
        tc = [0.0] * 36
        tc[0] = 0.02
        tc[7] = 0.01
        tc[14] = tc[21] = tc[28] = 1e3
        tc[35] = 0.02
        odom.twist.covariance = tc
        self.pub_odom.publish(odom)

        self.pub_speed.publish(Float32(data=float(speed)))
        self.pub_steer.publish(Float32(data=float(steer)))

        if self.publish_odom_tf:
            self.tf.sendTransform(self._make_tf('odom', 'base_link', (self.x, self.y, 0.0), q, stamp))
