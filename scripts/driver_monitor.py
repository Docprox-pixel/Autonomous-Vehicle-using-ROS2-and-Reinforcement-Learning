#!/usr/bin/env python3
"""Driver monitoring + emergency response: an unresponsive driver is taken to the nearest hospital.

The car drives itself, but a human must stay responsive (like the attention checks of today's
driver-assist systems). Any driver input counts as a response: /driver/input (std_msgs/String,
e.g. 'steering_touch', 'button', 'cancel'). In a real car this would be the steering-wheel torque
sensor, a button, or a driver-monitoring camera (eyes on road); here a test or a person publishes it.

Escalation (times since the last response, parameters below):
   0 .. warn_after        OK
   warn_after (10 s)      WARNING   visual alert                         /driver/alert
   alarm_after (20 s)     ALARM     audible alert, speed capped to 60 %  /driver/speed_limit
   emergency_after (30 s) EMERGENCY driver unresponsive:
                              hazard lights on                          /vehicle/hazard
                              emergency call logged (position, time)    /driver/emergency_call
                              destination := nearest hospital BY ROAD   /route/go_to 'hospital'
                              drives there obeying every traffic rule (the normal stack drives)
                              speed capped (emergency_speed) ...
   ARRIVED                stops at the hospital, hazard lights stay on, parks (speed 0)
HEALTH: /driver/vitals (String JSON {"heart_rate": bpm, "spo2": %}) from a wearable / seat sensor.
Readings outside heart rate 40..150 bpm or SpO2 >= 88 % for health_confirm (5 s) -> EMERGENCY at once
(a collapsing driver may still have a hand on the wheel). No vitals sensor -> responsiveness only.
A response before EMERGENCY resets to OK. During EMERGENCY only an explicit 'cancel' from the
driver ends it (a twitch is not proof of recovery); the car then stops at the side and waits.

Output: /driver/status (String JSON): state, seconds since the last response, destination.
"""
import json
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class DriverMonitor(Node):
    def __init__(self):
        super().__init__('driver_monitor')
        p = self.declare_parameter
        self.warn_after = p('warn_after', 10.0).value
        self.alarm_after = p('alarm_after', 20.0).value
        self.emergency_after = p('emergency_after', 30.0).value
        self.alarm_speed_factor = p('alarm_speed_factor', 0.6).value
        self.emergency_speed = p('emergency_speed', 11.1).value       # m/s: same rules as every drive
        self.enabled = p('enabled', True).value
        self.state = 'OK'
        self.last_input = self.now()
        self.destination = None
        self.route_state = None
        self.gps = None
        self.limit = 13.9
        self.pub_alert = self.create_publisher(String, '/driver/alert', 5)
        self.pub_status = self.create_publisher(String, '/driver/status', 5)
        self.pub_speed = self.create_publisher(Float32, '/driver/speed_limit', 5)
        self.pub_hazard = self.create_publisher(Bool, '/vehicle/hazard', 5)
        self.pub_goto = self.create_publisher(String, '/route/go_to', 5)
        self.pub_call = self.create_publisher(String, '/driver/emergency_call', 5)
        self.create_subscription(String, '/driver/input', self.on_input, 10)
        self.create_subscription(String, '/route/destination', self.on_destination, 5)
        self.create_subscription(String, '/route/status', lambda m: setattr(
            self, 'route_state', json.loads(m.data).get('state')), 5)
        self.create_subscription(PointStamped, '/gps/position', lambda m: setattr(self, 'gps', m.point), 5)
        self.create_subscription(Float32, '/behaviour/speed_limit', lambda m: setattr(self, 'limit', m.data), 5)
        # driver HEALTH (vitals): abnormal for health_confirm seconds -> emergency
        self.hr_min, self.hr_max = p('hr_min', 40.0).value, p('hr_max', 150.0).value
        self.spo2_min = p('spo2_min', 88.0).value
        self.health_confirm = p('health_confirm', 5.0).value
        self.vitals, self.vitals_t, self.abnormal_since, self.health_reason = None, None, None, None
        self.create_subscription(String, '/driver/vitals', self.on_vitals, 5)
        self.create_timer(0.5, self.loop)
        self.get_logger().info(f'driver_monitor: warning after {self.warn_after:.0f} s, alarm after '
                               f'{self.alarm_after:.0f} s, emergency (drive to hospital) after '
                               f'{self.emergency_after:.0f} s without driver response')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_input(self, m):
        if self.state == 'EMERGENCY' or self.state == 'ARRIVED':
            if m.data.strip().lower() == 'cancel':
                self.get_logger().warn('driver cancelled the emergency: stopping, hazard lights stay on')
                self.state = 'CANCELLED'
            return
        self.last_input = self.now()
        if self.state != 'OK':
            self.get_logger().info(f'driver responded ({m.data}): back to OK')
        self.state = 'OK'

    def on_vitals(self, m):
        try:
            v = json.loads(m.data)
        except ValueError:
            return
        self.vitals, self.vitals_t = v, self.now()
        hr, sp = v.get('heart_rate'), v.get('spo2')
        bad = []
        if hr is not None and not (self.hr_min <= hr <= self.hr_max):
            bad.append(f'heart rate {hr:.0f} bpm')
        if sp is not None and sp < self.spo2_min:
            bad.append(f'SpO2 {sp:.0f} %')
        if bad:
            self.abnormal_since = self.abnormal_since or self.vitals_t
            self.health_reason = ', '.join(bad)
        else:
            self.abnormal_since, self.health_reason = None, None

    def on_destination(self, m):
        d = json.loads(m.data)
        if self.state == 'EMERGENCY' and d.get('query') == 'hospital':
            if d.get('found'):
                self.destination = d
                self.get_logger().error(f"EMERGENCY: driving to {d['name']} ({d['route_length']:.0f} m by road)")
            else:
                self.get_logger().error('EMERGENCY: no hospital on the map: stopping at the roadside')
                self.state = 'CANCELLED'

    def loop(self):
        if not self.enabled:
            return
        t = self.now()
        quiet = t - self.last_input
        prev = self.state
        health_bad = (self.abnormal_since is not None and t - self.abnormal_since >= self.health_confirm
                      and self.vitals_t is not None and t - self.vitals_t < 5.0)
        if self.state in ('OK', 'WARNING', 'ALARM') and health_bad:
            self.get_logger().error(f'DRIVER HEALTH CRITICAL ({self.health_reason}): emergency')
            self.state = 'EMERGENCY'
        elif self.state in ('OK', 'WARNING', 'ALARM'):
            if quiet >= self.emergency_after:
                self.state = 'EMERGENCY'
            elif quiet >= self.alarm_after:
                self.state = 'ALARM'
            elif quiet >= self.warn_after:
                self.state = 'WARNING'
            else:
                self.state = 'OK'
        if self.state == 'EMERGENCY' and self.destination is not None and self.route_state == 'goal_reached':
            self.state = 'ARRIVED'
        if self.state != prev:
            self.on_state(prev, quiet)
        # speed cap: none / alarm fraction / emergency cap / stopped
        cap = {'OK': 1e3, 'WARNING': 1e3, 'ALARM': self.alarm_speed_factor * self.limit,
               'EMERGENCY': self.emergency_speed, 'ARRIVED': 0.0, 'CANCELLED': 0.0}[self.state]
        self.pub_speed.publish(Float32(data=float(cap)))
        if self.state == 'EMERGENCY' and self.destination is None and int(t * 2) % 10 == 0:
            self.pub_goto.publish(String(data='emergency:hospital'))   # emergency: no parking, stop at the door         # retry until the route planner answers
        self.pub_status.publish(String(data=json.dumps({
            'state': self.state, 'seconds_without_response': round(quiet, 1),
            'speed_cap': None if cap >= 1e3 else round(cap, 1),
            'destination': None if self.destination is None else self.destination.get('name'),
            'vitals': self.vitals, 'health_alarm': self.health_reason})))

    def on_state(self, prev, quiet):
        msg = {'WARNING': 'Please put your hands on the wheel (visual alert)',
               'ALARM': 'DRIVER NOT RESPONDING - audible alarm, slowing down',
               'EMERGENCY': 'DRIVER UNRESPONSIVE - emergency mode: hazard lights on, driving to the nearest hospital',
               'ARRIVED': 'Arrived at the hospital - parked, hazard lights on',
               'CANCELLED': 'Emergency cancelled - stopping',
               'OK': 'Driver attentive'}[self.state]
        text = f'{prev} -> {self.state} after {quiet:.0f} s without response: {msg}'
        if self.state in ('EMERGENCY', 'ARRIVED'):     # rclpy: one call site = one severity
            self.get_logger().error(text)
        else:
            self.get_logger().warn(text)
        self.pub_alert.publish(String(data=json.dumps({'level': self.state, 'message': msg})))
        if self.state == 'EMERGENCY':
            self.pub_hazard.publish(Bool(data=True))
            pos = None if self.gps is None else (round(self.gps.x, 1), round(self.gps.y, 1))
            call = {'type': 'medical_emergency', 'reason': 'driver unresponsive', 'position': pos,
                    'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            self.pub_call.publish(String(data=json.dumps(call)))
            self.get_logger().error(f'EMERGENCY CALL: {call}')
            self.pub_goto.publish(String(data='emergency:hospital'))   # emergency: no parking, stop at the door
        if self.state in ('ARRIVED', 'CANCELLED'):
            self.pub_hazard.publish(Bool(data=True))


def main():
    rclpy.init()
    node = DriverMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
