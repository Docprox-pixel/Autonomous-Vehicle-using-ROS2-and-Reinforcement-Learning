"""Frenet planner (auto_car_core/frenet.py) in a closed loop with a kinematic car: lane keeping in
tight turns, passing a parked car inside the own lane, stopping for a blocked lane, highway speed."""
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_car_core.frenet import FrenetParams, plan_frenet  # noqa: E402


def route_turn(R, deg, sign):
    th = np.linspace(0, math.radians(deg), 80)
    arc = np.c_[R * np.sin(th), sign * (R - R * np.cos(th))]
    h = sign * math.radians(deg)
    return np.r_[np.c_[np.arange(-40, 0, 0.5), np.zeros(80)], arc,
                 arc[-1] + np.outer(np.arange(0.5, 40.5, 0.5), [math.cos(h), math.sin(h)])]


def box(cx, cy, L=4.5, W=1.8):
    xs, ys = np.meshgrid(np.arange(cx - L / 2, cx + L / 2, 0.2), np.arange(cy - W / 2, cy + W / 2, 0.2))
    return np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 0.5)]


def drive(P, world_obs=None, steps=900, highway=False, v_cap=None):
    x, y, yaw, v, steer = P[5, 0], P[5, 1], 0.0, 0.0, 0.0
    worst, vmax, stopped_at = 0.0, 0.0, None
    for _ in range(steps):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rb = np.c_[c * dx + s * dy, -s * dx + c * dy]
        obs = np.zeros((0, 3))
        if world_obs is not None:
            ox, oy = world_obs[:, 0] - x, world_obs[:, 1] - y
            obs = np.c_[c * ox + s * oy, -s * ox + c * oy, world_obs[:, 2]]
        st, vc, info = plan_frenet(rb, obs, None, ego_v=v, highway=highway, speed_cap=v_cap)
        steer += float(np.clip(st - steer, -0.06, 0.06))
        v += float(np.clip(vc - v, -0.38, 0.2))
        x += v * math.cos(yaw) * 0.1
        y += v * math.sin(yaw) * 0.1
        yaw += v * math.tan(steer) / 2.995 * 0.1
        vmax = max(vmax, v)
        for side in (1.0, -1.0):
            for ax in (0.0, 2.995):
                wx = x + ax * math.cos(yaw) - side * 0.8 * math.sin(yaw)
                wy = y + ax * math.sin(yaw) + side * 0.8 * math.cos(yaw)
                worst = max(worst, float(np.hypot(P[:, 0] - wx, P[:, 1] - wy).min()))
        if world_obs is not None and v < 0.05 and stopped_at is None and _ > 50:
            stopped_at = (x, y)
        if np.hypot(x - P[-1, 0], y - P[-1, 1]) < 6:
            break
    return worst, vmax, (x, y), stopped_at


def test_tight_turns_stay_in_lane():
    for R, deg, sg in ((7, 90, -1), (7, 90, 1), (10, 120, -1), (22, 90, 1)):
        worst, _, end, _ = drive(route_turn(R, deg, sg))
        assert worst < 1.6, (R, deg, sg, worst)          # every wheel inside the 3.5 m lane
        assert np.hypot(end[0] - route_turn(R, deg, sg)[-1, 0], end[1] - route_turn(R, deg, sg)[-1, 1]) < 8


def test_passes_parked_car_inside_own_lane():
    P = np.c_[np.arange(-40, 120, 0.5), np.zeros(320)]
    parked = box(30.0, -1.9, W=1.8)                      # sticks 0.65 m into our lane from the right
    worst, _, end, stopped = drive(P, parked)
    assert stopped is None and end[0] > 100              # passed it
    assert worst < 1.75 + 0.05                           # never left the lane


def test_stops_for_blocked_lane():
    P = np.c_[np.arange(-40, 120, 0.5), np.zeros(320)]
    blocked = box(40.0, 0.0)
    _, _, end, stopped = drive(P, blocked, steps=400)
    assert stopped is not None and 30.0 < stopped[0] + 4.05 < 38.0   # stops short of it


def test_highway_speed_on_straight():
    P = np.c_[np.arange(-40, 600, 0.5), np.zeros(1280)]
    _, vmax, _, _ = drive(P, steps=500, highway=True)
    assert vmax > 20.0
