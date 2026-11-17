"""Arc planner lane following (track_route: pure pursuit with corner-cut compensation) in a closed
loop with a kinematic car: every wheel stays in the 3.5 m lane in tight turns, no corner cutting,
no steering oscillation on straight road."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from auto_car_core.planner import plan  # noqa: E402
from test_frenet import route_turn  # noqa: E402


def drive(P, steps=900, y0=0.0):
    x, y, yaw, v, steer, prev = P[5, 0], P[5, 1] + y0, 0.0, 0.0, 0.0, 0.0
    worst, signed, flips, last_sign = 0.0, [], 0, 0
    for i in range(steps):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rb = np.c_[c * dx + s * dy, -s * dx + c * dy]
        st, vc, info = plan(np.zeros((0, 3)), prev, None, path=rb, n_physical=0, ego_v=v)
        prev = st
        steer += float(np.clip(st - steer, -0.06, 0.06))
        v += float(np.clip(vc - v, -0.38, 0.2))
        x += v * math.cos(yaw) * 0.1
        y += v * math.sin(yaw) * 0.1
        yaw += v * math.tan(steer) / 2.995 * 0.1
        for side in (1.0, -1.0):
            for ax in (0.0, 2.995):
                wx = x + ax * math.cos(yaw) - side * 0.8 * math.sin(yaw)
                wy = y + ax * math.sin(yaw) + side * 0.8 * math.cos(yaw)
                worst = max(worst, float(np.hypot(P[:, 0] - wx, P[:, 1] - wy).min()))
        if np.hypot(x - P[-1, 0], y - P[-1, 1]) < 6:
            break
    return worst, (x, y)


def test_tight_turns_stay_in_lane():
    for R, deg, sg in ((7, 90, -1), (7, 90, 1), (10, 120, -1), (22, 90, 1)):
        P = route_turn(R, deg, sg)
        worst, end = drive(P)
        print(R, deg, sg, round(worst, 2))
        assert worst < 1.6, (R, deg, sg, worst)
        assert np.hypot(end[0] - P[-1, 0], end[1] - P[-1, 1]) < 8


def test_converges_without_waves():
    P = np.c_[np.arange(-10, 200, 0.5), np.zeros(420)]
    x, y, yaw, v, steer = -5.0, 1.0, 0.0, 6.0, 0.0
    ys = []
    for _ in range(300):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rb = np.c_[c * dx + s * dy, -s * dx + c * dy]
        st, vc, _ = plan(np.zeros((0, 3)), steer, None, path=rb, n_physical=0, ego_v=v)
        steer += float(np.clip(st - steer, -0.06, 0.06))
        x += v * math.cos(yaw) * 0.1; y += v * math.sin(yaw) * 0.1
        yaw += v * math.tan(steer) / 2.995 * 0.1
        ys.append(y)
    ys = np.array(ys)
    assert abs(ys[-100:]).max() < 0.1                     # settled on the lane centre
    assert ys.min() > -0.35                               # at most a small overshoot, no waves


def test_stable_with_steering_lag():
    """Real car: ~1 s between steering command and response. The tracker must not oscillate."""
    P = np.c_[np.arange(-10, 400, 0.5), np.zeros(820)]
    x, y, yaw, v, steer, cmd = P[5, 0], 1.2, 0.1, 0.0, 0.0, 0.0
    errs = []
    for _ in range(900):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rb = np.c_[c * dx + s * dy, -s * dx + c * dy]
        st, vc, _ = plan(np.zeros((0, 3)), cmd, None, path=rb, n_physical=0, ego_v=v)
        cmd = float(np.clip(st, cmd - 0.08, cmd + 0.08))
        steer += (cmd - steer) * 0.1 / 1.2                   # 1.2 s first-order steering lag
        v += float(np.clip(vc - v, -0.38, 0.12))
        x += v * c * 0.1; y += v * s * 0.1; yaw += v * math.tan(steer) / 2.995 * 0.1
        errs.append(abs(y))
        if x > 380:
            break
    assert max(errs) < 1.7 and np.mean(errs[-150:]) < 0.4   # starts 1.2 m off, heading away
