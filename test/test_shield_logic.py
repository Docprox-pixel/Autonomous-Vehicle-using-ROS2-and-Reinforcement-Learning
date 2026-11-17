#!/usr/bin/env python3
"""Unit tests for the safety shield's pure decision logic (no ROS, no simulator).

  python3 -m pytest test/test_shield_logic.py -q      (or: python3 test/test_shield_logic.py)
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from auto_car_core import safety as shield  # noqa: E402

P = shield.Params()
FRONT = shield.CAR_FRONT
Z = 0.5   # obstacle point height in base_link (well inside the knee..roof band)


def wall(x, y0=-1.0, y1=1.0):
    """A row of obstacle points at base_link x across y0..y1."""
    ys = np.arange(y0, y1 + 1e-9, 0.1)
    return np.c_[np.full_like(ys, x), ys, np.full_like(ys, Z)]


def car(x, y, vx, vy, L=4.5, W=1.8):
    return {'pos': (x, y), 'vel': (vx, vy), 'size': (L, W)}


# ---------------------------------------------------------------- static obstacles
def test_clear_road_passes_command_through():
    v, info = shield.evaluate(10.0, 0.0, 10.0, np.zeros((0, 3)), [])
    assert v == 10.0 and info['reason'] == 'clear'


def test_obstacle_ahead_caps_speed_to_stopping_distance():
    d = 20.0
    v, info = shield.evaluate(15.0, 0.0, 15.0, wall(FRONT + d), [])
    assert abs(info['static_distance'] - d) < 0.11
    # v*latency + v^2/(2a) must equal the room left before the margin
    assert abs(v * P.latency + v * v / (2 * P.a_brake) - (d - P.margin)) < 0.05
    assert info['reason'] == 'static_obstacle' or v >= 14.6


def test_obstacle_closer_than_margin_stops():
    v, _ = shield.evaluate(5.0, 0.0, 5.0, wall(FRONT + 1.5), [])
    assert v == 0.0


def test_obstacle_beside_corridor_is_ignored():
    v, info = shield.evaluate(10.0, 0.0, 10.0, wall(FRONT + 5, 2.0, 4.0), [])
    assert v == 10.0 and math.isinf(info['static_distance'])


def test_low_and_overhead_points_ignored():
    pts = wall(FRONT + 5)
    low, high = pts.copy(), pts.copy()
    low[:, 2] = -0.25      # 7 cm above the road
    high[:, 2] = 3.0       # sign/bridge above the roof
    v, _ = shield.evaluate(10.0, 0.0, 10.0, np.r_[low, high], [])
    assert v == 10.0


def test_turning_left_sees_obstacle_on_the_curve_not_straight_ahead():
    steer = 0.3
    R = P.wheelbase / math.tan(steer)                 # ~9.6 m
    th = 1.2                                          # rad along the arc
    on_curve = np.array([[R * math.sin(th), R * (1 - math.cos(th)), Z]])
    straight_ahead = np.array([[FRONT + 8.0, 0.0, Z]])
    _, a = shield.evaluate(5.0, steer, 5.0, on_curve, [])
    _, b = shield.evaluate(5.0, steer, 5.0, straight_ahead, [])
    assert abs(a['static_distance'] - (th * R - FRONT)) < 0.2
    assert b['static_distance'] > a['static_distance']     # straight-ahead point is off the curve


def test_turning_right_mirrors_left():
    steer = 0.3
    R = P.wheelbase / math.tan(steer)
    th = 1.0
    left = np.array([[R * math.sin(th), R * (1 - math.cos(th)), Z]])
    right = left * np.array([1, -1, 1])
    _, a = shield.evaluate(5.0, steer, 5.0, left, [])
    _, b = shield.evaluate(5.0, -steer, 5.0, right, [])
    assert abs(a['static_distance'] - b['static_distance']) < 1e-6


# ---------------------------------------------------------------- moving obstacles
def test_car_crossing_our_path_triggers_stop():
    # we drive 10 m/s; a car 15 m ahead crosses from the left at 8 m/s -> meets us in ~1 s
    v, info = shield.evaluate(10.0, 0.0, 10.0, None, [car(FRONT + 15, 8.0, 0.0, -8.0)])
    assert info['ttc'] < P.ttc_stop and v == 0.0 and info['reason'] == 'moving_obstacle'


def test_car_crossing_far_ahead_only_slows():
    v, info = shield.evaluate(10.0, 0.0, 10.0, None, [car(FRONT + 30, 12.0, 0.0, -5.0)])
    assert P.ttc_stop <= info['ttc'] < P.ttc_slow and 0.0 < v < 10.0


def test_car_moving_away_is_ignored():
    v, info = shield.evaluate(10.0, 0.0, 10.0, None, [car(FRONT + 10, 0.0, 15.0, 0.0)])
    assert math.isinf(info['ttc']) and v == 10.0


def test_oncoming_car_in_other_lane_is_ignored():
    v, info = shield.evaluate(10.0, 0.0, 10.0, None, [car(FRONT + 20, 3.7, -10.0, 0.0)])
    assert math.isinf(info['ttc']) and v == 10.0


def test_oncoming_car_in_our_lane_stops():
    v, info = shield.evaluate(10.0, 0.0, 10.0, None, [car(FRONT + 20, 0.0, -10.0, 0.0)])
    assert info['ttc'] < P.ttc_stop and v == 0.0


def test_car_passing_behind_is_ignored():
    v, info = shield.evaluate(5.0, 0.0, 5.0, None, [car(-10.0, 8.0, 0.0, -8.0)])
    assert math.isinf(info['ttc']) and v == 5.0


def test_slow_lead_car_in_lane_is_followed_not_rammed():
    # lead car 12 m ahead at 4 m/s, we want 10 m/s: closing at 6 m/s -> TTC ~2 s -> slow down
    v, info = shield.evaluate(10.0, 0.0, 10.0, None, [car(FRONT + 12 + 2.25, 0.0, 4.0, 0.0)])
    assert info['ttc'] < P.ttc_slow and v < 10.0


# ---------------------------------------------------------------- following / tracker mistakes
def test_lead_car_driving_away_at_our_speed_is_followed_without_hard_braking():
    gap = 20.0
    lead = car(FRONT + gap + 2.25, 0.0, 10.0, 0.0)          # rear bumper 20 m ahead, 10 m/s
    v, info = shield.evaluate(10.0, 0.0, 10.0, wall(FRONT + gap), [lead])
    assert info['lead_credit'] > 0 and v == 10.0


def test_stopped_car_ahead_gets_no_credit():
    gap = 20.0
    v, info = shield.evaluate(10.0, 0.0, 10.0, wall(FRONT + gap), [])   # not a moving track
    assert info['lead_credit'] == 0 and v < 10.0
    assert abs(v * P.latency + v * v / (2 * P.a_brake) - (gap - P.margin)) < 0.05


def test_obstacle_hidden_by_wrong_moving_label_still_stops_the_car():
    # tracker wrongly calls a static box 'moving' but SIDEWAYS: it must not vanish for the shield
    box = car(FRONT + 3.3, 0.0, 0.0, 3.0, L=0.6, W=1.8)
    v, info = shield.evaluate(8.0, 0.0, 8.0, wall(FRONT + 3.0), [box])
    assert info['lead_credit'] == 0 and v == 0.0


# ---------------------------------------------------------------- turns: swept footprint
def legacy_band_distance(points, steer, p=P):
    """The previous check: a band +/- (half width + margin) around the REAR-AXLE arc, ahead only."""
    R = p.wheelbase / math.tan(steer)
    d = np.hypot(points[:, 0], points[:, 1] - R)
    ang = np.arctan2(points[:, 0], np.sign(R) * (R - points[:, 1]))
    inside = (np.abs(d - abs(R)) < shield.HALF_WIDTH + p.corridor_margin) & (ang > 0) & (ang < math.pi)
    return float((ang[inside] * abs(R) - FRONT).min()) if inside.any() else math.inf


def pole(x, y):
    """A 0.2 m thick pole: a few points around (x, y) at knee-to-roof height."""
    a = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    return np.c_[x + 0.1 * np.cos(a), y + 0.1 * np.sin(a), np.full(8, Z)]


STEER = 0.35                                    # sharp left turn
R = P.wheelbase / math.tan(STEER)               # ~8.2 m rear-axle radius
HW = shield.HALF_WIDTH + P.corridor_margin


def test_front_corner_swing_out_hits_pole_outside_the_curve():
    # outer front corner sweeps radius sqrt(FRONT^2 + (R + HW)^2); the axle band only reaches R + HW
    r_band, r_front = R + HW, math.hypot(FRONT, R + HW)
    r = (r_band + r_front) / 2                  # in between: only the swinging front corner hits it
    th = 1.0
    x, y = r * math.sin(th), R - r * math.cos(th)
    assert math.isinf(legacy_band_distance(pole(x, y), STEER))           # old check: misses
    d = shield.static_distance(pole(x, y), STEER, P)
    assert math.isfinite(d), 'swept footprint must see the pole the front corner swings into'
    v, _ = shield.evaluate(6.0, STEER, 6.0, pole(x, y), [])
    assert v < 6.0


def test_tail_swing_out_hits_pole_beside_rear_doors():
    # turning left, the overhang behind the rear axle swings RIGHT. For the BmwX5 (rear corner
    # 1.55 m behind the axle incl. margin, R = 8.2 m) the swing is sqrt(1.55^2 + 9.6^2) - 9.6
    # = 0.12 m: small for a car, large for a bus. A pole 0.05 m outside the margin gets hit.
    rx = shield.CAR_REAR - P.corridor_margin
    swing = math.hypot(rx, R + HW) - (R + HW)
    assert 0.1 < swing < 0.15
    # The swinging rear corner's circle reaches furthest out beside the REAR AXLE (x = 0),
    # i.e. next to the rear doors: a pole there is clipped as soon as the turn starts.
    pts = pole(0.0, -(HW + 0.05 + 0.1))           # pole surface 0.05 m outside the margin
    assert math.isinf(legacy_band_distance(pts, STEER))                  # old check: misses
    d = shield.static_distance(pts, STEER, P)
    assert math.isfinite(d) and d < 3.0, f'tail swing contact expected within 3 m, got {d}'


def test_inside_of_curve_still_caught():
    # a kerb post just inside the rear-axle circle: both methods must catch it
    r, th = R - HW + 0.2, 1.2
    pts = pole(r * math.sin(th), R - r * math.cos(th))
    assert math.isfinite(legacy_band_distance(pts, STEER))
    assert math.isfinite(shield.static_distance(pts, STEER, P))


def test_turn_clear_of_poles_outside_the_swept_area():
    # same turn, pole well outside the outer front-corner circle: no false alarm
    r, th = math.hypot(FRONT, R + HW) + 1.0, 1.0
    pts = pole(r * math.sin(th), R - r * math.cos(th))
    assert math.isinf(shield.static_distance(pts, STEER, P))
    v, _ = shield.evaluate(6.0, STEER, 6.0, pts, [])
    assert v == 6.0


def test_parallel_wall_beside_car_does_not_block_straight_driving():
    ys = np.full(60, -(shield.HALF_WIDTH + P.corridor_margin + 0.2))
    wall_side = np.c_[np.linspace(-3, 30, 60), ys, np.full(60, Z)]
    assert math.isinf(shield.static_distance(wall_side, 0.0, P))



def test_obstacle_closer_than_hard_margin_beside_door_stops():
    # regression for the drive-test scrape: something 0.1 m beside the door was invisible
    pts = pole(1.5, -(shield.HALF_WIDTH + 0.1 + 0.1))   # pole surface 0.1 m from the body
    assert shield.static_distance(pts, 0.0, P) == 0.0
    single = pts[:1]                                      # ...but one stray return alone does not freeze it
    assert shield.static_distance(single, 0.0, P) > 0.0
    v, _ = shield.evaluate(5.0, 0.0, 5.0, pts, [])
    assert v == 0.0


def test_obstacle_in_soft_margin_beside_door_does_not_freeze_the_car():
    pts = pole(1.5, -(shield.HALF_WIDTH + 0.3 + 0.1))   # 0.3 m: inside the 0.4 m soft margin
    v, _ = shield.evaluate(5.0, 0.0, 5.0, pts, [])
    assert v == 5.0


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f'  PASS {n}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL {n} {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    raise SystemExit(1 if failed else 0)
