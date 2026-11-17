#!/usr/bin/env python3
"""Unit tests for the lane map (auto_car_core.lanemap) on the city road network.

  python3 test/test_lanemap.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from auto_car_core.lanemap import LaneMap  # noqa: E402

M = LaneMap(os.path.join(HERE, '..', 'worlds', 'city_net', 'sumo.net.xml'))
START, HEADING = (39.66, 0.03), -math.pi / 2      # car origin in city.wbt, heading south


def heading_at_end(path, back=5.0):
    d = np.hypot(*np.diff(path, axis=0).T)
    k = len(path) - 1
    acc = 0.0
    while k > 0 and acc < back:
        acc += d[k - 1]
        k -= 1
    v = path[-1] - path[k]
    return math.degrees(math.atan2(v[1], v[0]))


def test_map_loads():
    road = [l for l in M.lanes.values() if not l.internal]
    assert len(road) >= 60 and len(M.lanes) > len(road)


def test_start_lane_is_southbound_on_the_right():
    lane, s, d = M.nearest_lane(START, HEADING)
    assert d < 3.0, f'car should sit on a southbound lane, {d:.1f} m away'


def test_right_turn_route_ends_in_north_half_of_west_road():
    path = M.route(START, HEADING, (0.0, -45.0))
    assert path is not None
    west = path[path[:, 0] < 33.0]
    assert len(west) and np.all(west[:, 1] > -45.0), f'west-arm y values {west[:, 1].min():.1f}..{west[:, 1].max():.1f}'
    assert abs(abs(heading_at_end(path)) - 180) < 20, 'heading west at the end'


def test_left_turn_route_ends_in_south_half_of_east_road():
    # goal on the straight start of the east road (beyond x ~ 75 it curves north-east)
    path = M.route(START, HEADING, (72.0, -51.0))
    assert path is not None
    east = path[path[:, 0] > 57.0]
    assert len(east) and np.all(east[:, 1] < -45.0), f'east-arm y values {east[:, 1].min():.1f}..{east[:, 1].max():.1f}'
    assert abs(heading_at_end(path)) < 25, f'heading east at the end, got {heading_at_end(path):.0f}'


def test_straight_through_junction():
    path = M.route(START, HEADING, (45.0 - 2.7, -90.0))
    assert path is not None and path[-1][1] < -80
    assert np.all(path[:, 0] < 45.0 + 0.5), 'stays west of the road centre (right-hand traffic, southbound)'


def test_path_is_continuous():
    path = M.route(START, HEADING, (0.0, -45.0))
    gaps = np.hypot(*np.diff(path, axis=0).T)
    assert gaps.max() < 20.0, f'largest gap {gaps.max():.1f} m'



def min_radius(path):
    """Smallest turning radius along a path (m), from heading change per metre over 2 m chords."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    grid = np.arange(0.0, s[-1], 0.5)
    pts = np.c_[np.interp(grid, s, path[:, 0]), np.interp(grid, s, path[:, 1])]
    h = np.unwrap(np.arctan2(*np.diff(pts, axis=0).T[::-1]))
    k = np.abs(h[4:] - h[:-4]) / 2.0
    return 1.0 / max(k.max(), 1e-6)


def test_right_turn_route_is_drivable_for_the_car():
    path = M.route(START, HEADING, (0.0, -41.0))
    r = min_radius(path)
    assert r > 6.4, f'tightest curve {r:.1f} m: the BmwX5 needs >= 6.2 m'


def test_right_turn_route_keeps_off_the_corner():
    # junction corner (road edges) at about (34.25, -34.25); car half width 1 m + margin
    path = M.route(START, HEADING, (0.0, -41.0))
    near = np.hypot(path[:, 0] - 34.25, path[:, 1] + 34.25)
    assert near.min() > 2.0, f'route passes {near.min():.1f} m from the corner'


def test_no_infinite_loop_with_lane_change():
    # a goal that needs the other lane of the same road (forces a lane change in the route)
    path = M.route((36.8, -10.0), HEADING, (72.0, -51.0))
    assert path is not None and len(path) > 2


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
