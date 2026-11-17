#!/usr/bin/env python3
"""Unit tests for the local planner (auto_car_core.planner), no ROS / simulator.

  python3 test/test_planner_logic.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from auto_car_core import planner  # noqa: E402

Z = 0.5


def line(x0, y0, x1, y1, n=None):
    n = n or max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.2))
    return np.c_[np.linspace(x0, x1, n), np.linspace(y0, y1, n), np.full(n, Z)]


def arc_wall(radius, centre_y, th0, th1, n=400):
    th = np.linspace(th0, th1, n)
    return np.c_[radius * np.sin(th), centre_y - radius * np.cos(th) * np.sign(centre_y), np.full(n, Z)]


def straight_road(left=8.0, right=-3.0, length=80):
    return np.r_[line(-5, left, length, left), line(-5, right, length, right)]


def test_open_road_goes_straight_at_full_speed():
    st, v, info = planner.plan(np.zeros((0, 3)))
    assert abs(st) < 1e-6 and v == planner.PlannerParams().v_max


def test_straight_road_between_barriers_stays_straight():
    st, v, _ = planner.plan(straight_road())
    assert abs(st) < 0.05 and v > 6.0


def test_follows_left_curve():
    # road curving left, centre of curvature at y = +40.5. Edges at radius 32.5 (y = +8) and
    # 44.5 (y = -4): the car at y = 0 is in its own (right) half, so it should simply follow the
    # curve. (An earlier version put the car in the LEFT half; the lane rule then correctly
    # moved it right first.)
    pts = np.r_[arc_wall(32.5, 40.5, 0.0, 1.2), arc_wall(44.5, 40.5, 0.0, 1.2)]
    st, v, info = planner.plan(pts)
    road_k = 1 / 40.5
    assert st > 0.02, f'should steer left, got {st}'
    assert abs(math.tan(st) / 2.995 - road_k) < 0.02, 'curvature close to the road curvature'


def test_follows_right_curve():
    pts = np.r_[arc_wall(36.5, -40.5, 0.0, 1.2), arc_wall(48.5, -40.5, 0.0, 1.2)]   # mirror: y -4 / +8
    st, _, _ = planner.plan(pts)
    assert st < -0.02


def drive(world_pts, goal_world=None, steps=60, ds=1.0):
    """Closed loop: plan, move ds along the chosen arc, repeat. Returns final (x, y, yaw), min free."""
    x = y = yaw = 0.0
    st, min_free = 0.0, math.inf
    ps = planner.Params()
    for _ in range(steps):
        c, s_ = math.cos(yaw), math.sin(yaw)
        dx, dy = world_pts[:, 0] - x, world_pts[:, 1] - y
        local = np.c_[c * dx + s_ * dy, -s_ * dx + c * dy, world_pts[:, 2]]
        goal = None
        if goal_world is not None:
            gx, gy = goal_world[0] - x, goal_world[1] - y
            goal = (c * gx + s_ * gy, -s_ * gx + c * gy)
        st, v, info = planner.plan(local, prev_steer=st, goal=goal)
        if v == 0.0:
            break
        min_free = min(min_free, info['free'])
        ex, ey, eyaw = planner.path_poses(1.0, st, ps, ds)
        x, y, yaw = x + c * ex - s_ * ey, y + s_ * ex + c * ey, yaw + eyaw
    return x, y, math.degrees(yaw), min_free


def corner_right():
    """Corridor 6 m wide; at x = 14 a road opens to the right; wall straight ahead at x = 30."""
    return np.r_[line(-5, 3.0, 30, 3.0), line(30, 3.0, 30, -40),
                 line(-5, -3.0, 14, -3.0), line(14, -3.0, 14, -40)]


def t_junction():
    """Wall ahead at x = 25, side roads open both ways from x = 15."""
    return np.r_[line(-5, 3, 15, 3), line(15, 3, 15, 40), line(-5, -3, 15, -3), line(15, -3, 15, -40),
                 line(25, -50, 25, 50)]


def test_drives_to_the_corner_then_turns_right():
    x, y, yaw, _ = drive(corner_right())
    assert yaw < -60 and y < -10, f'should have turned right into the side road: ({x:.1f},{y:.1f}) yaw {yaw:.0f}'
    assert 14 < x < 30, 'between the side road edges'


def test_goal_picks_left_or_right_at_t_junction():
    _, yl, yaw_l, _ = drive(t_junction(), goal_world=(20.0, 60.0))
    _, yr, yaw_r, _ = drive(t_junction(), goal_world=(20.0, -60.0))
    assert yaw_l > 60 and yl > 10, f'left goal: y {yl:.1f} yaw {yaw_l:.0f}'
    assert yaw_r < -60 and yr < -10, f'right goal: y {yr:.1f} yaw {yaw_r:.0f}'


def test_drives_round_a_left_curve_without_touching():
    pts = np.r_[arc_wall(36.0, 40.5, -0.2, 1.8), arc_wall(45.0, 40.5, -0.2, 1.8)]
    x, y, yaw, min_free = drive(pts, steps=50)
    assert yaw > 50, f'should follow the curve, yaw {yaw:.0f}'
    r = math.hypot(x, y - 40.5)
    assert 36.0 + 1.0 < r < 45.0 - 1.0, f'stayed inside the road (radius {r:.1f})'


def test_obstacle_in_lane_is_avoided_or_slowed_for():
    pts = np.r_[straight_road(), line(15, -1.2, 15, 1.2)]          # box across our path
    st, v, info = planner.plan(pts)
    assert info['free'] > 11.0 or v < 4.0


def test_boxed_in_stops():
    pts = np.r_[line(5.5, -6, 5.5, 6), line(-2, 3, 6, 3), line(-2, -3, 6, -3)]
    st, v, info = planner.plan(pts)
    assert v == 0.0 and info['reason'] == 'no_free_arc'


def test_slows_in_tight_curve():
    st, v, info = planner.plan(np.r_[arc_wall(8.0, 12.0, 0.0, 2.0), arc_wall(16.0, 12.0, 0.0, 2.0)])
    assert v < 6.0 and v <= info['v_curve'] + 1e-6



def test_slows_down_before_the_corner_not_in_it():
    """Closed loop toward the right-hand corner: record speed when the turn starts."""
    world = corner_right()
    x = y = yaw = 0.0
    st, ps, speeds = 0.0, planner.Params(), []
    for _ in range(40):
        c, s_ = math.cos(yaw), math.sin(yaw)
        dx, dy = world[:, 0] - x, world[:, 1] - y
        local = np.c_[c * dx + s_ * dy, -s_ * dx + c * dy, world[:, 2]]
        st, v, info = planner.plan(local, prev_steer=st)
        speeds.append((x, st, v))
        ex, ey, eyaw = planner.path_poses(1.0, st, ps, 1.0)
        x, y, yaw = x + c * ex - s_ * ey, y + s_ * ex + c * ey, yaw + eyaw
    first_turn = next(i for i, (_, st_, _) in enumerate(speeds) if st_ < -0.08)
    v_at_turn = speeds[first_turn][2]
    v_far = speeds[0][2]
    assert v_at_turn <= planner.PlannerParams().v_turn + 0.5, f'entered the turn at {v_at_turn:.1f} m/s'
    assert v_far > v_at_turn, 'should have been faster before the corner was near'



def test_unmarked_wide_road_drives_in_middle_of_right_half():
    """22 m wide road, NO centre line: start in the middle, end at y ~ -11 + 22/4 = -5.5."""
    world = np.r_[line(-5, 11.0, 140, 11.0), line(-5, -11.0, 140, -11.0)]
    x, y, yaw, _ = drive(world, steps=80)
    assert -7.0 < y < -4.0, f'should settle in the right half, got y = {y:.1f}'
    assert abs(yaw) < 10, 'and drive straight along the road'


def test_unmarked_narrow_road_keeps_right_without_hitting_edge():
    """8 m road (two 4 m lanes): target centre y = -4 + 8/4 = -2."""
    world = np.r_[line(-5, 4.0, 140, 4.0), line(-5, -4.0, 140, -4.0)]
    x, y, yaw, min_free = drive(world, steps=80)
    assert -3.0 < y < -1.0, f'right lane centre expected near -2, got {y:.1f}'



def follow_route(world_path, start=(39.66, 0.03), yaw=-math.pi / 2, steps=140, ds=0.5):
    """Closed loop on an EMPTY world: plan with the route only, move ds per step."""
    x, y = start
    st, ps, worst = 0.0, planner.Params(), 0.0
    for _ in range(steps):
        c, s_ = math.cos(yaw), math.sin(yaw)
        dx, dy = world_path[:, 0] - x, world_path[:, 1] - y
        local = np.c_[c * dx + s_ * dy, -s_ * dx + c * dy]
        st, v, info = planner.plan(np.zeros((0, 3)), prev_steer=st, path=local)
        worst = max(worst, planner.path_error(local, 0.0, 0.0, 0.0)[0])
        ex, ey, eyaw = planner.path_poses(1.0, st, ps, ds)
        x, y, yaw = x + c * ex - s_ * ey, y + s_ * ex + c * ey, yaw + eyaw
        if math.hypot(world_path[-1, 0] - x, world_path[-1, 1] - y) < 2.0:
            break
    return x, y, math.degrees(yaw), worst


def test_route_right_turn_ends_in_correct_lane():
    from auto_car_core.lanemap import LaneMap
    m = LaneMap(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'worlds', 'city_net', 'sumo.net.xml'))
    path = m.route((39.66, 0.03), -math.pi / 2, (0.0, -45.0))
    x, y, yaw, worst = follow_route(path)
    assert x < 33 and y > -45, f'should end on the north half of the west road, at ({x:.1f},{y:.1f})'
    assert abs(abs(yaw) - 180) < 30 or abs(yaw + 180) < 30, f'heading west, got {yaw:.0f}'
    assert worst < 3.0, f'stayed within 3 m of the route (max {worst:.1f} m)'


def test_route_left_turn_ends_in_correct_lane():
    from auto_car_core.lanemap import LaneMap
    m = LaneMap(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'worlds', 'city_net', 'sumo.net.xml'))
    path = m.route((39.66, 0.03), -math.pi / 2, (72.0, -51.0))
    x, y, yaw, worst = follow_route(path)
    assert x > 57 and y < -45, f'should end on the south half of the east road, at ({x:.1f},{y:.1f})'
    assert worst < 3.0, f'stayed within 3 m of the route (max {worst:.1f} m)'



def test_pole_with_points_at_many_heights_is_not_lost():
    """Regression: thinning before the height filter made poles vanish from the planner."""
    zs = np.r_[np.linspace(2.5, 6.0, 8), np.linspace(-0.07, 1.8, 4)]       # mostly above the roof
    pole = np.c_[np.full(len(zs), 9.0), np.full(len(zs), 0.0), zs][::-1]   # high points first
    st, v, info = planner.plan(pole, 0.0)
    straight = info['arcs'][len(info['arcs']) // 2]
    assert straight['free'] < 6.0, f"straight arc should hit the pole ~5 m ahead, free {straight['free']:.1f}"


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


# ---- car-size passability (config/sensor_layout.yaml 'vehicle') -----------------------------
def _slab(x, y0, y1, z0, z1, G=-0.32):
    ys, zs = np.arange(y0, y1, 0.1), np.arange(z0, z1 + 1e-9, 0.05)
    return np.array([[x + dx, yy, G + z] for dx in (0, 0.2) for yy in ys for z in zs])


def _wall(x0, x1, y, z0, z1, G=-0.32):
    return np.array([[x, y, G + z] for x in np.arange(x0, x1, 0.1) for z in np.arange(z0, z1, 0.1)])


def test_passes_under_high_ceiling():
    assert planner.passability(_slab(10, -3, 3, 2.8, 3.0), 0.0)['verdict'] == 'pass'


def test_barrier_lower_than_roof_sensor_blocks():
    assert planner.passability(_slab(10, -3, 3, 2.3, 2.5), 0.0)['verdict'] == 'too_low'


def test_drives_over_speed_bump():
    assert planner.passability(_slab(10, -3, 3, 0.0, 0.10), 0.0)['verdict'] == 'pass'


def test_block_above_ground_clearance_blocks():
    assert planner.passability(_slab(10, -0.5, 0.5, 0.0, 0.30), 0.0)['verdict'] == 'blocked'


def test_gap_wider_than_car_passes():
    r = planner.passability(np.r_[_wall(6, 15, 1.5, 0, 1.5), _wall(6, 15, -1.5, 0, 1.5)], 0.0)
    assert r['verdict'] == 'pass' and r['gap'] >= 2.9


def test_gap_narrower_than_car_does_not_pass():
    r = planner.passability(np.r_[_wall(6, 15, 1.05, 0, 1.5), _wall(6, 15, -1.05, 0, 1.5)], 0.0)
    assert r['verdict'] == 'too_narrow'


# ---- highway speed --------------------------------------------------------------------------
def _straight_route(L=300.0):
    return np.c_[np.arange(0.0, L, 1.0), np.zeros(int(L))]


def _guardrails(L=150.0, y=4.5):
    xs = np.arange(0.0, L, 0.5)
    return np.array([[x, s * y, z] for x in xs for s in (1, -1) for z in (0.2, 0.5)])


def test_highway_open_road_fast_town_road_capped():
    pts = _guardrails()
    _, v_hw, _ = planner.plan(pts, 0.0, path=_straight_route(), highway=True)
    _, v_town, _ = planner.plan(pts, 0.0, path=_straight_route(), highway=False)
    assert v_hw > 20.0, v_hw                       # ~100 km/h class on a clear highway
    assert v_town <= planner.PlannerParams().v_max + 1e-6


def test_highway_slows_for_stopped_car_ahead():
    car = np.array([[80.0 + dx, dy, 0.5] for dx in np.arange(0, 4.5, 0.25) for dy in np.arange(-0.9, 0.9, 0.2)])
    _, v, _ = planner.plan(np.r_[_guardrails(), car], 0.0, path=_straight_route(), highway=True)
    from auto_car_core.safety import safe_speed, Params
    assert v <= safe_speed(80.0, Params()) + 0.5 and v < 23.0


def test_highway_brakes_before_hairpin():
    # straight 100 m then a 22 m radius half circle: speed now must allow braking to ~5.7 m/s there
    th = np.linspace(0, math.pi, 70)
    route = np.r_[_straight_route(100.0), np.c_[100.0 + 22 * np.sin(th), 22 - 22 * np.cos(th)]]
    _, v, _ = planner.plan(_guardrails(90.0), 0.0, path=route, highway=True)
    assert v <= math.sqrt(1.5 / (1 / 22.0) + 2 * 1.5 * 100.0) + 1.0
