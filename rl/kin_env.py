"""Fast kinematic driving environment for RL, built from the same lane map as the real stack.

Why not train in Webots: with the full stack it runs at about real time, so the few hundred
thousand steps SAC needs would take days. This environment reproduces what the local planner sees
and controls, 100-1000x faster, and is used for training; Webots is used to verify.

What is shared with the real stack (so a policy transfers):
  * lane-level routes      auto_car_core.lanemap (same SUMO network, same fillets)
  * safety shield          auto_car_core.safety.evaluate (swept footprint, TTC, braking 3.2 m/s^2)
  * vehicle geometry       base_link at the rear axle, body x -1.15..4.05, |y| <= 1.0, wheelbase 2.995
  * observation builder    obs_from() is used by both this env and scripts/rl_planner.py
Simplified here: kinematic bicycle model (no tyre slip), no occlusion, obstacles are boxes.

Observation (50): route ahead (10 points every 2 m, base_link, /20), 24 lidar-like sector ranges
(/30), speed (/10), steering (/0.45), previous action (2), heading error to route, lateral error.
Action (2): steering in [-1, 1] * 0.45 rad, target speed in [-1, 1] -> [0, 8] m/s.
Reward per step (0.1 s): progress along the route (m) - 0.3 * lateral error - 0.5 * |steering
change| - 0.05 * |accel| - 2 if the shield had to cut speed; -50 on collision; +50 at the goal.
"""
import math
import os

import gymnasium as gym
import numpy as np
from scipy.spatial import cKDTree

from auto_car_core.lanemap import LaneMap
from auto_car_core.safety import CAR_FRONT, CAR_REAR, HALF_WIDTH, Params, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.1
WHEELBASE = 2.995
MAX_STEER = 0.45
V_MAX = 8.0
N_ROUTE, ROUTE_STEP = 10, 2.0
N_SECTORS, SECTOR_RANGE = 24, 30.0
OBS_DIM = 2 * N_ROUTE + N_SECTORS + 2 + 2 + 2


def to_base(pts, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    return np.c_[c * dx + s * dy, -s * dx + c * dy]


def route_ahead(route_base):
    """N_ROUTE points every ROUTE_STEP m along the route, starting at the point nearest the car."""
    i = int(np.argmin(np.hypot(route_base[:, 0], route_base[:, 1])))
    r = route_base[i:]
    if len(r) < 2:
        r = np.vstack([r, r[-1:] + [0.1, 0.0]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(r, axis=0).T))]
    g = np.arange(1, N_ROUTE + 1) * ROUTE_STEP
    return np.c_[np.interp(g, s, r[:, 0]), np.interp(g, s, r[:, 1])], r


def sectors(points_base):
    """Minimum range in N_SECTORS angular sectors around the car (body-relative, capped)."""
    out = np.full(N_SECTORS, SECTOR_RANGE)
    if len(points_base):
        rx = points_base[:, 0] - (CAR_FRONT + CAR_REAR) / 2          # measure from the body centre
        r = np.hypot(rx, points_base[:, 1])
        k = ((np.arctan2(points_base[:, 1], rx) + math.pi) / (2 * math.pi) * N_SECTORS).astype(int) % N_SECTORS
        m = r < SECTOR_RANGE
        np.minimum.at(out, k[m], r[m])
    return out


def obs_from(route_base, points_base, v, steer, prev_action):
    """Observation vector, shared by the training env and the ROS rl_planner node."""
    ahead, r = route_ahead(route_base)
    lat = float(np.min(np.hypot(r[:, 0], r[:, 1]))) if len(r) else 0.0
    side = np.sign(r[0, 1]) if len(r) else 0.0
    d = r[min(2, len(r) - 1)] - r[0] if len(r) > 1 else np.array([1.0, 0.0])
    head = math.atan2(d[1], d[0])
    return np.concatenate([
        (ahead / 20.0).ravel(), sectors(points_base) / SECTOR_RANGE,
        [v / 10.0, steer / MAX_STEER], np.asarray(prev_action, float),
        [head / math.pi, side * min(lat, 5.0) / 5.0]]).astype(np.float32)


def action_to_command(a):
    return float(np.clip(a[0], -1, 1)) * MAX_STEER, (float(np.clip(a[1], -1, 1)) + 1.0) / 2.0 * V_MAX


class KinematicDrivingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, net_file=None, obstacle_rate=0.6, lead_car_rate=0.4, seed=None):
        super().__init__()
        self.map = LaneMap(net_file or os.path.join(HERE, '..', 'worlds', 'city_net', 'sumo.net.xml'))
        self.obstacle_rate, self.lead_car_rate = obstacle_rate, lead_car_rate
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.ps = Params()
        self.np_random = np.random.default_rng(seed)
        # road edges: outer side of every lane with index 0 (the rightmost lane of each direction)
        edge_pts = []
        for lane in self.map.road_lanes:
            if lane.index != 0:
                continue
            for i in range(len(lane.pts) - 1):
                a, b = lane.pts[i], lane.pts[i + 1]
                d = b - a
                L = float(np.hypot(*d))
                if L < 1e-6:
                    continue
                n = np.array([d[1], -d[0]]) / L                       # right-hand normal
                # leave the first/last 6 m of every lane free: junction corners are rounded kerbs,
                # not sharp walls (sharp corner points blocked every turning arc at junctions)
                for t in np.arange(0.0, L, 0.5):
                    if (lane.s[i] + t) < 6.0 or (lane.s[i] + t) > lane.length - 6.0:
                        continue
                    edge_pts.append(a + d * (t / L) + n * (lane.width / 2 + 0.3))
        self.edges = np.array(edge_pts)
        self.edge_tree = cKDTree(self.edges)

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        rng = self.np_random
        route = None
        for attempt in range(500):            # keep sampling until a valid route exists
            lane = self.map.road_lanes[rng.integers(len(self.map.road_lanes))]
            if lane.length < 20:
                continue
            s0 = rng.uniform(2.0, lane.length - 10.0)
            p0 = lane.at(s0)
            d = lane.at(s0 + 1.0) - p0
            yaw = math.atan2(d[1], d[0])
            goal_lane = self.map.road_lanes[rng.integers(len(self.map.road_lanes))]
            goal = goal_lane.at(rng.uniform(0.3, 0.7) * goal_lane.length)
            route = self.map.route(tuple(p0), yaw, tuple(goal))
            if route is None:
                continue
            length = float(np.hypot(*np.diff(route, axis=0).T).sum())
            if 60.0 < length < 300.0 or (attempt > 300 and length > 30.0):
                break
            route = None
        if route is None:
            raise RuntimeError('no valid route found on this map after 500 samples')
        self.route, self.route_len = route, length
        self.route_s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
        self.x, self.y = float(p0[0]) + rng.normal(0, 0.3), float(p0[1]) + rng.normal(0, 0.3)
        self.yaw = yaw + rng.normal(0, 0.05)
        self.v, self.steer = 0.0, 0.0
        self.prev_action = np.zeros(2)
        self.progress = 0.0
        self.t = 0
        # static obstacles (cones / parked cars) beside or on the route, and slower lead cars
        self.boxes = []
        for _ in range(rng.poisson(3 * self.obstacle_rate)):
            s = rng.uniform(25.0, max(30.0, length - 20.0))
            c = self.route_point(s)
            size = rng.choice([0.5, 1.8])
            near = self.map.nearest_lane(tuple(c), include_internal=False)
            half_lane = (near[0].width / 2) if near else 1.6
            # beside the lane, never sticking into it (village lanes are only 2.5 m wide)
            off = rng.choice([-1, 1]) * (half_lane + size / 2 + rng.uniform(0.3, 1.5))
            h = self.route_heading(s)
            c = c + off * np.array([-math.sin(h), math.cos(h)])
            self.boxes.append({'c': c, 'size': size, 'v': 0.0, 's': s, 'off': off})
        self.leads = []
        if rng.random() < self.lead_car_rate:
            self.leads.append({'s': rng.uniform(20.0, 45.0), 'v': rng.uniform(2.0, 5.0)})
        return self._obs(), {}

    def route_point(self, s):
        return np.array([np.interp(s, self.route_s, self.route[:, 0]), np.interp(s, self.route_s, self.route[:, 1])])

    def route_heading(self, s):
        a, b = self.route_point(s), self.route_point(s + 1.0)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    # -------------------------------------------------------------- sensing
    def _obstacle_world(self):
        pts = [self.edges[self.edge_tree.query_ball_point([self.x, self.y], 40.0)]]
        for b in self.boxes:
            h = b['size'] / 2
            g = np.linspace(-h, h, max(int(b['size'] / 0.25), 2))
            pts.append(np.c_[b['c'][0] + np.r_[g, g, np.full_like(g, -h), np.full_like(g, h)],
                             b['c'][1] + np.r_[np.full_like(g, -h), np.full_like(g, h), g, g]])
        for car in self.leads:
            c, hd = self.route_point(car['s']), self.route_heading(car['s'])
            L, W = 4.5, 1.8
            u, w = np.array([math.cos(hd), math.sin(hd)]), np.array([-math.sin(hd), math.cos(hd)])
            gl, gw = np.linspace(-L / 2, L / 2, 18), np.linspace(-W / 2, W / 2, 8)
            pts.append(np.vstack([c + np.outer(gl, u) + side * W / 2 * w for side in (-1, 1)] +
                                 [c + np.outer(gw, w) + side * L / 2 * u for side in (-1, 1)]))
        return np.vstack([p for p in pts if len(p)]) if pts else np.zeros((0, 2))

    def _physical_points(self):
        n_edges = getattr(self, '_n_edge_pts', 0)
        return self._last_points[n_edges:] if len(self._last_points) else self._last_points

    def _sensed(self):
        """Sensed points (base_link): road-edge (virtual) points FIRST, then physical obstacles;
        self._n_edge_pts tells where the physical part starts."""
        world = self._obstacle_world()
        n_e = len(self.edge_tree.query_ball_point([self.x, self.y], 40.0))
        base = to_base(world, self.x, self.y, self.yaw)
        keep = np.hypot(base[:, 0], base[:, 1]) < 40.0
        self._n_edge_pts = int(keep[:n_e].sum())
        base = base[keep]
        base = base + self.np_random.normal(0, 0.03, base.shape)
        return np.c_[base, np.full(len(base), 0.5)]

    def _tracks(self):
        out = []
        for car in self.leads:
            c = to_base(self.route_point(car['s'])[None], self.x, self.y, self.yaw)[0]
            hd = self.route_heading(car['s']) - self.yaw
            out.append({'pos': (c[0], c[1]), 'vel': (car['v'] * math.cos(hd), car['v'] * math.sin(hd)), 'size': (4.5, 1.8)})
        return out

    def _obs(self):
        rb = to_base(self.route, self.x, self.y, self.yaw)
        self._last_points = self._sensed()
        return obs_from(rb, self._last_points[:, :2], self.v, self.steer, self.prev_action)

    # ----------------------------------------------------------------- step
    def step(self, action):
        steer_cmd, v_cmd = action_to_command(action)
        # the SAME safety shield as on the car
        # the shield sees PHYSICAL obstacles only (boxes, cars); road edges are virtual walls for the
        # planner, like the camera road boundary on the car (kerbs are drive-over height)
        v_safe, info = evaluate(v_cmd, steer_cmd, self.v, self._physical_points(), self._tracks(), self.ps)
        intervened = v_safe < v_cmd - 0.5
        # actuators: steering rate limit, speed tracking with measured accel/brake limits
        self.steer += float(np.clip(steer_cmd - self.steer, -0.6 * DT, 0.6 * DT))
        dv = float(np.clip(v_safe - self.v, -3.8 * DT, 2.0 * DT))
        self.v = max(0.0, self.v + dv)
        yaw_rate = self.v * math.tan(self.steer) / WHEELBASE
        self.x += self.v * math.cos(self.yaw + yaw_rate * DT / 2) * DT
        self.y += self.v * math.sin(self.yaw + yaw_rate * DT / 2) * DT
        self.yaw += yaw_rate * DT
        for car in self.leads:
            car['s'] += car['v'] * DT
        # a lead car that reaches the end of our route has turned away: remove it (otherwise it
        # stays parked at the goal while still reporting its speed - a phantom moving obstacle)
        self.leads = [c for c in self.leads if c['s'] < self.route_len - 3.0]
        self.t += 1
        # progress / errors along the route
        d = np.hypot(self.route[:, 0] - self.x, self.route[:, 1] - self.y)
        i = int(np.argmin(d))
        s_now = float(self.route_s[i])
        gained = max(0.0, s_now - self.progress)
        self.progress = max(self.progress, s_now)
        lat = float(d[i])
        rb = to_base(self.route, self.x, self.y, self.yaw)
        self._last_points = self._sensed()                    # sense AFTER moving
        collision = self._collided()
        reached = self.route_len - s_now < 5.0 and lat < 3.0
        reward = (gained - 0.3 * min(lat, 5.0) - 0.5 * abs(action[0] - self.prev_action[0])
                  - 0.05 * abs(dv / DT) - (2.0 if intervened else 0.0))
        if collision:
            reward -= 50.0
        off_route = lat > 8.0
        if off_route:
            # Leaving the route ends the episode: it must cost as much as a crash. Without this the
            # first SAC policy learned to drive fast and simply 'end' episodes at turns (13 of 15
            # evaluation episodes ended off-route, success 10%).
            reward -= 50.0
        if reached:
            reward += 50.0
        self.prev_action = np.asarray(action, float)
        terminated = collision or reached or off_route
        truncated = self.t >= 1800          # 180 s: enough for 300 m at turn speeds
        info = {'collision': collision, 'reached': reached, 'off_route': off_route, 'lat': lat, 'intervened': intervened,
                'progress': self.progress, 'route_len': self.route_len}
        obs = obs_from(rb, self._last_points[:, :2], self.v, self.steer, self.prev_action)
        return obs, float(reward), terminated, truncated, info

    def _collided(self):
        p = self._last_points
        if len(p) == 0:
            return False
        inside = (p[:, 0] > CAR_REAR) & (p[:, 0] < CAR_FRONT) & (np.abs(p[:, 1]) < HALF_WIDTH)
        return bool(inside.sum() >= 2)


def planner_action(env, prev_steer=0.0):
    """The hand-written planner's command in this env, as an action (the RL baseline)."""
    from auto_car_core.planner import plan
    rb = to_base(env.route, env.x, env.y, env.yaw)
    st, v, _ = plan(env._last_points, prev_steer, path=rb)
    return np.array([st / MAX_STEER, v / V_MAX * 2.0 - 1.0], np.float32)
