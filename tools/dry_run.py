#!/usr/bin/env python3
"""DRY RUN: predict how the car will drive a route BEFORE driving it.

  python3 tools/dry_run.py [--to hospital|X,Y] [--from X,Y,YAW_DEG] [--speeds 5.5,8.3,11.1] [--lag 0.5]

1. Plans the route on the lane map exactly like route_planner (lanes, roundabout spiral, smoothing).
2. Finds every CORNER (curvature > 0.02 1/m): its tightest radius R, how much it turns, and the
   speed a driver would take it at: v = sqrt(a_lat * R), a_lat = 1.5 m/s^2 (comfortable).
3. Simulates the real planner (auto_car_core.planner.plan) in a closed loop with a kinematic car
   that has the measured steering lag and rate limit, at each open-road cruise speed, and reports
   per corner the predicted entry speed and the worst distance from the lane centre.
4. Recommends the fastest cruise speed with which no corner is predicted to leave the lane
   (worst wheel within the lane: < 0.75 m from its centre).
"""
import argparse
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import auto_car_core.planner as PL  # noqa: E402
from auto_car_core.lanemap import LaneMap  # noqa: E402

A_LAT = 1.5


def curvature(P):
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    h = np.unwrap(np.arctan2(*np.gradient(P, axis=0).T[::-1]))
    # heading change over +-1.5 m
    k = np.zeros(len(P))
    for i in range(len(P)):
        a = np.searchsorted(s, s[i] - 1.5)
        b = min(np.searchsorted(s, s[i] + 1.5), len(P) - 1)
        if s[b] - s[a] > 0.5:
            k[i] = (h[b] - h[a]) / (s[b] - s[a])
    return s, h, k


def corners(P):
    s, h, k = curvature(P)
    out, i = [], 0
    while i < len(P):
        if abs(k[i]) > 0.02:
            j = i
            while j + 1 < len(P) and abs(k[j + 1]) > 0.02:
                j += 1
            kmax = float(np.max(np.abs(k[i:j + 1])))
            turn = math.degrees(h[j] - h[i])
            if abs(turn) > 10 or kmax > 0.06:
                R = 1.0 / kmax
                out.append({'s0': s[i], 's1': s[j], 'xy': P[(i + j) // 2], 'R': R, 'turn': turn,
                            'side': 'left' if turn > 0 else 'right', 'v_rec': math.sqrt(A_LAT * R)})
            i = j + 1
        else:
            i += 1
    return out, s


def simulate(P, cruise, lag, rate=0.08, dt=0.1):
    pp = PL.PlannerParams()
    pp.v_max = cruise
    x, y = P[0]
    yaw = math.atan2(*(P[3] - P[0])[::-1])
    v, steer, cmd, t = 0.0, 0.0, 0.0, 0.0
    log = []
    sP = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    for _ in range(int(3000 / dt)):
        c, s_ = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rb = np.c_[c * dx + s_ * dy, -s_ * dx + c * dy]
        st, vc, _ = PL.plan(np.zeros((0, 3)), cmd, None, pp=pp, path=rb, n_physical=0, ego_v=v)
        k = abs(math.tan(cmd)) / 2.995
        if k > 1e-3:
            vc = min(vc, max(2.5, math.sqrt(1.5 / k)))
        cmd = float(np.clip(st, cmd - rate, cmd + rate))
        steer += (cmd - steer) * min(1.0, dt / lag)
        v += float(np.clip(vc - v, -1.5 * dt, 1.2 * dt))
        x += v * c * dt
        y += v * s_ * dt
        yaw += v * math.tan(steer) / 2.995 * dt
        # TRUE distance of the body centre from the route LINE (segments), not from the nearest
        # route vertex: vertices are 1 m apart, the vertex distance read up to 0.5 m on the line
        bx, by = x + 1.45 * c, y + 1.45 * s_
        i = int(np.argmin(np.hypot(P[:, 0] - bx, P[:, 1] - by)))
        lo, hi = max(i - 3, 0), min(i + 4, len(P))
        e = abs(float(PL.signed_offset(P[lo:hi], [bx], [by])[0])) if hi - lo >= 2 else 0.0
        log.append((sP[i], e, v))
        if i >= len(P) - 3:
            break
    return np.array(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='worlds/village_net/sumo.net.xml')
    ap.add_argument('--from', dest='frm', default='88.4,-479.8,80')
    ap.add_argument('--to', default='hospital')
    ap.add_argument('--speeds', default='5.5,8.3,11.1')
    ap.add_argument('--lag', type=float, default=0.2)   # measured: tools/steer_id.py
    a = ap.parse_args()
    m = LaneMap(a.net)
    fx, fy, fh = (float(v) for v in a.frm.split(','))
    if ',' in a.to:
        goal = tuple(float(v) for v in a.to.split(','))
    else:
        pois = yaml.safe_load(open(os.path.join(os.path.dirname(a.net), 'pois.yaml')))['places']
        p = [q for q in pois if a.to.lower() in (q.get('kind', '') + ' ' + q.get('name', '')).lower()][0]
        goal = (p['x'], p['y'])
    P = m.route_near((fx, fy), math.radians(fh), goal, max_distance=200.0)
    cs, sP = corners(P)
    print(f'route {sP[-1]:.0f} m, {len(cs)} corners')
    speeds = [float(v) for v in a.speeds.split(',')]
    runs = {v: simulate(P, v, a.lag) for v in speeds}
    hdr = ' '.join(f'| {v * 3.6:4.0f} km/h: entry  worst' for v in speeds)
    print(f"{'corner':>6} {'at (x,y)':>14} {'turn':>6} {'R m':>6} {'rec km/h':>8} {hdr}")
    ok = {v: True for v in speeds}
    for n, c in enumerate(cs):
        row = f"{n:6d} ({c['xy'][0]:5.0f},{c['xy'][1]:5.0f}) {c['turn']:+5.0f}d {c['R']:6.1f} {c['v_rec'] * 3.6:8.0f} "
        for v in speeds:
            L = runs[v]
            sel = (L[:, 0] >= c['s0'] - 3) & (L[:, 0] <= c['s1'] + 3)
            if not sel.any():
                row += '|      -       -     '
                continue
            ent = L[sel][0, 2] * 3.6
            worst = L[sel][:, 1].max()
            ok[v] &= worst < 0.75
            row += f"| {ent:7.0f}  {worst:5.2f}{' !' if worst >= 0.75 else '  '}   "
        print(row)
    for v in speeds:
        L = runs[v]
        print(f'cruise {v * 3.6:4.0f} km/h: whole route worst {L[:, 1].max():.2f} m, mean {L[:, 1].mean():.2f} m, '
              f'time {len(L) * 0.1:.0f} s -> {"OK" if ok[v] else "leaves the lane in a corner"}')
    good = [v for v in speeds if ok[v]]
    print(f'RECOMMENDED cruise: {max(good) * 3.6:.0f} km/h' if good else 'no tested cruise keeps every corner in lane')


if __name__ == '__main__':
    main()
