#!/usr/bin/env python3
"""Check EVERY turn of the road network the way the car will drive it.

  python3 tools/audit_turns.py [--net worlds/village_net/sumo.net.xml] [--lag 0.2] [--show 25]

For every connection between two road lanes (U-turns excluded): the path the planner builds
(last 30 m of the arriving lane + the junction link + first 30 m of the leaving lane, through
LaneMap.smooth_kinks - fillets, wide turns), then
  * sharpest curvature vs the car's limit 1/6.2 m,
  * how far the path leaves the map lanes (corner cut / swing),
  * a closed-loop simulation of the real planner (auto_car_core.planner.plan) with the measured
    steering delay, at the speed it would choose there: worst distance of the car from the path.
PASS = steerable, within 1.0 m of the lanes, simulated car within 0.6 m of the path, and STRAIGHT in
its own lane up to the turn and after it (<= 0.25 m off the lane centre, <= 3 deg to the lane).
"""
import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import auto_car_core.planner as PL  # noqa: E402
from auto_car_core.lanemap import LaneMap  # noqa: E402

K_CAR = math.tan(0.45) / 2.995


def _cl(P):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def clip_len(P, length, from_end):
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if from_end:
        return P[s >= s[-1] - length]
    return P[s <= length]


def lane_err(L, x, y, yaw):
    """Offset of (x, y) from lane polyline L and the car's yaw relative to the lane there."""
    i = int(np.argmin(np.hypot(L[:, 0] - x, L[:, 1] - y)))
    lo, hi = max(i - 2, 0), min(i + 3, len(L))
    if hi - lo < 2:
        return 0.0, 0.0
    off = abs(float(PL.signed_offset(L[lo:hi], [x], [y])[0]))
    d = L[min(i + 1, len(L) - 1)] - L[max(i - 1, 0)]
    dy = math.atan2(math.sin(yaw - math.atan2(d[1], d[0])), math.cos(yaw - math.atan2(d[1], d[0])))
    return off, abs(math.degrees(dy))


def simulate(P, lag, dt=0.1, A=None, B=None):
    """Closed loop on P. Returns (worst offset from P, worst offset / yaw from the arriving lane A on
    its last 25 m before the junction, the same on the leaving lane B from 12 m after it)."""
    pp = PL.PlannerParams()
    app = [0.0, 0.0]
    ext = [0.0, 0.0]
    x, y = P[0]
    yaw = math.atan2(*(P[min(3, len(P) - 1)] - P[0])[::-1])
    v, steer, cmd, worst = 4.0, 0.0, 0.0, 0.0
    sP = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    for _ in range(600):
        c, s_ = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - x, P[:, 1] - y
        rb = np.c_[c * dx + s_ * dy, -s_ * dx + c * dy]
        st, vc, _ = PL.plan(np.zeros((0, 3)), cmd, None, pp=pp, path=rb, n_physical=0, ego_v=v)
        cmd = float(np.clip(st, cmd - 0.08, cmd + 0.08))
        steer += (cmd - steer) * min(1.0, dt / lag)
        v += float(np.clip(vc - v, -1.5 * dt, 1.0 * dt))
        x += v * c * dt
        y += v * s_ * dt
        yaw += 0.95 * v * math.tan(steer) / 2.995 * dt
        i = int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))
        lo, hi = max(i - 3, 0), min(i + 4, len(P))
        if hi - lo >= 2 and 5.0 < sP[i] < sP[-1] - 5.0:
            worst = max(worst, abs(float(PL.signed_offset(P[lo:hi], [x], [y])[0])))
        if A is not None and len(A) >= 6 and len(B) >= 6:
            dA = np.hypot(A[:, 0] - x, A[:, 1] - y)
            jA = int(np.argmin(dA))
            if 1 <= jA <= len(A) - 2 and dA[jA] < 3.0 and sP[i] > 5.0:     # still on the approach
                o, y_ = lane_err(A, x, y, yaw)
                app = [max(app[0], o), max(app[1], y_)]
            dB = np.hypot(B[:, 0] - x, B[:, 1] - y)
            jB = int(np.argmin(dB))
            if jB >= 2 and jB <= len(B) - 3 and dB[jB] < 3.0:              # settled on the new lane
                o, y_ = lane_err(B, x, y, yaw)
                ext = [max(ext[0], o), max(ext[1], y_)]
        if i >= len(P) - 3:
            break
    return worst, app, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='worlds/village_net/sumo.net.xml')
    ap.add_argument('--lag', type=float, default=0.2)
    ap.add_argument('--show', type=int, default=25)
    a = ap.parse_args()
    m = LaneMap(a.net)
    root = ET.parse(a.net).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    lane = {ln.get('id'): np.array([[float(v) for v in q.split(',')[:2]] for q in ln.get('shape').split()]) - off
            for e in root.findall('edge') for ln in e.findall('lane')}
    rev = lambda e: e[1:] if e.startswith('-') else '-' + e
    rows = []
    for c in root.findall('connection'):
        f, t, via = c.get('from'), c.get('to'), c.get('via')
        if f.startswith(':') or t == rev(f) or not via:
            continue
        A, B, V = lane.get(f'{f}_{c.get("fromLane")}'), lane.get(f'{t}_{c.get("toLane")}'), lane.get(via)
        if A is None or B is None or V is None:
            continue
        raw = np.vstack([clip_len(A, 30.0, True), V, clip_len(B, 30.0, False)])
        raw = raw[np.r_[True, np.hypot(*np.diff(raw, axis=0).T) > 1e-6]]
        # resample 1 m (like the route)
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(raw, axis=0).T))]
        ss = np.arange(0.0, s[-1], 1.0)
        raw = np.c_[np.interp(ss, s, raw[:, 0]), np.interp(ss, s, raw[:, 1])]
        if len(raw) < 8:
            continue
        P = m.smooth_kinks(raw)
        g = np.diff(P, axis=0)
        L = np.hypot(*g.T)
        h = np.unwrap(np.arctan2(g[:, 1], g[:, 0]))
        # curvature over 1.5 m windows (a 5 deg bend across a 5 cm gap is not a hairpin)
        sP = np.r_[0.0, np.cumsum(L)]
        k = np.zeros(1)
        for q in range(len(h)):
            q2 = min(int(np.searchsorted(sP, sP[q] + 1.5)), len(h) - 1)
            if sP[q2] - sP[q] > 1.0:
                k = np.r_[k, abs(h[q2] - h[q]) / (sP[q2] - sP[q])]
        turn = math.degrees(h[-1] - h[0])
        dev = max(float(np.min(np.hypot(*(raw - q).T))) for q in P[::2])
        rs = lambda Q: np.c_[np.interp(np.arange(0, _cl(Q)[-1], 1.0), _cl(Q), Q[:, 0]),
                             np.interp(np.arange(0, _cl(Q)[-1], 1.0), _cl(Q), Q[:, 1])]
        # only where the ROUTE itself is still straight in the lane (before the turn's arc begins,
        # after it ends): there the CAR must be straight in the lane centre too
        Aa, Bb = rs(clip_len(A, 25.0, True)), rs(clip_len(B, 30.0, False))
        if len(Aa) >= 2 and len(Bb) >= 2:
            onP = lambda Q: np.array([float(np.min(np.hypot(*(P - q).T))) for q in Q]) < 0.08
            mA, mB = onP(Aa), onP(Bb)
            kA = int(np.argmin(mA)) if not mA.all() else len(Aa)
            kB = len(Bb) - int(np.argmin(mB[::-1])) if not mB.all() else 0
            Aa, Bb = Aa[:max(kA - 1, 0)], Bb[min(kB + 1, len(Bb)):]
        worst, app, ext = simulate(P, a.lag, A=Aa, B=Bb)
        straight = app[0] <= 0.25 and app[1] <= 3.0 and ext[0] <= 0.25 and ext[1] <= 3.0
        ok = k.max() <= K_CAR * 1.02 and dev <= 1.0 and worst <= 0.6 and straight
        rows.append((ok, f, t, turn, float(k.max()), dev, worst, raw[len(raw) // 2], app, ext))
    bad = [r for r in rows if not r[0]]
    print(f'{len(rows)} turns checked: {len(rows) - len(bad)} PASS, {len(bad)} need attention')
    A_ = np.array([r[8] for r in rows])
    E_ = np.array([r[9] for r in rows])
    print(f'approach (last 25 m in the own lane): worst offset {A_[:, 0].max():.2f} m, worst angle '
          f'{A_[:, 1].max():.1f} deg, median {np.median(A_[:, 0]):.2f} m / {np.median(A_[:, 1]):.1f} deg')
    print(f'exit (new lane, after the arc): worst offset {E_[:, 0].max():.2f} m, worst angle {E_[:, 1].max():.1f} deg, '
          f'median {np.median(E_[:, 0]):.2f} m / {np.median(E_[:, 1]):.1f} deg')
    for ok, f, t, turn, kmax, dev, worst, p, app, ext in sorted(bad, key=lambda r: -r[6])[:a.show]:
        why = []
        if kmax > K_CAR * 1.02:
            why.append(f'sharper than the car (R {1 / kmax:.1f} m)')
        if dev > 1.0:
            why.append(f'leaves the lanes by {dev:.1f} m')
        if worst > 0.6:
            why.append(f'car off the path by {worst:.2f} m')
        if app[0] > 0.25 or app[1] > 3.0:
            why.append(f'approach not straight ({app[0]:.2f} m, {app[1]:.0f} deg)')
        if ext[0] > 0.25 or ext[1] > 3.0:
            why.append(f'exit not straight ({ext[0]:.2f} m, {ext[1]:.0f} deg)')
        print(f'  {f:>14} -> {t:<14} turn {turn:+5.0f} deg at ({p[0]:.0f},{p[1]:.0f}): ' + '; '.join(why))


if __name__ == '__main__':
    main()
