#!/usr/bin/env python3
"""Add PEOPLE and ANIMALS to a world: pedestrians that walk the pavements and use the zebra
crossings, and animals in the gardens, fields and at the forest edge.

  python3 tools/add_life.py worlds/village.wbt [--walkers 6] [--check]

Pedestrians (Webots 'Pedestrian' + its 'pedestrian' controller, which loops a waypoint list):
  * one per zebra crossing: walks along the pavement on one side, crosses ON the zebra, walks
    along the other side and back - so it is on the road only a small part of the time, at the
    crossing (the car must yield to it; it never stands on the road),
  * --walkers more on the pavements of the longest roads (2.2 m beyond the road edge), back and forth.
Animals (static): a dog and a cat by houses, a flock of sheep, cows and a horse in an open field,
a deer and a fox at the forest edge, rabbits - all >= 8 m from every lane (dogs/cats >= 6 m),
off buildings, trees, water and the hill.
Re-running replaces what it added. Backup .before_life.
"""
import argparse
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from matplotlib.path import Path
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fix_lane_offsets import drawn_roads  # noqa: E402
from terrain_height import height_grid  # noqa: E402

BASE = 'https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/'
PROTOS = {'Pedestrian': 'humans/pedestrian/protos/Pedestrian.proto',
          'Dog': 'objects/animals/protos/Dog.proto', 'Cat': 'objects/animals/protos/Cat.proto',
          'Cow': 'objects/animals/protos/Cow.proto', 'Sheep': 'objects/animals/protos/Sheep.proto',
          'Horse': 'objects/animals/protos/Horse.proto', 'Deer': 'objects/animals/protos/Deer.proto',
          'Fox': 'objects/animals/protos/Fox.proto', 'Rabbit': 'objects/animals/protos/Rabbit.proto'}
SHIRTS = ['0.25 0.55 0.85', '0.8 0.2 0.2', '0.2 0.6 0.3', '0.9 0.75 0.2', '0.5 0.3 0.6', '0.95 0.95 0.95',
          '0.3 0.3 0.3', '0.85 0.45 0.15']
PANTS = ['0.15 0.15 0.3', '0.25 0.25 0.25', '0.4 0.3 0.2', '0.1 0.1 0.1']
SKIN = ['1.0 0.75 0.7', '0.85 0.6 0.45', '0.55 0.38 0.27', '0.95 0.8 0.65']
TAG = 'add_life'


def rot(axis_angle):
    x, y, z, a = axis_angle
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def water_zones(text):
    """(points, buffer) keep-out sets for the river and the lakes."""
    zones = []
    for m in re.finditer(r'(?ms)^DEF (RIVER|SMALL_LAKE|BIG_LAKE) Pose \{\n(.*?)^\}', text):
        b = m.group(2)
        tr = np.array([float(v) for v in re.search(r'(?m)^  translation (.+)$', b).group(1).split()])
        r_ = re.search(r'(?m)^  rotation (.+)$', b)
        R = rot([float(v) for v in r_.group(1).split()]) if r_ else np.eye(3)
        sp = re.search(r'spine \[(.*?)\]', b, re.S)
        if sp:
            P = np.array(sp.group(1).split(), float).reshape(-1, 3)
            buf = 7.0
        else:
            pt = re.search(r'point \[(.*?)\]', b, re.S)
            if not pt:
                continue
            P = np.array(pt.group(1).replace(',', ' ').split(), float).reshape(-1, 3)
            buf = 4.0
        W = P @ R.T + tr
        # densify the spine / outline so a distance test works between vertices
        out = [W[0, :2]]
        for a_, b_ in zip(W[:-1, :2], W[1:, :2]):
            n = max(int(np.hypot(*(b_ - a_)) / 1.0), 1)
            out += [a_ + (b_ - a_) * t for t in np.linspace(0, 1, n + 1)[1:]]
        zones.append((np.array(out), buf, Path(W[:, :2]) if not sp and len(W) > 2 else None))
    return zones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--walkers', type=int, default=6)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    text = open(a.world).read()
    text = re.sub(r'(?ms)^(?:Pedestrian|Dog|Cat|Cow|Sheep|Horse|Deer|Fox|Rabbit) \{\n  translation [^\n]*\n'
                  r'(?:  [^\n]*\n)*?  name "' + TAG + r'[^"]*"\n(?:.*?\n)*?^\}\n', '', text)
    net_dir = os.path.splitext(a.world)[0] + '_net'
    root = ET.parse(os.path.join(net_dir, 'sumo.net.xml')).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    lane_pts = []
    for e in root.findall('edge'):
        for ln in e.findall('lane'):
            P = np.array([[float(v) for v in q.split(',')[:2]] for q in ln.get('shape').split()]) - off
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
            ss = np.arange(0.0, s[-1] + 1e-9, 0.5)
            lane_pts.append(np.c_[np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])])
    lanes = cKDTree(np.vstack(lane_pts))
    roads = drawn_roads(a.world)
    places = yaml.safe_load(open(os.path.join(net_dir, 'pois.yaml')))['places']
    foots = [Path(np.array(p['footprint'])) for p in places]
    houses = [p for p in places if p['kind'] in ('house', 'residential')]
    obj = []
    for m in re.finditer(r'(?ms)^(?:DEF \S+ )?([A-Z]\w*) \{\n  translation ([-\d.e]+) ([-\d.e]+)', text):
        if m.group(1) not in ('Road', 'Solid', 'Pose', 'Crossroad', 'RoadIntersection', 'Roundabout', 'Forest',
                              'PedestrianCrossing', 'StraightRoadSegment', 'ParkingLines'):
            obj.append((float(m.group(2)), float(m.group(3))))
    objs = cKDTree(np.array(obj))
    water = water_zones(text)
    forests = [np.array([float(x), float(y)]) for x, y in re.findall(r'(?m)^Forest \{\n  translation ([-\d.e]+) ([-\d.e]+)', text)]
    discs = [(np.array([float(x), float(y)]), float(r)) for x, y, r in re.findall(
        r'translation ([-\d.e]+) ([-\d.e]+) [-\d.e]+\n  name "turning circle\([^)]*\)".*?radius ([\d.]+)', text, re.S)]

    def ok(q, lane_gap, obj_gap=1.5, bldg_gap=1.5):
        if lanes.query(q)[0] < lane_gap:
            return False
        if any(f.contains_point(q) for f in foots):
            return False
        if bldg_gap > 0 and any(float(np.min(np.hypot(*(f.vertices - q).T))) < bldg_gap for f in foots):
            return False
        if objs.query(q)[0] < obj_gap:
            return False
        for P, buf, poly in water:
            if float(np.min(np.hypot(*(P - q).T))) < buf or (poly is not None and poly.contains_point(q)):
                return False
        for o, r in discs:
            if np.hypot(*(q - o)) < r + 1.0:
                return False
        return True

    def flat(q, r=2.0):
        g = height_grid(a.world, q[0], q[1], r)
        return not g or max(g.values()) < 0.2

    def road_frame(q):
        """Nearest drawn road: (unit direction, unit left normal, half width, foot point)."""
        best = None
        for C, W, _ in roads.values():
            d = np.hypot(*(C - q).T)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                j0, j1 = max(i - 2, 0), min(i + 2, len(C) - 1)
                t = C[j1] - C[j0]
                t = t / max(np.hypot(*t), 1e-9)
                best = (d[i], t, np.array([-t[1], t[0]]), W / 2, C[i])
        return best[1:]

    def pavement(p0, t, n, side, hw, length):
        """Walkable points along the road side: from p0 along t for `length`, hw + 2.2 m out."""
        pts = []
        for s in np.linspace(0.0, length, 9):
            q = p0 + s * t + side * (hw + 2.2) * n
            if not ok(q, lane_gap=2.85, obj_gap=0.8, bldg_gap=0.8):
                break
            pts.append(q)
        return pts

    people, animals = [], []
    # ---- 1) one pedestrian per zebra crossing
    for m in re.finditer(r'(?ms)^PedestrianCrossing \{\n  translation ([-\d.e]+) ([-\d.e]+)', text):
        c = np.array([float(m.group(1)), float(m.group(2))])
        t, n, hw, foot = road_frame(c)
        A, B = c + (hw + 1.2) * n, c - (hw + 1.2) * n            # kerb points on both sides
        walk_a = pavement(c, t, n, +1, hw, 16.0)
        walk_b = pavement(c, -t, n, -1, hw, 16.0)
        if not walk_a or not walk_b:
            continue
        traj = [walk_a[-1], walk_a[0], A, B, walk_b[0], walk_b[-1], walk_b[0], B, A, walk_a[0]]
        people.append((traj, 1.2, f'crossing at ({c[0]:.0f},{c[1]:.0f})'))
    # ---- 2) walkers on the pavements of the longest roads
    longest = sorted(roads.items(), key=lambda kv: -float(np.sum(np.hypot(*np.diff(kv[1][0], axis=0).T))))
    for rid, (C, W, _) in longest:
        if len(people) >= 5 + a.walkers or rid.startswith(('link', 'conn')):
            continue
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
        mid = s[-1] / 2
        i = int(np.searchsorted(s, mid))
        if i <= 0 or i >= len(C) - 1:
            continue
        side = 1 if len(people) % 2 else -1
        # follow the road's curve on the pavement for ~40 m
        pts = []
        for sq in np.linspace(mid - 20, mid + 20, 9):
            j = min(max(int(np.searchsorted(s, sq)), 1), len(C) - 1)
            t = C[j] - C[j - 1]
            t = t / max(np.hypot(*t), 1e-9)
            q = C[j] + side * (W / 2 + 2.2) * np.array([-t[1], t[0]])
            if ok(q, lane_gap=2.85, obj_gap=0.8, bldg_gap=0.8):
                pts.append(q)
            elif len(pts) >= 3:
                break
            else:
                pts = []
        if len(pts) >= 3:
            people.append((pts + pts[-2:0:-1], 1.0 + 0.1 * (len(people) % 4), f'pavement of road {rid}'))
    # ---- 3) animals
    xs = np.array([p for P in lane_pts for p in P[::10]])
    lo, hi = xs.min(0) - 20, xs.max(0) + 20
    grid = np.mgrid[lo[0]:hi[0]:6.0, lo[1]:hi[1]:6.0].reshape(2, -1).T
    gd = lanes.query(grid)[0]
    field = grid[(gd > 14) & (gd < 40)]

    def place(kind, cands, lane_gap, n=1, spread=3.0, why=''):
        for q in cands:
            if not ok(q, lane_gap) or not flat(q):
                continue
            herd = [q]
            for k in range(1, n):
                ang = 2.4 * k
                p = q + spread * math.sqrt(k) * np.array([math.cos(ang), math.sin(ang)])
                if ok(p, lane_gap) and flat(p):
                    herd.append(p)
            for k, p in enumerate(herd):
                animals.append((kind, p, (k * 1.7 + len(animals)) % (2 * math.pi), why))
            return True
        return False

    rng = np.random.default_rng(3)
    order = rng.permutation(len(field))
    place('Sheep', field[order], 12.0, n=5, spread=2.5, why='flock in a field')
    place('Cow', field[order[::-1]], 12.0, n=3, spread=4.0, why='cows in a field')
    place('Horse', field[order[len(order) // 2:]], 12.0, n=1, why='horse in a field')
    for f in forests[:2]:
        ring = [f + r * np.array([math.cos(u), math.sin(u)]) for r in (18, 24, 30) for u in np.linspace(0, 6.28, 16)]
        place('Deer', ring, 10.0, why='forest edge')
    if forests:
        f = forests[-1]
        place('Fox', [f + r * np.array([math.cos(u), math.sin(u)]) for r in (16, 22, 28) for u in np.linspace(0, 6.28, 16)],
              10.0, why='forest edge')
    for kind, hs in (('Dog', houses[:6]), ('Cat', houses[6:12])):
        for h in hs:
            F = np.array(h['footprint'])
            c = F.mean(0)
            cand = [c + (np.hypot(*(F - c).T).max() + 2.5) * np.array([math.cos(u), math.sin(u)])
                    for u in np.linspace(0, 6.28, 12)]
            if place(kind, cand, 6.0, why=f'garden of {h["name"]}'):
                break
    place('Rabbit', field[order[len(order) // 3:]], 9.0, n=2, spread=1.5, why='field')
    # ---- report / write
    for traj, sp, why in people:
        L = sum(float(np.hypot(*(b - a_))) for a_, b in zip(traj, traj[1:] + traj[:1]))
        print(f'pedestrian: {why}, loop {L:.0f} m at {sp:.1f} m/s')
    for kind, p, yaw, why in animals:
        print(f'{kind}: ({p[0]:.1f},{p[1]:.1f}) {why}, {lanes.query(p)[0]:.0f} m from the nearest lane')
    if a.check:
        return
    out = []
    for k, (traj, sp, why) in enumerate(people):
        tr = ', '.join(f'{q[0]:.2f} {q[1]:.2f}' for q in traj)
        h = math.atan2(*(traj[1] - traj[0])[::-1])
        out.append(f'Pedestrian {{\n  translation {traj[0][0]:.3f} {traj[0][1]:.3f} 1.27\n  rotation 0 0 1 {h:.3f}\n'
                   f'  name "{TAG} pedestrian({k})"\n  controllerArgs [\n    "--trajectory={tr}"\n    "--speed={sp:.2f}"\n  ]\n'
                   f'  shirtColor {SHIRTS[k % len(SHIRTS)]}\n  pantsColor {PANTS[k % len(PANTS)]}\n'
                   f'  skinColor {SKIN[k % len(SKIN)]}\n}}\n')
    for k, (kind, p, yaw, why) in enumerate(animals):
        out.append(f'{kind} {{\n  translation {p[0]:.3f} {p[1]:.3f} 0\n  rotation 0 0 1 {yaw:.3f}\n'
                   f'  name "{TAG} {kind.lower()}({k})"\n}}\n')
    used = {'Pedestrian'} | {k for k, *_ in animals}
    lines = text.split('\n')
    last = max(i for i, l in enumerate(lines) if l.startswith('EXTERNPROTO'))
    for kind in sorted(used):
        ext = f'EXTERNPROTO "{BASE}{PROTOS[kind]}"'
        if ext not in text:
            lines.insert(last + 1, ext)
    text = '\n'.join(lines)
    shutil.copy(a.world, a.world + '.before_life')
    open(a.world, 'w').write(text.rstrip('\n') + '\n' + ''.join(out))
    print(f'{len(people)} pedestrians, {len(animals)} animals -> {a.world} (backup .before_life)')


if __name__ == '__main__':
    main()
