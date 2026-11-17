#!/usr/bin/env python3
"""Move every building that stands on (or too close to) a road / turning circle the least needed.

  python3 tools/clear_buildings.py worlds/village.wbt [--gap 2.5] [--check]

Uses the real footprints in <world>_net/pois.yaml (tools/extract_pois.py, PROTO collision outlines).
A building is in conflict when its footprint comes within --gap m of a drawn road's edge or of a
turning-circle disc. For each one, the smallest shift (0.5 m grid, up to 30 m) is chosen that clears
every road and disc, keeps 1.5 m from every other building and stays off the hill. The world's
translation is changed; run tools/extract_pois.py again afterwards.
"""
import argparse
import math
import os
import re
import shutil
import sys

import numpy as np
import yaml
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_lane_offsets import drawn_roads  # noqa: E402
from terrain_height import height_grid  # noqa: E402

SKIP = ('bus_stop', 'toilet')


def poly_pts(F, step=1.0):
    F = np.asarray(F)
    edge = []
    for a, b in zip(F, np.roll(F, -1, axis=0)):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        edge += [a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False)]
    lo, hi = F.min(0), F.max(0)
    g = np.mgrid[lo[0]:hi[0]:step, lo[1]:hi[1]:step].reshape(2, -1).T
    return np.vstack([np.array(edge), g[Path(F).contains_points(g)]]) if len(g) else np.array(edge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--gap', type=float, default=2.5)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    text = open(a.world).read()
    net_dir = os.path.splitext(a.world)[0] + '_net'
    places = yaml.safe_load(open(os.path.join(net_dir, 'pois.yaml')))['places']
    roads = drawn_roads(a.world)
    R = np.vstack([np.c_[C, np.full(len(C), W / 2)] for C, W, _ in roads.values()])   # centre samples
    # densify the road centre lines to 1 m
    dense = []
    for C, W, _ in roads.values():
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
        ss = np.arange(0, s[-1] + 1e-9, 1.0)
        dense.append(np.c_[np.interp(ss, s, C[:, 0]), np.interp(ss, s, C[:, 1]), np.full(len(ss), W / 2)])
    R = np.vstack(dense)
    discs = np.array([[float(x), float(y), float(r)] for x, y, r in re.findall(
        r'translation ([-\d.e]+) ([-\d.e]+) [-\d.e]+\n  name "turning circle\([^)]*\)".*?radius ([\d.]+)', text, re.S)])
    trees = np.array([[float(x), float(y)] for x, y in re.findall(
        r'(?m)^SimpleTree \{\n  translation ([-\d.e]+) ([-\d.e]+)', text)])

    def clearance(Q):
        d = np.hypot(Q[:, None, 0] - R[None, :, 0], Q[:, None, 1] - R[None, :, 1]) - R[None, :, 2]
        c = float(d.min())
        if len(discs):
            c = min(c, float((np.hypot(Q[:, None, 0] - discs[None, :, 0], Q[:, None, 1] - discs[None, :, 1])
                              - discs[None, :, 2]).min()))
        return c

    foots = {i: np.array(p['footprint']) for i, p in enumerate(places)}
    edits = []
    for i, p in enumerate(places):
        if p['kind'] in SKIP:
            continue
        Q = poly_pts(foots[i])
        c0 = clearance(Q)
        if c0 >= a.gap:
            continue
        others = [poly_pts(foots[j], 1.5) for j in foots if j != i]
        best = None
        for r in np.arange(0.5, 30.01, 0.5):
            for ang in np.arange(0, 360, 10):
                v = r * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
                Qv = Q + v
                if clearance(Qv) < a.gap:
                    continue
                if any(Path(foots[j] + 0).contains_points(Qv).any() or
                       Path(foots[i] + v).contains_points(o).any() for j, o in zip([j for j in foots if j != i], others)):
                    continue
                if len(trees) and (Path(foots[i] + v).contains_points(trees).any()):
                    continue
                c = p['x'] + v[0], p['y'] + v[1]
                g = height_grid(a.world, c[0], c[1], float(np.max(np.hypot(*(foots[i] - [p['x'], p['y']]).T))) + 1)
                if g and max(g.values()) > 0.3:
                    continue
                best = v
                break
            if best is not None:
                break
        if best is None:
            print(f"{p['name']} ({p['webots_name']}): {c0:.1f} m from the road, NO free spot within 30 m")
            continue
        foots[i] = foots[i] + best
        print(f"{p['name']} ({p['webots_name']}) at ({p['x']},{p['y']}): {c0:.1f} m from a road -> moved "
              f"{np.hypot(*best):.1f} m to ({p['x'] + best[0]:.2f},{p['y'] + best[1]:.2f})")
        edits.append((p, best))
    if a.check or not edits:
        return
    for p, v in edits:
        pat = re.compile(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)$')
        done = False
        for m in pat.finditer(text):
            x, y = float(m.group(1)), float(m.group(2))
            if abs(x - p['x']) < 0.006 and abs(y - p['y']) < 0.006:
                text = text[:m.start()] + f'  translation {x + v[0]:.3f} {y + v[1]:.3f} {m.group(3)}' + text[m.end():]
                done = True
                break
        if not done:
            print(f"  could not find the translation of {p['webots_name']}")
    shutil.copy(a.world, a.world + '.before_buildings')
    open(a.world, 'w').write(text)
    print(f'{len(edits)} buildings moved -> {a.world} (backup .before_buildings); now run tools/extract_pois.py')


if __name__ == '__main__':
    main()
