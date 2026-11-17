#!/usr/bin/env python3
"""Check every SUMO lane against the road Webots actually DRAWS and move misplaced lanes back.

  python3 tools/fix_lane_offsets.py worlds/village.wbt [--check]

For each Road with an `id` (2-lane two-way roads), the drawn centre line is computed exactly like
Webots (B-spline, tools/spline_lanes.py). A lane of edge `id` / `-id` must lie half a lane width to
the RIGHT of the centre line (-W/4). Road 25 in the village had both lanes ~1.6 m too far right:
one direction ran on the kerb/sidewalk and the other on the centre line, and the car, following the
map exactly, drove on the kerb. Misplaced lanes (> 0.5 m off) are shifted by the measured error
along the local normal; the shift fades out over the last 5 m at each end so junction connections
still meet. The original file is kept as sumo.net.xml.before_offsets.
"""
import argparse
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from spline_lanes import bspline3  # noqa: E402


def drawn_roads(world):
    text = open(world).read()
    roads = {}
    for m in re.finditer(r'(?ms)^Road \{\n(.*?)^\}', text):
        b = m.group(1)
        rid = re.search(r'(?m)^  id "([^"]+)"', b)
        wp = re.search(r'wayPoints \[\n(.*?)\n  \]', b, re.S)
        if not rid or not wp:
            continue
        tr = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+)', b)
        rot = re.search(r'(?m)^  rotation 0 0 (-?1) ([-\d.e]+)', b)
        sub = re.search(r'(?m)^  splineSubdivision (-?\d+)', b)
        sub = int(sub.group(1)) if sub else 4
        wd = re.search(r'(?m)^  width ([\d.]+)', b)
        nl = re.search(r'(?m)^  numberOfLanes (\d+)', b)
        P = np.array([[float(v) for v in l.split()[:2]] for l in wp.group(1).strip().splitlines()])
        ang = float(rot.group(2)) * float(rot.group(1)) if rot else 0.0
        c, s = math.cos(ang), math.sin(ang)
        P = np.c_[c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1]] + (
            [float(tr.group(1)), float(tr.group(2))] if tr else [0, 0])
        C = bspline3(list(P), sub) if sub > 0 and len(P) >= 3 else P
        roads[rid.group(1)] = (C, float(wd.group(1)) if wd else 7.0, int(nl.group(1)) if nl else 2)
    return roads


def offsets(pts, C):
    """Signed offset (+ left of the lane's own direction) of every lane point from centre line C."""
    fwd = np.dot(pts[-1] - pts[0], C[-1] - C[0]) > 0
    out = []
    for q in pts:
        d = np.hypot(*(C - q).T)
        j = int(np.argmin(d))
        t = C[min(j + 1, len(C) - 1)] - C[max(j - 1, 0)]
        t = t / max(np.hypot(*t), 1e-9)
        out.append(float((q - C[j]) @ np.array([-t[1], t[0]])) * (1 if fwd else -1))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--check', action='store_true', help='report only')
    ap.add_argument('--from-backup', action='store_true', help='start again from sumo.net.xml.before_offsets')
    a = ap.parse_args()
    net_file = a.world.replace('.wbt', '_net/sumo.net.xml')
    roads = drawn_roads(a.world)
    src = net_file + '.before_offsets' if a.from_backup and os.path.exists(net_file + '.before_offsets') else net_file
    tree = ET.parse(src)
    root = tree.getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    fixed, checked, moves = [], 0, []
    for e in root.findall('edge'):
        if e.get('function') == 'internal':
            continue
        rid = e.get('id').lstrip('-')
        if rid not in roads:
            continue
        C, W, NL = roads[rid]
        if NL != 2:
            continue                                       # only 2-lane two-way roads have a clear rule
        for ln in e.findall('lane'):
            pts = np.array([[float(v) for v in p.split(',')[:2]] for p in ln.get('shape').split()]) - off
            if len(pts) < 2:
                continue
            checked += 1
            o = offsets(pts, C)
            mid = o[1:-1] if len(o) > 2 else o
            err = float(np.median(mid)) - (-W / 4.0)
            if abs(err) <= 0.5:
                continue
            fixed.append((ln.get('id'), round(float(np.median(mid)), 2), round(-W / 4.0, 2)))
            if a.check:
                continue
            t = np.gradient(pts, axis=0)
            t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
            nl_ = np.c_[-t[:, 1], t[:, 0]]
            new = pts - err * nl_                      # the WHOLE lane moves (no fade: that left an
            # S-shaped jog across the road at the junctions); the junction lanes follow below
            moves.append((pts[0], new[0] - pts[0]))
            moves.append((pts[-1], new[-1] - pts[-1]))
            ln.set('shape', ' '.join(f'{x + off[0]:.2f},{y + off[1]:.2f}' for x, y in new))
        if not a.check and e.get('shape') is not None:
            del e.attrib['shape']
    # junction (internal) lanes that started/ended at a moved lane end: move that end with it,
    # blending linearly to their other end, so the connections stay continuous and smooth
    n_int = 0
    if moves and not a.check:
        for e in root.findall('edge'):
            if e.get('function') != 'internal':
                continue
            for ln in e.findall('lane'):
                pts = np.array([[float(v) for v in p.split(',')[:2]] for p in ln.get('shape').split()]) - off
                if len(pts) < 2:
                    continue
                s_ = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
                u = s_ / max(s_[-1], 1e-9)
                delta = np.zeros_like(pts)
                for q, mv in moves:
                    if np.hypot(*(pts[0] - q)) < 0.6:
                        delta += (1 - u)[:, None] * mv
                    if np.hypot(*(pts[-1] - q)) < 0.6:
                        delta += u[:, None] * mv
                if np.abs(delta).max() > 1e-3:
                    pts = pts + delta
                    ln.set('shape', ' '.join(f'{x + off[0]:.2f},{y + off[1]:.2f}' for x, y in pts))
                    n_int += 1
    print(f'{checked} lanes checked; {len(fixed)} misplaced (> 0.5 m): {fixed}; {n_int} junction lanes adjusted')
    if fixed and not a.check:
        if not os.path.exists(net_file + '.before_offsets'):
            shutil.copy(net_file, net_file + '.before_offsets')
        tree.write(net_file, xml_declaration=True, encoding='UTF-8')
        print(f'fixed -> {net_file} (backup .before_offsets)')


if __name__ == '__main__':
    main()
