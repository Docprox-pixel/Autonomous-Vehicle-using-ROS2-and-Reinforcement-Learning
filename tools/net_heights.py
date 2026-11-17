#!/usr/bin/env python3
"""Give the SUMO network the HEIGHTS of the roads Webots draws (bridges, ramps).

  python3 tools/net_heights.py worlds/village.wbt [--check]

The Webots export / our plain-XML rebuild has flat (x,y) lanes, so SUMO vehicles drove along the
ground under the bridge decks and through the ramps (SumoInterface enableHeight TRUE uses SUMO's z):
our car's lidar saw them sticking out of the ramp in front of it and the shield held it ("lane
blocked"). For every lane point (internal lanes too) the height of the drawn road under it is
taken - the road's wayPoints z interpolated along the same B-spline Webots uses - and written as
x,y,z. Points on no raised road keep z = 0 (and stay 2-D). Backup .before_heights.
"""
import argparse
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spline_lanes import bspline3  # noqa: E402


def raised_roads(world):
    text = open(world).read()
    out = []
    for m in re.finditer(r'(?ms)^(?:DEF \S+ )?Road \{\n(.*?)^\}', text):
        b = m.group(1)
        wp = re.search(r'wayPoints \[\n(.*?)\n  \]', b, re.S)
        if not wp:
            continue
        P = np.array([[float(v) for v in l.split()[:3]] for l in wp.group(1).strip().splitlines()])
        tr = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)', b)
        rot = re.search(r'(?m)^  rotation 0 0 (-?1) ([-\d.e]+)', b)
        sub = re.search(r'(?m)^  splineSubdivision (-?\d+)', b)
        sub = int(sub.group(1)) if sub else 4
        wd = re.search(r'(?m)^  width ([\d.]+)', b)
        rid = re.search(r'(?m)^  id "([^"]+)"', b)
        ang = float(rot.group(2)) * float(rot.group(1)) if rot else 0.0
        c, s = np.cos(ang), np.sin(ang)
        P = np.c_[c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1], P[:, 2]]
        if tr:
            P = P + [float(tr.group(1)), float(tr.group(2)), float(tr.group(3))]
        if P[:, 2].max() < 0.05:
            continue
        C = bspline3(list(P), sub) if sub > 0 and len(P) >= 3 else P
        # densify to 0.5 m
        s_ = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C[:, :2], axis=0).T))]
        ss = np.arange(0, s_[-1] + 1e-9, 0.5)
        C = np.c_[[np.interp(ss, s_, C[:, k]) for k in range(3)]].T
        out.append((C, float(wd.group(1)) if wd else 7.0, rid.group(1) if rid else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    net = os.path.join(os.path.splitext(a.world)[0] + '_net', 'sumo.net.xml')
    roads = raised_roads(a.world)
    tree = ET.parse(net)
    root = tree.getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])

    raised = {r[2] for r in roads if r[2]}
    # junctions a raised road's edge touches: their internal links are on the deck / ramp too
    deck_j = set()
    for e in root.findall('edge'):
        if e.get('function') != 'internal' and e.get('id').lstrip('-') in raised:
            deck_j |= {e.get('from'), e.get('to')}

    def z_at(x, y):
        best = None
        for C, W, _ in roads:
            d = np.hypot(C[:, 0] - x, C[:, 1] - y)
            i = int(np.argmin(d))
            if d[i] < W / 2 + 1.0 and (best is None or d[i] < best[0]):
                best = (d[i], C[i, 2])
        return 0.0 if best is None else float(best[1])

    n_lanes, zmax = 0, 0.0
    for e in root.findall('edge'):
        eid = e.get('id')
        if e.get('function') == 'internal':
            if not any(eid[1:].startswith(j + '_') for j in deck_j):
                continue
        elif eid.lstrip('-') not in raised:
            continue                                   # a road under a bridge stays on the ground
        for ln in e.findall('lane'):
            pts = np.array([[float(v) for v in q.split(',')[:2]] for q in ln.get('shape').split()])
            # resample every 2 m: SUMO interpolates z linearly between shape points, the drawn road
            # follows a spline - with the few original points cars sank into the ramps
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
            if s[-1] > 2.0:
                ss = np.unique(np.r_[np.arange(0.0, s[-1], 2.0), s[-1]])
                pts = np.c_[np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])]
            pts = pts.tolist()
            z = [z_at(x - off[0], y - off[1]) for x, y in pts]
            if max(z) < 0.05:
                continue
            n_lanes += 1
            zmax = max(zmax, max(z))
            ln.set('shape', ' '.join(f'{x:.2f},{y:.2f},{h:.2f}' for (x, y), h in zip(pts, z)))
    print(f'{len(roads)} raised roads; {n_lanes} lanes get heights (max {zmax:.2f} m)')
    if a.check:
        return
    shutil.copy(net, net + '.before_heights')
    tree.write(net, xml_declaration=True, encoding='UTF-8')


if __name__ == '__main__':
    main()
