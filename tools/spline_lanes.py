#!/usr/bin/env python3
"""Put SUMO lanes on the curves Webots actually DRAWS (Road B-spline), so routes stay on the road.

  python3 tools/spline_lanes.py worlds/village.wbt

Webots draws a Road through its way points with a uniform cubic B-spline (wbgeometry.bSpline3,
splineSubdivision 4 by default); the SUMO network (shipped, or from Webots' exporter) joins the
way points with STRAIGHT segments. On tight bends these differ by up to ~3 m: the route then ran
over the kerb and the car, following it, drove off the road into a garden.

For every Road with an `id` matching a SUMO edge (`id` and `-id`): compute the drawn centre line
(same formula as Webots), measure each lane's signed offset from the straight-segment centre line,
and rewrite the lane (and edge) shape as the drawn centre line shifted by that offset. Lane end
points stay where they were, so junction connections remain valid.
"""
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np


def bspline3(points, sub):
    """Port of Webots wbgeometry.bSpline3 (2-D)."""
    P = [np.asarray(p, float) for p in points]
    n = len(P)
    P.append(P[n - 1] + (P[n - 1] - P[n - 2]))
    P.append(P[n] + (P[n] - P[n - 1]))
    P.insert(0, P[0] + (P[0] - P[1]))
    P.insert(0, P[0] + (P[0] - P[1]))
    out = [P[2]]
    for i in range(2, n + 1):
        a0 = (-P[i - 1] + 3 * P[i] - 3 * P[i + 1] + P[i + 2]) / 6.0
        a1 = (3 * P[i - 1] - 6 * P[i] + 3 * P[i + 1]) / 6.0
        a2 = (-3 * P[i - 1] + 3 * P[i + 1]) / 6.0
        a3 = (P[i - 1] + 4 * P[i] + P[i + 1]) / 6.0
        for j in range(1, sub + 1):
            t = j / sub
            out.append((a2 + t * (a1 + t * a0)) * t + a3)
    return np.array(out)


def signed_offset(pts, line):
    """Mean signed lateral offset (+ = left) of pts from polyline `line`."""
    vals = []
    for p in pts:
        a, b = line[:-1], line[1:]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
        q = a + t[:, None] * ab
        d = np.hypot(*(p - q).T)
        k = int(np.argmin(d))
        n = np.array([-ab[k, 1], ab[k, 0]]) / max(np.hypot(*ab[k]), 1e-9)
        vals.append(float((p - q[k]) @ n))
    return float(np.median(vals))


def offset_line(line, d):
    t = np.gradient(line, axis=0)
    t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
    return line + d * np.c_[-t[:, 1], t[:, 0]]


def main():
    world = sys.argv[1]
    net_file = world.replace('.wbt', '_net/sumo.net.xml')
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
        sub_m = re.search(r'(?m)^  splineSubdivision (-?\d+)', b)
        sub = int(sub_m.group(1)) if sub_m else 4
        P = np.array([[float(v) for v in l.split()[:2]] for l in wp.group(1).strip().splitlines()])
        ang = float(rot.group(2)) * float(rot.group(1)) if rot else 0.0
        c, s = math.cos(ang), math.sin(ang)
        P = np.c_[c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1]] + ([float(tr.group(1)), float(tr.group(2))] if tr else [0, 0])
        if sub <= 0 or len(P) < 3:
            continue
        roads[rid.group(1)] = (P, bspline3(list(P), sub))
    tree = ET.parse(net_file)
    root = tree.getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    moved, worst = 0, 0.0
    for e in root.findall('edge'):
        if e.get('function') == 'internal':
            continue
        rid = e.get('id').lstrip('-')
        if rid not in roads:
            continue
        raw, spl = roads[rid]
        for ln in e.findall('lane'):
            pts = np.array([[float(v) for v in p.split(',')[:2]] for p in ln.get('shape').split()]) - off
            fwd = np.dot(pts[-1] - pts[0], raw[-1] - raw[0]) > 0
            base_raw, base_spl = (raw, spl) if fwd else (raw[::-1], spl[::-1])
            d = signed_offset(pts, base_raw)
            new = offset_line(base_spl, d)
            # keep the original end points (junction connections), blend over 3 m at each end
            s_new = np.r_[0.0, np.cumsum(np.hypot(*np.diff(new, axis=0).T))]
            proj0 = int(np.argmin(np.hypot(*(new - pts[0]).T)))
            proj1 = int(np.argmin(np.hypot(*(new - pts[-1]).T)))
            new = new[proj0:proj1 + 1] if proj1 > proj0 else new
            new = np.vstack([pts[0], new[1:-1], pts[-1]]) if len(new) > 2 else pts
            dev = max(min(np.hypot(*(q - new).T)) for q in pts[1:-1]) if len(pts) > 2 else 0.0
            worst = max(worst, dev)
            ln.set('shape', ' '.join(f'{x + off[0]:.2f},{y + off[1]:.2f}' for x, y in new))
            moved += 1
        if e.get('shape') is not None:
            del e.attrib['shape']
    shutil.copy(net_file, net_file + '.before_spline')
    tree.write(net_file, xml_declaration=True, encoding='UTF-8')
    print(f'{moved} lanes moved onto the drawn road curves ({len(roads)} roads); largest correction {worst:.2f} m')


if __name__ == '__main__':
    main()
