#!/usr/bin/env python3
"""Which poles / signs / props stand too close to the paths the car really drives?

  python3 tools/path_clearance.py worlds/village.wbt [--need 1.9] [--fix]

For every turn of the network (the planner's path: last 30 m of the arriving lane + junction link +
first 30 m of the leaving lane, through LaneMap.smooth_kinks) and every plain lane, the distance
from each pole to the path centre is compared with what the car needs: half its width (1.0 m) +
off-tracking of the inner side in a curve (wheelbase^2 / 2R) + the safety shield's side margin
(0.75 m) -> --need at least. --fix moves each offender straight away from the path until it has
--need + 0.3 m (and stays off the asphalt). Backup .before_clearance.
"""
import argparse
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from auto_car_core.lanemap import LaneMap  # noqa: E402
from audit_turns import clip_len  # noqa: E402
from clear_corners import PROPS  # noqa: E402

WB = 2.995


def paths(net):
    m = LaneMap(net)
    root = ET.parse(net).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    lane = {ln.get('id'): np.array([[float(v) for v in q.split(',')[:2]] for q in ln.get('shape').split()]) - off
            for e in root.findall('edge') for ln in e.findall('lane')}
    rev = lambda e: e[1:] if e.startswith('-') else '-' + e
    # roundabout ring lanes are not driven as drawn (LaneMap.fix_roundabouts drives its own circle)
    ring = {e.get('id') for e in root.findall('edge') if e.get('id', '').startswith('gneE')}
    out = [L for k, L in lane.items() if not k.startswith(':') and k.rsplit('_', 1)[0] not in ring]
    for c in root.findall('connection'):
        f, t, via = c.get('from'), c.get('to'), c.get('via')
        if f.startswith(':') or t == rev(f) or not via or f in ring or t in ring:
            continue
        A, B, V = lane.get(f'{f}_{c.get("fromLane")}'), lane.get(f'{t}_{c.get("toLane")}'), lane.get(via)
        if A is None or B is None or V is None:
            continue
        raw = np.vstack([clip_len(A, 30.0, True), V, clip_len(B, 30.0, False)])
        raw = raw[np.r_[True, np.hypot(*np.diff(raw, axis=0).T) > 1e-6]]
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(raw, axis=0).T))]
        ss = np.arange(0.0, s[-1], 1.0)
        raw = np.c_[np.interp(ss, s, raw[:, 0]), np.interp(ss, s, raw[:, 1])]
        if len(raw) >= 8:
            out.append(m.smooth_kinks(raw))
    return out


def need_at(P, i, base):
    j0, j1 = max(i - 2, 0), min(i + 2, len(P) - 1)
    a = np.arctan2(*(P[min(j0 + 1, len(P) - 1)] - P[j0])[::-1])
    b = np.arctan2(*(P[j1] - P[max(j1 - 1, 0)])[::-1])
    L = max(float(np.sum(np.hypot(*np.diff(P[j0:j1 + 1], axis=0).T))), 1e-3)
    k = abs(math.atan2(math.sin(b - a), math.cos(b - a))) / L
    return base + min(0.8, WB * WB * k / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--need', type=float, default=1.75)
    ap.add_argument('--fix', action='store_true')
    a = ap.parse_args()
    net = os.path.join(os.path.splitext(a.world)[0] + '_net', 'sumo.net.xml')
    PS = paths(net)
    text = open(a.world).read()
    kinds = [k for k in PROPS if k != 'SimpleTree'] + ['SimpleTree', 'StreetLight', 'BusStop', 'Fence']
    moves = []
    for m in re.finditer(r'(?ms)^(?:DEF \S+ )?(' + '|'.join(sorted(set(kinds))) + r') \{\n(.*?)^\}', text):
        tr = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)', m.group(2))
        if not tr:
            continue
        q = np.array([float(tr.group(1)), float(tr.group(2))])
        worst = None
        for P in PS:
            d = np.hypot(*(P - q).T)
            i = int(np.argmin(d))
            if d[i] > 4.0:
                continue
            short = need_at(P, i, a.need) - float(d[i])
            if short > 0 and (worst is None or short > worst[0]):
                worst = (short, P, i, float(d[i]))
        if worst is None:
            continue
        short, P, i, d = worst
        n = m.group(2)
        nm = re.search(r'(?m)^  name "([^"]+)"', n)
        away = (q - P[i]) / max(d, 1e-6)
        new = q + (short + 0.3) * away
        print(f'{m.group(1):<20} {nm.group(1) if nm else "":<24} ({q[0]:.1f},{q[1]:.1f}): {d:.2f} m from a path, '
              f'needs {d + short:.2f} -> move {short + 0.3:.1f} m')
        moves.append((m.start(2) + tr.start(), m.start(2) + tr.end(), f'  translation {new[0]:.3f} {new[1]:.3f} {tr.group(3)}'))
    if a.fix and moves:
        for s, e, new in sorted(moves, key=lambda x: -x[0]):
            text = text[:s] + new + text[e:]
        shutil.copy(a.world, a.world + '.before_clearance')
        open(a.world, 'w').write(text)
        print(f'{len(moves)} moved -> {a.world} (backup .before_clearance)')


if __name__ == '__main__':
    main()
