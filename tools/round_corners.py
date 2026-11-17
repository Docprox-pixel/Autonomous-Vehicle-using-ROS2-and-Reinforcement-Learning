#!/usr/bin/env python3
"""Give the junctions of a world REAL corner radii (kerb returns), as junctions are built.

  python3 tools/round_corners.py worlds/village.wbt [--radius 8] [--check]

Webots' RoadIntersection (and the plain Crossroad joins) is a square crossing with sharp kerbed corners: a car (min turning
radius 6.2 m at the rear axle, inner rear wheel 1 m inside it) cannot turn right around such a
corner from its lane without its inner wheels climbing the kerb. Real urban corners are built with
a corner radius (kerb return) of ~6-10 m. For every RoadIntersection this tool
  * takes the edges of each pair of neighbouring connected roads (their drawn direction over the
    first 5 m, offset by half the road width),
  * fills the corner between them with an asphalt FILLET of --radius tangent to both edges,
  * removes the square sidewalk kerbs of the intersection (startRoadBorder FALSE).
The fillets are one Solid per intersection ("corner fillets(<id>)"), Asphalt appearance.
"""
import argparse
import math
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fix_lane_offsets import drawn_roads  # noqa: E402

ASPHALT = ('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/appearances/'
           'protos/Asphalt.proto"')


def meet(p, d, q, e):
    A = np.c_[d, -e]
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--radius', type=float, default=8.0)
    ap.add_argument('--shallow-run', type=float, default=16.0)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    text = open(a.world).read()
    roads = drawn_roads(a.world)
    solids, n_patch = [], 0
    for m in re.finditer(r'(?ms)^(?:RoadIntersection|Crossroad) \{\n(.*?)^\}', text):
        b = m.group(1)
        if 'connectedRoadIDs' not in b or not re.search(r'(?m)^  id "', b):
            continue
        jid = re.search(r'(?m)^  id "([^"]+)"', b).group(1)
        T = np.array([float(v) for v in re.search(r'(?m)^  translation (.+)$', b).group(1).split()[:2]])
        ids = re.findall(r'"([^"]+)"', re.search(r'connectedRoadIDs \[(.*?)\]', b, re.S).group(1))
        arms = []
        for rid in ids:
            if rid not in roads:
                continue
            C, W, _ = roads[rid]
            if np.hypot(*(C[0] - T)) > np.hypot(*(C[-1] - T)):
                C = C[::-1]
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
            q = np.array([np.interp(5.0, s, C[:, 0]), np.interp(5.0, s, C[:, 1])])
            d = (q - C[0]) / max(np.hypot(*(q - C[0])), 1e-9)
            arms.append((math.atan2(d[1], d[0]), rid, C[0], d, W))
        arms.sort()
        patches = []
        for k in range(len(arms)):
            angA, ridA, pA, dA, WA = arms[k]
            angB, ridB, pB, dB, WB = arms[(k + 1) % len(arms)]
            phi = (angB - angA) % (2 * math.pi)                       # CCW from road A to road B
            if phi < math.radians(30) or phi > math.radians(170):
                continue
            nA = np.array([-dA[1], dA[0]])                            # left of A (towards B)
            nB = np.array([-dB[1], dB[0]])
            K = meet(pA + WA / 2 * nA, dA, pB - WB / 2 * nB, dB)       # the corner of the two edges
            if K is None or np.hypot(*(K - T)) > 14.0:
                continue
            R = a.radius
            Td = R / math.tan(phi / 2)
            if phi > math.radians(140) and Td < a.shallow_run:
                # SHALLOW corner (roads meeting almost in line, a Y-junction): an 8 m return would be
                # a sliver; a car bending through it cuts over the sand. Long flat return instead,
                # tangent a.shallow_run m along both edges.
                Td = a.shallow_run
                R = Td * math.tan(phi / 2)
            t1, t2 = K + Td * dA, K + Td * dB
            bis = (dA + dB) / max(np.hypot(*(dA + dB)), 1e-9)
            Cc = K + R / math.sin(phi / 2) * bis
            a1, a2 = math.atan2(*(t1 - Cc)[::-1]), math.atan2(*(t2 - Cc)[::-1])
            da = math.atan2(math.sin(a2 - a1), math.cos(a2 - a1))
            arc = [Cc + R * np.array([math.cos(a1 + da * u), math.sin(a1 + da * u)]) for u in np.linspace(0, 1, 16)]
            patches.append((ridA, ridB, math.degrees(phi), K, arc))
        print(f'intersection {jid}: {len(patches)} corners rounded with R = {a.radius} m '
              + ', '.join(f'{p[0]}/{p[1]} ({p[2]:.0f} deg)' for p in patches))
        if not patches:
            continue
        pts, idx = [], []
        for (_, _, _, K, arc) in patches:
            base = len(pts)
            pts.append(K)
            pts.extend(arc)
            for i in range(len(arc) - 1):
                idx.append(f'{base} {base + 1 + i} {base + 2 + i} -1')
        n_patch += len(patches)
        coord = ', '.join(f'{p[0]:.3f} {p[1]:.3f} 0' for p in pts)
        solids.append(f'Solid {{\n  translation 0 0 0.004\n  name "corner fillets({jid})"\n  children [\n'
                      f'    Shape {{\n      appearance Asphalt {{\n      }}\n      geometry IndexedFaceSet {{\n'
                      f'        coord Coordinate {{\n          point [ {coord} ]\n        }}\n'
                      f'        coordIndex [ {" ".join(idx)} ]\n      }}\n    }}\n  ]\n}}\n')
    if a.check:
        return
    shutil.copy(a.world, a.world + '.before_corners')
    text = re.sub(r'(?ms)^Solid \{\n  translation 0 0 0\.004\n  name "corner fillets\([^)]*\)".*?^\}\n', '', text)
    # remove the square sidewalk kerbs of the intersections
    def no_border(mm):
        blk = mm.group(0)
        if re.search(r'(?m)^  startRoadBorder ', blk):
            return re.sub(r'(?m)^  startRoadBorder (TRUE|FALSE)', '  startRoadBorder FALSE', blk)
        return blk.replace('\n}', '\n  startRoadBorder FALSE\n}', 1) if blk.endswith('}') else blk
    text = re.sub(r'(?ms)^RoadIntersection \{\n.*?^\}', no_border, text)
    if 'Asphalt.proto' not in text:
        lines = text.split('\n')
        last = max(i for i, l in enumerate(lines) if l.startswith('EXTERNPROTO'))
        lines.insert(last + 1, ASPHALT)
        text = '\n'.join(lines)
    open(a.world, 'w').write(text.rstrip('\n') + '\n' + ''.join(solids))
    print(f'{n_patch} corner fillets added, intersection kerbs removed -> {a.world} (backup .before_corners)')


if __name__ == '__main__':
    main()
