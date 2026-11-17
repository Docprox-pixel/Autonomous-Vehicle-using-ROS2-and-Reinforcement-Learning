#!/usr/bin/env python3
"""Clear the rounded corners and the new road joins of kerbs and poles.

  python3 tools/clear_corners.py worlds/village.wbt [--check]

After round_corners.py / connect_road_ends.py the asphalt fillets and new roads overlap things
that were placed for the old square corners:
  * the KERBS (road borders) of the roads meeting a rounded corner or a new road run straight up
    to the old square edge and cut across the fillet - a turning car climbs them. The borders of
    those roads are switched off (rightBorder / leftBorder FALSE).
  * POLES and props (yield / speed signs, street lights, traffic lights, bins, boxes, parked bikes)
    stand on the fillet or the road edge. Each one on the asphalt or within --margin of it is
    moved straight outwards to --clear m beyond the asphalt edge (fillet: towards the kerb-return
    centre; road: perpendicular to its centre line).
"""
import argparse
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fix_lane_offsets import drawn_roads  # noqa: E402

PROPS = ('YieldSign', 'StopSign', 'SpeedLimitSign', 'CautionSign', 'StreetLight', 'GenericTrafficLight',
         'TrashBin', 'PublicBin', 'SolidBox', 'MotorbikeSimple', 'ScooterSimple', 'SignPole', 'HighwayPole',
         'Bench', 'SpotLight', 'SimpleTree')


def circle(a, b, c):
    """Centre and radius of the circle through three points."""
    A = 2 * np.array([b - a, c - a])
    r = np.array([b @ b - a @ a, c @ c - a @ a])
    o = np.linalg.solve(A, r)
    return o, float(np.hypot(*(a - o)))


def seg_dist(P, q):
    """Distance of q from polyline P and the unit normal pointing from the line to q."""
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    f = a + t[:, None] * ab
    d = np.hypot(*(q - f).T)
    i = int(np.argmin(d))
    n = (q - f[i]) / max(d[i], 1e-9)
    end = (i == 0 and t[0] == 0) or (i == len(a) - 1 and t[-1] == 1)
    return float(d[i]), n, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--margin', type=float, default=0.8)
    ap.add_argument('--clear', type=float, default=1.2)
    ap.add_argument('--kinds', default=None, help='comma list: only these PROTOs (e.g. a wider --clear for props)')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    text = open(a.world).read()
    roads = drawn_roads(a.world)
    # fillets: K + 16 arc points per corner
    fillets = []
    for m in re.finditer(r'(?ms)name "corner fillets\(([^)]*)\)".*?point \[ (.*?) \]', text):
        pts = np.array([[float(v) for v in p.split()[:2]] for p in m.group(2).split(',')])
        for k in range(0, len(pts), 17):
            K, arc = pts[k], pts[k + 1:k + 17]
            o, R = circle(arc[0], arc[8], arc[15])
            fillets.append((m.group(1), K, arc, o, R))
    circles = [(np.array([float(x), float(y)]), float(r)) for x, y, r in re.findall(
        r'(?ms)translation ([-\d.e]+) ([-\d.e]+) [-\d.e]+\n  name "turning circle\([^)]*\)".*?radius ([\d.]+)', text)]

    import os
    import yaml
    rb_file = os.path.join(os.path.splitext(a.world)[0] + '_net', 'roundabouts.yaml')
    rings = [(np.array([r['x'], r['y']]), r['inner'], r['outer'])
             for r in (yaml.safe_load(open(rb_file)).get('roundabouts', []) if os.path.exists(rb_file) else [])]

    def on_fillet(q):
        for _, K, arc, o, R in fillets:
            d = float(np.hypot(*(q - o)))
            # inside the corner wedge (between the two tangent rays seen from the centre)
            u1, u2, uq = arc[0] - o, arc[-1] - o, q - o
            cross = lambda x, y: x[0] * y[1] - x[1] * y[0]
            inside = cross(u1, uq) * cross(u1, u2) >= 0 and cross(uq, u2) * cross(u1, u2) >= 0
            if inside and R - a.margin < d < np.hypot(*(K - o)) + a.margin:
                return o + (R - a.clear) * uq / max(d, 1e-9)
        return None

    def on_road(q):
        for rid, (C, W, _) in roads.items():
            d, n, end = seg_dist(C, q)
            if d < W / 2 + a.margin and (not end or d < W / 2 + 0.3):
                return q + (W / 2 + a.clear - d) * n, rid
        for o, r in circles:
            d = float(np.hypot(*(q - o)))
            if d < r + a.margin:
                return o + (r + a.clear) * (q - o) / max(d, 1e-9), 'turning circle'
        for o, ri, ro in rings:
            d = float(np.hypot(*(q - o)))
            if (ri + ro) / 2 <= d < ro + a.margin:
                return o + (ro + a.clear) * (q - o) / max(d, 1e-9), 'roundabout ring'
            if ri - a.margin < d < (ri + ro) / 2:
                return o + max(ri - a.clear, 0.5) * (q - o) / max(d, 1e-9), 'roundabout ring'
        return None, None

    # ---- 1) props
    moves = []
    kinds = a.kinds.split(',') if a.kinds else PROPS
    for m in re.finditer(r'(?ms)^(?:DEF \S+ )?(' + '|'.join(kinds) + r') \{\n(.*?)^\}', text):
        tr = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)$', m.group(2))
        if not tr:
            continue
        q0 = np.array([float(tr.group(1)), float(tr.group(2))])
        q, why = q0.copy(), None
        for _ in range(6):
            f = on_fillet(q)
            if f is not None:
                q, why = f, why or 'corner fillet'
                continue
            r, rid = on_road(q)
            if r is not None:
                q, why = r, why or f'road {rid}'
                continue
            break
        if why:
            nm = re.search(r'(?m)^  name "([^"]+)"', m.group(2))
            moves.append((m.start(2) + tr.start(), m.start(2) + tr.end(),
                          f'  translation {q[0]:.3f} {q[1]:.3f} {tr.group(3)}'))
            print(f'{m.group(1)} {nm.group(1) if nm else ""} at ({q0[0]:.1f},{q0[1]:.1f}) on the {why}: '
                  f'moved {np.hypot(*(q - q0)):.1f} m to ({q[0]:.1f},{q[1]:.1f})')
    # ---- 2) kerbs of roads that end at a rounded corner or a new road
    joins = [K for _, K, *_ in fillets]
    for rid in ('conn0', 'conn1'):
        if rid in roads:
            joins += [roads[rid][0][0], roads[rid][0][-1]]
    kerbs = []
    for m in re.finditer(r'(?ms)^Road \{\n(.*?)^\}', text):
        b = m.group(1)
        rid = re.search(r'(?m)^  id "([^"]+)"', b)
        if not rid or rid.group(1) not in roads:
            continue
        hb = re.search(r'(?m)^  roadBorderHeight (-?[\d.]+)', b)
        if hb and float(hb.group(1)) <= 0:
            continue                                                 # sunk below ground: invisible
        if 'rightBorder FALSE' in b and 'leftBorder FALSE' in b:
            continue
        C = roads[rid.group(1)][0]
        if min(min(np.hypot(*(C[0] - j)), np.hypot(*(C[-1] - j))) for j in joins) < 18.0:
            kerbs.append((m.start(1), m.end(1), rid.group(1)))
    print(f'kerbs removed on roads: {", ".join(k[2] for k in kerbs) or "none"}')
    if a.check:
        return
    edits = moves[:]
    for s, e, _ in kerbs:
        b = text[s:e]
        for side in ('rightBorder', 'leftBorder'):
            if re.search(rf'(?m)^  {side} ', b):
                b = re.sub(rf'(?m)^  {side} \w+', f'  {side} FALSE', b)
            else:
                b += f'  {side} FALSE\n'
        edits.append((s, e, b))
    for s, e, new in sorted(edits, key=lambda x: -x[0]):
        text = text[:s] + new + text[e:]
    shutil.copy(a.world, a.world + '.before_clear')
    open(a.world, 'w').write(text)
    print(f'{len(moves)} props moved, kerbs removed on {len(kerbs)} roads -> {a.world} (backup .before_clear)')


if __name__ == '__main__':
    main()
