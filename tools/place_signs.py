#!/usr/bin/env python3
"""Put traffic lights and road signs where real ones stand (right-hand traffic).

  python3 tools/place_signs.py worlds/village.wbt [--edge 1.0] [--check]

  * junction signal heads ("signal <junction>-<k>"): FAR-SIDE, at the far right corner of the
    junction - where the straight-ahead lane leaves it - --edge m outside the road edge / corner
    fillet, facing the approach (seen straight ahead from the stop line, within the front camera's
    view). At a T stem (no straight-ahead road) on the far kerb straight ahead.
  * speed-limit / caution signs and the remaining traffic lights: on the RIGHT of the lane they
    serve (the lane nearest that the sign faces), --edge m outside the road edge, square to it,
    at the same place along the road.
  * yield / stop signs: kept at their stop line, turned square to their approach.
Asphalt = drawn roads, corner fillets, turning-circle discs. Backup .before_signs.
"""
import argparse
import math
import os
import re
import shutil
import sys

import numpy as np
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from auto_car_core.lanemap import LaneMap  # noqa: E402
from fix_lane_offsets import drawn_roads  # noqa: E402
from clear_corners import seg_dist  # noqa: E402
from audit_signs import items  # noqa: E402

wrap = lambda a: math.atan2(math.sin(a), math.cos(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--edge', type=float, default=1.0)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    text = open(a.world).read()
    lm = LaneMap(os.path.splitext(a.world)[0] + '_net/sumo.net.xml')
    roads = drawn_roads(a.world)
    fill = []
    for m in re.finditer(r'(?ms)name "corner fillets\([^)]*\)".*?point \[ (.*?) \]', text):
        pts = np.array([[float(v) for v in p.split()[:2]] for p in m.group(1).split(',')])
        fill += [Path(pts[k:k + 17]) for k in range(0, len(pts), 17)]
    discs = [(np.array([float(x), float(y)]), float(r)) for x, y, r in re.findall(
        r'translation ([-\d.e]+) ([-\d.e]+) [-\d.e]+\n  name "turning circle\([^)]*\)".*?radius ([\d.]+)', text, re.S)]

    import yaml
    rb_file = os.path.splitext(a.world)[0] + '_net/roundabouts.yaml'
    rings = [(np.array([r['x'], r['y']]), r['inner'], r['outer'])
             for r in (yaml.safe_load(open(rb_file)).get('roundabouts', []) if os.path.exists(rb_file) else [])]

    # every lane of the network, junction links included (drawn roads stop at the junction mouths:
    # a pole 'off the road' could stand on the junction asphalt right next to a lane)
    import xml.etree.ElementTree as ET2
    from scipy.spatial import cKDTree
    net_root = ET2.parse(os.path.splitext(a.world)[0] + '_net/sumo.net.xml').getroot()
    net_off = np.array([float(v) for v in net_root.find('location').get('netOffset').split(',')])
    lane_pts = []
    for e_ in net_root.findall('edge'):
        if e_.get('id', '').startswith(('gneE', ':Custom24', ':Custom25', ':Custom26')):
            continue                                   # roundabout ring: driven as its own circle
        for ln_ in e_.findall('lane'):
            P_ = np.array([[float(v) for v in s.split(',')[:2]] for s in ln_.get('shape').split()]) - net_off
            sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P_, axis=0).T))]
            sq_ = np.arange(0.0, sl[-1] + 1e-9, 0.5)
            lane_pts.append(np.c_[np.interp(sq_, sl, P_[:, 0]), np.interp(sq_, sl, P_[:, 1])])
    lane_tree = cKDTree(np.vstack(lane_pts))

    def gap(q):
        """Distance from q to the nearest asphalt (negative = on it)."""
        g = min(seg_dist(C, q)[0] - W / 2 for C, W, _ in roads.values())
        g = min(g, float(lane_tree.query(q)[0]) - 1.75)          # lane centre + half a lane
        for o, ri, ro in rings:                        # roundabout ring (island in the middle is off-road)
            d = float(np.hypot(*(q - o)))
            g = min(g, d - ro if d >= (ri + ro) / 2 else ri - d)
        for o, r in discs:
            g = min(g, float(np.hypot(*(q - o))) - r)
        if any(f.contains_point(q) for f in fill):
            g = min(g, -0.1)
        elif fill:
            # distance to the fillet polygons' vertices (dense arcs) as an approximation
            g = min(g, min(float(np.min(np.hypot(*(f.vertices - q).T))) for f in fill))
        return g

    def off_road(q, n, max_out=25.0):
        """Walk from q along unit n until --edge m clear of all asphalt."""
        for s in np.arange(0.0, max_out, 0.25):
            p = q + s * n
            if gap(p) >= a.edge:
                return p
        return None

    road_lanes = [ln for ln in lm.lanes.values() if not ln.internal and len(ln.pts) >= 2]
    import xml.etree.ElementTree as ET
    ends = {e.get('id'): (e.get('from'), e.get('to')) for e in
            ET.parse(os.path.splitext(a.world)[0] + '_net/sumo.net.xml').getroot().findall('edge')
            if e.get('function') != 'internal'}

    def heading_at(ln, i):
        i = min(max(i, 0), len(ln.pts) - 2)
        d = ln.pts[i + 1] - ln.pts[i]
        return math.atan2(d[1], d[0])

    moves = []
    for m, kind, name, pos, yaw in items(text):
        drive = wrap(yaw - math.pi)                     # the direction of the traffic it is meant for
        new, nyaw, why = None, None, ''
        if kind == 'GenericTrafficLight' and name.startswith('signal '):
            jid = name.split()[1].rsplit('-', 1)[0]
            # its approach: road lane ending at this junction, heading closest to `drive`
            best = None
            for ln in road_lanes:
                if ends.get(ln.edge, (None, None))[1] != jid:
                    continue
                h = heading_at(ln, len(ln.pts) - 2)
                dh = abs(wrap(h - drive))
                d = float(np.hypot(*(ln.pts[-1] - pos)))
                if dh < math.radians(40) and d < 60:
                    sc = d + 30 * dh
                    if best is None or sc < best[0]:
                        best = (sc, ln, h)
            if best is None:
                continue
            _, A, h = best
            p = A.pts[-1]
            dvec = np.array([math.cos(h), math.sin(h)])
            right = np.array([math.sin(h), -math.cos(h)])
            # straight-ahead lane leaving the junction
            out = None
            for ln in road_lanes:
                if ln.edge == A.edge or ends.get(ln.edge, (None, None))[0] != jid or ln.edge.lstrip('-') == A.edge.lstrip('-'):
                    continue
                v = ln.pts[0] - p
                along = float(v @ dvec)
                if 2.0 < along < 45 and abs(float(v @ right)) < 8 and abs(wrap(heading_at(ln, 0) - h)) < math.radians(40):
                    if out is None or along < out[0]:
                        out = (along, ln)
            if out is not None:
                b = out[1].pts[0] + 1.0 * dvec
                new = off_road(b, right)
                why = f'far-side corner beyond {out[1].id}'
            else:
                # T stem: far kerb straight ahead (slightly right of the lane line)
                q = p.copy()
                for s in np.arange(1.0, 45.0, 0.5):
                    if gap(p + s * dvec + 1.0 * right) >= a.edge:
                        q = p + s * dvec + 1.0 * right
                        break
                new, why = q, 'far kerb across the T'
            nyaw = wrap(h + math.pi)
        elif kind == 'CautionSign' and 'roundabout.jpg' in m.group(0) and rings:
            # ADVANCE WARNING "roundabout ahead" (W2-6): on the right of the approach, ~40 m before
            # the ring - not at the entry corner, where cars leaving the ring sweep past it
            best = None
            for ln in road_lanes:
                o, ri, ro = min(rings, key=lambda r: float(np.hypot(*(ln.pts[-1] - r[0]))))
                if float(np.hypot(*(ln.pts[-1] - o))) > ro + 8:
                    continue
                h = heading_at(ln, len(ln.pts) - 2)
                dh = abs(wrap(h - drive))
                d = float(np.min(np.hypot(*(ln.pts - pos).T)))
                if dh < math.radians(60) and d < 20 and (best is None or d + 10 * dh < best[0]):
                    best = (d + 10 * dh, ln)
            if best is None:
                continue
            ln = best[1]
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ln.pts, axis=0).T))]
            sq = max(s[-1] - 40.0, min(5.0, s[-1] / 2))
            foot = np.array([np.interp(sq, s, ln.pts[:, 0]), np.interp(sq, s, ln.pts[:, 1])])
            i = int(np.searchsorted(s, sq))
            h = heading_at(ln, max(i - 1, 0))
            new = off_road(foot, np.array([math.sin(h), -math.cos(h)]))
            nyaw = wrap(h + math.pi)
            why = f'{s[-1] - sq:.0f} m before the ring on {ln.id}'
        elif kind in ('SpeedLimitSign', 'CautionSign', 'GenericTrafficLight'):
            best = None
            for ln in road_lanes:
                d = np.hypot(*(ln.pts - pos).T)
                i = int(np.argmin(d))
                h = heading_at(ln, i)
                dh = abs(wrap(h - drive))
                if d[i] < 15 and dh < math.radians(60):
                    sc = d[i] + 10 * dh
                    if best is None or sc < best[0]:
                        best = (sc, ln, i, h)
            if best is None:
                continue
            _, ln, i, h = best
            right = np.array([math.sin(h), -math.cos(h)])
            # foot point on the lane, then out to the right
            seg = ln.pts[min(i + 1, len(ln.pts) - 1)] - ln.pts[i]
            t = float(np.clip((pos - ln.pts[i]) @ seg / max(seg @ seg, 1e-9), 0, 1))
            foot = ln.pts[i] + t * seg
            new = off_road(foot, right)
            nyaw = wrap(h + math.pi)
            why = f'right of {ln.id}'
        elif kind in ('YieldSign', 'StopSign'):
            best = None
            for ln in road_lanes:
                h = heading_at(ln, len(ln.pts) - 2)
                d = float(np.hypot(*(ln.pts[-1] - pos)))
                if d < 15 and abs(wrap(h - drive)) < math.radians(45) and (best is None or d < best[0]):
                    best = (d, ln, h)
            if best is None:
                continue
            new, nyaw, why = pos, wrap(best[2] + math.pi), f'squared to {best[1].id}'
        if new is None:
            print(f'{kind} {name}: no spot found')
            continue
        dmove = float(np.hypot(*(new - pos)))
        dyaw = abs(math.degrees(wrap(nyaw - yaw)))
        if dmove < 0.3 and dyaw < 3:
            continue
        moves.append((m, new, nyaw))
        print(f'{kind:<20} {name:<26} {why:<34} moved {dmove:5.1f} m, turned {dyaw:3.0f} deg')
    if a.check:
        return
    for m, new, nyaw in sorted(moves, key=lambda x: -x[0].start()):
        blk = m.group(0)
        blk = re.sub(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)$',
                     lambda q: f'  translation {new[0]:.3f} {new[1]:.3f} {q.group(3)}', blk, count=1)
        if re.search(r'(?m)^  rotation ', blk):
            blk = re.sub(r'(?m)^  rotation .*$', f'  rotation 0 0 1 {nyaw:.4f}', blk, count=1)
        else:
            blk = blk.replace('\n', f'\n  rotation 0 0 1 {nyaw:.4f}\n', 1)
        text = text[:m.start()] + blk + text[m.end():]
    shutil.copy(a.world, a.world + '.before_signs')
    open(a.world, 'w').write(text)
    print(f'{len(moves)} lights / signs placed -> {a.world} (backup .before_signs)')


if __name__ == '__main__':
    main()
