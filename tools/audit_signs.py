#!/usr/bin/env python3
"""Where does every traffic light and road sign stand relative to the approach it serves?

  python3 tools/audit_signs.py worlds/village.wbt

For each GenericTrafficLight / YieldSign / StopSign / SpeedLimitSign / CautionSign: the approach
lane it faces (a road lane ending at a junction, heading opposite to the sign's facing), the
distance along that lane from its stop line (+ = past the line, into/over the junction), which
side of the road (R/L) and the gap to the drawn road edge, and how far its facing is off from
"square to the approaching driver".
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_car_core.lanemap import LaneMap  # noqa: E402
from fix_lane_offsets import drawn_roads  # noqa: E402
from clear_corners import seg_dist  # noqa: E402

KINDS = ('GenericTrafficLight', 'YieldSign', 'StopSign', 'SpeedLimitSign', 'CautionSign')
# PROTO facing: the yaw (rotation about z) for which the sign/lamps face a car driving +x is
# FACE[k] (GenericTrafficLight lamps face +x, so a car driving +x sees them at rotation pi)
FACE = {'GenericTrafficLight': math.pi, 'YieldSign': math.pi, 'StopSign': math.pi,
        'SpeedLimitSign': math.pi, 'CautionSign': math.pi}


def items(text):
    for m in re.finditer(r'(?ms)^(' + '|'.join(KINDS) + r') \{\n(.*?)^\}', text):
        b = m.group(2)
        t = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+)', b)
        r = re.search(r'(?m)^  rotation 0 0 (-?1) ([-\d.e]+)', b)
        n = re.search(r'(?m)^  name "([^"]+)"', b)
        yield m, m.group(1), (n.group(1) if n else m.group(1)), np.array([float(t.group(1)), float(t.group(2))]), \
            (float(r.group(2)) * float(r.group(1)) if r else 0.0)


def approaches(lm):
    """Road lanes that end at a junction: (lane, stop-line point, heading)."""
    out = []
    for ln in lm.lanes.values():
        if ln.internal or len(ln.pts) < 2:
            continue
        p, q = ln.pts[-1], ln.pts[-2]
        out.append((ln, p, math.atan2(p[1] - q[1], p[0] - q[0])))
    return out


def relation(pos, yaw, kind, apps, roads):
    drive = yaw - FACE[kind]                      # the driving direction this sign is meant for
    best = None
    for ln, p, h in apps:
        dh = abs(math.atan2(math.sin(h - drive), math.cos(h - drive)))
        if dh > math.radians(45):
            continue
        v = pos - p
        along = v[0] * math.cos(h) + v[1] * math.sin(h)
        side = -v[0] * math.sin(h) + v[1] * math.cos(h)      # + left of the driver
        if not -60 < along < 40 or abs(side) > 15:
            continue
        score = abs(along) * 0.5 + abs(side) + math.degrees(dh) * 0.05
        if best is None or score < best[0]:
            best = (score, ln, p, h, along, side, math.degrees(dh))
    if best is None:
        return None
    _, ln, p, h, along, side, dh = best
    gap = min(seg_dist(C, pos)[0] - W / 2 for C, W, _ in roads.values())
    return ln.id, along, side, gap, dh


def main():
    world = sys.argv[1]
    net = os.path.splitext(world)[0] + '_net/sumo.net.xml'
    lm = LaneMap(net)
    roads = drawn_roads(world)
    apps = approaches(lm)
    text = open(world).read()
    for _, kind, name, pos, yaw in items(text):
        r = relation(pos, yaw, kind, apps, roads)
        if r is None:
            print(f'{kind:<20} {name:<28} ({pos[0]:7.1f},{pos[1]:7.1f}): faces no approach')
            continue
        lid, along, side, gap, dh = r
        print(f'{kind:<20} {name:<28} lane {lid:<10} {along:+6.1f} m from stop line, '
              f'{"L" if side > 0 else "R"} {abs(side):4.1f} m, edge gap {gap:+5.1f} m, facing off {dh:3.0f} deg')


if __name__ == '__main__':
    main()
