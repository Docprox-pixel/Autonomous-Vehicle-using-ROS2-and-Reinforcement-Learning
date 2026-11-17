#!/usr/bin/env python3
"""Put a traffic light on every approach of every real intersection of a world (from its lane map).

  python3 tools/add_traffic_lights.py worlds/village.wbt [--min-approaches 3]

For each SUMO junction with >= --min-approaches incoming roads: one GenericTrafficLight per
incoming road, 1.2 m right of the lane end (the stop line), facing the approaching car, green 10 s /
red 10 s. Crossing directions alternate: approaches roughly along one axis start green, the
crossing axis starts red, so they are never green together. Junctions that already have a light
within 20 m are skipped. Afterwards worlds/<world>_net/signals.yaml is regenerated (the HD-map
signal layer the traffic-rules node uses), and the SUMO network gets matching static signals
(10 s green, 2 s yellow per axis) so SUMO traffic also stops at red.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from auto_car_core.lanemap import LaneMap  # noqa: E402

PROTO = ('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/'
         'traffic/protos/GenericTrafficLight.proto"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--min-approaches', type=int, default=3)
    a = ap.parse_args()
    world = os.path.abspath(a.world)
    net_dir = os.path.realpath(os.path.splitext(world)[0] + '_net')
    net = os.path.join(net_dir, 'sumo.net.xml')
    text = open(world).read()
    existing = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(
        r'(?:GenericTrafficLight|CrossRoadsTrafficLight) \{\n(?:  [^\n]*\n)*?  translation ([-\d.e]+) ([-\d.e]+)', text)]
    lm = LaneMap(net)
    root = ET.parse(net).getroot()
    added, tls_ids, names = [], [], set(re.findall(r'name "([^"]+)"', text))
    for j in root.findall('junction'):
        if j.get('type') in ('internal', 'dead_end') or j.get('id', '').startswith(':'):
            continue
        inc = [l for l in (j.get('incLanes') or '').split() if l in lm.lanes and not lm.lanes[l].internal]
        edges = sorted({lm.lanes[l].edge for l in inc})
        if len(edges) < a.min_approaches:
            continue
        jx, jy = float(j.get('x')), float(j.get('y'))
        if any(math.hypot(ex - jx, ey - jy) < 20.0 for ex, ey in existing):
            continue
        tls_ids.append(j.get('id'))
        for e in edges:
            lane = min(lm.edge_lanes[e], key=lambda l: l.index)          # rightmost lane of that road
            p, q = lane.pts[-1], lane.pts[-2]
            h = math.atan2(p[1] - q[1], p[0] - q[0])                    # driving direction
            right = (math.sin(h), -math.cos(h))
            # FAR-SIDE signal head (across the junction, on the right of the road the approach
            # leads into): visible straight ahead from far away AND while waiting at the stop line.
            # (A near-side head 1 m before the line is ~50 deg to the side of a car at the line,
            # outside the front camera's +/-34 deg: the car could never see its light turn green.)
            cross = min(max(2.0 * math.hypot(jx - p[0], jy - p[1]), 8.0), 30.0)
            x = p[0] + math.cos(h) * cross + right[0] * (lane.width / 2 + 1.5)
            y = p[1] + math.sin(h) * cross + right[1] * (lane.width / 2 + 1.5)
            # never ON a road: push outward (and if needed back toward the junction) until the pole
            # is >= 0.5 m from every lane edge (a bending road beyond the junction can lie there)
            def clearance(px, py):
                return min(L.project(np.array([px, py]))[1] - L.width / 2 for L in lm.lanes.values())
            if clearance(x, y) < 0.5:
                spot = None
                for back in (0.0, 3.0, 6.0, 9.0):
                    for out in np.arange(0.5, 8.01, 0.5):
                        cx_ = x + right[0] * out - math.cos(h) * back
                        cy_ = y + right[1] * out - math.sin(h) * back
                        if clearance(cx_, cy_) >= 0.5:
                            spot = (cx_, cy_)
                            break
                    if spot:
                        break
                if spot is None:
                    print(f'  could not place a light for {e} at junction {j.get("id")}: skipped')
                    continue
                x, y = spot
            axis = int(round(((h % math.pi) / (math.pi / 2)))) % 2         # 0: ~E-W, 1: ~N-S
            k = 0
            while f'signal {j.get("id")}-{k}' in names:
                k += 1
            name = f'signal {j.get("id")}-{k}'
            names.add(name)
            # The lamps face the PROTO's +x axis (lamp offset +0.12 m along x): rotate by h + pi so
            # they face the approaching car (rotation h made them face away: cars saw only the backs)
            rot = math.atan2(math.sin(h + math.pi), math.cos(h + math.pi))
            added.append(f'GenericTrafficLight {{\n  translation {x:.2f} {y:.2f} 0\n  rotation 0 0 1 {rot:.4f}\n'
                         f'  name "{name}"\n  startGreen {"TRUE" if axis == 0 else "FALSE"}\n'
                         f'  greenTime 10\n  redTime 10\n}}\n')
    if PROTO not in text:
        text = text.replace('\nWorldInfo {', '\n' + PROTO + '\nWorldInfo {', 1)
    text = text.rstrip('\n') + '\n' + ''.join(added)
    open(world, 'w').write(text)
    print(f'{len(added)} traffic lights at {len(tls_ids)} intersections -> {world}')
    subprocess.check_call([sys.executable, os.path.join(HERE, 'extract_signals.py'), world])
    have = {j.get('id') for j in root.findall('junction') if j.get('type') == 'traffic_light'}
    tls_ids = [j for j in tls_ids if j not in have]   # netconvert crashes re-setting existing signals
    if tls_ids:   # SUMO traffic obeys matching signals
        out = net + '.tls.xml'
        subprocess.check_call(['netconvert', '-s', net, '-o', out, '--tls.set', ','.join(tls_ids),
                               '--tls.default-type', 'static', '--tls.green.time', '10', '--tls.yellow.time', '2',
                               '--no-warnings'])
        os.replace(out, net)
        print(f'SUMO network: static signals at {len(tls_ids)} junctions (10 s green, 2 s yellow)')


if __name__ == '__main__':
    main()
