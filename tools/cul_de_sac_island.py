#!/usr/bin/env python3
"""Make every turning circle a real CUL-DE-SAC: a raised grass island with a kerb in the centre that
the one-way loop drives round, and a dashed centre line from the road end up to the island.

  python3 tools/cul_de_sac_island.py worlds/village.wbt [--island 4.5] [--check]

The island is centred on the loop the routes actually drive (circle fitted to the tc_<jid>_a/_b
lanes in <world>_net/tc.edg.xml), radius --island (default 4.5 m: with the loop at ~7 m the inner
wheels keep >= 1.7 m from the kerb). Kerb 0.15 m high, 0.25 m wide, with a collision shape (lidar
and physics see it). The centre line continues the road's dashed line between the in- and out-lane.
Re-running replaces the islands / lines it made before.
"""
import argparse
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_lane_offsets import drawn_roads  # noqa: E402

GRASS = 'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/appearances/protos/Grass.proto"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--island', type=float, default=4.5)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    net_dir = os.path.splitext(a.world)[0] + '_net'
    off = np.array([float(v) for v in ET.parse(os.path.join(net_dir, 'sumo.net.xml')).getroot()
                    .find('location').get('netOffset').split(',')])
    tc = open(os.path.join(net_dir, 'tc.edg.xml')).read()
    roads = drawn_roads(a.world)
    text = open(a.world).read()
    text = re.sub(r'(?ms)^Solid \{\n  translation [^\n]*\n  name "cul-de-sac (island|centre line)\([^)]*\)".*?^\}\n', '', text)
    add = []
    for jid in sorted(set(re.findall(r'id="tc_(\w+?)_a"', tc))):
        P = np.vstack([np.array([[float(v) for v in q.split(',')[:2]] for q in
                                 re.search(rf'id="tc_{jid}_{s}"[^>]*shape="([^"]+)"', tc).group(1).split()])
                       for s in 'ab']) - off
        Q = P[np.hypot(*(P - P[0]).T) > 5]
        c = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]
        O = c[:2]
        Rloop = float(np.hypot(*(Q - O).T).min())
        # the road that ends here: its drawn centre-line end nearest the loop's start
        best = None
        for rid, (C, W, _) in roads.items():
            for end, nxt in ((C[0], C[min(3, len(C) - 1)]), (C[-1], C[max(len(C) - 4, 0)])):
                d = float(np.hypot(*(end - P[0])))
                if best is None or d < best[0]:
                    best = (d, rid, end, nxt)
        _, rid, E, nxt = best
        u = O - E
        dist = float(np.hypot(*u))
        u /= dist
        r_i = a.island
        print(f'{jid} (end of road {rid}): loop radius {Rloop:.1f} m -> island r {r_i} m at ({O[0]:.2f},{O[1]:.2f}), '
              f'inner-wheel margin ~{Rloop - 1.0 - r_i:.1f} m; centre line {dist - r_i - 0.3:.1f} m')
        if Rloop - 1.0 - r_i < 1.2:
            print('  WARNING: island too big for this loop')
        add.append(f'''Solid {{
  translation {O[0]:.3f} {O[1]:.3f} 0
  name "cul-de-sac island({jid})"
  children [
    Pose {{
      translation 0 0 0.075
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColor 0.72 0.72 0.70
            roughness 0.9
            metalness 0
          }}
          geometry Cylinder {{
            height 0.15
            radius {r_i:.2f}
            subdivision 48
          }}
        }}
      ]
    }}
    Pose {{
      translation 0 0 0.08
      children [
        Shape {{
          appearance Grass {{
          }}
          geometry Cylinder {{
            height 0.16
            radius {r_i - 0.25:.2f}
            subdivision 48
          }}
        }}
      ]
    }}
  ]
  boundingObject Pose {{
    translation 0 0 0.075
    children [
      Cylinder {{
        height 0.15
        radius {r_i:.2f}
      }}
    ]
  }}
}}
''')
        # dashed centre line: 3 m dash, 3 m gap, 0.15 m wide, from the road end to the kerb
        n = np.array([-u[1], u[0]])
        L = dist - r_i - 0.3
        quads, pts = [], []
        s = 1.0
        while s < L:
            s2 = min(s + 3.0, L)
            for q in (E + u * s - 0.075 * n, E + u * s2 - 0.075 * n, E + u * s2 + 0.075 * n, E + u * s + 0.075 * n):
                pts.append(q)
            k = len(pts) - 4
            quads.append(f'{k} {k + 1} {k + 2} {k + 3} -1')
            s += 6.0
        if quads:
            add.append(f'''Solid {{
  translation 0 0 0.012
  name "cul-de-sac centre line({jid})"
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.95 0.95 0.95
        roughness 0.6
        metalness 0
      }}
      geometry IndexedFaceSet {{
        coord Coordinate {{
          point [ {", ".join(f"{q[0]:.3f} {q[1]:.3f} 0" for q in pts)} ]
        }}
        coordIndex [ {" ".join(quads)} ]
      }}
    }}
  ]
}}
''')
    if a.check:
        return
    if 'Grass.proto' not in text:
        lines = text.split('\n')
        last = max(i for i, l in enumerate(lines) if l.startswith('EXTERNPROTO'))
        lines.insert(last + 1, GRASS)
        text = '\n'.join(lines)
    shutil.copy(a.world, a.world + '.before_islands')
    open(a.world, 'w').write(text.rstrip('\n') + '\n' + ''.join(add))
    print(f'{len(add)} cul-de-sac parts -> {a.world} (backup .before_islands)')


if __name__ == '__main__':
    main()
