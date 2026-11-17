#!/usr/bin/env python3
"""Build the raised approach roads of the bridges as SOLID brick embankments.

  python3 tools/bridge_embankments.py worlds/village.wbt [--check]

The ramps up to the bridges were plain Roads whose kerb borders hung down a FIXED depth
(roadBorderHeight -1.7 / -6 m) as brick 'walls': at the low end of a ramp they went into the
ground, at the high end they stopped short of it, and the ends were open - a hollow shell. For every
raised Road that is not a deck ('bottom TRUE' roads span the water and stay open bridges) this
tool adds one closed solid under the road, from 0.1 m below the ground up to the road surface,
following the road's own spline (same B-spline as Webots), road width + 0.3 m each side, with
brick sides and end walls; the hanging borders of those roads become a 0.1 m kerb.
Re-running replaces the embankments it made. Backup .before_embankments.
"""
import argparse
import math
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net_heights import raised_roads  # noqa: E402

DECK_DEPTH = 0.8   # m: brick slab under a bridge span (same width / kerb as the ramps: no step)
BRICK = ('https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/default/worlds/textures/'
         'old_brick_wall.jpg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--side', type=float, default=0.3)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    text = open(a.world).read()
    decks = set()
    blocks = {}
    for m in re.finditer(r'(?ms)^(?:DEF \S+ )?Road \{\n(.*?)^\}', text):
        rid = re.search(r'(?m)^  id "([^"]+)"', m.group(1))
        if rid:
            blocks[rid.group(1)] = m
            if re.search(r'(?m)^  bottom TRUE', m.group(1)):
                decks.add(rid.group(1))
    text = re.sub(r'(?ms)^Solid \{\n  name "embankment\([^)]*\)".*?^\}\n', '', text)
    solids = []
    ramps = []
    for C, W, rid in raised_roads(a.world):
        if rid is None:
            continue
        deck = rid in decks                            # spans the water: a slab, open underneath
        ramps.append(rid)
        # tangent / left normal
        t = np.gradient(C[:, :2], axis=0)
        t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
        n = np.c_[-t[:, 1], t[:, 0]]
        hw = W / 2 + a.side
        top = np.maximum(C[:, 2] - 0.01, 0.0)
        # bottom of the solid: the ground for a ramp, a DECK_DEPTH slab under a bridge span
        bot = np.maximum(top - DECK_DEPTH, 0.0) if deck else np.full(len(C), -0.1)
        L = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C[:, :2], axis=0).T))]
        # vertices: per sample  0 left-bottom, 1 left-top, 2 right-top, 3 right-bottom
        V = []
        for i in range(len(C)):
            lx, ly = C[i, :2] + hw * n[i]
            rx, ry = C[i, :2] - hw * n[i]
            V += [(lx, ly, bot[i]), (lx, ly, top[i]), (rx, ry, top[i]), (rx, ry, bot[i])]
        faces, tex, tidx = [], [], []

        def quad(a_, b_, c_, d_, uv):
            k = len(tex)
            tex.extend(uv)
            faces.append(f'{a_} {b_} {c_} {d_} -1')
            tidx.append(f'{k} {k + 1} {k + 2} {k + 3} -1')

        for i in range(len(C) - 1):
            j = i + 1
            u0, u1 = L[i] / 2.0, L[j] / 2.0                       # brick texture: 2 m per tile
            # left wall (outward = +n), right wall (outward = -n), top (under the asphalt)
            quad(4 * i, 4 * j, 4 * j + 1, 4 * i + 1, [(u0, 0), (u1, 0), (u1, top[j] + 0.1), (u0, top[i] + 0.1)])
            quad(4 * i + 3, 4 * i + 2, 4 * j + 2, 4 * j + 3, [(u0, 0), (u0, top[i] + 0.1), (u1, top[j] + 0.1), (u1, 0)])
            quad(4 * i + 1, 4 * j + 1, 4 * j + 2, 4 * i + 2, [(u0, 0), (u1, 0), (u1, hw), (u0, hw)])
            if deck:                                   # underside of the slab
                quad(4 * i, 4 * i + 3, 4 * j + 3, 4 * j, [(u0, 0), (u0, hw), (u1, hw), (u1, 0)])
        e = len(C) - 1
        for i, flip in ((0, True), (e, False)):                   # end walls (closed)
            if top[i] < 0.05:
                continue
            idx = [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
            if flip:
                idx = idx[::-1]
            quad(*idx, [(0, 0), (0, top[i] + 0.1), (2 * hw / 2.0, top[i] + 0.1), (2 * hw / 2.0, 0)])
        pts = ', '.join(f'{x:.3f} {y:.3f} {z:.3f}' for x, y, z in V)
        tc = ', '.join(f'{u:.3f} {v:.3f}' for u, v in tex)
        solids.append(f'''Solid {{
  name "embankment({rid})"
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColorMap ImageTexture {{
          url [
            "{BRICK}"
          ]
        }}
        roughness 0.9
        metalness 0
      }}
      geometry IndexedFaceSet {{
        coord Coordinate {{
          point [ {pts} ]
        }}
        texCoord TextureCoordinate {{
          point [ {tc} ]
        }}
        coordIndex [ {" ".join(faces)} ]
        texCoordIndex [ {" ".join(tidx)} ]
        solid FALSE
      }}
    }}
  ]
}}
''')
        print(f'road {rid}: {"deck slab" if deck else "embankment"} {L[-1]:.0f} m long, up to {top.max():.1f} m, '
              f'{2 * hw:.1f} m wide')
    print(f'bridge decks (open underneath, {DECK_DEPTH} m slab): {", ".join(sorted(decks)) or "none"}')
    if a.check:
        return
    # the hanging fixed-depth borders of the ramps -> a low kerb on top of the embankment
    for rid in sorted(ramps, key=lambda r: -blocks[r].start()):
        m = blocks[rid]
        b = re.sub(r'(?m)^  roadBorderHeight -?[\d.]+', '  roadBorderHeight 0.1', m.group(0))
        text = text[:m.start()] + b + text[m.end():]
    shutil.copy(a.world, a.world + '.before_embankments')
    open(a.world, 'w').write(text.rstrip('\n') + '\n' + ''.join(solids))
    print(f'{len(solids)} embankments -> {a.world} (backup .before_embankments)')


if __name__ == '__main__':
    main()
