#!/usr/bin/env python3
"""Grade the hill mesh (DEF MONTAIN) flat under new asphalt discs, like a cut for a cul-de-sac.

  python3 tools/grade_terrain.py worlds/village.wbt X Y RADIUS [X Y RADIUS ...]

Iteratively lowers the mesh vertices of every triangle that rises above the ground inside a disc
(+0.3 m shoulder), halving their height each pass (towards -0.05 m), so the slope steepens into an
embankment around the disc instead of the hill being removed. Coincident vertex copies move
together (no cracks). Backup .before_grade.
"""
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from terrain_height import _rot  # noqa: E402


def main():
    world = sys.argv[1]
    discs = [(float(sys.argv[i]), float(sys.argv[i + 1]), float(sys.argv[i + 2])) for i in range(2, len(sys.argv), 3)]
    text = open(world).read()
    m = re.search(r'(?m)^DEF MONTAIN Solid \{', text)
    s0 = m.end()
    tr = np.array([float(v) for v in re.search(r'(?m)^  translation (.+)$', text[s0:]).group(1).split()])
    R = _rot(*[float(v) for v in re.search(r'(?m)^  rotation (.+)$', text[s0:]).group(1).split()])
    sc = np.array([float(v) for v in re.search(r'scale ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)', text[s0:]).groups()])
    pm = re.compile(r'point \[\s*(.*?)\]', re.S).search(text, s0)
    im = re.compile(r'coordIndex \[\s*(.*?)\]', re.S).search(text, s0)
    W = (np.array(pm.group(1).replace(',', ' ').split(), float).reshape(-1, 3) * sc) @ R.T + tr
    F = np.array(im.group(1).replace(',', ' ').split(), int).reshape(-1, 4)[:, :3]
    key = {}
    group = np.array([key.setdefault((round(x, 2), round(y, 2)), len(key)) for x, y, _ in W])
    for cx, cy, rad in discs:
        for it in range(40):
            T = W[F]
            # sample each triangle near the disc on a barycentric grid
            u = np.linspace(0, 1, 9)
            bu, bv = np.meshgrid(u, u)
            ok = bu + bv <= 1
            bu, bv = bu[ok], bv[ok]
            c = T.mean(1)
            near = np.flatnonzero(np.hypot(c[:, 0] - cx, c[:, 1] - cy) < rad + 12)
            bad = set()
            for f in near:
                a, b, cc = T[f]
                S = a + bu[:, None] * (b - a) + bv[:, None] * (cc - a)
                inside = np.hypot(S[:, 0] - cx, S[:, 1] - cy) < rad + 0.3
                if inside.any() and S[inside, 2].max() > 0.02:
                    bad.add(f)
            if not bad:
                print(f'disc ({cx},{cy}) r {rad}: clear after {it} passes')
                break
            gs = {group[v] for f in bad for v in F[f] if W[v, 2] > 0.0}
            for g in gs:
                sel = group == g
                W[sel, 2] = np.maximum(-0.05, W[sel, 2] * 0.5 - 0.03)
    loc = ((W - tr) @ R) / sc
    new = 'point [\n                ' + ', '.join(f'{a:.7g} {b:.7g} {c:.7g}' for a, b, c in loc) + '\n              ]'
    shutil.copy(world, world + '.before_grade')
    open(world, 'w').write(text[:pm.start()] + new + text[pm.end():])


if __name__ == '__main__':
    main()
