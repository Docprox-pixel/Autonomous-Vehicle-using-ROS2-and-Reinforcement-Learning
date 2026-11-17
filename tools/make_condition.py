#!/usr/bin/env python3
"""Make lighting/weather variants of a world (same roads, same lane map, same car).

  python3 tools/make_condition.py worlds/city.wbt --night           -> worlds/city_night_v.wbt
  python3 tools/make_condition.py worlds/city.wbt --fog 35          -> worlds/city_fog35.wbt
  python3 tools/make_condition.py worlds/city.wbt --night --fog 60  -> worlds/city_night_fog60.wbt

Night: the textured sky and its light are replaced by a near-black Background (as Webots'
city_night.wbt does). Fog: Fog.visibilityRange. A <name>_net link to the source world's lane
map is created so routing works unchanged.
"""
import argparse
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--night', action='store_true')
    ap.add_argument('--fog', type=float)
    a = ap.parse_args()
    src = os.path.abspath(a.world)
    text = open(src).read()
    tags = []
    if a.night:
        text = re.sub(r'(?m)^TexturedBackground \{[^}]*\}\n', 'Background {\n  skyColor [\n    0.01 0.01 0.015\n  ]\n}\n', text, count=1)
        text = re.sub(r'(?m)^TexturedBackgroundLight \{[^}]*\}\n', '', text, count=1)
        tags.append('night_v')
    if a.fog:
        text, n = re.subn(r'(Fog \{[^}]*visibilityRange )[\d.]+', rf'\g<1>{a.fog:g}', text, count=1)
        if n == 0:
            text += f'Fog {{\n  color 0.38 0.35 0.32\n  visibilityRange {a.fog:g}\n}}\n'
        tags.append(f'fog{int(a.fog)}')
    base = os.path.splitext(os.path.basename(src))[0]
    name = f"{base}_{'_'.join(tags)}"
    out = os.path.join(os.path.dirname(src), name + '.wbt')
    open(out, 'w').write(text)
    link = os.path.join(os.path.dirname(src), name + '_net')
    if not os.path.exists(link):
        os.symlink(base + '_net', link)
    print(f'{out}  (lane map: {link} -> {base}_net)')


if __name__ == '__main__':
    main()
