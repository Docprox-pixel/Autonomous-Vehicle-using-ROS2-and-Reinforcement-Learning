#!/usr/bin/env python3
"""Write the traffic lights of a world next to its lane map (<world>_net/signals.yaml).

SUMO networks exported by Webots have no traffic lights (every junction is 'priority'), so the
lane map cannot tell which junctions are signalised. This is the HD-map 'signal' layer: the
traffic_rules node only applies light colours at stop lines of junctions listed here (otherwise a
light seen elsewhere, e.g. at night when lights are the brightest thing in view, could be applied
to a junction without lights and hold the car forever).

  python3 tools/extract_signals.py worlds/city.wbt
"""
import os
import re
import sys

import yaml


def main():
    world = sys.argv[1]
    text = open(world).read()
    lights = []
    for m in re.finditer(r'(?m)^(CrossRoadsTrafficLight|GenericTrafficLight) \{', text):
        seg = text[m.end():m.end() + 400]
        t = re.search(r'translation ([-\d.e]+) ([-\d.e]+)', seg)
        r = re.search(r'rotation 0 0 (-?1) ([-\d.e]+)', seg)
        yaw = (float(r.group(2)) * float(r.group(1))) if r else 0.0      # lamps face the light's +x
        blk = text[m.end():text.index('\n}', m.end())]
        g = re.search(r'greenTime ([\d.]+)', blk)
        rr = re.search(r'redTime ([\d.]+)', blk)
        sg = re.search(r'startGreen (TRUE|FALSE)', blk)
        lights.append({'type': m.group(1), 'x': float(t.group(1)), 'y': float(t.group(2)), 'yaw': round(yaw, 4),
                       # schedule (Webots generic_traffic_light: green -> orange 1.5 s -> red -> orange 1.5 s,
                       # 512 ms controller tick) = the V2I SPaT data of this signal
                       'green': float(g.group(1)) if g else 60.0, 'red': float(rr.group(1)) if rr else 15.0,
                       'start_green': (sg.group(1) == 'TRUE') if sg else True})
    net_dir = os.path.splitext(world)[0] + '_net'
    net_dir = os.path.realpath(net_dir)
    out = os.path.join(net_dir, 'signals.yaml')
    yaml.safe_dump({'source': os.path.basename(world), 'lights': lights}, open(out, 'w'), sort_keys=False)
    print(f'{len(lights)} traffic lights -> {out}')


if __name__ == '__main__':
    main()
