#!/usr/bin/env python3
"""Parking bays of a world (Webots ParkingLines nodes) -> worlds/<world>_net/parking.yaml.

  python3 tools/extract_parking.py worlds/village.wbt

ParkingLines (Webots PROTO): `numberOfCarParks` bays side by side along the node's local x axis,
each `carParkWidth` long in x and `carParkLength` deep in y, starting at the node origin. In the
village they line the roads as PARALLEL-parking bays (5.8 m x 3.1 m). Each bay is written as its
centre (x, y) and the heading `yaw` of a car parked in it (along local x).
"""
import math
import os
import re
import sys

import numpy as np
import yaml


def main():
    world = sys.argv[1]
    text = open(world).read()
    bays = []
    for k, m in enumerate(re.finditer(r'(?ms)^ParkingLines \{\n(.*?)^\}', text)):
        b = m.group(1)
        tr = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+)', b)
        rot = re.search(r'(?m)^  rotation 0 0 (-?1) ([-\d.e]+)', b)
        n = int((re.search(r'(?m)^  numberOfCarParks (\d+)', b) or [0, 5])[1])
        L = float((re.search(r'(?m)^  carParkLength ([\d.]+)', b) or [0, 4.8])[1])
        W = float((re.search(r'(?m)^  carParkWidth ([\d.]+)', b) or [0, 2.4])[1])
        tx, ty = (float(tr.group(1)), float(tr.group(2))) if tr else (0.0, 0.0)
        a = float(rot.group(2)) * float(rot.group(1)) if rot else 0.0
        c, s = math.cos(a), math.sin(a)
        for i in range(n):
            lx, ly = (i + 0.5) * W, 0.5 * L
            bays.append({'lot': k, 'index': i, 'x': round(tx + c * lx - s * ly, 3), 'y': round(ty + s * lx + c * ly, 3),
                         'yaw': round(a, 4), 'length': W, 'width': L})
    # bays already taken by vehicles parked in the world (static *Simple models: cars, trucks,
    # buses, motorbikes): a truck in bay 0/0 blocked the reverse manoeuvre
    parked = []
    for m in re.finditer(r'(?ms)^(?:DEF \S+ )?([A-Z][A-Za-z0-9]*(?:Simple|Truck|Trailer|Bus)[A-Za-z0-9]*) \{\n(.*?)^\}', text):
        tr = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+)', m.group(2))
        if tr:
            L = {'TruckSimple': 14.0, 'BusSimple': 12.0}.get(m.group(1), 4.5)
            rot = re.search(r'(?m)^  rotation 0 0 (-?1) ([-\d.e]+)', m.group(2))
            parked.append((float(tr.group(1)), float(tr.group(2)), L,
                           float(rot.group(2)) * float(rot.group(1)) if rot else 0.0, m.group(1)))
    for b in bays:
        c, s = math.cos(b['yaw']), math.sin(b['yaw'])
        for (px, py, L, pa, kind) in parked:
            # vehicle as a segment along its heading (front .. back), tested against the bay box
            for u in np.linspace(-0.5, 0.5, 9):
                qx, qy = px + u * L * math.cos(pa), py + u * L * math.sin(pa)
                lx, ly = c * (qx - b['x']) + s * (qy - b['y']), -s * (qx - b['x']) + c * (qy - b['y'])
                if abs(lx) < b['length'] / 2 + 0.3 and abs(ly) < b['width'] / 2 + 0.3:
                    b['occupied'] = kind
                    break
            if b.get('occupied'):
                break
    out = world.replace('.wbt', '_net/parking.yaml')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    yaml.safe_dump({'source': os.path.basename(world), 'bays': bays}, open(out, 'w'), sort_keys=False)
    print(f"{len(bays)} parking bays -> {out}; occupied: {[(b['lot'], b['index'], b['occupied']) for b in bays if b.get('occupied')]}")


if __name__ == '__main__':
    main()
