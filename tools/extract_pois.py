#!/usr/bin/env python3
"""Points of interest + building footprints of a world -> <world>_net/pois.yaml (the map's POI layer).

Like the 'places' layer of a real map: every building with its footprint, height, kind and a
readable name (shop brand, museum, bus stop, ...). The route planner draws them in RViz and the
driver-safety node looks up the nearest place of a kind ('hospital') to drive there.

  python3 tools/extract_pois.py worlds/village.wbt --hospital "building(1)"

Footprints: SimpleBuilding has explicit `corners`; other building PROTOs get the outline of their
own boundingObject (tools/proto_footprint.py), else a rectangle from their size fields (or a typical
size for their kind). Coordinates are world/map coordinates.
"""
import argparse
import math
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto_footprint import footprint as proto_footprint  # noqa: E402

# PROTO -> (kind, default footprint length x width, height, readable label)
KINDS = {
    'SimpleBuilding': ('building', (16, 16), None, 'Building'),
    'ResidentialBuilding': ('residential', (20, 14), 15, 'Apartments'),
    'SimpleTwoFloorsHouse': ('house', (10, 8), 7, 'House'),
    'BungalowStyleHouse': ('house', (12.5, 12), 6, 'House'),   # its boundingObject is unscaled (2.5x smaller)
    'ModernHouse': ('house', (12, 10), 7, 'House'),
    'FastFoodRestaurant': ('restaurant', (12, 12), 5, 'Restaurant'),
    'Museum': ('museum', (30, 20), 12, 'Museum'),
    'TheThreeTowers': ('landmark', (40, 40), 60, 'The Three Towers'),
    'PublicToilet': ('toilet', (3, 2), 3, 'Public toilet'),
    'BusStop': ('bus_stop', (4, 1.5), 3, 'Bus stop'),
    'GasStation': ('fuel', (20, 15), 6, 'Highway fuel station'),
    'BuildingUnderConstruction': ('construction', (20, 20), 20, 'Construction site'),
    'Church': ('church', (25, 15), 20, 'Church'),
    'Hotel': ('hotel', (14, 24), 20, 'Highway Hotel'),
    'CommercialBuilding': ('shop', (27, 42), 12, 'Shopping centre'),
    'Warehouse': ('warehouse', (30, 20), 8, 'Warehouse'),
    'Carwash': ('car_wash', (15, 8), 5, 'Car wash'),
}


def blocks(text, proto):
    for m in re.finditer(rf'(?m)^{proto} \{{', text):
        depth, j = 0, m.end() - 1
        while True:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        yield text[m.end():j]


def num(block, field, default=None):
    m = re.search(rf'(?m)^  {field} ([-\d.e]+)', block)
    return float(m.group(1)) if m else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--hospital', default=None, help='name "..." of the building that is the hospital')
    a = ap.parse_args()
    text = open(a.world).read()
    places, counts = [], {}
    for proto, (kind, (L, W), h_default, label) in KINDS.items():
        for b in blocks(text, proto):
            t = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)', b)
            if not t:
                continue
            x, y = float(t.group(1)), float(t.group(2))
            r = re.search(r'(?m)^  rotation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)', b)
            yaw = float(r.group(4)) * (1 if not r or float(r.group(3)) >= 0 else -1) if r else 0.0
            name_m = re.search(r'(?m)^  name "([^"]+)"', b)
            wb_name = name_m.group(1) if name_m else proto
            corners = re.search(r'corners \[\s*([^\]]+)\]', b)
            if corners:
                v = [float(s) for s in corners.group(1).split()]
                local = list(zip(v[0::2], v[1::2]))
            elif proto not in ('BungalowStyleHouse', 'BusStop', 'PublicToilet') and proto_footprint(proto) and num(b, 'scale', 1.0) == 1.0 and num(b, 'length') is None:
                local = proto_footprint(proto)           # the PROTO's real collision outline
            else:
                L_, W_ = num(b, 'length', L), num(b, 'width', W)
                local = [(L_ / 2, W_ / 2), (-L_ / 2, W_ / 2), (-L_ / 2, -W_ / 2), (L_ / 2, -W_ / 2)]
            c, s = math.cos(yaw), math.sin(yaw)
            foot = [[round(x + c * px - s * py, 2), round(y + s * px + c * py, 2)] for px, py in local]
            floors, fh = num(b, 'floorNumber', 3), num(b, 'floorHeight', 3.0)
            height = num(b, 'height', h_default) if h_default is not None else floors * fh
            brand = re.search(r'(?m)^  brand "([^"]+)"', b)
            counts[kind] = counts.get(kind, 0) + 1
            title = f'{brand.group(1).title()} restaurant' if brand else (
                label if kind not in ('building', 'house', 'residential') else f'{label} {counts[kind]}')
            place = {'name': title, 'kind': kind, 'webots_name': wb_name, 'x': round(x, 2), 'y': round(y, 2),
                     'height': round(float(height), 1), 'footprint': foot}
            if a.hospital and wb_name == a.hospital:
                place.update(name='Village Hospital', kind='hospital')
            places.append(place)
    net_dir = os.path.realpath(os.path.splitext(a.world)[0] + '_net')
    out = os.path.join(net_dir, 'pois.yaml')
    yaml.safe_dump({'source': os.path.basename(a.world), 'places': places}, open(out, 'w'), sort_keys=False)
    kinds = {}
    for p in places:
        kinds[p['kind']] = kinds.get(p['kind'], 0) + 1
    print(f'{len(places)} places -> {out}: {kinds}')


if __name__ == '__main__':
    main()
