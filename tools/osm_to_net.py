#!/usr/bin/env python3
"""OpenStreetMap -> SUMO lane map aligned with the Webots OSM importer's world coordinates.

  python3 tools/osm_to_net.py worlds/osm_source/metro_city.osm worlds/metro_city_net

The Webots importer (resources/osm_importer) projects lon/lat with a UTM projection centred on the
map (lon_0 = bbox centre) and then shifts everything so the projected bbox centre is (0, 0). The
SUMO network must use the SAME projection and shift, otherwise routes would not lie on the roads.

netconvert options: car roads only, junction clusters joined (OSM draws dual carriageways and big
junctions as many nodes), ramps guessed on motorways, traffic lights from OSM 'traffic_signals'
(and guessed where OSM has none), no U-turns.
"""
import math
import os
import re
import subprocess
import sys

import pyproj


def main():
    osm, out_dir = sys.argv[1], sys.argv[2]
    head = open(osm).read(4000)
    b = re.search(r'<bounds minlat="([-\d.]+)" minlon="([-\d.]+)" maxlat="([-\d.]+)" maxlon="([-\d.]+)"', head)
    minlat, minlon, maxlat, maxlon = (float(v) for v in b.groups())
    lat0, lon0 = 0.5 * (minlat + maxlat), 0.5 * (minlon + maxlon)
    zone = 1 + math.floor((lon0 + 180) / 6)
    hemi = 'south' if lat0 < 0 else 'north'
    proj_str = (f'+proj=utm +{hemi} +zone={zone} +lon_0={lon0:f} +lat_0={lat0:f} +x_0=0 +y_0=0 '
                f'+ellps=WGS84 +units=m +no_defs')
    P = pyproj.Proj(proj_str)
    x1, y1 = P(minlon, minlat)
    x2, y2 = P(maxlon, maxlat)
    xo, yo = (x1 + x2) / 2, (y1 + y2) / 2
    os.makedirs(out_dir, exist_ok=True)
    net = os.path.join(out_dir, 'sumo.net.xml')
    cmd = ['netconvert', '--osm-files', osm, '-o', net,
           '--proj', proj_str, '--offset.disable-normalization', 'true',
           '--offset.x', f'{-xo:.3f}', '--offset.y', f'{-yo:.3f}',
           '--keep-edges.by-vclass', 'passenger', '--remove-edges.isolated',
           '--geometry.remove', '--ramps.guess', '--junctions.join', '--junctions.join-dist', '12',
           '--tls.guess-signals', '--tls.discard-simple', '--tls.join', '--tls.default-type', 'static',
           '--no-turnarounds.except-deadend', '--osm.sidewalks', 'false', '--osm.crossings', 'false',
           '--roundabouts.guess', '--output.street-names', '--no-warnings']
    print(' '.join(cmd[:4]), '...')
    subprocess.check_call(cmd)
    # Coordinates are already in the Webots world frame; Webots' SumoInterface (and our LaneMap)
    # compute world = sumo - netOffset, so the stored netOffset must be zero.
    text = open(net).read()
    text = re.sub(r'netOffset="[^"]*"', 'netOffset="0.00,0.00"', text, count=1)
    open(net, 'w').write(text)
    open(os.path.join(out_dir, 'projection.txt'), 'w').write(
        f'{proj_str}\noffset {xo:.3f} {yo:.3f}\nbbox {minlat} {minlon} {maxlat} {maxlon}\n')
    print(f'-> {net}  (projection {proj_str}, offset {xo:.1f}, {yo:.1f})')


if __name__ == '__main__':
    main()
