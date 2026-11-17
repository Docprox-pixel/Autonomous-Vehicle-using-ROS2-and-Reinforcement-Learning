#!/usr/bin/env python3
"""Regenerate the SUMO background traffic for the CURRENT network.

  python3 tools/make_traffic.py worlds/village_net [--vehicles 1800] [--every 2.0] [--seed 5]

randomTrips (min 150 m trips, validated by duarouter) over the whole network - new roads, cul-de-sac
loops, bridges - then drops through-traffic in the car parks (aisles 12, 23 and their link_ roads:
parked cars and our parking bays are there) and departs the rest every --every s.
Run after every network change (removed turns invalidate old routes). Backup sumo.rou.xml.prev.
"""
import argparse
import os
import re
import shutil
import subprocess

CARPARK = ('12', '23')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('net_dir')
    ap.add_argument('--vehicles', type=int, default=1800)
    ap.add_argument('--every', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()
    d = a.net_dir
    rou = os.path.join(d, 'sumo.rou.xml')
    if os.path.exists(rou):
        shutil.copy(rou, rou + '.prev')
    tools = os.path.join(os.environ.get('SUMO_HOME', '/usr/share/sumo'), 'tools')
    subprocess.check_call(['python3', os.path.join(tools, 'randomTrips.py'), '-n', os.path.join(d, 'sumo.net.xml'),
                           '-o', os.path.join(d, 'sumo.trip.xml'), '-r', rou, '--seed', str(a.seed),
                           '--min-distance', '150', '--period', str(a.every * 0.8), '-e', str(a.vehicles * a.every),
                           '--validate'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = open(rou).read()
    head = t.split('<routes', 1)[0]
    veh = re.findall(r'(?s)    <vehicle id="[^"]+" depart="[^"]+">\n        <route edges="([^"]+)"/>\n    </vehicle>\n', t)
    keep = [r for r in veh if not any(e.startswith('link_') or e in CARPARK for e in r.split())][:a.vehicles]
    out = head + ('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                  'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
                  '    <!-- tools/make_traffic.py: whole current network, no through-traffic in the car parks -->\n')
    for i, r in enumerate(keep):
        out += f'    <vehicle id="{i}" depart="{a.every * i:.2f}">\n        <route edges="{r}"/>\n    </vehicle>\n'
    open(rou, 'w').write(out + '</routes>\n')
    print(f'{len(veh)} routed trips, {len(keep)} kept -> {rou}')


if __name__ == '__main__':
    main()
