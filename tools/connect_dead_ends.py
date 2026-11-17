#!/usr/bin/env python3
"""Connect every dead-end road of a world to the road network (no road may simply end).

  python3 tools/connect_dead_ends.py worlds/village.wbt [--max 8] [--check]

For each SUMO junction of type dead_end, the nearest normal junction within --max metres gets a
short connector in the right direction:
  * a road that STARTS at the dead end gets an entry link  junction -> dead end
  * a road that ENDS at the dead end gets an exit link      dead end -> junction
(In the village the two car-park lanes, roads 12 and 23, were connected to nothing: the car could
not drive into the car parks.) The connectors are
  * merged into worlds/<world>_net/sumo.net.xml with netconvert (backup .before_connect), and
  * drawn in the world as Road nodes (same width as the car-park lane, no markings).
"""
import argparse
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--max', type=float, default=8.0)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    net_dir = a.world.replace('.wbt', '_net')
    net = os.path.join(net_dir, 'sumo.net.xml')
    root = ET.parse(net).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    J = {j.get('id'): (np.array([float(j.get('x')), float(j.get('y'))]), j.get('type'))
         for j in root.findall('junction') if j.get('type') != 'internal'}
    edges = [e for e in root.findall('edge') if e.get('function') != 'internal']
    links = []
    for jid, (p, typ) in J.items():
        if typ != 'dead_end':
            continue
        outs = [e for e in edges if e.get('from') == jid]
        ins = [e for e in edges if e.get('to') == jid]
        cand = sorted((float(np.hypot(*(q - p))), k) for k, (q, t) in J.items() if k != jid and t != 'dead_end')
        if not cand or cand[0][0] > a.max:
            print(f'{jid}: no junction within {a.max} m - left as is')
            continue
        dist, k = cand[0]
        ref = (outs or ins)[0]
        lane = ref.find('lane')
        width = float(lane.get('width', 3.2))
        if outs:
            links.append((f'link_{k}_{jid}', k, jid, width, dist))
        if ins:
            links.append((f'link_{jid}_{k}', jid, k, width, dist))
    for l in links:
        print(f'connector {l[0]}: {l[1]} -> {l[2]} ({l[4]:.1f} m, width {l[3]})')
    if a.check or not links:
        return
    edg = os.path.join(net_dir, 'connect.edg.xml')
    with open(edg, 'w') as f:
        f.write('<edges>\n')
        for eid, fr, to, w, _ in links:
            f.write(f'  <edge id="{eid}" from="{fr}" to="{to}" numLanes="1" speed="5.0" width="{w}"/>\n')
        f.write('</edges>\n')
    # explicit connections: netconvert does not add turn connections for new edges at existing
    # junctions, so traffic arriving at the junction could not take the link (and vice versa)
    con = os.path.join(net_dir, 'connect.con.xml')
    new_ids = {l[0] for l in links}
    with open(con, 'w') as f:
        f.write('<connections>\n')
        for eid, fr, to, w, _ in links:
            into = [e.get('id') for e in edges if e.get('to') == fr and e.get('id') not in new_ids]
            outof = [e.get('id') for e in edges if e.get('from') == to and e.get('id') not in new_ids]
            for e_in in into:
                f.write(f'  <connection from="{e_in}" to="{eid}" fromLane="0" toLane="0"/>\n')
            for e_out in outof:
                f.write(f'  <connection from="{eid}" to="{e_out}" fromLane="0" toLane="0"/>\n')
        f.write('</connections>\n')
    if not os.path.exists(net + '.before_connect'):
        shutil.copy(net, net + '.before_connect')
    src = net + '.before_connect'
    tmp = net + '.tmp.xml'
    subprocess.check_call(['netconvert', '-s', src, '-e', edg, '-x', con, '-o', tmp, '--no-warnings',
                           '--offset.disable-normalization', 'true', '--no-turnarounds.except-deadend'])
    os.replace(tmp, net)
    # draw them in the world
    text = open(a.world).read()
    add = []
    for eid, fr, to, w, _ in links:
        if f'id "{eid}"' in text:
            continue
        p0, p1 = J[fr][0] - off, J[to][0] - off
        add.append(f'Road {{\n  translation 0 0 0.01\n  name "road({eid})"\n  id "{eid}"\n  width {w}\n'
                   f'  numberOfLanes 1\n  numberOfForwardLanes 1\n  lines []\n  rightBorder FALSE\n  leftBorder FALSE\n'
                   f'  wayPoints [\n    {p0[0]:.2f} {p0[1]:.2f} 0\n    {p1[0]:.2f} {p1[1]:.2f} 0\n  ]\n'
                   f'  splineSubdivision 0\n}}\n')
    if add:
        shutil.copy(a.world, a.world + '.before_connect')
        open(a.world, 'w').write(text.rstrip('\n') + '\n' + ''.join(add))
    print(f'{len(links)} connectors added to {net} and {len(add)} roads drawn in {a.world}')


if __name__ == '__main__':
    main()
