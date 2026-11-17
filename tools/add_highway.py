#!/usr/bin/env python3
"""Add a highway (dual carriageway, 2 lanes each way, guardrails) to a world AND its SUMO lane map.

  python3 tools/add_highway.py worlds/village.wbt --from-junction Custom13 \
      --entry 450 -430 --loop "560,-430 580,-410 580,-210 560,-190 440,-190 420,-210 420,-400" \
      --speed 27.78 --radius 20

Geometry (world coordinates): a two-way connector road from an existing junction to a new
interchange node at --entry, then a closed highway loop through --loop corners and back to the
entry, corners rounded with --radius. The SAME polyline is used for
  * the Webots Road nodes (numberOfLanes 4, 2 forward, guardrails, speed limit), and
  * the SUMO edges (net coordinates = world + netOffset), merged into sumo.net.xml with netconvert,
so the lane map, SUMO traffic and the 3D road coincide. SUMO trips are regenerated so traffic
uses the highway too.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROAD_PROTO = ('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/'
              'road/protos/Road.proto"')


def rounded(points, radius, closed, step=2.0):
    """Polyline with corners replaced by circular arcs of `radius`."""
    P = [np.asarray(p, float) for p in points]
    n = len(P)
    out = []
    rng = range(n) if closed else range(1, n - 1)
    if not closed:
        out.append(P[0])
    for i in rng:
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        u, v = (a - b) / np.linalg.norm(a - b), (c - b) / np.linalg.norm(c - b)
        ang = math.acos(max(-1.0, min(1.0, float(u @ v))))
        if ang > math.pi - 1e-3:
            out.append(b)
            continue
        t = radius / math.tan(ang / 2)
        t = min(t, 0.45 * np.linalg.norm(a - b), 0.45 * np.linalg.norm(c - b))
        r = t * math.tan(ang / 2)                                  # radius actually used
        p1, p2 = b + u * t, b + v * t
        bis = (u + v) / np.linalg.norm(u + v)
        centre = b + bis * (r / math.sin(ang / 2))
        a1, a2 = math.atan2(*(p1 - centre)[::-1]), math.atan2(*(p2 - centre)[::-1])
        d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        k = max(2, int(abs(d) * radius / step))
        for s in np.linspace(0, 1, k + 1):
            out.append(centre + r * np.array([math.cos(a1 + d * s), math.sin(a1 + d * s)]))
    if not closed:
        out.append(P[-1])
    return np.array(out)


def road_node(name, pts, lanes, forward, width, speed, barriers):
    wp = '\n'.join(f'    {x:.3f} {y:.3f} 0' for x, y in pts)
    return (f'Road {{\n  translation 0 0 0.02\n  name "{name}"\n  width {width}\n  numberOfLanes {lanes}\n'
            f'  numberOfForwardLanes {forward}\n  speedLimit {speed}\n'
            f'  lines [\n    RoadLine {{\n    }}\n    RoadLine {{\n      color 0.85 0.75 0.3\n      type "double"\n    }}\n  ]\n'
            f'  rightBorder FALSE\n  leftBorder FALSE\n'
            f'  rightBarrier {"TRUE" if barriers else "FALSE"}\n  leftBarrier {"TRUE" if barriers else "FALSE"}\n'
            f'  wayPoints [\n{wp}\n  ]\n  splineSubdivision 0\n}}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--from-junction', required=True)
    ap.add_argument('--entry', type=float, nargs=2, required=True)
    ap.add_argument('--loop', required=True, help='"x,y x,y ..." corners of the loop after the entry')
    ap.add_argument('--speed', type=float, default=27.78)
    ap.add_argument('--radius', type=float, default=20.0)
    a = ap.parse_args()
    world = os.path.abspath(a.world)
    net_dir = os.path.realpath(os.path.splitext(world)[0] + '_net')
    net = os.path.join(net_dir, 'sumo.net.xml')
    root = ET.parse(net).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    j0 = next(j for j in root.findall('junction') if j.get('id') == a.from_junction)
    p0 = np.array([float(j0.get('x')), float(j0.get('y'))]) - off            # world
    entry = np.array(a.entry)
    corners = [tuple(map(float, c.split(','))) for c in a.loop.split()]
    # entry -> corners -> entry; corners rounded, the entry is a junction (no rounding there)
    loop = rounded([tuple(entry)] + corners + [tuple(entry)], a.radius, closed=False)
    i_far = int(np.argmax(np.hypot(*(loop - entry).T)))           # split: SUMO edges need 2 end nodes
    leg1, leg2 = loop[:i_far + 1], loop[i_far:]
    far = loop[i_far]
    conn = np.array([p0, entry])

    def sumo_xy(pts):
        return ' '.join(f'{x + off[0]:.2f},{y + off[1]:.2f}' for x, y in pts)
    nod = (f'<nodes>\n  <node id="HW_entry" x="{entry[0] + off[0]:.2f}" y="{entry[1] + off[1]:.2f}" type="priority"/>\n'
           f'  <node id="HW_far" x="{far[0] + off[0]:.2f}" y="{far[1] + off[1]:.2f}" type="priority"/>\n</nodes>\n')
    lane = 3.5
    E = []
    for eid, a_, b_, pts, nl, sp in (('hw_link', a.from_junction, 'HW_entry', conn, 1, 13.89),
                                     ('hw_east', 'HW_entry', 'HW_far', leg1, 2, a.speed),
                                     ('hw_west', 'HW_far', 'HW_entry', leg2, 2, a.speed)):
        E.append(f'  <edge id="{eid}" from="{a_}" to="{b_}" numLanes="{nl}" speed="{sp:.2f}" width="{lane}" '
                 f'shape="{sumo_xy(pts)}" spreadType="right" name="Village Highway"/>')
        E.append(f'  <edge id="-{eid}" from="{b_}" to="{a_}" numLanes="{nl}" speed="{sp:.2f}" width="{lane}" '
                 f'shape="{sumo_xy(pts[::-1])}" spreadType="right" name="Village Highway"/>')
    edg = '<edges>\n' + '\n'.join(E) + '\n</edges>\n'
    open(os.path.join(net_dir, 'highway.nod.xml'), 'w').write(nod)
    open(os.path.join(net_dir, 'highway.edg.xml'), 'w').write(edg)
    out = net + '.hw.xml'
    subprocess.check_call(['netconvert', '-s', net, '-n', os.path.join(net_dir, 'highway.nod.xml'),
                           '-e', os.path.join(net_dir, 'highway.edg.xml'), '-o', out,
                           '--offset.disable-normalization', 'true', '--no-turnarounds', '--no-warnings'])
    os.replace(out, net)
    # merging keeps the junction's OLD connection list: road(s) ending at the junction get no move
    # onto the new link (only their U-turn). Connect every incoming road lane to the link.
    r2 = ET.parse(net).getroot()
    inc = [e.get('id') for e in r2.findall('edge') if e.get('to') == a.from_junction
           and e.get('function') != 'internal' and e.get('id') != '-hw_link']
    con = '<connections>\n' + ''.join(f'  <connection from="{e}" to="hw_link" fromLane="0" toLane="0"/>\n' for e in inc) + '</connections>\n'
    open(os.path.join(net_dir, 'highway.con.xml'), 'w').write(con)
    subprocess.check_call(['netconvert', '-s', net, '-x', os.path.join(net_dir, 'highway.con.xml'), '-o', out,
                           '--offset.disable-normalization', 'true', '--no-warnings'])
    os.replace(out, net)
    print(f'SUMO: highway loop {np.hypot(*np.diff(np.vstack([leg1, leg2]), axis=0).T).sum():.0f} m + link '
          f'{np.hypot(*(entry - p0)):.0f} m merged into {net}')

    text = open(world).read()
    if 'Road.proto"' not in text:
        text = text.replace('\nWorldInfo {', '\n' + ROAD_PROTO + '\nWorldInfo {', 1)
    text = re.sub(r'(?ms)^Road \{\n  translation 0 0 0.02\n  name "(?:highway|highway link)".*?^\}\n', '', text)
    full = np.vstack([leg1, leg2[1:]])
    text += road_node('highway', full, 4, 2, 4 * lane, a.speed, True)
    text += road_node('highway link', conn, 2, 1, 2 * lane, 13.89, False)
    open(world, 'w').write(text)
    print(f'Webots: highway Road ({len(full)} way points, 4 lanes, guardrails) + link road -> {world}')

    tdir = net_dir
    subprocess.check_call([sys.executable, os.path.join(os.environ.get('SUMO_HOME', '/usr/share/sumo'), 'tools',
                                                       'randomTrips.py'), '-n', net, '-o', os.path.join(tdir, 'sumo.trip.xml'),
                           '-r', os.path.join(tdir, 'sumo.rou.xml'), '-e', '3600', '-p', '2', '--seed', '5',
                           '--min-distance', '150', '--validate'], stdout=subprocess.DEVNULL)
    alt = os.path.join(tdir, 'sumo.rou.alt.xml')
    if os.path.exists(alt):
        os.remove(alt)
    print('SUMO traffic regenerated (routes over the whole network incl. the highway)')


if __name__ == '__main__':
    main()
