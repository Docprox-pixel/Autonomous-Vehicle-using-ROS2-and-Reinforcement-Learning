#!/usr/bin/env python3
"""Ring motorway with a real ramp interchange, in the Webots world AND its SUMO lane map.

  python3 tools/add_ring_highway.py worlds/village.wbt --from-junction Custom13

How highways join other roads (FHWA / NCHRP / state DOT design guidance, AASHTO Green Book):
never at-grade T-junctions, but RAMPS. Entering traffic uses an on-ramp that meets the carriageway
at a shallow angle, accelerates in a parallel ACCELERATION LANE, then merges where that lane ends
(taper). Leaving traffic moves into a DECELERATION LANE, then diverges onto the off-ramp.
A two-direction interchange needs a bridge (one road over the other), impossible on this flat map,
so the motorway is a ONE-WAY ring (2 lanes, counter-clockwise) with both ramps on its village side.

Layout (world coordinates, ring corners rounded with --radius):
  ring: (450,-430) -> (580,-430) -> (580,-190) -> (450,-190) -> back   (west leg runs southbound)
  on the west leg, top to bottom:  X1 -[3 lanes: deceleration lane]- X2 -[2]- M -[3 lanes:
  acceleration lane]- T -[2 lanes: rest of the ring]- X1
  off-ramp X2 -> L, on-ramp L -> M, both 1 lane; two-way link road L <-> --from-junction.
SUMO lanes: ring edges use spreadType 'right' with the shape on the carriageway's LEFT edge, so
lane 0 (rightmost) of a 3-lane piece is the extra lane outside the 2-lane carriageway, exactly
where the Webots ramp roads run alongside.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from add_highway import rounded  # noqa: E402

LANE = 3.5
ROAD_PROTO = ('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/'
              'road/protos/Road.proto"')


def resample(pts, step=2.0):
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    g = np.r_[np.arange(0.0, s[-1], step), s[-1]]
    return np.c_[np.interp(g, s, pts[:, 0]), np.interp(g, s, pts[:, 1])], g


def offset_left(pts, d):
    """Polyline shifted d metres to the LEFT of its direction of travel."""
    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
    return pts + d * np.c_[-t[:, 1], t[:, 0]]


def cut(pts, s, a, b):
    m = (s >= a - 1e-9) & (s <= b + 1e-9)
    return pts[m]


def road_node(name, pts, lanes, width, speed, barrier_left=False, barrier_right=False, lines='dashed'):
    wp = '\n'.join(f'    {x:.3f} {y:.3f} 0' for x, y in pts)
    return (f'Road {{\n  translation 0 0 0.02\n  name "{name}"\n  width {width}\n  numberOfLanes {lanes}\n'
            f'  numberOfForwardLanes {lanes}\n  speedLimit {speed}\n'
            f'  lines [\n    RoadLine {{\n      type "{lines}"\n    }}\n  ]\n'
            f'  rightBorder FALSE\n  leftBorder FALSE\n'
            f'  rightBarrier {"TRUE" if barrier_right else "FALSE"}\n  leftBarrier {"TRUE" if barrier_left else "FALSE"}\n'
            f'  wayPoints [\n{wp}\n  ]\n  splineSubdivision 0\n}}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--from-junction', default='Custom13')
    ap.add_argument('--radius', type=float, default=22.0)
    ap.add_argument('--speed', type=float, default=27.78)
    a = ap.parse_args()
    world = os.path.abspath(a.world)
    net_dir = os.path.realpath(os.path.splitext(world)[0] + '_net')
    net = os.path.join(net_dir, 'sumo.net.xml')
    root = ET.parse(net).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    j0 = next(j for j in root.findall('junction') if j.get('id') == a.from_junction)
    p0 = np.array([float(j0.get('x')), float(j0.get('y'))]) - off

    # --- ring centreline, counter-clockwise, starting at the top of the west leg (going south)
    corners = [(450, -190), (450, -430), (580, -430), (580, -190)]
    ring = rounded(corners, a.radius, closed=True)
    # rotate so it starts where the west leg's straight begins (just after the NW corner)
    k = int(np.argmin(np.hypot(ring[:, 0] - 450, ring[:, 1] - (-190 - a.radius))))
    ring = np.vstack([np.roll(ring, -k, axis=0), np.roll(ring, -k, axis=0)[:1]])
    ring, s = resample(ring, 2.0)
    L = s[-1]
    # Interchange where every movement is FORWARD (no U-turn, no bridge needed):
    #   exit  on the WEST leg (runs south; the village junction is to the south-west):
    #         deceleration lane X1 (s=120) -> X2 (s=160), off-ramp X2 -> village junction
    #   entry on the SOUTH leg (runs east; the village junction is to the west):
    #         on-ramp village junction -> M (s=245), acceleration lane M -> T (s=305)
    sX1, sX2, sM, sT = 120.0, 160.0, 245.0, 305.0
    outer = offset_left(ring, -LANE * 1.5)            # centre of the extra (3rd) lane, right side

    def at(arr, sv):
        return arr[min(np.searchsorted(s, sv), len(arr) - 1)]

    def heading(sv):
        d = at(ring, sv + 5.0) - at(ring, sv - 5.0)
        return d / np.linalg.norm(d)
    nodes = {'M': at(ring, sM), 'T': at(ring, sT), 'X1': at(ring, sX1), 'X2': at(ring, sX2)}
    dw, ds = heading(sX2), heading(sM)                # west leg (south), south leg (east)
    off_ramp = rounded([at(outer, sX2), at(outer, sX2) + dw * 15.0, p0 + np.array([12.0, 10.0]), p0],
                       10.0, closed=False)
    on_ramp = rounded([p0, p0 + np.array([14.0, -14.0]), at(outer, sM) - ds * 18.0, at(outer, sM)],
                      10.0, closed=False)

    def sumo(pts):
        return ' '.join(f'{x + off[0]:.2f},{y + off[1]:.2f}' for x, y in pts)
    E = []

    def edge(eid, fr, to, pts, nl, sp, spread='right', name='Ring Motorway'):
        E.append(f'  <edge id="{eid}" from="{fr}" to="{to}" numLanes="{nl}" speed="{sp:.2f}" width="{LANE}" '
                 f'shape="{sumo(pts)}" spreadType="{spread}" name="{name}"/>')
    ring2 = np.vstack([ring, ring[1:]])
    s2 = np.r_[s, s[1:] + L]
    le, le2 = offset_left(ring, LANE), offset_left(ring2, LANE)       # carriageway left edge
    edge('ring_decel', 'X1', 'X2', cut(le, s, sX1, sX2), 3, a.speed)
    edge('ring_mid', 'X2', 'M', cut(le, s, sX2, sM), 2, a.speed)
    edge('ring_accel', 'M', 'T', cut(le, s, sM, sT), 3, a.speed)
    edge('ring_main', 'T', 'X1', cut(le2, s2, sT, L + sX1), 2, a.speed)
    edge('ramp_off', 'X2', a.from_junction, off_ramp, 1, 16.67, spread='center', name='Exit ramp')
    edge('ramp_on', a.from_junction, 'M', on_ramp, 1, 16.67, spread='center', name='Entry ramp')
    incoming = [e.get('id') for e in root.findall('edge') if e.get('to') == a.from_junction and e.get('function') != 'internal']
    outgoing = [e.get('id') for e in root.findall('edge') if e.get('from') == a.from_junction and e.get('function') != 'internal']
    con = ('<connections>\n'
           '  <connection from="ring_main" to="ring_decel" fromLane="0" toLane="0"/>\n'
           '  <connection from="ring_main" to="ring_decel" fromLane="0" toLane="1"/>\n'
           '  <connection from="ring_main" to="ring_decel" fromLane="1" toLane="2"/>\n'
           '  <connection from="ring_decel" to="ramp_off" fromLane="0" toLane="0"/>\n'
           '  <connection from="ring_decel" to="ring_mid" fromLane="1" toLane="0"/>\n'
           '  <connection from="ring_decel" to="ring_mid" fromLane="2" toLane="1"/>\n'
           '  <connection from="ring_mid" to="ring_accel" fromLane="0" toLane="1"/>\n'
           '  <connection from="ring_mid" to="ring_accel" fromLane="1" toLane="2"/>\n'
           '  <connection from="ramp_on" to="ring_accel" fromLane="0" toLane="0"/>\n'
           '  <connection from="ring_accel" to="ring_main" fromLane="0" toLane="0"/>\n'
           '  <connection from="ring_accel" to="ring_main" fromLane="1" toLane="0"/>\n'
           '  <connection from="ring_accel" to="ring_main" fromLane="2" toLane="1"/>\n'
           + ''.join(f'  <connection from="{e}" to="ramp_on" fromLane="0" toLane="0"/>\n' for e in incoming)
           + ''.join(f'  <connection from="ramp_off" to="{e}" fromLane="0" toLane="0"/>\n' for e in outgoing)
           + '</connections>\n')
    nod = '<nodes>\n' + ''.join(
        f'  <node id="{n}" x="{p[0] + off[0]:.2f}" y="{p[1] + off[1]:.2f}" type="priority"/>\n' for n, p in nodes.items()) + '</nodes>\n'
    edg = '<edges>\n' + '\n'.join(E) + '\n</edges>\n'

    for name, body in (('ring.nod.xml', nod), ('ring.edg.xml', edg), ('ring.con.xml', con)):
        open(os.path.join(net_dir, name), 'w').write(body)
    out = net + '.ring.xml'
    subprocess.check_call(['netconvert', '-s', net, '-n', os.path.join(net_dir, 'ring.nod.xml'),
                           '-e', os.path.join(net_dir, 'ring.edg.xml'), '-x', os.path.join(net_dir, 'ring.con.xml'),
                           '-o', out, '--offset.disable-normalization', 'true', '--no-turnarounds', '--no-warnings'])
    os.replace(out, net)
    print(f'SUMO: ring motorway {L:.0f} m (one-way, 2 lanes), acceleration lane {sT - sM:.0f} m, '
          f'deceleration lane {sX2 - sX1:.0f} m, entry/exit ramps, link road')

    text = open(world).read()
    if 'Road.proto"' not in text:
        text = text.replace('\nWorldInfo {', '\n' + ROAD_PROTO + '\nWorldInfo {', 1)
    text = re.sub(r'(?ms)^Road \{\n  translation 0 0 0\.02\n  name "(?:highway|highway link|ring motorway|'
                  r'deceleration lane|acceleration lane|exit ramp|entry ramp|motorway link|motorway link north)".*?^\}\n', '', text)
    text += road_node('ring motorway', ring, 2, 2 * LANE, a.speed, barrier_left=True)
    text += road_node('deceleration lane', cut(outer, s, sX1, sX2 + 15), 1, LANE, a.speed, lines='continuous')
    text += road_node('acceleration lane', cut(outer, s, sM - 18, sT), 1, LANE, a.speed, lines='continuous')
    text += road_node('exit ramp', off_ramp, 1, LANE, 16.67)
    text += road_node('entry ramp', on_ramp, 1, LANE, 16.67)
    open(world, 'w').write(text)
    print(f'Webots: ring motorway + deceleration/acceleration lanes + ramps + link -> {world}')


if __name__ == '__main__':
    main()
