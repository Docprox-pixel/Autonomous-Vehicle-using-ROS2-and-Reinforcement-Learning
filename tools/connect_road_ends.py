#!/usr/bin/env python3
"""Connect road ends so no road simply stops (a barrier / red X with sand behind it).

  python3 tools/connect_road_ends.py worlds/village.wbt [--check] [--reach 75] [--rmin 15]

Finds every junction where traffic arriving can only U-turn, then:
  * two such ends at the SAME spot (roads that touch but were never joined): short links both ways
  * an end with another end or a junction AHEAD (within --reach m, within 100 deg of the road's
    direction): a new two-way road (7 m, 2 lanes) along a smooth curve - tangent to the road at the
    end, entering the target along its own road direction - accepted only if its tightest radius
    is >= --rmin and it keeps 6 m from every building.
The new roads are drawn in the world (Road nodes) and merged into the SUMO network with all turn
connections; the existing lanes keep their exact (spline-corrected) shapes.
Ends with no acceptable target are reported (left for a manual design).
"""
import argparse
import collections
import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

BUILDING = re.compile(r'(?ms)^(?:DEF \S+ )?([A-Z][A-Za-z0-9]*(?:House|Building|Tower|Hospital|Church|Barn|'
                      r'Museum|School|Hotel|Mall|Shop|Restaurant|Store|Station|Factory|Warehouse|Silo|Hall|'
                      r'Office|Apartment|Residential|UnderConstruction|Bungalow|Pub|Cafe)[A-Za-z0-9]*) \{\n(.*?)^\}')


def bezier(p0, h0, p3, h3, n=60, f=0.4, f1=None):
    d = float(np.hypot(*(p3 - p0)))
    f1 = f if f1 is None else f1
    c1 = p0 + f * d * np.array([math.cos(h0), math.sin(h0)])
    c2 = p3 - f1 * d * np.array([math.cos(h3), math.sin(h3)])
    u = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * p3


def min_radius(C):
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
    t = np.gradient(C, axis=0)
    h = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    k = np.abs(np.gradient(h, s))
    return 1.0 / max(float(k.max()), 1e-6)


def offset(C, d):
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
    return C + d * np.c_[-t[:, 1], t[:, 0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--reach', type=float, default=75.0)
    ap.add_argument('--rmin', type=float, default=15.0)
    ap.add_argument('--plan', help='JSON {end: {target, f0, f1}}: connect exactly these (after moving buildings)')
    a = ap.parse_args()
    net_dir = a.world.replace('.wbt', '_net')
    net = os.path.join(net_dir, 'sumo.net.xml')
    root = ET.parse(net).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    edges = {e.get('id'): e for e in root.findall('edge') if e.get('function') != 'internal'}
    J = {j.get('id'): np.array([float(j.get('x')), float(j.get('y'))]) - off
         for j in root.findall('junction') if j.get('type') != 'internal'}
    lane_pts = {ln.get('id'): np.array([[float(v) for v in q.split(',')[:2]] for q in ln.get('shape').split()]) - off
                for e in root.findall('edge') for ln in e.findall('lane')}
    rev = lambda eid: eid[1:] if eid.startswith('-') else '-' + eid
    conns = collections.defaultdict(set)
    for c in root.findall('connection'):
        if c.get('from') in edges and c.get('to') in edges:
            conns[c.get('from')].add(c.get('to'))
    ends = {}
    for jid in J:
        ins = [e for e in edges if edges[e].get('to') == jid]
        if ins and not [t for e in ins for t in conns[e] if t != rev(e)]:
            e_in = ins[0]
            P = lane_pts[e_in + '_0']
            k = max(len(P) - 4, 0)
            d = P[-1] - P[k]
            ends[jid] = {'p': J[jid], 'h': math.atan2(d[1], d[0]), 'in': ins, 'out': [rev(e) for e in ins if rev(e) in edges]}
    text = open(a.world).read()
    buildings = []
    for m in BUILDING.finditer(text):
        tr = re.search(r'(?m)^  translation ([-\d.e]+) ([-\d.e]+)', m.group(2))
        if tr:
            buildings.append(np.array([float(tr.group(1)), float(tr.group(2))]))
    plans, used = [], set()
    # 1) ends at the same spot
    ids = list(ends)
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            if x in used or y in used:
                continue
            if np.hypot(*(ends[x]['p'] - ends[y]['p'])) < 1.0:
                plans.append(('join', x, y, None))
                used |= {x, y}
    # 2a) explicit plans (designed by hand, e.g. after moving a building out of the way)
    if a.plan:
        import json
        for x, pl in json.load(open(a.plan)).items():
            if x not in ends or x in used:
                continue
            y = pl['target']
            q = J[y]
            h_in = ends[y]['h'] + math.pi if y in ends else math.atan2(*(q - ends[x]['p'])[::-1])
            C = bezier(ends[x]['p'], ends[x]['h'], q, h_in, n=80, f=pl['f0'], f1=pl['f1'])
            plans.append(('road', x, y, C))
            used.add(x)
            if y in ends:
                used.add(y)
            print(f'{x}: new road to {y} (planned, tightest radius {min_radius(C):.1f} m)')
    # 2b) a new road to a target ahead
    for x in ids:
        if x in used:
            continue
        E = ends[x]
        cands = []
        targets = [(k, J[k], None) for k in J if k != x] + [(k, ends[k]['p'], ends[k]['h']) for k in ends if k != x and k not in used]
        for k, q, hk in targets:
            v = q - E['p']
            dist = float(np.hypot(*v))
            if dist < 8.0 or dist > a.reach:
                continue
            ang = math.atan2(v[1], v[0])
            if abs(math.atan2(math.sin(ang - E['h']), math.cos(ang - E['h']))) > math.radians(100):
                continue
            h_in = (hk + math.pi) if hk is not None else ang                # into the target
            C = bezier(E['p'], E['h'], q, h_in)
            R = min_radius(C)
            if R < a.rmin:
                continue
            if buildings and min(float(np.min(np.hypot(*(C - b).T))) for b in buildings) < 6.0:
                continue
            cands.append((dist + (0.0 if hk is not None else 5.0), k, C, R))
        if cands:
            dist, k, C, R = min(cands, key=lambda c: c[0])
            plans.append(('road', x, k, C))
            used.add(x)
            if k in ends:
                used.add(k)
            print(f'{x}: new road to {k} ({dist:.0f} m, tightest radius {R:.0f} m)')
        else:
            print(f'{x}: no acceptable target ahead within {a.reach} m (needs a manual design)')
    for kind, x, y, _ in plans:
        if kind == 'join':
            print(f'{x} <-> {y}: roads touch, joined with short links')
    if a.check or not plans:
        return
    # ---- SUMO plain edges + connections
    edg = os.path.join(net_dir, 'ends.edg.xml')
    con = os.path.join(net_dir, 'ends.con.xml')
    new_edges, con_lines = [], []
    for n, (kind, x, y, C) in enumerate(plans):
        if kind == 'join':
            for fr, to in ((x, y), (y, x)):
                eid = f'join_{fr}_{to}'
                new_edges.append(f'  <edge id="{eid}" from="{fr}" to="{to}" numLanes="1" speed="13.89" width="3.5"/>')
                for e_in in [e for e in edges if edges[e].get('to') == fr]:
                    con_lines.append(f'  <connection from="{e_in}" to="{eid}"/>')
                for e_out in [e for e in edges if edges[e].get('from') == to]:
                    con_lines.append(f'  <connection from="{eid}" to="{e_out}"/>')
            continue
        eid = f'conn{n}'
        for fr, to, CC in ((x, y, C), (y, x, C[::-1])):
            e2 = eid if fr == x else '-' + eid
            shape = ' '.join(f'{px + off[0]:.2f},{py + off[1]:.2f}' for px, py in offset(CC, -1.75))
            new_edges.append(f'  <edge id="{e2}" from="{fr}" to="{to}" numLanes="1" speed="13.89" width="3.5" '
                             f'shape="{shape}" spreadType="center"/>')
            for e_in in [e for e in edges if edges[e].get('to') == fr]:
                con_lines.append(f'  <connection from="{e_in}" to="{e2}"/>')
            for e_out in [e for e in edges if edges[e].get('from') == to]:
                con_lines.append(f'  <connection from="{e2}" to="{e_out}"/>')
    open(edg, 'w').write('<edges>\n' + '\n'.join(new_edges) + '\n</edges>\n')
    open(con, 'w').write('<connections>\n' + '\n'.join(con_lines) + '\n</connections>\n')
    if not os.path.exists(net + '.before_ends'):
        shutil.copy(net, net + '.before_ends')
    prev = net + '.prev'                                   # always build on the CURRENT network
    shutil.copy(net, prev)
    tmp = net + '.tmp.xml'
    subprocess.check_call(['netconvert', '-s', prev, '-e', edg, '-x', con, '-o', tmp, '--no-warnings',
                           '--offset.disable-normalization', 'true', '--no-turnarounds.except-deadend'])
    # restore every existing lane's exact shape (netconvert straightens them)
    old = ET.parse(prev).getroot()
    shapes = {ln.get('id'): ln.get('shape') for e in old.findall('edge') for ln in e.findall('lane')}
    eshape = {e.get('id'): e.get('shape') for e in old.findall('edge')}
    tree = ET.parse(tmp)
    for e in tree.getroot().findall('edge'):
        for ln in e.findall('lane'):
            if ln.get('id') in shapes:
                ln.set('shape', shapes[ln.get('id')])
        if e.get('id') in eshape:
            if eshape[e.get('id')] is None and e.get('shape') is not None:
                del e.attrib['shape']
            elif eshape[e.get('id')] is not None:
                e.set('shape', eshape[e.get('id')])
    tree.write(net, xml_declaration=True, encoding='UTF-8')
    os.remove(tmp)
    # ---- draw the new roads
    add = []
    for n, (kind, x, y, C) in enumerate(plans):
        if kind != 'road':
            continue
        wp = '\n'.join(f'    {px:.2f} {py:.2f} 0' for px, py in C[::3])
        add.append(f'Road {{\n  translation 0 0 0.005\n  name "road(conn{n})"\n  id "conn{n}"\n  width 7\n'
                   f'  numberOfLanes 2\n  numberOfForwardLanes 1\n  speedLimit 13.89\n'
                   f'  lines [\n    RoadLine {{\n      type "dashed"\n    }}\n  ]\n  rightBorder FALSE\n  leftBorder FALSE\n'
                   f'  wayPoints [\n{wp}\n  ]\n  splineSubdivision 0\n}}\n')
    if add:
        shutil.copy(a.world, a.world + '.prev')
        open(a.world, 'w').write(text.rstrip('\n') + '\n' + ''.join(add))
    print(f'{len(plans)} connections applied: {net}, {len(add)} roads drawn in {a.world}')


if __name__ == '__main__':
    main()
