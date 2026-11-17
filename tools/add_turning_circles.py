#!/usr/bin/env python3
"""Turn the remaining dead ends into proper CUL-DE-SACS (turning circles), as real streets end.

  python3 tools/add_turning_circles.py worlds/village.wbt [--path-radius 7.5] [--bulb 11] [--check]

For every junction where arriving traffic can only make an (impossible, zero-radius) U-turn:
  * world: an asphalt bulb of --bulb m radius just beyond the road end,
  * SUMO network: a one-way loop around the bulb on a --path-radius circle (the car's minimum
    turning radius is 6.2 m), driven anticlockwise (right-hand traffic: keep right, the centre on
    the left), as two edges  end -> far point -> end,  connected
        arriving lane -> loop -> loop -> departing lane,
so the planner routes a real turnaround instead of a U-turn on the spot. Existing lanes keep their
exact shapes.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--path-radius', type=float, default=7.5)
    ap.add_argument('--bulb', type=float, default=11.0)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    net_dir = a.world.replace('.wbt', '_net')
    net = os.path.join(net_dir, 'sumo.net.xml')
    root = ET.parse(net).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    edges = {e.get('id'): e for e in root.findall('edge') if e.get('function') != 'internal'}
    J = {j.get('id'): np.array([float(j.get('x')), float(j.get('y'))]) - off
         for j in root.findall('junction') if j.get('type') != 'internal'}
    lane = {ln.get('id'): np.array([[float(v) for v in q.split(',')[:2]] for q in ln.get('shape').split()]) - off
            for e in root.findall('edge') for ln in e.findall('lane')}
    rev = lambda e: e[1:] if e.startswith('-') else '-' + e
    conns = collections.defaultdict(set)
    for c in root.findall('connection'):
        if c.get('from') in edges and c.get('to') in edges:
            conns[c.get('from')].add(c.get('to'))
    nodes, new_edges, con_lines, bulbs = [], [], [], []
    for jid, p in J.items():
        ins = [e for e in edges if edges[e].get('to') == jid]
        if not ins or [t for e in ins for t in conns[e] if t != rev(e)]:
            continue
        e_in = ins[0]
        outs = [e for e in edges if edges[e].get('from') == jid]
        P = lane[e_in + '_0']
        d = P[-1] - P[max(len(P) - 4, 0)]
        d /= np.hypot(*d)
        n = np.array([-d[1], d[0]])                       # left
        Rc = a.path_radius
        O = p + (Rc + 1.0) * d                            # bulb centre just beyond the end
        B = O + Rc * d                                    # far point of the loop
        # loop: from the arriving lane (right side) round the far side anticlockwise ... to the
        # departing lane (left side): angles measured around O
        th_in = math.atan2(*(-n)[::-1])                   # right side point of the circle
        start = O + Rc * np.array([math.cos(th_in), math.sin(th_in)])
        arc = [O + Rc * np.array([math.cos(th_in + u), math.sin(th_in + u)]) for u in np.linspace(0, 2 * math.pi * 0.5, 20)]
        arc2 = [O + Rc * np.array([math.cos(th_in + math.pi + u), math.sin(th_in + math.pi + u)]) for u in np.linspace(0, math.pi, 20)]
        P_in_end = P[-1]
        seg1 = np.vstack([P_in_end, start, *arc])          # end -> far point (right half)
        dep = lane[outs[0] + '_0'][0] if outs else p + 1.75 * n
        seg2 = np.vstack([*arc2, dep])                    # far point -> end (left half)
        bid = f'tc_{jid}'
        nodes.append(f'  <node id="{bid}" x="{B[0] + off[0]:.2f}" y="{B[1] + off[1]:.2f}"/>')
        sh = lambda S: ' '.join(f'{x + off[0]:.2f},{y + off[1]:.2f}' for x, y in S)
        new_edges.append(f'  <edge id="{bid}_a" from="{jid}" to="{bid}" numLanes="1" speed="4.0" width="3.5" shape="{sh(seg1)}"/>')
        new_edges.append(f'  <edge id="{bid}_b" from="{bid}" to="{jid}" numLanes="1" speed="4.0" width="3.5" shape="{sh(seg2)}"/>')
        con_lines.append(f'  <connection from="{e_in}" to="{bid}_a"/>')
        con_lines.append(f'  <connection from="{bid}_a" to="{bid}_b"/>')
        for e_out in outs:
            con_lines.append(f'  <connection from="{bid}_b" to="{e_out}"/>')
        bulbs.append((jid, O))
        print(f'{jid} (end of road {e_in.lstrip("-")}): turning circle, loop radius {Rc} m, bulb radius {a.bulb} m at ({O[0]:.1f},{O[1]:.1f})')
    if a.check or not bulbs:
        return
    open(os.path.join(net_dir, 'tc.nod.xml'), 'w').write('<nodes>\n' + '\n'.join(nodes) + '\n</nodes>\n')
    open(os.path.join(net_dir, 'tc.edg.xml'), 'w').write('<edges>\n' + '\n'.join(new_edges) + '\n</edges>\n')
    open(os.path.join(net_dir, 'tc.con.xml'), 'w').write('<connections>\n' + '\n'.join(con_lines) + '\n</connections>\n')
    prev = net + '.prev'
    shutil.copy(net, prev)
    tmp = net + '.tmp.xml'
    subprocess.check_call(['netconvert', '-s', prev, '-n', os.path.join(net_dir, 'tc.nod.xml'),
                           '-e', os.path.join(net_dir, 'tc.edg.xml'), '-x', os.path.join(net_dir, 'tc.con.xml'),
                           '-o', tmp, '--no-warnings', '--offset.disable-normalization', 'true',
                           '--no-turnarounds.except-deadend'])
    old = ET.parse(prev).getroot()
    shapes = {ln.get('id'): ln.get('shape') for e in old.findall('edge') for ln in e.findall('lane')}
    eshape = {e.get('id'): e.get('shape') for e in old.findall('edge')}
    tree = ET.parse(tmp)
    for e in tree.getroot().findall('edge'):
        if e.get('function') == 'internal':
            continue
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
    text = open(a.world).read()
    shutil.copy(a.world, a.world + '.prev')
    add = []
    for jid, O in bulbs:
        if f'"turning circle({jid})"' in text:
            continue
        add.append(f'Solid {{\n  translation {O[0]:.3f} {O[1]:.3f} 0.004\n  name "turning circle({jid})"\n  children [\n'
                   f'    Shape {{\n      appearance Asphalt {{\n      }}\n      geometry Cylinder {{\n'
                   f'        height 0.008\n        radius {a.bulb}\n        subdivision 48\n      }}\n    }}\n  ]\n}}\n')
    open(a.world, 'w').write(text.rstrip('\n') + '\n' + ''.join(add))
    print(f'{len(bulbs)} turning circles added -> {net}, {a.world}')


if __name__ == '__main__':
    main()
