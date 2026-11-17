#!/usr/bin/env python3
"""Make SUMO traffic obey the SAME traffic lights as our car (Webots and SUMO on one schedule).

  python3 tools/sync_signals.py worlds/village.wbt [--replace-crossroads]

Webots' GenericTrafficLight controller runs green -> orange 1.5 s -> red -> orange 1.5 s on a
512 ms tick; with greenTime = redTime = 10 every phase really lasts 10.24 s and every orange
1.536 s (cycle 23.552 s). This tool
  1. (--replace-crossroads) replaces each CrossRoadsTrafficLight (12 lamps, no way to tell which
     lamp faces which road) by one GenericTrafficLight per approach (tools/add_traffic_lights.py),
  2. assigns every GenericTrafficLight to the junction approach it stands at and to an axis
     (approaches ~E-W = axis 0, ~N-S = axis 1); axis 0 starts green, axis 1 red - so crossing
     directions are never green together (the village's original lights all started green),
  3. writes a SUMO signal program for each of those junctions with exactly that schedule
     (axis 0: G 10.24 / y 1.536 / r 10.24 / u 1.536, axis 1 shifted by half a cycle),
  4. sets SumoInterface 'step 128' so 10.24 s and 1.536 s are whole SUMO steps (no drift).
Afterwards worlds/<world>_net/signals.yaml is regenerated.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, ORANGE = 10.24, 1.536


def axis_of(h):
    return int(round(((h % math.pi) / (math.pi / 2)))) % 2


def lane_end_heading(shape):
    pts = [tuple(map(float, p.split(','))) for p in shape.split()]
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return (x1, y1), math.atan2(y1 - y0, x1 - x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--replace-crossroads', action='store_true')
    a = ap.parse_args()
    world = os.path.abspath(a.world)
    net_dir = os.path.realpath(os.path.splitext(world)[0] + '_net')
    net = os.path.join(net_dir, 'sumo.net.xml')
    sys.path.insert(0, HERE)
    from port_world import remove_nodes

    text = open(world).read()
    if a.replace_crossroads and 'CrossRoadsTrafficLight {' in text:
        text, n = remove_nodes(text, 'CrossRoadsTrafficLight')
        open(world, 'w').write(text)
        print(f'removed {n} CrossRoadsTrafficLight; adding one light per approach')
        subprocess.check_call([sys.executable, os.path.join(HERE, 'add_traffic_lights.py'), world])
        text = open(world).read()

    root = ET.parse(net).getroot()
    off = [float(v) for v in root.find('location').get('netOffset').split(',')]
    # incoming road lanes: end point (world frame) -> (junction id, heading)
    ends = []
    for e in root.findall('edge'):
        if e.get('function') == 'internal':
            continue
        for ln in e.findall('lane'):
            (x, y), h = lane_end_heading(ln.get('shape'))
            ends.append((x - off[0], y - off[1], e.get('to'), h))

    # assign lights to approaches; fix startGreen per axis
    junction_axes = {}

    def fix(m):
        block = m.group(0)
        t = re.search(r'translation ([-\d.e]+) ([-\d.e]+)', block)
        lx, ly = float(t.group(1)), float(t.group(2))
        r = re.search(r'rotation 0 0 (-?1) ([-\d.e]+)', block)
        yaw = float(r.group(2)) * float(r.group(1)) if r else 0.0
        app = yaw + math.pi                                 # the approach it faces drives this way
        # the approach this light controls: a lane end within 40 m BEHIND the light that points at it
        def score(e):
            dx, dy = lx - e[0], ly - e[1]
            along = dx * math.cos(app) + dy * math.sin(app)
            dh = abs(math.atan2(math.sin(e[3] - app), math.cos(e[3] - app)))
            return (dh < math.radians(35) and 0.0 < along < 40.0, -math.hypot(dx, dy))
        cand = max(ends, key=score)
        if not score(cand)[0]:
            cand = min(ends, key=lambda e: math.hypot(e[0] - lx, e[1] - ly))   # near-side / old lights
            if math.hypot(cand[0] - lx, cand[1] - ly) > 15.0:
                return block                               # not at a junction approach
        ax = axis_of(cand[3])
        junction_axes.setdefault(cand[2], set()).add(ax)
        block = re.sub(r'\n  startGreen [A-Z]+', '', block)
        return block.replace('\n}', f'\n  startGreen {"TRUE" if ax == 0 else "FALSE"}\n}}', 1)
    text = re.sub(r'(?m)^GenericTrafficLight \{\n(?:  [^\n]*\n)*?\}', fix, text)
    text = re.sub(r'(SumoInterface \{\n(?:  [^\n]*\n)*?)\}', lambda m: re.sub(r'  step \d+\n', '', m.group(1)) + '  step 128\n}', text)
    open(world, 'w').write(text)
    tls = sorted(junction_axes)
    print(f'{sum(len(v) for v in junction_axes.values())} approach axes at {len(tls)} junctions; startGreen set per axis')

    # SUMO: signals at these junctions, then our phase program
    if tls:
        have = {j.get('id') for j in ET.parse(net).getroot().findall('junction') if j.get('type') == 'traffic_light'}
        new = [j for j in tls if j not in have]       # netconvert crashes re-setting existing signals
        if new:
            out = net + '.tmp.xml'
            subprocess.check_call(['netconvert', '-s', net, '-o', out, '--tls.set', ','.join(new),
                                   '--tls.default-type', 'static', '--no-warnings'])
            os.replace(out, net)
        root = ET.parse(net).getroot()
        heads = {}
        for e in root.findall('edge'):
            if e.get('function') != 'internal':
                ln = e.find('lane')
                heads[e.get('id')] = lane_end_heading(ln.get('shape'))[1]
        links = {}
        for c in root.findall('connection'):
            if c.get('tl') in tls and c.get('linkIndex') is not None:
                links.setdefault(c.get('tl'), {})[int(c.get('linkIndex'))] = axis_of(heads.get(c.get('from'), 0.0))
        txt = open(net).read()
        for j in tls:
            idx = links.get(j, {})
            n = max(idx) + 1 if idx else 0

            def state(s0, s1):
                return ''.join((s0 if idx.get(i, 0) == 0 else s1) for i in range(n))
            phases = [(GREEN, state('G', 'r')), (ORANGE, state('y', 'u')),
                      (GREEN, state('r', 'G')), (ORANGE, state('u', 'y'))]
            logic = (f'    <tlLogic id="{j}" type="static" programID="0" offset="0">\n' +
                     ''.join(f'        <phase duration="{d}" state="{s}"/>\n' for d, s in phases) +
                     '    </tlLogic>\n')
            txt, k = re.subn(rf'    <tlLogic id="{re.escape(j)}"[^>]*>.*?</tlLogic>\n', logic, txt, flags=re.S)
            if k == 0:
                txt = txt.replace('    <junction ', logic + '    <junction ', 1)
        open(net, 'w').write(txt)
        print(f'SUMO signal programs written for {len(tls)} junctions (cycle {2 * (GREEN + ORANGE):.3f} s)')
    subprocess.check_call([sys.executable, os.path.join(HERE, 'extract_signals.py'), world])


if __name__ == '__main__':
    main()
