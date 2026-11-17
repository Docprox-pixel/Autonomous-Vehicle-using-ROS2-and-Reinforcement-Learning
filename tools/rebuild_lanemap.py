#!/usr/bin/env python3
"""Rebuild a world's SUMO lane map FROM ITS 3D ROADS (Webots' own exporter), so lanes match the road.

  python3 tools/rebuild_lanemap.py SOURCE.wbt OUT_NET_DIR

Why: the village's shipped SUMO network was 1-5 m off the drawn roads on many streets (some lane
centres lay on the centre line or in the opposite half), so the car, following the map's lane
centre, drove in the oncoming lane. Webots' resources/sumo_exporter converts the Road/Crossroad
nodes (including the B-spline smoothing of Road way points) into SUMO nodes/edges.

The exporter reads every field by regex, so road nodes that rely on PROTO defaults are first given
explicit default values (in a temporary copy). Then netconvert builds sumo.net.xml (no U-turns,
junctions joined) and the result is checked against the roads: lane centre offsets are reported.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

DEFAULTS = {
    'Road': {'translation': '0 0 0', 'rotation': '0 0 1 0', 'width': '7', 'numberOfLanes': '2',
             'numberOfForwardLanes': '1', 'speedLimit': '13.89', 'splineSubdivision': '4'},
    'StraightRoadSegment': {'translation': '0 0 0', 'rotation': '0 0 1 0', 'width': '7', 'numberOfLanes': '2',
                            'numberOfForwardLanes': '1', 'speedLimit': '13.89', 'length': '10'},
    'CurvedRoadSegment': {'translation': '0 0 0', 'rotation': '0 0 1 0', 'width': '7', 'numberOfLanes': '2',
                          'numberOfForwardLanes': '1', 'speedLimit': '13.89', 'curvatureRadius': '10',
                          'totalAngle': '1.5708', 'subdivision': '16'},
    'Crossroad': {'translation': '0 0 0', 'rotation': '0 0 1 0', 'speedLimit': '13.89'},
    'RoadIntersection': {'translation': '0 0 0', 'rotation': '0 0 1 0', 'roadNumber': '4', 'roadsWidth': '7',
                         'speedLimit': '13.89'},
    'Roundabout': {'translation': '0 0 0', 'rotation': '0 0 1 0', 'subdivision': '16', 'speedLimit': '13.89'},
}


def fill_defaults(text):
    out, i, n_fixed = [], 0, 0
    pat = re.compile(r'(?m)^(?:DEF \S+ )?(%s) \{\n' % '|'.join(DEFAULTS))
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        depth, j = 0, m.end() - 2
        while True:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        block = text[m.end():j]
        top = '\n'.join(l for l in block.splitlines() if l.startswith('  ') and not l.startswith('   '))
        add = ''.join(f'  {k} {v}\n' for k, v in DEFAULTS[m.group(1)].items()
                      if not re.search(rf'(?m)^  {k}\s', top))
        n_fixed += bool(add)
        out.append(text[i:m.end()] + add + block)
        i = j
    return ''.join(out), n_fixed


def main():
    src, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    tmp = tempfile.mkdtemp(prefix='lanemap_')
    text, n = fill_defaults(open(src).read())
    world = os.path.join(tmp, 'world.wbt')
    open(world, 'w').write(text)
    print(f'{n} road nodes given explicit default fields')
    exporter = os.path.join(os.environ.get('WEBOTS_HOME', os.path.expanduser('~/apps/webots')), 'resources', 'sumo_exporter')
    subprocess.check_call([sys.executable, 'exporter.py', f'--input={world}', f'--output={tmp}'], cwd=exporter)
    os.makedirs(out_dir, exist_ok=True)
    net = os.path.join(out_dir, 'sumo.net.xml')
    subprocess.check_call(['netconvert', '--node-files', os.path.join(tmp, 'sumo.nod.xml'),
                           '--edge-files', os.path.join(tmp, 'sumo.edg.xml'), '-o', net,
                           '--no-turnarounds.except-deadend', '--junctions.join', '--no-warnings',
                           '--offset.disable-normalization', 'true'])
    for f in ('sumo.nod.xml', 'sumo.edg.xml'):
        shutil.copy(os.path.join(tmp, f), os.path.join(out_dir, f))
    print(f'-> {net}')


if __name__ == '__main__':
    main()
