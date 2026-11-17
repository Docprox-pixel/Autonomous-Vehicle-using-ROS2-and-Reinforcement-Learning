#!/usr/bin/env python3
"""Rebuild the SUMO network CONSISTENTLY: exact lane lines + junctions fitted to them + additions.

  python3 tools/rebuild_network.py worlds/village_net [--base sumo.net.xml.before_connect]

Why: lanes had been corrected (spline curves, offsets) inside sumo.net.xml without regenerating
the junctions, so many junction links no longer met their lanes (gaps up to 7 m) - the route
jumped at junctions and the turns looked wrong. This tool
  1. exports the base network to plain XML,
  2. gives every single-lane edge its exact corrected lane line as the EDGE shape (spread centre),
  3. drops the old junction outlines and moves each junction onto the centre of its lane ends,
     so netconvert recomputes every junction to fit,
  4. merges the additions kept in the net directory (if present):
       connect.edg/.con.xml (car-park entries/exits), ends_join.nod/.con.xml (roads that touched),
       ends.edg/.con.xml (new connecting roads), tc.nod/.edg/.con.xml (cul-de-sac loops),
       restrict.con.xml (<delete> turn restrictions),
  5. builds sumo.net.xml (backup .before_rebuild) and reports the junction fit.
"""
import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


def merge(dst, src, tag):
    """Append the <tag> children of src into the root of dst (both plain XML files)."""
    if not os.path.exists(src):
        return 0
    t, s = ET.parse(dst), ET.parse(src)
    ids = {e.get('id') for e in t.getroot().findall(tag)}
    n = 0
    for e in s.getroot():
        if e.tag != tag:
            continue
        if e.get('id') is not None and e.get('id') in ids:
            continue
        if tag == 'connection':
            # lane-level like the exported ones (edge-level ones were ignored next to them)
            e.set('fromLane', e.get('fromLane', '0'))
            e.set('toLane', e.get('toLane', '0'))
        t.getroot().append(e)
        n += 1
    t.write(dst)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('net_dir')
    ap.add_argument('--base', default='sumo.net.xml.before_connect')
    a = ap.parse_args()
    d = a.net_dir
    base = os.path.join(d, a.base)
    out = os.path.join(d, 'sumo.net.xml')
    pl = os.path.join(d, 'rebuild')
    shutil.rmtree(pl, ignore_errors=True)
    os.makedirs(pl)
    subprocess.check_call(['netconvert', '-s', base, '--plain-output-prefix', os.path.join(pl, 'p'),
                           '--offset.disable-normalization', 'true', '--no-warnings'], stdout=subprocess.DEVNULL)
    net = ET.parse(base).getroot()
    good = {ln.get('id'): ln.get('shape') for e in net.findall('edge') if e.get('function') != 'internal'
            for ln in e.findall('lane')}
    edg = os.path.join(pl, 'p.edg.xml')
    t = ET.parse(edg)
    for e in t.getroot().findall('edge'):
        lid = f"{e.get('id')}_0"
        if int(e.get('numLanes', '1')) == 1 and lid in good:
            for l in e.findall('lane'):
                e.remove(l)
            e.set('shape', good[lid])
            e.set('spreadType', 'center')
    t.write(edg)
    # additions
    nod, con = os.path.join(pl, 'p.nod.xml'), os.path.join(pl, 'p.con.xml')
    added = {}
    for pre in ('connect', 'ends', 'tc'):
        added[pre + '.edg'] = merge(edg, os.path.join(d, pre + '.edg.xml'), 'edge')
        added[pre + '.con'] = merge(con, os.path.join(d, pre + '.con.xml'), 'connection')
    for f in ('tc.nod.xml', 'ends_join.nod.xml'):
        added[f] = merge(nod, os.path.join(d, f), 'node') + merge(nod, os.path.join(d, f), 'join')
    added['ends_join.con'] = merge(con, os.path.join(d, 'ends_join.con.xml'), 'connection')
    # turn restrictions (e.g. hairpin movements at an acute Y-junction): <delete from= to=/>
    rs = os.path.join(d, 'restrict.con.xml')
    if os.path.exists(rs):
        added['restrict.con'] = merge(con, rs, 'connection')
        bans = {(x.get('from'), x.get('to')) for x in ET.parse(rs).getroot().findall('delete')}
        tc = ET.parse(con)
        for x in list(tc.getroot()):
            if x.tag == 'connection' and (x.get('from'), x.get('to')) in bans:
                tc.getroot().remove(x)
        for f_, t_ in bans:
            ET.SubElement(tc.getroot(), 'delete', {'from': f_, 'to': t_})
        tc.write(con)
        added['restrict'] = len(bans)
    # junctions: no old outline, centred on their lane ends
    tn = ET.parse(nod)
    N = {n.get('id'): n for n in tn.getroot().findall('node')}
    ends = {k: [] for k in N}
    for e in ET.parse(edg).getroot().findall('edge'):
        sh = e.get('shape')
        if sh and e.get('from') in ends and e.get('to') in ends:
            P = [tuple(map(float, q.split(','))) for q in sh.split()]
            ends[e.get('from')].append(P[0])
            ends[e.get('to')].append(P[-1])
    # junctions that new roads attach to are no longer dead ends (netconvert allows no turns at a
    # 'dead_end' junction: the new connections were silently dropped)
    touched = set()
    for f in ('connect.edg.xml', 'ends.edg.xml', 'tc.edg.xml'):
        fp = os.path.join(d, f)
        if os.path.exists(fp):
            for e in ET.parse(fp).getroot().findall('edge'):
                touched |= {e.get('from'), e.get('to')}
    for k, n in N.items():
        if n.get('type') == 'dead_end' and k in touched:
            n.set('type', 'priority')
        if n.get('shape') is not None:
            del n.attrib['shape']
        if ends[k]:
            c = np.mean(ends[k], axis=0)
            if np.hypot(c[0] - float(n.get('x')), c[1] - float(n.get('y'))) > 1.0:
                n.set('x', f'{c[0]:.2f}')
                n.set('y', f'{c[1]:.2f}')
    tn.write(nod)
    if os.path.exists(out):
        shutil.copy(out, out + '.before_rebuild')
    subprocess.check_call(['netconvert', '-n', nod, '-e', edg, '-x', con, '-i', os.path.join(pl, 'p.tll.xml'),
                           '-o', out, '--offset.disable-normalization', 'true',
                           '--no-turnarounds.except-deadend', '--no-warnings'])
    # report
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from auto_car_core.lanemap import LaneMap
    m = LaneMap(out)
    r = ET.parse(out).getroot()
    rev = lambda e: e[1:] if e.startswith('-') else '-' + e
    gaps = []
    for c in r.findall('connection'):
        if c.get('from').startswith(':') or not c.get('via') or c.get('to') == rev(c.get('from')):
            continue
        A, B, V = (f"{c.get('from')}_{c.get('fromLane')}", f"{c.get('to')}_{c.get('toLane')}", c.get('via'))
        if A in m.lanes and B in m.lanes and V in m.lanes:
            gaps.append(max(np.hypot(*(m.lanes[A].pts[-1] - m.lanes[V].pts[0])),
                            np.hypot(*(m.lanes[V].pts[-1] - m.lanes[B].pts[0]))))
    g = np.array(gaps) if gaps else np.zeros(1)
    print(f'additions merged: { {k: v for k, v in added.items() if v} }')
    print(f'rebuilt {out}: {len(g)} turning links, worst lane-to-junction gap {g.max():.2f} m '
          f'({(g > 0.3).sum()} over 0.3 m), traffic lights {len(r.findall("tlLogic"))}')


if __name__ == '__main__':
    main()
