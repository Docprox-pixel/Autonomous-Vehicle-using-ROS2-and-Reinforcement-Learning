#!/usr/bin/env python3
"""Ground footprint of a Webots PROTO from its own boundingObject (the collision geometry).

  python3 tools/proto_footprint.py BungalowStyleHouse [Museum ...]

Reads the PROTO from the Webots asset cache (~/.cache/Cyberbotics/Webots/assets) or $WEBOTS_HOME,
walks the boundingObject tree (Pose/Transform translation + axis-angle rotation, nested), takes the
corners of every Box and the rim of every Cylinder / Capsule, keeps the parts that reach above
0.3 m (not floors, ramps, kerbs) and returns the convex hull in the PROTO's own frame (x, y).
Template expressions are evaluated with the PROTO's default field values (scale 1 etc.).
"""
import glob
import math
import os
import re
import sys

import numpy as np

_CACHE = {}


def _find(proto):
    dirs = [os.path.expanduser('~/.cache/Cyberbotics/Webots/assets'),
            os.path.join(os.environ.get('WEBOTS_HOME', os.path.expanduser('~/apps/webots')), 'projects')]
    for d in dirs:
        for f in (glob.glob(os.path.join(d, '*')) if d.endswith('assets') else
                  glob.glob(os.path.join(d, '**', proto + '.proto'), recursive=True)):
            try:
                with open(f, errors='ignore') as fh:
                    head = fh.read()
            except (IsADirectoryError, OSError):
                continue
            if re.search(rf'(?m)^PROTO {proto}\s*\[', head):
                return head
    return None


def _defaults(text):
    env = {}
    for m in re.finditer(r'(?m)^\s*field\s+SF(?:Float|Int32)\s+(\w+)\s+([-\d.e]+)', text):
        env[m.group(1)] = float(m.group(2))
    return env


def _expand(text, env):
    """Evaluate %<= expr >% with the defaults; drop %< statements >% (all branches kept)."""
    def ev(m):
        e = m.group(1)
        e = re.sub(r'fields\.(\w+)\.value(?:\.\w+)?', lambda q: repr(env.get(q.group(1), 1.0)), e)
        e = re.sub(r'\b([A-Za-z_]\w*)\b', lambda q: repr(env[q.group(1)]) if q.group(1) in env else q.group(1), e)
        e = e.replace('Math.', 'math.')
        try:
            return repr(float(eval(e, {'math': math})))
        except Exception:
            return '0'
    text = re.sub(r'%<=(.*?)>%', ev, text, flags=re.S)
    return re.sub(r'%<(?!=).*?>%', '', text, flags=re.S)


def _rot(axis_angle):
    x, y, z, a = axis_angle
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _block(text, start):
    depth, j = 0, text.index('{', start)
    k = j
    while True:
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[j + 1:k], k + 1
        k += 1


def _walk(text, R, T, pts):
    i = 0
    while True:
        m = re.search(r'\b(Pose|Transform|Box|Cylinder|Capsule|Group|Shape|Solid)\s*\{', text[i:])
        if not m:
            return
        kind = m.group(1)
        body, end = _block(text, i + m.start())
        i += m.start() + (end - (i + m.start()))
        own = re.sub(r'\{.*\}', '', body, flags=re.S)             # fields of this node only
        if kind in ('Pose', 'Transform', 'Solid'):
            t = re.search(r'translation\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', own)
            r = re.search(r'rotation\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', own)
            t = np.array([float(v) for v in t.groups()]) if t else np.zeros(3)
            r = _rot([float(v) for v in r.groups()]) if r else np.eye(3)
            _walk(body, R @ r, T + R @ t, pts)
        elif kind == 'Box':
            s = re.search(r'size\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', body)
            if s:
                h = np.array([float(v) for v in s.groups()]) / 2
                for cx in (-1, 1):
                    for cy in (-1, 1):
                        for cz in (-1, 1):
                            pts.append(T + R @ (h * [cx, cy, cz]))
        elif kind in ('Cylinder', 'Capsule'):
            rr = re.search(r'radius\s+([-\d.e]+)', body)
            hh = re.search(r'height\s+([-\d.e]+)', body)
            if rr:
                r_, h_ = float(rr.group(1)), float(hh.group(1)) if hh else 2.0
                for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                    for z in (-h_ / 2, h_ / 2):
                        pts.append(T + R @ np.array([r_ * math.cos(a), r_ * math.sin(a), z]))  # Webots R2025a: z axis
        else:
            _walk(body, R, T, pts)


def hull(P):
    P = sorted(set(map(tuple, np.round(P, 3))))
    if len(P) < 3:
        return P
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def footprint(proto):
    """Convex hull [(x, y), ...] of the PROTO's boundingObject in its own frame, or None."""
    if proto in _CACHE:
        return _CACHE[proto]
    text = _find(proto)
    res = None
    if text:
        text = _expand(text, _defaults(text))
        m = re.search(r'boundingObject\s+(?:DEF\s+\S+\s+)?\w+\s*\{', text)
        if m:
            _, end = _block(text, m.start())
            body = text[m.start() + len('boundingObject'):end]     # the node itself, with its own pose
            env = _defaults(text)
            body = _expand(body, env)
            # the boundingObject is in the frame of the Solid that owns it (old y-up models rotate it)
            R0, T0 = np.eye(3), np.zeros(3)
            depth, k = 0, m.start()
            while k > 0:
                k -= 1
                if text[k] == '}':
                    depth += 1
                elif text[k] == '{':
                    if depth == 0:
                        break
                    depth -= 1
            own = re.sub(r'\{.*\}', '', _expand(text[k + 1:m.start()], env), flags=re.S)
            tt = re.search(r'translation\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', own)
            rr = re.search(r'rotation\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', own)
            if rr:
                R0 = _rot([float(v) for v in rr.groups()])
            if tt:
                T0 = np.array([float(v) for v in tt.groups()])
            pts = []
            _walk(body, R0, T0, pts)
            # keep the boxes standing up (walls), not floors/ramps
            P = np.array(pts)
            if len(P):
                P = P[P[:, 2] > 0.3] if (P[:, 2] > 0.3).sum() >= 3 else P
                res = [(float(x), float(y)) for x, y in hull(P[:, :2])]
    _CACHE[proto] = res
    return res


if __name__ == '__main__':
    for p in sys.argv[1:]:
        f = footprint(p)
        if f:
            A = np.array(f)
            print(f'{p}: {len(f)} corners, x {A[:, 0].min():.1f}..{A[:, 0].max():.1f}, y {A[:, 1].min():.1f}..{A[:, 1].max():.1f}')
        else:
            print(f'{p}: no boundingObject found')
