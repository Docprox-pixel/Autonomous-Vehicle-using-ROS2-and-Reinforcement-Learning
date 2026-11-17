"""Lane-level road map and routing (pure Python, no ROS).

Real autonomous cars do not work out lanes inside junctions from the camera: Autoware uses a
Lanelet2 HD map, Waymo its own HD maps. The map says which path through a junction leads to which
lane; perception handles obstacles, other cars and signals along that path.

This module reads a SUMO network (sumo.net.xml). SUMO networks are exactly a lane graph: every
lane's centreline in driving order, plus internal lanes that connect lanes THROUGH each junction.
Webots ships one per city world (worlds/<world>_net/), and for unknown places a network can be
built after mapping. Coordinates are converted to the simulator world frame (world = sumo - netOffset).

  m = LaneMap('worlds/city_traffic_net/sumo.net.xml')
  path = m.route((x0, y0), heading0, (gx, gy))     # list of (x, y) along lane centrelines
"""
import heapq
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

LANE_CHANGE_COST = 40.0       # m-equivalent: prefer routes without lane changes
LANE_CHANGE_RUN = 25.0        # m over which a lane change is spread (a 5.4 m lane shift: radius > 6.2 m)
MIN_TURN_RADIUS = 7.0         # m: BmwX5 minimum is 6.2 m at full lock (wheelbase 2.995 / tan 0.45)


def _polyline(shape, off):
    pts = np.array([[float(v) for v in p.split(',')[:2]] for p in shape.split()])
    return pts - off


def _heading(v):
    return math.atan2(v[1], v[0])


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def fillet(p_in, d_in, p_out, d_out, radius, n=24):
    """Arc of `radius` tangent to the line (p_in, d_in) and the line (p_out, d_out).

    Returns (T1, T2, arc points) or None if the lines are parallel. T1 lies on the incoming line,
    T2 on the outgoing line."""
    d_in, d_out = d_in / np.linalg.norm(d_in), d_out / np.linalg.norm(d_out)
    cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
    if abs(cross) < 1e-3:
        return None
    # intersection P = p_in + a d_in = p_out + b d_out
    A = np.array([d_in, -d_out]).T
    a, _ = np.linalg.solve(A, p_out - p_in)
    P = p_in + a * d_in
    turn = _wrap(_heading(d_out) - _heading(d_in))
    t = radius * math.tan(abs(turn) / 2)
    T1, T2 = P - t * d_in, P + t * d_out
    side = 1.0 if turn > 0 else -1.0                       # left turn: centre on the left
    C = T1 + side * radius * np.array([-d_in[1], d_in[0]])
    a0 = math.atan2(*(T1 - C)[::-1])
    arc = np.array([C + radius * np.array([math.cos(a0 + side * k), math.sin(a0 + side * k)])
                    for k in np.linspace(0.0, abs(turn), n)])
    return T1, T2, arc


def blend(lane_a, lane_b, s0, run, step=1.0):
    """Smooth S-shaped lane change from lane_a to lane_b between arc lengths s0 and s0 + run."""
    out = []
    for u in np.linspace(0.0, 1.0, max(int(run / step), 2) + 1):
        w = (1 - math.cos(math.pi * u)) / 2
        out.append((1 - w) * lane_a.at(s0 + u * run) + w * lane_b.at(s0 + u * run))
    return np.array(out)


def resample(pts, step):
    """Evenly spaced points (every `step` m) along a polyline."""
    s = _cumlen(pts)
    if s[-1] < step:
        return pts
    g = np.r_[np.arange(0.0, s[-1], step), s[-1]]
    return np.c_[np.interp(g, s, pts[:, 0]), np.interp(g, s, pts[:, 1])]


def _cumlen(pts):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]


class Lane:
    def __init__(self, lane_id, edge_id, index, pts, width, speed, internal):
        self.id, self.edge, self.index = lane_id, edge_id, index
        self.pts, self.width, self.speed, self.internal = pts, width, speed, internal
        self.s = _cumlen(pts)
        self.length = float(self.s[-1])

    def project(self, p):
        """(distance along lane, lateral distance, heading at that point) of point p."""
        best = (math.inf, 0.0, 0.0, 0.0)
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 == 0 else min(max(float((p - a) @ ab) / L2, 0.0), 1.0)
            q = a + t * ab
            d = float(np.hypot(*(p - q)))
            if d < best[0]:
                best = (d, self.s[i] + t * math.sqrt(L2), math.atan2(ab[1], ab[0]), 0.0)
        return best[1], best[0], best[2]

    def slice(self, s0, s1):
        """Points of the lane between arc lengths s0 and s1 (inclusive ends interpolated)."""
        s0, s1 = max(0.0, s0), min(self.length, s1)
        if s1 <= s0:
            return np.zeros((0, 2))
        inner = self.pts[(self.s > s0) & (self.s < s1)]
        return np.vstack([self.at(s0), inner, self.at(s1)])

    def at(self, s):
        s = min(max(s, 0.0), self.length)
        i = min(int(np.searchsorted(self.s, s, side='right')) - 1, len(self.pts) - 2)
        seg = self.s[i + 1] - self.s[i]
        t = 0.0 if seg == 0 else (s - self.s[i]) / seg
        return self.pts[i] + t * (self.pts[i + 1] - self.pts[i])


class LaneMap:
    def __init__(self, net_file):
        root = ET.parse(net_file).getroot()
        loc = root.find('location')
        off = np.array([float(v) for v in loc.get('netOffset').split(',')])
        self.lanes = {}
        self.edge_lanes = {}
        self.edge_names = {}            # edge id -> street name (OSM nets: netconvert --output.street-names)
        for e in root.findall('edge'):
            internal = e.get('function') == 'internal'
            if e.get('name'):
                self.edge_names[e.get('id')] = e.get('name')
            for ln in e.findall('lane'):
                lane = Lane(ln.get('id'), e.get('id'), int(ln.get('index')), _polyline(ln.get('shape'), off),
                            float(ln.get('width', 3.2)), float(ln.get('speed', 13.9)), internal)
                self.lanes[lane.id] = lane
                self.edge_lanes.setdefault(lane.edge, []).append(lane)
        self.next = {lid: set() for lid in self.lanes}
        self.conn_state = {}         # internal (via) lane -> SUMO right-of-way state ('M' major, 'm' yield, 's' stop, ...)
        self.junction_xy = {}
        for j in root.findall('junction'):
            if j.get('x') is not None and not j.get('id', '').startswith(':'):
                self.junction_xy[j.get('id')] = (float(j.get('x')) - off[0], float(j.get('y')) - off[1])
        for c in root.findall('connection'):
            a = f"{c.get('from')}_{c.get('fromLane')}"
            b = f"{c.get('to')}_{c.get('toLane')}"
            via = c.get('via')
            if via:
                self.conn_state[via] = c.get('state', 'M')
            if a not in self.lanes or b not in self.lanes:
                continue
            # NO REVERSING LINKS: a link that turns back on itself (> 150 deg between the arriving
            # and the leaving lane - a U-turn or two roads meeting in a sharp V) is not drivable;
            # routes never use it (real turnarounds are the cul-de-sac loops 'tc_*')
            if not (c.get('from').startswith('tc_') or c.get('to').startswith('tc_')):
                A, B = self.lanes[a].pts, self.lanes[b].pts
                seq = [A[-2:], self.lanes[via].pts if via and via in self.lanes else np.zeros((0, 2)), B[:2]]
                Pj = np.vstack([q for q in seq if len(q)])
                if len(Pj) > 1:                                   # drop sub-0.5 m joints (they point anywhere)
                    keep = [0]
                    for q in range(1, len(Pj)):
                        if np.hypot(*(Pj[q] - Pj[keep[-1]])) > 0.5:
                            keep.append(q)
                    Pj = Pj[keep]
                if len(Pj) >= 3:
                    hh = np.unwrap(np.arctan2(*np.diff(Pj, axis=0).T[::-1]))
                    total = float(np.sum(np.abs(np.diff(hh))))            # all turning inside the link
                    net = abs(float(hh[-1] - hh[0]))
                    if net > math.radians(150) or total > math.radians(200) or total - net > math.radians(90):
                        continue                                          # doubles back: not drivable
            if via and via in self.lanes:
                self.next[a].add(via)
                self.next[via].add(b)
            else:
                self.next[a].add(b)
        # A link SUMO splits at an internal junction (yielding turns) is a CHAIN of internal lanes:
        # 'A -> B via :j_4_0' and ':j_4 -> B via :j_7_0'. The first one also made :j_4_0 -> B, a
        # shortcut that skipped :j_7_0 - the route jumped 13 m straight across junction 38.
        for lid in list(self.next):
            if not self.lanes[lid].internal:
                continue
            inner = [s for s in self.next[lid] if self.lanes[s].internal]
            for s in inner:
                self.next[lid] -= {b for b in self.next[s] if not self.lanes[b].internal}
        self.road_lanes = [l for l in self.lanes.values() if not l.internal]
        # Signalised junctions (HD-map signal layer, tools/extract_signals.py). None = unknown.
        self.signal_junctions = None
        self.single_lights = []
        sig = os.path.join(os.path.dirname(os.path.realpath(net_file)), 'signals.yaml')
        if os.path.exists(sig):
            import yaml
            lights = (yaml.safe_load(open(sig)) or {}).get('lights', [])
            self.single_lights = [(L['x'], L['y'], L.get('yaw')) for L in lights if L['type'] != 'CrossRoadsTrafficLight']
            self.signal_junctions = set()        # junctions whose EVERY approach is signalised
            for j in root.findall('junction'):
                if j.get('id', '').startswith(':') or j.get('x') is None:
                    continue
                jx, jy = float(j.get('x')) - off[0], float(j.get('y')) - off[1]
                if any(L['type'] == 'CrossRoadsTrafficLight' and math.hypot(L['x'] - jx, L['y'] - jy) < 15.0
                       for L in lights):
                    self.signal_junctions.add(j.get('id'))
        # No U-turns through junctions: legal in SUMO, but a poor and risky manoeuvre for a car this
        # size. Measured across the WHOLE junction (road lane in -> road lane out), because a U-turn is
        # often split into two ~90 deg connection pieces (found by RL training on random routes).
        for road in self.road_lanes:
            h_in = _heading(road.pts[-1] - road.pts[-2])
            stack = [(nid, [road.id]) for nid in self.next[road.id]]
            while stack:
                lid, trail = stack.pop()
                lane = self.lanes[lid]
                if lane.internal:
                    stack.extend((n, trail + [lid]) for n in self.next[lid] if n not in trail)
                    continue
                h_out = _heading(lane.pts[1] - lane.pts[0])
                if abs(_wrap(h_out - h_in)) > math.radians(150):
                    self.next[trail[-1]].discard(lid)       # cut the last piece into the U-turn
        self.min_radius = MIN_TURN_RADIUS
        # Roundabouts (worlds/<world>_net/roundabouts.yaml): centre + island/outer radius of the ring.
        # Routes through them are moved onto the middle of the ring (the map's internal lanes ran at
        # the island kerb, which the car's body cannot follow).
        self.roundabouts = []
        rb = os.path.join(os.path.dirname(os.path.realpath(net_file)), 'roundabouts.yaml')
        if os.path.exists(rb):
            import yaml
            self.roundabouts = (yaml.safe_load(open(rb)) or {}).get('roundabouts', [])
        self.last_stop_lines = []   # junction entry points of the last route: [(x, y, heading, speed)]
        self.last_junctions = []    # per stop line: junction centre + whether we must give way
        self.last_speeds = None     # speed limit (m/s) per point of the last route

    # ------------------------------------------------------------- queries
    def nearest_lane(self, p, heading=None, max_angle=math.radians(60), include_internal=True):
        p = np.asarray(p, float)
        best = None
        for lane in self.lanes.values():
            if lane.internal and not include_internal:
                continue
            s, d, h = lane.project(p)
            if heading is not None:
                dh = abs(math.atan2(math.sin(h - heading), math.cos(h - heading)))
                if dh > max_angle:
                    continue
            if best is None or d < best[2]:
                best = (lane, s, d)
        return best

    def fix_roundabouts(self, path):
        """Roundabout path the way a driver takes it:
          1. straight along the approach lane up to and over the give-way line,
          2. straight on into the ring until the car is well inside (2.2 m in from the outer edge),
          3. turn onto the ring and follow its CIRCLE in the traffic's direction (the map's ring
             direction - in this world the traffic circulates clockwise) as far as the exit,
          4. leave the circle and straighten onto the exit lane.
        (The map's own ring lanes run over the island kerb; an earlier spiral cut the entry corner by
        the sign.) The joins are rounded afterwards by smooth_kinks, only as much as needed."""
        if path is None or not self.roundabouts:
            return path
        P = np.array(path, float)
        for r in self.roundabouts:
            c = np.array([r['x'], r['y']])
            dist = np.hypot(*(P - c).T)
            idx = np.flatnonzero(dist < r['outer'] + 1.0)
            if len(idx) < 3:
                continue
            a, b = int(idx[0]), int(idx[-1])
            if a < 1 or b > len(P) - 2 or dist[a + 1:b].min() > r['outer']:
                continue                                         # passes by, does not enter
            ang = np.unwrap(np.arctan2(P[a:b + 1, 1] - c[1], P[a:b + 1, 0] - c[0]))
            direction = r.get('direction', 'map')
            sgn = (1.0 if direction == 'ccw' else -1.0) if direction in ('ccw', 'cw') else \
                (1.0 if ang[-1] - ang[0] > 0 else -1.0)           # follow the map / traffic direction
            rc = r['outer'] - 2.2                                  # circle the car's centre follows

            def straight_in(q, h):
                """From q along heading h until the radius is rc (or the closest approach)."""
                d = np.array([math.cos(h), math.sin(h)])
                w = q - c
                bq, cq = float(w @ d), float(w @ w) - rc * rc
                disc = bq * bq - cq
                t = -bq - math.sqrt(disc) if disc > 0 else max(-bq, 0.0)
                t = max(t, 0.0)
                n = max(int(t / 0.5), 1)
                return q + np.outer(np.linspace(0.0, t, n + 1)[1:], d)

            he = math.atan2(*(P[a] - P[a - 1])[::-1])
            hx = math.atan2(*(P[b + 1] - P[b])[::-1])
            entry = straight_in(P[a], he)
            exit_rev = straight_in(P[b], hx + math.pi)[::-1]       # from the circle out to P[b]
            th0 = math.atan2(*(entry[-1] - c)[::-1])
            th1 = math.atan2(*(exit_rev[0] - c)[::-1])
            sweep = (sgn * (th1 - th0)) % (2 * math.pi)
            if sweep < 0.3:
                sweep += 2 * math.pi                             # back the way it came: full circle
            th = th0 + sgn * np.linspace(0.0, sweep, max(int(sweep * rc / 0.5), 8))
            ring = np.c_[c[0] + rc * np.cos(th), c[1] + rc * np.sin(th)]
            P = np.vstack([P[:a + 1], entry, ring, exit_rev, P[b:]])
        return P

    # Swing-out turns (_wide_turn) were for SQUARE kerbed corners. The junctions now have real corner
    # radii (tools/round_corners.py, 8 m kerb returns, kerbs cleared by tools/clear_corners.py): a
    # plain fillet from lane centre to lane centre stays ~3 m from the kerb, so the car keeps STRAIGHT
    # in the middle of its own lane right up to the turn and after it, like every other car.
    WIDE_TURN = False

    @staticmethod
    def smooth_kinks(path, max_turn=math.radians(12), radius=7.0, r_min=6.3):
        """Round every corner of a route the way a driver turns: a FILLET - the incoming and outgoing
        lane lines are extended to their intersection and joined by a circular arc of `radius`
        (7 m, just above the car's 6.2 m turning circle) tangent to both. The car stays in its lane
        straight up to the point where it must start turning ('go further, then turn') and ends the
        turn exactly on the new lane; nothing cuts across the corner early. If there is not enough
        straight lane before/after the corner for that radius, the radius shrinks to fit (>= r_min).
        """
        if path is None or len(path) < 5:
            return path
        P = np.array(path, float)
        P = P[np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-6]]
        s_from = 0.0
        for _ in range(60):
            seg = np.diff(P, axis=0)
            Ls = np.hypot(*seg.T)
            s = np.r_[0.0, np.cumsum(Ls)]
            h = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
            found = None
            for i in range(max(1, int(np.searchsorted(s, s_from))), len(h)):
                j = min(int(np.searchsorted(s, s[i] + 1.0)), len(h) - 1)
                if abs(h[j] - h[i - 1]) > max_turn:
                    found = i
                    break
            if found is None:
                break
            i = found
            # extent of the turn: from where the heading starts changing to where it is steady again
            a0 = max(i - 1, 0)
            b0 = i
            while b0 + 2 < len(h) and s[b0] - s[i] < 25.0 and abs(h[min(b0 + 2, len(h) - 1)] - h[b0]) > math.radians(3):
                b0 += 1
            b0 = min(b0 + 1, len(h) - 1)
            h_in, h_out = h[a0], h[b0]
            dlt = math.atan2(math.sin(h_out - h_in), math.cos(h_out - h_in))
            # only REAL kinks: a lane curve the car can already follow (curvature <= 1/r_min over
            # 1.5 m windows) stays exactly as the map draws it
            kloc = 0.0
            for q in range(a0, b0 + 1):
                q2 = min(int(np.searchsorted(s, s[q] + 1.5)), len(h) - 1)
                if s[q2] - s[q] > 0.5:
                    kloc = max(kloc, abs(h[q2] - h[q]) / (s[q2] - s[q]))
            if abs(dlt) < math.radians(4) or kloc <= 1.0 / r_min:
                s_from = s[b0] + 0.5
                continue
            d_in = np.array([math.cos(h_in), math.sin(h_in)])
            d_out = np.array([math.cos(h_out), math.sin(h_out)])
            # intersection V of the incoming line (through P[a0]) and outgoing line (through P[b0+1])
            A = np.c_[d_in, -d_out]
            try:
                tt = np.linalg.solve(A, P[b0 + 1] - P[a0])
            except np.linalg.LinAlgError:
                s_from = s[b0] + 0.5
                continue
            if tt[0] < -0.5 or tt[1] > 0.5:
                s_from = s[b0] + 0.5                     # lines meet behind: not a simple corner
                continue
            V = P[a0] + tt[0] * d_in
            if LaneMap.WIDE_TURN and abs(dlt) > math.radians(35):
                Q = LaneMap._wide_turn(P, s, i, a0, b0, d_in, d_out, dlt, V)
                if Q is not None:
                    P = Q
                    s_from = s[a0] + 12.0
                    continue
            # room on the incoming / outgoing lanes for the tangent points
            room_in = float(np.hypot(*(V - P[0])))
            room_out = float(np.hypot(*(P[-1] - V)))
            R = radius
            # FOLLOW THE JUNCTION'S OWN TURN: the map's link (fitted by netconvert to the junction and
            # its corner radii) bends round the corner at some distance from V; a fixed 7 m arc cut
            # 1.2-1.5 m inside it and put the car's side over the kerb return. The fillet radius is
            # chosen so that the arc's middle meets the link's middle (same tangents, >= radius).
            bis = d_out - d_in
            if np.hypot(*bis) > 1e-6:
                bis = bis / np.hypot(*bis)                 # from V towards the inside of the corner
                seg_pts = P[a0:b0 + 2]
                rel = seg_pts - V
                on_bis = np.abs(rel @ np.array([-bis[1], bis[0]]))
                k_mid = int(np.argmin(on_bis))
                d_mid = float(rel[k_mid] @ bis)
                sag = 1.0 / max(math.cos(abs(dlt) / 2.0), 1e-6) - 1.0
                if d_mid > 0.2 and on_bis[k_mid] < 1.5 and sag > 1e-3:
                    R = float(np.clip(d_mid / sag, radius, 30.0))
            T = R * math.tan(abs(dlt) / 2.0)
            lim = min(room_in, room_out) - 0.5
            if T > lim:
                R = max(r_min, lim / max(math.tan(abs(dlt) / 2.0), 1e-6))
                T = R * math.tan(abs(dlt) / 2.0)
            p0, p1 = V - T * d_in, V + T * d_out
            # replace the path between the tangent points by the arc
            ia = int(np.argmin(np.hypot(*(P[:i + 1] - p0).T)))
            ib = i + int(np.argmin(np.hypot(*(P[i:] - p1).T)))
            if ib <= ia + 1 or np.hypot(*(P[ia] - p0)) > 1.5 or np.hypot(*(P[ib] - p1)) > 1.5:
                s_from = s[b0] + 0.5                     # tangent points not on the path: skip
                continue
            sg = 1.0 if dlt > 0 else -1.0
            centre = p0 + sg * R * np.array([-d_in[1], d_in[0]])
            a_s = math.atan2(*(p0 - centre)[::-1])
            n = max(int(abs(dlt) * R / 0.5), 4)
            th = a_s + sg * np.linspace(0.0, abs(dlt), n + 1)
            arc = np.c_[centre[0] + R * np.cos(th), centre[1] + R * np.sin(th)]
            P = np.vstack([P[:ia], arc, P[ib + 1:]])
            s_from = s[ia] + R * abs(dlt) + 0.5
        # final clean-up of the few kinks a fillet could not handle (S-bends, joins at the
        # roundabout): relax ONLY points sharper than the car can steer toward their neighbours
        P = LaneMap._relax_sharp(P, 1.0 / r_min)
        return P

    @staticmethod
    def _wide_turn(P, s, i, a0, b0, d_in, d_out, dlt, V, R=6.4, half_lane=1.75, body=1.0, clear=0.3):
        """A real junction turn around a (nearly square) kerb corner: the car's rear axle turns on a
        circle of radius R and its INNER rear wheel runs `body` m inside it; that wheel must keep
        `clear` m from the inner kerb corner K. From lane centre to lane centre this is impossible
        around a square corner (the rear axle alone would pass ~0.4 m over the corner at R = 7 m),
        so - like a driver - the car swings wide: up to 0.6 m toward the outside of its own lane on
        the approach (u), and up to 2.0 m outward into the junction / new road (w) coming out of the
        turn, easing back into its lane afterwards. Smallest (u, w) that clears the corner."""
        sg = 1.0 if dlt > 0 else -1.0
        n1 = sg * np.array([-d_in[1], d_in[0]])          # inner normal of the incoming lane
        n2 = sg * np.array([-d_out[1], d_out[0]])        # inner normal of the outgoing lane

        def meet(p_a, d_a, p_b, d_b):
            A = np.c_[d_a, -d_b]
            try:
                t_ = np.linalg.solve(A, p_b - p_a)
            except np.linalg.LinAlgError:
                return None
            return p_a + t_[0] * d_a

        K = meet(V + half_lane * n1, d_in, V + half_lane * n2, d_out)  # inner kerb corner
        if K is None:
            return None
        # real-world rule: turn from the own lane INTO the own lane of the new road (a right turn
        # ends in the lane nearest the kerb; the next lane of a two-lane road is ONCOMING traffic) -
        # so the swing-out keeps the whole body inside its lanes: u, w <= half_lane - body (0.75 m)
        best, best_margin = None, -math.inf
        w_max = half_lane - body
        for u in (0.0, 0.3, 0.6):
            for w in np.arange(0.0, w_max + 1e-9, 0.25):
                C = meet(V - u * n1 + R * n1, d_in, V - w * n2 + R * n2, d_out)
                if C is None:
                    continue
                margin = (R - body) - float(np.hypot(*(C - K)))        # inner rear wheel to corner
                if margin >= clear and best is not None and best_margin >= clear:
                    break
                if margin > best_margin + 1e-6 or (margin >= clear and best_margin < clear):
                    best, best_margin = (u, w, C), margin
                if margin >= clear:
                    break
            if best_margin >= clear:
                break
        if best is None:
            return None
        u, w, C = best
        if u == 0.0 and w == 0.0:
            return None                                   # the plain fillet already clears it
        p0 = C - R * n1                                   # arc start (on the shifted incoming line)
        p1 = C - R * n2                                   # arc end (on the shifted outgoing line)
        # where those lie on the original path
        ia = int(np.argmin(np.hypot(*(P[:i + 1] - (p0 + u * n1)).T)))
        ib = i + int(np.argmin(np.hypot(*(P[i:] - (p1 + w * n2)).T)))
        if ib <= ia + 1 or np.hypot(*(P[ia] - (p0 + u * n1))) > 2.0 or np.hypot(*(P[ib] - (p1 + w * n2))) > 2.0:
            return None
        Q = P.copy()
        # approach: ease out to the outside of the own lane over 10 m before the turn
        for j in range(ia, -1, -1):
            dd = s[ia] - s[j]
            if dd > 10.0:
                break
            Q[j] = P[j] - u * (0.5 + 0.5 * math.cos(math.pi * dd / 10.0)) * n1
        # exit: ease back from the wide line into the lane over 14 m
        for j in range(ib, len(P)):
            dd = s[j] - s[ib]
            if dd > 14.0:
                break
            Q[j] = P[j] - w * (0.5 + 0.5 * math.cos(math.pi * dd / 14.0)) * n2
        a_s = math.atan2(*(p0 - C)[::-1])
        nseg = max(int(abs(dlt) * R / 0.5), 4)
        th = a_s + sg * np.linspace(0.0, abs(dlt), nseg + 1)
        arc = np.c_[C[0] + R * np.cos(th), C[1] + R * np.sin(th)]
        return np.vstack([Q[:ia], arc, Q[ib + 1:]])

    @staticmethod
    def _relax_sharp(P, k_lim, iters=200):
        P = np.array(P, float)
        for _ in range(iters):
            seg = np.diff(P, axis=0)
            L = np.hypot(*seg.T)
            h = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
            # curvature at interior points over +-1 segment
            k = np.abs(np.diff(h)) / np.maximum(0.5 * (L[1:] + L[:-1]), 0.05)
            bad = np.flatnonzero(k > k_lim) + 1
            if len(bad) == 0:
                break
            for j in bad:
                lo, hi = max(j - 2, 0), min(j + 2, len(P) - 1)
                if hi - lo < 2:
                    continue
                P[j] = 0.5 * P[j] + 0.25 * (P[j - 1] + P[j + 1])
                # spread a little to the neighbours so the bend widens instead of moving the corner
                if j - 1 > 0:
                    P[j - 1] = 0.75 * P[j - 1] + 0.125 * (P[j - 2] + P[j])
                if j + 1 < len(P) - 1:
                    P[j + 1] = 0.75 * P[j + 1] + 0.125 * (P[j] + P[j + 2])
        return P

    def route_near(self, start, heading, goal, max_distance=60.0):
        """Route to the goal; if the lane nearest the goal cannot be reached (one-way streets,
        U-turns are banned), route to the nearest REACHABLE lane within max_distance of the goal.
        Used for places (a hospital beside a one-way street is still a valid destination)."""
        path = self.route(start, heading, goal)
        if path is not None:
            return self.smooth_kinks(self.fix_roundabouts(path))
        g = np.asarray(goal, float)
        cands = sorted(((lane.project(g)[1], lane) for lane in self.road_lanes), key=lambda c: c[0])
        for d, lane in cands:
            if d > max_distance:
                break
            s_g = lane.project(g)[0]
            path = self.route(start, heading, tuple(lane.at(s_g)))
            if path is not None:
                return self.smooth_kinks(self.fix_roundabouts(path))
        return None

    def road_types(self):
        """edge id -> 'highway' | 'two_way' | 'one_way'. Two-way: an edge in the opposite direction
        joins the same two junctions. Highway: >= 2 lanes per direction and speed >= 22 m/s (80 km/h),
        or speed >= 25 m/s."""
        if getattr(self, '_types', None) is not None:
            return self._types
        ends = {}
        for e, lanes in self.edge_lanes.items():
            l0 = lanes[0]
            ends[e] = (tuple(np.round(l0.pts[0], 0)), tuple(np.round(l0.pts[-1], 0)), l0)
        types = {}
        for e, lanes in self.edge_lanes.items():
            sp, n = max(l.speed for l in lanes), len(lanes)
            if (n >= 2 and sp >= 22.0) or sp >= 25.0:
                types[e] = 'highway'
                continue
            rev = ('-' + e) if not e.startswith('-') else e[1:]
            two = rev in self.edge_lanes
            if not two:                      # nets without the '-id' convention: reversed end points
                a, b, _ = ends[e]
                two = any(np.hypot(*(np.subtract(a, b2))) < 25 and np.hypot(*(np.subtract(b, a2))) < 25
                          for f, (a2, b2, _) in ends.items() if f != e)
            types[e] = 'two_way' if two else 'one_way'
        self._types = types
        return types

    def road_info(self, p, heading):
        """What road the car is on: type (one_way / two_way / highway), street name, lanes in our
        direction, our lane (1 = rightmost), speed limit. None in a junction / off the map."""
        hit = self.nearest_lane(p, heading, max_angle=math.radians(45), include_internal=True)
        if hit is None or hit[2] > 3.0:
            return None
        lane = hit[0]
        if lane.internal:
            return {'type': 'junction', 'name': '', 'lanes': 0, 'lane': 0, 'speed_limit': lane.speed}
        same = self.edge_lanes[lane.edge]
        return {'type': self.road_types().get(lane.edge, 'two_way'), 'name': self.edge_names.get(lane.edge, ''),
                'lanes': len(same), 'lane': lane.index + 1, 'speed_limit': max(l.speed for l in same)}

    def left_boundary(self, path, margin=0.15, step=1.0):
        """Points along the LEFT edge of the lane the path is in (the centre line on a two-way road,
        the kerb on a one-way road), where there is no same-direction lane to the left. Junction
        interiors are skipped (turns must stay possible). As virtual walls these keep the car out of
        oncoming traffic even when the camera misreads the road (tunnel lights)."""
        if path is None or len(path) < 2:
            return np.zeros((0, 2))
        if getattr(self, '_segs', None) is None:
            A, B, L = [], [], []
            for lane in self.road_lanes:
                for a, b in zip(lane.pts[:-1], lane.pts[1:]):
                    A.append(a), B.append(b), L.append(lane)
            self._segs = (np.array(A), np.array(B), L)
        A, B, L = self._segs
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-9)
        hseg = np.arctan2(AB[:, 1], AB[:, 0])
        out = []
        for i in range(0, len(path) - 1):
            p, q = path[i], path[i + 1]
            h = math.atan2(q[1] - p[1], q[0] - p[0])
            t = np.clip(((p - A) * AB).sum(1) / L2, 0, 1)
            d = np.hypot(*(A + t[:, None] * AB - p).T)
            ok = np.abs(np.arctan2(np.sin(hseg - h), np.cos(hseg - h))) < math.radians(30)
            d = np.where(ok, d, np.inf)
            j = int(np.argmin(d))
            if d[j] > 1.0:
                continue                                   # in a junction or a lane change blend
            lane = L[j]
            if any(l.index == lane.index + 1 for l in self.edge_lanes[lane.edge]):
                continue                                   # a same-direction lane on the left
            s_on = lane.project(A[j] + t[j] * AB[j])[0]
            if s_on < 8.0 or s_on > lane.length - 8.0:
                continue                                   # lane lines end before junctions: the car
                                                           # must be able to swing through the turn
            nl = np.array([-math.sin(hseg[j]), math.cos(hseg[j])])
            base = A[j] + t[j] * AB[j]
            out.append(base + nl * (lane.width / 2 + margin))
        return np.array(out) if out else np.zeros((0, 2))

    def lane_changes(self, lane):
        """Adjacent lanes on the same (non-internal) edge."""
        if lane.internal:
            return []
        return [l for l in self.edge_lanes[lane.edge] if abs(l.index - lane.index) == 1]

    # --------------------------------------------------------------- route
    def route(self, start, heading, goal):
        """Lane-level path (Nx2, world frame) from a pose to the lane point nearest the goal."""
        st = self.nearest_lane(start, heading)
        if st is not None and st[0].internal:
            # Starting inside a junction: if the rest of this connection is a turn tighter than the
            # car can drive from here, start on the road lane straight ahead instead (and re-route
            # from there, like a driver who has missed a turn).
            lane, s0, _ = st
            rest = lane.slice(s0, lane.length)
            if len(rest) >= 2:
                h0 = _heading(rest[1] - rest[0]) if len(rest) > 2 else heading
                h1 = _heading(rest[-1] - rest[-2])
                turn = abs(_wrap(h1 - heading))
                if turn > math.radians(30) and (lane.length - s0) / turn < self.min_radius * 1.5:
                    ahead = (start[0] + 8 * math.cos(heading), start[1] + 8 * math.sin(heading))
                    st = self.nearest_lane(ahead, heading, max_angle=math.radians(35), include_internal=False) or st
        gl = self.nearest_lane(goal, include_internal=False)
        if st is None or gl is None:
            return None
        start_lane, s_start, _ = st
        goal_lane, s_goal, _ = gl
        # Dijkstra over lanes; state = lane id, cost = remaining length of the start lane + lanes
        dist = {start_lane.id: 0.0}
        prev = {}
        pq = [(0.0, start_lane.id)]
        target = goal_lane.id
        while pq:
            c, lid = heapq.heappop(pq)
            if lid == target and (lid != start_lane.id or s_goal >= s_start):
                break
            if c > dist.get(lid, math.inf):
                continue
            lane = self.lanes[lid]
            run = lane.length - (s_start if lid == start_lane.id else 0.0)
            for nid in self.next[lid]:
                nc = c + run
                if nc < dist.get(nid, math.inf):
                    dist[nid], prev[nid] = nc, (lid, 'follow')
                    heapq.heappush(pq, (nc, nid))
            for adj in self.lane_changes(lane):
                nc = c + LANE_CHANGE_COST
                if nc < dist.get(adj.id, math.inf):
                    dist[adj.id], prev[adj.id] = nc, (lid, 'change')
                    heapq.heappush(pq, (nc, adj.id))
        if target not in dist:
            return None
        chain = [(target, None)]
        while chain[-1][0] != start_lane.id:
            p_id, how = prev[chain[-1][0]]
            chain[-1] = (chain[-1][0], how)
            chain.append((p_id, None))
        chain.reverse()                                   # [(lane, how we arrived)], first = start
        return self._stitch(chain, s_start, s_goal)

    def drivable_connection(self, before, conn, after):
        """Replace a junction connection tighter than the car can drive by a fillet arc.

        Tries every lane of the outgoing road; keeps one whose arc lies inside the junction (starts
        after `before` ends, ends before the target lane starts + 0.5 m), preferring the original
        target lane. Returns (arc points, landing lane, arc length along landing lane) or None."""
        turn = abs(_wrap(_heading(after.pts[1] - after.pts[0]) - _heading(before.pts[-1] - before.pts[-2])))
        if turn < math.radians(20) or conn.length / max(turn, 1e-6) >= self.min_radius:
            return None
        p_in, d_in = before.pts[-1], before.pts[-1] - before.pts[-2]
        options = []
        for cand in self.edge_lanes[after.edge]:
            f = fillet(p_in, d_in, cand.pts[0], cand.pts[1] - cand.pts[0], self.min_radius)
            if f is None:
                continue
            T1, T2, arc = f
            back = float((p_in - T1) @ (d_in / np.linalg.norm(d_in)))      # >0: arc starts before lane end
            s_land, lat, _ = cand.project(T2)
            if back <= 0.5 and lat < 0.5:
                options.append((cand.id != after.id, abs(back), arc, cand, s_land))
        if not options:
            return None
        options.sort(key=lambda o: (o[0], o[1]))
        _, _, arc, cand, s_land = options[0]
        return arc, cand, s_land

    def _stitch(self, chain, s_start, s_goal):
        chain = list(chain)
        pieces, s_from = [], s_start
        self.last_stop_lines = []
        self.last_junctions = []
        speeds = []
        k = 0
        while k < len(chain):
            lid = chain[k][0]
            lane = self.lanes[lid]
            if lane.internal and k > 0 and not self.lanes[chain[k - 1][0]].internal:
                # Stop line = where the route leaves a road lane and enters the junction
                prev = self.lanes[chain[k - 1][0]]
                d = prev.pts[-1] - prev.pts[-2]
                junction = lane.edge[1:].rsplit('_', 1)[0]          # ':16_0' -> '16'
                # per APPROACH: a crossroads light controls all approaches of its junction, a
                # single light controls the approach whose stop line is next to it
                end = prev.pts[-1]
                h_app = _heading(d)

                def controls(lx, ly, yaw):
                    # near-side light next to the stop line, or a far-side light up to 40 m ahead
                    # whose lamps face this approach (yaw = facing direction = approach + pi)
                    if math.hypot(lx - end[0], ly - end[1]) < 14.0:
                        return True
                    if yaw is None:
                        return False
                    along = (lx - end[0]) * math.cos(h_app) + (ly - end[1]) * math.sin(h_app)
                    facing = abs(_wrap(yaw - (h_app + math.pi))) < math.radians(35)
                    return facing and 0.0 < along < 40.0 and math.hypot(lx - end[0], ly - end[1]) < 45.0
                signal = None if self.signal_junctions is None else (
                    junction in self.signal_junctions or any(controls(*L) for L in self.single_lights))
                self.last_stop_lines.append((float(prev.pts[-1][0]), float(prev.pts[-1][1]),
                                             _heading(d), prev.speed, signal))
                # right of way through this junction: minor ('m' yield, 's' stop, 'w' roundabout
                # entry...) means GIVE WAY to crossing/circulating traffic
                st_ = self.conn_state.get(lane.id, 'M')
                jx, jy = self.junction_xy.get(junction, (float(prev.pts[-1][0]), float(prev.pts[-1][1])))
                # SUMO link states: M/O = priority; m = yield, o = yield (unregulated, blinking),
                # s = stop sign, w = zipper/roundabout entry, = all-way: give way
                self.last_junctions.append({'x': jx, 'y': jy, 'minor': st_ in ('m', 'o', 's', 'w', '='),
                                            'state': st_})
            if (lane.internal and 0 < k < len(chain) - 1 and chain[k + 1][1] != 'change'):
                before, after = self.lanes[chain[k - 1][0]], self.lanes[chain[k + 1][0]]
                fix = self.drivable_connection(before, lane, after)
                if fix is not None:
                    arc, land, s_land = fix
                    pieces.append(arc)
                    if land.id == after.id:
                        s_from = s_land
                        k += 1
                        continue
                    # If the route later changes into the lane we landed in anyway, stay there.
                    later = [j for j in range(k + 1, len(chain)) if chain[j][0] == land.id]
                    if later and all(self.lanes[chain[j][0]].edge == land.edge for j in range(k + 1, later[0] + 1)):
                        s_from = s_land
                        k = later[0]
                        continue
                    # landed in a neighbouring lane: drive it a little, then blend into the plan
                    run_end = min(s_land + LANE_CHANGE_RUN, land.length)
                    pieces.append(land.slice(s_land, run_end))
                    if k + 1 == len(chain) - 1:          # the goal lane itself: stay where we landed
                        pieces.append(land.slice(run_end, s_goal))
                        break
                    pieces.append(blend(land, after, run_end, LANE_CHANGE_RUN))
                    s_from = min(run_end + LANE_CHANGE_RUN, after.length)
                    k += 1
                    continue
            nxt = chain[k + 1] if k + 1 < len(chain) else None
            s_to = s_goal if nxt is None else lane.length
            if nxt is not None and nxt[1] == 'change':
                # leave this lane part-way with a smooth S-curve into the neighbour
                s_to = min(s_from + 5.0, lane.length)
                pieces.append(lane.slice(s_from, s_to))
                pieces.append(blend(lane, self.lanes[nxt[0]], s_to, LANE_CHANGE_RUN))
                s_from = min(s_to + LANE_CHANGE_RUN, self.lanes[nxt[0]].length)
                k += 1
                continue
            pieces.append(lane.slice(s_from, s_to))
            s_from = 0.0
            k += 1
        pieces = [p for p in pieces if len(p)]
        if not pieces:
            return None                      # e.g. goal behind the start on the same lane
        pts = np.vstack(pieces)
        keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 0.05]
        if keep.sum() < 2:
            return None
        return resample(pts[keep], 1.0)
