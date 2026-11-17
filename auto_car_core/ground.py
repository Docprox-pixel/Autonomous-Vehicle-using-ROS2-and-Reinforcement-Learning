"""Slope-aware road surface from one lidar scan (grid region growing, as Autoware's ground filters).

height_above_road(p, h_plane) -> height of every point above the road surface under it.

The road is grown outwards from the car over a CELL m grid: a cell is road when its lowest return
is within MAX_STEP of an adjacent road cell (a drivable gradient: 0.2 m per 1 m ~ 11 deg) and the
cell is flat (its returns span < FLAT m vertically - a car, wall or post cell spans much more and
is never taken as ground). Seeds are the flat cells next to the car on the fitted plane.
A single plane (plus 'near the plane' cells) turned a bridge RAMP into an obstacle: 5-8 m ahead
the rising asphalt was 0.2 m above the flat road behind and the car stopped at every ramp.
Points over no road cell take the nearest road cell within REACH m, else the fitted plane.
"""
import math
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

CELL = 1.0        # m
MAX_STEP = 0.20   # m between adjacent road cells
FLAT = 0.30       # m: max vertical spread of the returns in a road cell
SEED_R = 5.0      # m around the car
REACH = 6.0       # m
MAX_R = 45.0      # m
GAP = 3           # cells: the road may be grown across empty cells (sparse far lidar rings)


def height_above_road(p, h_plane):
    """p: Nx3 (base_link), h_plane: N heights above the fitted road plane. Returns N heights."""
    if len(p) == 0:
        return h_plane
    r = np.hypot(p[:, 0], p[:, 1])
    use = r < MAX_R
    ij = np.floor(p[:, :2] / CELL).astype(np.int64)
    key = ij[:, 0] * 100003 + ij[:, 1]
    idx = np.flatnonzero(use)
    if len(idx) < 10:
        return h_plane
    k = key[idx]
    z = p[idx, 2]
    order = np.lexsort((z, k))
    ks = k[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    last = np.r_[ks[1:] != ks[:-1], True]
    lo_i = idx[order[first]]                     # lowest return per cell
    hi_i = idx[order[last]]                      # highest return per cell
    cells = {}
    for a, b in zip(lo_i, hi_i):
        cells[(int(ij[a, 0]), int(ij[a, 1]))] = (float(p[a, 2]), float(p[b, 2] - p[a, 2]), float(h_plane[a]))
    road = {}
    q = deque()
    for c, (zl, spread, hp) in cells.items():
        cx, cy = (c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL
        if spread < FLAT and abs(hp) < 0.25 and cx * cx + cy * cy < SEED_R * SEED_R:
            road[c] = zl
            q.append(c)
    if not road:
        return h_plane
    # neighbours up to GAP cells away: beyond ~10 m the lidar rings hit the road 1-2 m apart, the
    # cells between them are EMPTY - growing only to adjacent cells stopped at the first gap and the
    # next ring up a bridge ramp looked 0.2-0.3 m 'tall' (the car stopped at every ramp)
    offs = [(dx, dy, math.hypot(dx, dy)) for dx in range(-GAP, GAP + 1) for dy in range(-GAP, GAP + 1)
            if (dx or dy)]
    offs.sort(key=lambda o: o[2])
    while q:
        c = q.popleft()
        zc = road[c]
        for dx, dy, dist in offs:
            n = (c[0] + dx, c[1] + dy)
            if n in road or n not in cells:
                continue
            zl, spread, _ = cells[n]
            if spread < FLAT and abs(zl - zc) < MAX_STEP * dist * CELL + 0.03:
                road[n] = zl
                q.append(n)
    keys = np.array(list(road.keys()), np.int64)
    g_z = np.array(list(road.values()))
    # own cell first (exact), else the nearest road cell within REACH, else the plane
    rk = keys[:, 0] * 100003 + keys[:, 1]
    srt = np.argsort(rk)
    rk_s, gz_s = rk[srt], g_z[srt]
    pos = np.clip(np.searchsorted(rk_s, key), 0, len(rk_s) - 1)
    own = rk_s[pos] == key
    dist, nn = cKDTree((keys + 0.5) * CELL).query(p[:, :2])
    plane_z = p[:, 2] - h_plane
    ground_here = np.where(own, gz_s[pos], np.where(dist < REACH, g_z[nn], plane_z))
    return p[:, 2] - ground_here
