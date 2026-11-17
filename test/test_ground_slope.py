"""Slope-aware ground: a bridge ramp is road, a car on it and a wall beside it stay obstacles."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_car_core.ground import height_above_road  # noqa: E402

G = -0.32          # road level in base_link


def scene(grade=0.07, ramp_from=4.0):
    rng = np.random.default_rng(0)
    pts = []
    # road surface: dense returns on a 0.25 m grid, x 0..40, y -4..4 (flat, then the ramp)
    for x in np.arange(0.5, 40, 0.25):
        for y in np.arange(-4, 4, 0.25):
            z = G + max(0.0, x - ramp_from) * grade
            pts.append((x, y, z + rng.normal(0, 0.01)))
    road = len(pts)
    # a car on the ramp at x 18..22.5, y -1..1, body 0.3..1.5 m above the ramp
    car = []
    for x in np.arange(18, 22.5, 0.2):
        for y in np.arange(-1, 1, 0.2):
            for hz in (0.3, 0.8, 1.4):
                car.append((x, y, G + (x - ramp_from) * grade + hz))
    # a wall at y = 5 along the flat part
    wall = [(x, 5.0, G + hz) for x in np.arange(0.5, 4, 0.2) for hz in np.arange(0.0, 2.0, 0.2)]
    # a 0.12 m kerb at y = -4.2
    kerb = [(x, -4.2, G + max(0.0, x - ramp_from) * grade + 0.12) for x in np.arange(0.5, 30, 0.25)]
    P = np.array(pts + car + wall + kerb, float)
    return P, slice(0, road), slice(road, road + len(car)), slice(road + len(car), road + len(car) + len(wall))


def test_ramp_is_road_car_and_wall_are_not():
    P, road, car, wall = scene()
    h_plane = P[:, 2] - G                      # the fitted plane: the flat road (the failing case)
    h = height_above_road(P, h_plane)
    assert np.percentile(np.abs(h[road]), 99) < 0.12, 'ramp asphalt must be ground'
    assert (h[road] > 0.2).sum() == 0, 'no ramp point may become an obstacle'
    assert np.median(h[car]) > 0.5 and (h[car] > 0.2).mean() > 0.95, 'the car on the ramp stays an obstacle'
    assert (h[wall] > 0.2).mean() > 0.85, 'the wall stays an obstacle'


def test_flat_road_unchanged():
    P, road, car, wall = scene(grade=0.0)
    h = height_above_road(P, P[:, 2] - G)
    assert np.abs(h[road]).max() < 0.1
    assert (h[car] > 0.2).mean() > 0.95


def test_sparse_rings_on_a_ramp_are_road():
    """Real lidar: rings hit the road ever further apart (1-2 m at 10-20 m). The ramp starts 6 m ahead."""
    pts = []
    ranges = [3.0, 3.6, 4.3, 5.1, 6.0, 7.1, 8.4, 10.0, 11.9, 14.2, 17.0, 20.4, 24.5]
    for r in ranges:
        for ang in np.linspace(-0.5, 0.5, 120):
            x, y = r * np.cos(ang), r * np.sin(ang)
            pts.append((x, y, G + max(0.0, x - 6.0) * 0.05))
    P = np.array(pts)
    h = height_above_road(P, P[:, 2] - G)
    assert (h > 0.15).sum() == 0, f'{(h > 0.15).sum()} ramp points look like obstacles'
