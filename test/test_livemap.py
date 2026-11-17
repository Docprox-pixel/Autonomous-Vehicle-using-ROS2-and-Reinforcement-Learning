"""Live map: static obstacles confirmed after 2 scans, one-scan ripples never appear, objects that
left fade out, and the map stays fixed in the world while the car moves."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_car_core.livemap import LiveMap  # noqa: E402


def test_livemap():
    m = LiveMap()
    wall = np.c_[np.full(20, 10.0), np.linspace(-1, 1, 20), np.full(20, 0.5)]
    ripple = np.array([[5.0, 0.0, 0.3]])
    m.update(np.r_[wall, ripple], (0.0, 0.0, 0.0))
    assert len(m.occupied((0.0, 0.0, 0.0))) == 0                      # nothing confirmed after 1 scan
    m.update(wall - [1.0, 0, 0], (1.0, 0.0, 0.0))                     # car moved 1 m, wall fixed in world
    occ = m.occupied((1.0, 0.0, 0.0))
    assert len(occ) > 0 and np.all(np.abs(occ[:, 0] - 9.0) < 0.3)    # wall 9 m ahead now, ripple absent
    for _ in range(6):
        m.update(np.zeros((0, 3)), (1.0, 0.0, 0.0))                   # wall gone (car drove away)
    assert len(m.occupied((1.0, 0.0, 0.0))) == 0
