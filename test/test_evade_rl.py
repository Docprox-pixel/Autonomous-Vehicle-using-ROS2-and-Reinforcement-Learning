"""RL sudden-obstacle layer: active only for something in our lane; the oncoming-traffic envelope
always holds, whatever the policy wants."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_car_core import evade_rl as E  # noqa: E402


class GoAroundLeft:
    def predict(self, o, deterministic=True):
        return np.array([0.0, 1.0]), None             # lateral target +3.5 m (oncoming lane)


def straight_route():
    return np.c_[np.arange(-2.0, 60.0, 1.0), np.zeros(62)]


def box(x, y, n=12):
    g = np.mgrid[x:x + 0.6:0.2, y - 0.3:y + 0.3:0.2].reshape(2, -1).T
    return np.c_[g, np.full(len(g), 0.5)]


def make():
    ev = E.Evader(model_path=None)
    ev.model = GoAroundLeft()
    return ev


def test_inactive_when_lane_clear():
    ev = make()
    side = box(20.0, -4.5)                             # on the pavement, not in our lane
    act, a, y, why = ev.step(straight_route(), side, [], 8.0, 0.0)
    assert not act


def test_envelope_blocks_centre_line_with_oncoming():
    ev = make()
    ped = box(20.0, 0.0)
    oncoming = [(40.0, 3.5, -10.0, 0.0, 'car')]       # 40 m away, closing at 18 m/s -> meets in ~2 s
    act, a, y, why = ev.step(straight_route(), ped, oncoming, 8.0, 0.0)
    assert act and y <= E.CENTRE + 1e-6 and why.startswith('envelope')


def test_may_go_round_when_other_lane_free():
    ev = make()
    act, a, y, why = ev.step(straight_route(), box(20.0, 0.0), [], 8.0, 0.0)
    assert act and y > E.CENTRE
