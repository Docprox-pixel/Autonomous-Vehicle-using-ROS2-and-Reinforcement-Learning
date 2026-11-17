"""RL decision layer: the car builds the SAME observation the policy was trained on, and the hard
rules (committed past the line, 1.5 s envelope) always override the policy."""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'rl'))
from auto_car_core import decision as D  # noqa: E402
from decision_env import LANE_DIR, LANES_X, GiveWayEnv  # noqa: E402


def env_tracks(env):
    """The env's crossing cars as the tracker reports them: base_link (x, y, vx, vy). base_link is
    the rear axle, 4.05 m behind the front bumper; the env's x is the bumper position."""
    out = []
    for c in env.cars:
        out.append((LANES_X[c.lane] - env.x + 4.05, c.y, 0.0, LANE_DIR[c.lane] * c.v))
    return out


def test_observation_matches_training():
    env = GiveWayEnv(seed=3, rate_range=(0.25, 0.3))
    for ep in range(5):
        obs, _ = env.reset(seed=50 + ep)
        for _ in range(20):
            obs, *_ = env.step(0)
        tr = env_tracks(env)
        # junction point per lane: the conflict point of each lane in base_link
        for side, lane in ((0, 0), (1, 1)):
            arr = D.arrivals([t for t in tr if (t[3] < 0) == (lane == 0)], LANES_X[lane] - env.x + 4.05, 0.0)
            ref = env._arrivals(lane)
            got = arr[0] + arr[1]
            assert len(got) >= min(len(ref), 1) - 0
            if ref and got:
                assert abs(got[0][0] - ref[0][0]) < 0.05 and abs(got[0][1] - ref[0][1]) < 0.05


def test_side_assignment_left_is_lane0():
    # a car coming from the left (+y) moving -y towards the junction ahead
    arr = D.arrivals([(8.0, 20.0, 0.0, -10.0)], 8.0, 0.0)
    assert len(arr[0]) == 1 and len(arr[1]) == 0
    arr = D.arrivals([(8.0, -20.0, 0.0, 10.0)], 8.0, 0.0)
    assert len(arr[1]) == 1 and len(arr[0]) == 0


def test_hard_rules_override_policy():
    dec = D.GiveWayDecider(model_path=None)            # rule fallback; envelope / commit still apply
    # committed: past the line -> GO even with a car about to arrive
    a, why = dec.decide(-1.0, 3.0, 0.0, [(8.0, 8.0, 0.0, -10.0)], 8.0, 0.0)
    assert a == D.GO and why == 'committed'
    # a car 0.5 s from the conflict point, we are at the line: never GO

    class AlwaysGo:
        def predict(self, o, deterministic=True):
            return np.array(D.GO), None
    dec.model = AlwaysGo()
    a, why = dec.decide(0.5, 0.0, 3.0, [(6.0, 8.0, 0.0, -12.0)], 6.0, 0.0)
    assert a == D.YIELD and why == 'envelope'
    # empty road: the policy's GO stands
    a, why = dec.decide(0.5, 0.0, 3.0, [], 6.0, 0.0)
    assert a == D.GO
