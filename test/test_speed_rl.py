"""RL speed control: whatever the policy wants, the speed never exceeds the planner's target."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_car_core.speed_rl import SpeedController  # noqa: E402


class FullThrottle:
    def predict(self, o, deterministic=True):
        return np.array([1.0]), None


def test_never_above_target():
    c = SpeedController(model_path=None)
    c.model = FullThrottle()
    for v, vt in ((5.0, 5.0), (5.0, 4.0), (0.0, 0.0), (3.0, 3.05)):
        a = c.accel(v, 0.0, vt, 200.0, 1e9, 13.9)
        # at or below the target after the step, or braking as hard as allowed when it dropped
        assert v + a * 0.1 <= vt + 1e-6 or a == -3.0


def test_rule_fallback_brakes_for_stop_target():
    c = SpeedController(model_path=None)
    a = c.accel(8.0, 0.0, 2.0, 200.0, 10.0, 13.9, firm=True)
    assert -3.0 <= a < -2.9


def test_observation_shape():
    o = SpeedController.observation(5.0, 0.2, 6.0, 30.0, 2.0, 200.0, 13.9, False)
    assert o.shape == (10,) and np.isfinite(o).all()
