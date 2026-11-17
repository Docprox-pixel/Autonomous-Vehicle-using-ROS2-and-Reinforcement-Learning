"""RL DECISION LAYER: when to go at a give-way point (minor-road junction, roundabout entry).

The planner drives; this module only answers the question a driver answers at the line:
YIELD (stop / stay at the line), CREEP (edge up to the line slowly), GO (enter and clear it).
Policy: PPO trained in rl/decision_env.py (rl/train_decision.py), evaluated against the old
4 s gap rule (rl/eval_decision.py). The observation is built here exactly as in training.

Always wrapped by hard rules (an RL policy is never the last word on safety):
  * committed: once the bumper is past the line the answer is GO (never stop inside a junction),
  * envelope:  never GO near the line while a priority vehicle is < SAFE_TTC s from the conflict,
  * the safety shield downstream still brakes for anything in the swept path.
Without the model file (or stable-baselines3) it falls back to the 4 s gap rule.
"""
import math
import os

import numpy as np

YIELD, CREEP, GO = 0, 1, 2
NAMES = {YIELD: 'yield', CREEP: 'creep', GO: 'go'}
SAFE_TTC = 1.5
HALF_W, OTHER_L = 1.0, 4.6
DEFAULT_MODEL = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'rl', 'models',
                                              'giveway_ppo.zip'))


def arrivals(tracks, jx, jy):
    """Per side (0: from the left, 1: from the right) the vehicles approaching the junction point
    (jx, jy) (base_link): sorted (time to conflict, distance, speed)."""
    out = ([], [])
    for tx, ty, tvx, tvy in tracks:
        sp = math.hypot(tvx, tvy)
        if sp < 0.5:
            continue
        ux, uy = tvx / sp, tvy / sp
        rx, ry = jx - tx, jy - ty
        d = rx * ux + ry * uy                       # distance still to go to the conflict point
        lateral = abs(-rx * uy + ry * ux)
        if d < -(OTHER_L + HALF_W + 1.0) or lateral > 8.0:
            continue
        if ux > 0.85:                                # going our way (queued behind / ahead of us)
            continue
        side = 0 if tvy < 0 else 1                   # from the left moves -y (right-wards), as in training
        out[side].append((max(d - (HALF_W + OTHER_L / 2), 0.0) / sp, d, sp))
    for o in out:
        o.sort(key=lambda q: q[1])
    return out


def observation(d_stop, v, waited, arr):
    o = [float(np.clip(-d_stop / 30.0, -1.5, 1.5)), v / 10.0, min(waited / 20.0, 1.5)]
    for side in (0, 1):
        a = arr[side][:3]
        for k in range(3):
            if k < len(a):
                ttc, d, sp = a[k]
                o += [min(ttc, 10.0) / 10.0, float(np.clip(d / 50.0, -1.0, 1.0)), sp / 15.0]
            else:
                o += [1.0, 1.0, 0.0]
    return np.array(o, np.float32)


A_GO, V_GO, CAR_LEN, OTHER_W, LANE_OFF = 2.0, 7.0, 5.2, 1.9, (2.75, 6.25)


def t_clear(dist, v, a=A_GO, vmax=V_GO):
    if dist <= 0:
        return 0.0
    t_acc = max(vmax - v, 0.0) / a
    d_acc = v * t_acc + 0.5 * a * t_acc * t_acc
    if d_acc >= dist:
        return (-v + math.sqrt(v * v + 2 * a * dist)) / a
    return t_acc + (dist - d_acc) / vmax


def gap_ok(d_stop, v, arr, margin=0.5):
    """Envelope (same as training): for each crossing side the next vehicle arrives only after our
    car (rear bumper) has cleared that lane, accelerating from its current speed."""
    for side in (0, 1):
        need = t_clear(d_stop + LANE_OFF[side] + OTHER_W / 2 + CAR_LEN, v) + margin
        for ttc, d, sp in arr[side]:
            if d > -(OTHER_L / 2 + HALF_W) and ttc < need:
                return False
    return True


class GiveWayDecider:
    def __init__(self, model_path=DEFAULT_MODEL, rule_horizon=4.0):
        self.model = None
        self.rule_horizon = rule_horizon
        self.source = 'rule'
        self.last = YIELD
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path, device='cpu')
                self.source = 'rl'
            except Exception as e:                    # pragma: no cover - missing deps
                self.error = str(e)

    def decide(self, d_stop, v, waited, tracks, jx, jy):
        """d_stop: bumper to the give-way line (m, < 0 past it); tracks: (x, y, vx, vy) base_link.
        Returns (action, reason)."""
        if d_stop < -0.5 or (self.last == GO and d_stop - v * v / (2 * 3.5) < 0.3):
            self.last = GO
            return GO, 'committed'           # past the line, or a GO that can no longer stop before it
        arr = arrivals(tracks, jx, jy)
        min_ttc = min([q[0] for side in arr for q in side if q[1] > -(OTHER_L / 2 + HALF_W)] or [math.inf])
        if self.model is not None:
            a = int(self.model.predict(observation(d_stop, v, waited, arr), deterministic=True)[0])
            why = 'rl'
        else:
            a = GO if min_ttc > self.rule_horizon else YIELD
            why = 'rule'
        if a == GO and not gap_ok(d_stop, v, arr):
            a, why = (CREEP if d_stop > 3.0 else YIELD), 'envelope'
        self.last = a
        return a, why
