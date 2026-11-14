"""
Telemetry autopilot used as the *teacher* for the CNN (no human driving needed).

It uses privileged simulator information that the CNN never sees:
  * track centre-line nodes (DonkeyClient.fetch_track, `node_position` message),
  * active_node, yaw, cte, speed from telemetry.

Steering: aim at the track direction a few nodes ahead (anticipates turns)
          + a small correction of the cross-track error.
Throttle: speed target that drops before sharp turns.
Noise:    perturbations are added to the *applied* steering while the clean
          command is recorded as the label -> the dataset contains recoveries.

Verified conventions (2026-11-13): node heading = atan2(dx, dz) in degrees,
same frame as telemetry yaw; positive steering turns right = yaw increases;
cte sign: steering must be -k * cte.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class AutopilotSettings:
    lookahead: int = 3          # nodes ahead used to aim (nodes are ~2.66 units apart)
    k_heading: float = 1 / 30   # steering per degree of heading error
    k_cte: float = 0.25         # steering per unit of cte
    cte_sign: float = -1.0
    speed_lookahead: int = 7    # nodes ahead used to anticipate turns
    v_max: float = 4.0          # target speed on straights (sim units)
    v_min: float = 2.0          # target speed in the sharpest turns
    turn_full: float = 70.0     # degrees of upcoming turn that mean "sharpest"
    k_speed: float = 0.3        # throttle per unit of speed error
    throttle_ff: float = 0.25
    max_throttle: float = 0.7
    brake_margin: float = 1.0   # brake if faster than target by this much
    brake_gain: float = 0.0     # 0: fixed brake 0.3 (v1 behaviour); >0: brake = gain * overspeed (max 1)
    lookahead_per_speed: float = 0.0   # extra aim nodes per unit of speed (aim further when fast)


class TrackAutopilot:
    """`profile` (optional): target speed for each track node, e.g. learned by
    tools/optimize_profile.py. Without it, the speed target comes from the
    upcoming-turn heuristic (v_max .. v_min)."""

    def __init__(self, nodes: List[Dict], s: AutopilotSettings = AutopilotSettings(), profile=None):
        self.s = s
        self.profile = list(profile) if profile is not None else None
        self.n = len(nodes)
        xz = [(p["pos_x"], p["pos_z"]) for p in nodes]
        # direction of each segment i -> i+1
        self.seg = [math.degrees(math.atan2(xz[(i + 1) % self.n][0] - xz[i][0],
                                            xz[(i + 1) % self.n][1] - xz[i][1])) % 360
                    for i in range(self.n)]
        self.xz = xz

    def reset(self):
        pass

    def _chord(self, a: int, b: int) -> float:
        (x0, z0), (x1, z1) = self.xz[a % self.n], self.xz[b % self.n]
        return math.degrees(math.atan2(x1 - x0, z1 - z0)) % 360

    def upcoming_turn(self, node: int) -> float:
        """Largest heading change (deg) over the next `speed_lookahead` nodes."""
        h0 = self.seg[node % self.n]
        return max(abs(_wrap180(self.seg[(node + k) % self.n] - h0))
                   for k in range(1, self.s.speed_lookahead + 1))

    def target_speed(self, node: int) -> float:
        if self.profile is not None:
            return self.profile[node % self.n]
        turn = self.upcoming_turn(node)
        return self.s.v_max - (self.s.v_max - self.s.v_min) * min(1.0, turn / self.s.turn_full)

    def __call__(self, tel):
        s, node = self.s, max(tel.active_node, 0)
        look = s.lookahead + int(round(s.lookahead_per_speed * max(tel.speed, 0.0)))
        target = self._chord(node, node + look)
        heading_err = _wrap180(target - tel.yaw)
        steering = s.k_heading * heading_err + s.cte_sign * s.k_cte * tel.cte
        steering = max(-1.0, min(1.0, steering))
        v_target = self.target_speed(node)
        throttle = max(0.0, min(s.max_throttle, s.throttle_ff + s.k_speed * (v_target - tel.speed)))
        over = tel.speed - (v_target + s.brake_margin)
        if s.brake_gain > 0:
            brake = min(1.0, s.brake_gain * over) if over > 0 else 0.0
        else:
            brake = 0.3 if over > 0 else 0.0
        if brake:
            throttle = 0.0
        return steering, throttle, brake


class SteeringNoise:
    """Occasional steering perturbations (so the teacher demonstrates recoveries)."""

    def __init__(self, every_s=4.0, duration_s=0.5, amplitude=0.4, jitter=0.05):
        self.every_s, self.duration_s, self.amplitude, self.jitter = every_s, duration_s, amplitude, jitter
        self.t, self.until, self.value, self.next_at = 0.0, -1.0, 0.0, every_s

    def __call__(self, dt: float) -> float:
        self.t += dt
        if self.t >= self.next_at:
            self.value = random.choice((-1, 1)) * self.amplitude * random.uniform(0.5, 1.0)
            self.until = self.t + self.duration_s * random.uniform(0.5, 1.5)
            self.next_at = self.t + self.every_s * random.uniform(0.6, 1.4)
        burst = self.value if self.t < self.until else 0.0
        return burst + random.gauss(0.0, self.jitter)

    @property
    def active(self) -> bool:
        return self.t < self.until
