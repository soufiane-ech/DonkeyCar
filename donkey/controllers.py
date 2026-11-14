"""
Driving controllers. A controller turns inputs into a (steering, throttle, brake)
command; it does not talk to the simulator itself.

KeyboardController is pure logic (no pygame import) so it can be unit-tested:
the drive script reads the keys and passes booleans to update().
"""

from dataclasses import dataclass


def _approach(value: float, target: float, max_step: float) -> float:
    """Move `value` toward `target` by at most `max_step`."""
    if value < target:
        return min(target, value + max_step)
    return max(target, value - max_step)


@dataclass
class KeyboardSettings:
    # Steering: holding a key ramps toward full lock; releasing returns to center.
    steer_rate: float = 2.0        # units/s while a key is held  -> full lock in 0.5 s
    steer_return_rate: float = 3.0 # units/s back to 0 when released
    steer_reverse_rate: float = 5.0  # units/s when pressing the opposite direction
    max_steering: float = 1.0
    # Throttle
    cruise_throttle: float = 0.25  # constant speed used for v1 data collection
    boost_throttle: float = 0.35   # while holding accelerate
    throttle_rate: float = 1.0     # units/s up
    throttle_release_rate: float = 1.5  # units/s down when nothing is held
    cruise_step: float = 0.05
    cruise_limits: tuple = (0.05, 0.6)


class KeyboardController:
    """Smooth keyboard driving.

    Steering ramps continuously instead of jumping between -1/0/+1, so a short
    tap gives a small correction and a long press a sharp turn.

    Throttle: with cruise ON the car holds `cruise_throttle` without any key
    (recommended for collection: consistent speed). Accelerate = boost,
    brake = full brake and cancels cruise.
    """

    def __init__(self, settings: KeyboardSettings = None):
        self.s = settings or KeyboardSettings()   # own copy: adjust_cruise() mutates it
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.cruise = False

    def reset(self) -> None:
        self.steering = self.throttle = self.brake = 0.0
        self.cruise = False

    def toggle_cruise(self) -> bool:
        self.cruise = not self.cruise
        return self.cruise

    def adjust_cruise(self, direction: int) -> float:
        lo, hi = self.s.cruise_limits
        self.s.cruise_throttle = round(min(hi, max(lo, self.s.cruise_throttle
                                                   + direction * self.s.cruise_step)), 3)
        return self.s.cruise_throttle

    def update(self, left: bool, right: bool, accel: bool, brake: bool, dt: float):
        """Advance by `dt` seconds with the given key states.
        Returns (steering, throttle, brake)."""
        dt = max(0.0, min(dt, 0.1))   # avoid big jumps after a stall
        s = self.s

        # ---- steering
        target = (-s.max_steering if left else 0.0) + (s.max_steering if right else 0.0)
        if target == 0.0:
            rate = s.steer_return_rate
        elif self.steering * target < 0:      # pressing the opposite direction
            rate = s.steer_reverse_rate
        else:
            rate = s.steer_rate
        self.steering = _approach(self.steering, target, rate * dt)

        # ---- throttle / brake
        if brake:
            self.cruise = False
            self.throttle = 0.0
            self.brake = 1.0
        else:
            self.brake = 0.0
            if accel:
                t_target, rate = max(s.boost_throttle, s.cruise_throttle), s.throttle_rate
            elif self.cruise:
                t_target = s.cruise_throttle
                rate = s.throttle_rate if self.throttle < t_target else s.throttle_release_rate
            else:
                t_target, rate = 0.0, s.throttle_release_rate
            self.throttle = _approach(self.throttle, t_target, rate * dt)

        return round(self.steering, 4), round(self.throttle, 4), self.brake
