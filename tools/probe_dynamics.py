"""
Car dynamics probe on the start straight: speed reached vs throttle, and braking.
Steering is held on the track by the autopilot so the car stays on the straight.

    python tools/probe_dynamics.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey.autopilot import TrackAutopilot   # noqa: E402
from donkey.sim_client import DonkeyClient    # noqa: E402


def run(client, ap, throttle, seconds, brake=None, brake_s=2.0):
    client.reset_car()
    time.sleep(1.5)
    client.flush()
    t0, out = None, []
    phase = "accel"
    while True:
        tel = client.get_frame(timeout=1.0)
        if tel is None:
            continue
        if t0 is None:
            t0 = tel.sim_time
        t = tel.sim_time - t0
        steer, _, _ = ap(tel)
        if t < seconds:
            client.send_controls(steer, throttle, 0.0)
        elif brake is not None and t < seconds + brake_s:
            phase = "brake"
            client.send_controls(steer, 0.0, brake)
        else:
            break
        out.append((round(t, 2), phase, round(tel.speed, 2), tel.active_node, round(tel.cte, 2)))
    client.send_controls(0, 0, 1.0)
    return out


with DonkeyClient() as c:
    c.start()
    first = c.get_frame(timeout=2.0)
    ap = TrackAutopilot(c.fetch_track(first.total_nodes))
    for thr in (0.3, 0.5, 0.7, 1.0):
        rows = run(c, ap, thr, 3.0)
        print(f"throttle {thr}: speed at t=1,2,3 s ->",
              [r[2] for r in rows if r[0] in (1.0, 2.0, 2.95)], " node at end", rows[-1][3])
    rows = run(c, ap, 1.0, 2.5, brake=1.0, brake_s=2.0)
    b = [r for r in rows if r[1] == "brake"]
    print("brake 1.0 from", b[0][2], ":", [r[2] for r in b[::4]])
    rows = run(c, ap, 1.0, 2.5, brake=0.3, brake_s=2.0)
    b = [r for r in rows if r[1] == "brake"]
    print("brake 0.3 from", b[0][2], ":", [r[2] for r in b[::4]])
    rows = run(c, ap, 1.0, 2.5, brake=0.0, brake_s=2.0)
    b = [r for r in rows if r[1] == "brake"]
    print("coast    from", b[0][2], ":", [r[2] for r in b[::4]])
