"""
Validation of donkey.sim_client (Step 2 of the roadmap).

Connects, loads/configures the scene, then runs the same loop the recorder
will use: wait for each telemetry frame -> send one command. The car stays
stationary (throttle 0) while the steering follows a slow sine wave, which
lets us measure the command -> telemetry echo delay in frames.

Checks (PASS/FAIL):
  * N frames received, all with the configured image size and decodable;
  * simulator clock at ~20 Hz with no dropped frames (no 'time' gap > 75 ms);
  * the main loop handled every frame (no skipped frame_id);
  * all expected numeric fields parsed (no NaN), no callback errors;
  * echoed steering matches the commanded steering (after the measured delay).

Usage:
    python tools/check_sim_client.py                 # reuse the loaded scene
    python tools/check_sim_client.py --reload        # exit + reload the scene
    python tools/check_sim_client.py --frames 400
"""

import argparse
import logging
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey.config import CameraConfig, SimConfig, TELEMETRY_HZ   # noqa: E402
from donkey.sim_client import DonkeyClient, SimConnectionError    # noqa: E402

CORE_FIELDS = ("sim_time", "steering_angle", "throttle", "speed", "cte", "pos_x", "pos_y",
               "pos_z", "vel_x", "gyro_x", "accel_x", "roll", "pitch", "yaw")


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--scene", default=SimConfig().scene)
    ap.add_argument("--reload", action="store_true", help="exit and reload the scene first")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("PIL").setLevel(logging.INFO)   # silence per-frame PNG debug logs

    sim, cam = SimConfig(scene=args.scene), CameraConfig()
    rows, skipped = [], 0
    try:
        with DonkeyClient(sim=sim, cam=cam) as client:
            t = time.perf_counter()
            info = client.start(force_reload=args.reload)
            print(f"start() OK in {time.perf_counter() - t:.2f} s: {info}")

            last_id, t0, max_backlog = None, time.perf_counter(), 0
            while len(rows) < args.frames:
                tel = client.get_frame(timeout=1.0)
                max_backlog = max(max_backlog, client.backlog)
                if tel is None:
                    print("FAIL: no telemetry for 1 s (aborted=%s)" % client.aborted)
                    return 1
                if last_id is not None and tel.frame_id != last_id + 1:
                    skipped += tel.frame_id - last_id - 1
                last_id = tel.frame_id
                if len(rows) == 0:
                    first_image = tel.image()
                steer = round(0.8 * math.sin(2 * math.pi * 0.25 * (time.perf_counter() - t0)), 4)
                cmd = client.send_controls(steer, 0.0, 0.0)
                rows.append((tel, cmd["steering"], cmd["t_local"] - tel.t_local))
            callback_errors, lap_events = client.callback_errors, len(client.lap_events)
            dropped = client.frames_dropped
    except SimConnectionError as e:
        print("FAIL:", e)
        return 1

    tels = [r[0] for r in rows]
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")

    print()
    sizes = {(t.img_w, t.img_h) for t in tels}
    check("frames received", len(tels) == args.frames, f"{len(tels)}")
    check("image size", sizes == {(cam.img_w, cam.img_h)},
          f"{sizes}, format={tels[-1].image_format}, decoded={first_image.shape} {first_image.dtype}")

    sim_dt = [b.sim_time - a.sim_time for a, b in zip(tels, tels[1:])]
    sim_fps = (len(tels) - 1) / (tels[-1].sim_time - tels[0].sim_time)
    check("simulator rate", abs(sim_fps - TELEMETRY_HZ) < 1.0, f"{sim_fps:.2f} Hz")
    check("no dropped frames (sim clock)", max(sim_dt) < 0.075,
          f"sim dt min/max = {1000*min(sim_dt):.1f}/{1000*max(sim_dt):.1f} ms")
    check("every frame processed in order", skipped == 0 and dropped == 0,
          f"skipped ids={skipped}, buffer drops={dropped}, max backlog={max_backlog}")

    loc_dt = [b.t_local - a.t_local for a, b in zip(tels, tels[1:])]
    print(f"       local arrival dt: median {1000*statistics.median(loc_dt):.1f} ms, "
          f"p95 {1000*pct(loc_dt, .95):.1f} ms, max {1000*max(loc_dt):.1f} ms (info: jitter is ours)")
    react = [r[2] for r in rows]
    print(f"       frame->command reaction: median {1000*statistics.median(react):.2f} ms, "
          f"max {1000*max(react):.2f} ms")

    nan_fields = [f for f in CORE_FIELDS if any(math.isnan(getattr(t, f)) for t in tels)]
    check("all core fields parsed", not nan_fields and callback_errors == 0,
          f"NaN fields={nan_fields}, callback errors={callback_errors}")
    rp = [t.roll for t in tels] + [t.pitch for t in tels]
    check("roll/pitch wrapped to [-180,180)", all(-180 <= v < 180 for v in rp),
          f"range [{min(rp):.3f}, {max(rp):.3f}]")

    # Echo delay: the command sent after frame i shows up as steering_angle in frame i+k.
    # The delay varies with jitter, so for each frame j find which earlier command
    # it echoes exactly (float32 precision), searching k = 1..5.
    cmds, echo = [r[1] for r in rows], [t.steering_angle for t in tels]
    delays = []
    for j in range(6, len(rows)):
        ks = [k for k in range(1, 6) if abs(echo[j] - cmds[j - k]) < 1e-4]
        delays.append(ks[0] if ks else None)
    matched = [d for d in delays if d is not None]
    dist = {k: matched.count(k) for k in sorted(set(matched))}
    check("steering echo = a sent command", len(matched) / len(delays) > 0.98,
          f"{100*len(matched)/len(delays):.1f}% exact matches; delay (frames) distribution {dist}")

    print(f"\nextra keys: {sorted(tels[-1].extra)}  |  lap events: {lap_events}")
    print("\nRESULT:", "ALL PASS" if all(checks) else "SOME CHECKS FAILED")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
