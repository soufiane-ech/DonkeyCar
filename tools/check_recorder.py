"""
Validation of donkey.recorder (Step 3 of the roadmap).

Records ~30 s of driving with a smooth *random* controller. Random steering is
only used to exercise the pipeline: these sessions are NOT training data and
are written to data/test_sessions/ (not data/raw/).

The run exercises every recorder feature:
  * continuous recording at 20 Hz,
  * a 2 s recording pause (must appear as exactly one time gap),
  * car resets on collision / stuck / timeout (new episode_id),
  * discard_last(1.0) right before closing,
then validates the session folder and checks that two sessions created in
the same second never collide.

Usage:
    python tools/check_recorder.py
    python tools/check_recorder.py --duration 60 --throttle 0.3
"""

import argparse
import logging
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey.config import (CameraConfig, CarConfig, PROJECT_ROOT, SimConfig,   # noqa: E402
                           config_dict)
from donkey.recorder import SessionRecorder                                  # noqa: E402
from donkey.session import print_report, validate_session                    # noqa: E402
from donkey.sim_client import DonkeyClient, SimConnectionError               # noqa: E402

TEST_ROOT = PROJECT_ROOT / "data" / "test_sessions"


class SmoothRandomController:
    """Test-only controller: steering eases toward a random target that changes
    every 0.5-1.5 s; constant throttle."""

    def __init__(self, throttle: float, seed: int = 0):
        self.rng = random.Random(seed)
        self.throttle = throttle
        self.steering, self.target, self.next_change = 0.0, 0.0, 0.0

    def __call__(self, now: float):
        if now >= self.next_change:
            self.target = self.rng.uniform(-0.7, 0.7)
            self.next_change = now + self.rng.uniform(0.5, 1.5)
        self.steering += 0.15 * (self.target - self.steering)
        return round(self.steering, 4), self.throttle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--throttle", type=float, default=0.25)
    ap.add_argument("--pause", type=float, nargs=2, default=(10.0, 12.0),
                    metavar=("START", "END"), help="recording pause window (s)")
    ap.add_argument("--episode-timeout", type=float, default=12.0, help="reset car every N s")
    ap.add_argument("--keep-collision-dirs", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("PIL").setLevel(logging.INFO)

    sim, car, cam = SimConfig(), CarConfig(), CameraConfig()
    ctrl = SmoothRandomController(args.throttle)
    resets = {"hit": 0, "stuck": 0, "timeout": 0}
    try:
        with DonkeyClient(sim, car, cam) as client:
            start_info = client.start()
            client.reset_car()
            time.sleep(0.5)
            client.flush()
            rec = SessionRecorder(
                scene=sim.scene, tag="recorder_check", root=TEST_ROOT,
                config=config_dict(sim, car, cam),
                notes="Pipeline test with a random controller - NOT training data.",
                extra_meta={"controller": "SmoothRandomController (test only)",
                            "client_start": start_info})
            with rec:
                t0 = time.perf_counter()
                ep_start, slow_since, pause_seen = t0, None, False
                while (now := time.perf_counter()) - t0 < args.duration:
                    tel = client.get_frame(timeout=1.0)
                    if tel is None:
                        raise SimConnectionError("telemetry stopped")
                    steering, throttle = ctrl(now)
                    cmd = client.send_controls(steering, throttle)

                    recording = not (args.pause[0] <= now - t0 < args.pause[1])
                    pause_seen |= not recording
                    if recording:
                        rec.record(tel, cmd, lap_count=len(client.lap_events))

                    # reset rules (test only): collision, stuck, or timeout.
                    # The first frames after reset_car still report the old 'hit'
                    # and a meaningless cte (observed), hence the 1 s cooldown.
                    slow_since = (slow_since or now) if tel.speed < 0.05 else None
                    reason = None
                    if now - ep_start < 1.0 and rec.episode_id > 0:
                        pass
                    elif tel.hit != "none":
                        reason = "hit"
                    elif slow_since and now - slow_since > 2.0 and now - ep_start > 3.0:
                        reason = "stuck"
                    elif now - ep_start > args.episode_timeout:
                        reason = "timeout"
                    if reason:
                        resets[reason] += 1
                        client.reset_car()
                        rec.new_episode(reason)
                        ep_start, slow_since = now, None

                n_disc = rec.discard_last(1.0)
                recorded = rec.n_recorded
            session_dir = rec.dir
    except SimConnectionError as e:
        print("FAIL:", e)
        return 1

    print(f"\nrecorded={recorded}, discarded={n_disc}, resets={resets}, pause={pause_seen}\n")
    report = validate_session(session_dir)
    print_report(report)

    ok = report["ok"]
    extra = []
    written = report["samples"]
    extra.append(("written == recorded - discarded", written == recorded - n_disc,
                  f"{written} == {recorded} - {n_disc}"))
    extra.append(("discard removed ~1 s (18-22 frames)", 18 <= n_disc <= 22, f"{n_disc}"))
    big_gaps = [g for g in report["checks"]["time gaps (info)"]["gaps"] if g["dt_s"] > 1.0]
    small_gaps = [g for g in report["checks"]["time gaps (info)"]["gaps"] if g["dt_s"] <= 1.0]
    expected_pause = args.pause[1] - args.pause[0]
    # The pause is defined in real time, so compare its wall-clock length.
    extra.append(("pause = exactly one ~2 s gap",
                  len(big_gaps) == 1 and abs(big_gaps[0]["wall_dt_s"] - expected_pause) < 0.2,
                  f"{big_gaps}"))
    extra.append(("no other frame gaps", not small_gaps, f"{small_gaps}"))
    episodes = report["checks"]["episode_id non-decreasing"]["episodes"]
    extra.append(("episodes match resets", len(episodes) == 1 + sum(resets.values())
                  or len(episodes) == sum(resets.values()),  # last reset may fall in the discarded tail
                  f"episodes={episodes}, resets={sum(resets.values())}"))

    # Two sessions created within the same second must not collide.
    a = SessionRecorder(scene=sim.scene, tag="collision_test", root=TEST_ROOT)
    b = SessionRecorder(scene=sim.scene, tag="collision_test", root=TEST_ROOT)
    a.close(), b.close()
    extra.append(("unique session folders", a.dir != b.dir and a.dir.exists() and b.dir.exists(),
                  f"{a.dir.name} / {b.dir.name}"))
    if not args.keep_collision_dirs:
        for d in (a.dir, b.dir):   # empty folders created by this test only
            shutil.rmtree(d)

    print("\nRecorder-specific checks:")
    for name, passed, detail in extra:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<36} {detail}")
        ok &= passed
    print(f"\nSession folder: {session_dir}")
    print("RESULT:", "ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
