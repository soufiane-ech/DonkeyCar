"""
Drive the car with the autopilot (teacher) or the trained CNN, optionally record,
and measure performance (laps, lap times, collisions, track departures).

    # tune / check the teacher (no recording)
    python scripts/run_driver.py --driver autopilot --minutes 2
    # collect the training dataset (noise injection -> recoveries)
    python scripts/run_driver.py --driver autopilot --record --noise --minutes 10
    # autonomous driving with the CNN (camera only)
    python scripts/run_driver.py --driver cnn --model models/cnn.pt --minutes 3

The CNN only sees the camera image (+ current speed); cte/position are used
here only to measure performance and reset the car after a departure.
"""

import argparse
import dataclasses
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey.autopilot import AutopilotSettings, SteeringNoise, TrackAutopilot   # noqa: E402
from donkey.config import DATA_RAW_DIR, PROJECT_ROOT, SimConfig, config_dict  # noqa: E402
from donkey.recorder import SessionRecorder                                    # noqa: E402
from donkey.session import print_report, validate_session                      # noqa: E402
from donkey.sim_client import DonkeyClient                                     # noqa: E402


class LapCounter:
    """Lap = active_node wraps from the end of the track to the start after
    having visited most of the nodes (resets onto the start line do not count)."""

    def __init__(self):
        self.visited, self.prev, self.lap_start, self.laps, self.lap_times = set(), None, None, 0, []

    def restart(self):
        self.visited, self.prev, self.lap_start = set(), None, None

    def update(self, tel) -> bool:
        n, total = tel.active_node, tel.total_nodes
        if n < 0 or total <= 0:
            return False
        if self.lap_start is None:
            self.lap_start = tel.sim_time
        done = False
        if (self.prev is not None and self.prev >= total - 6 and n <= 5
                and len(self.visited) >= 0.8 * total):
            self.laps += 1
            self.lap_times.append(tel.sim_time - self.lap_start)
            self.lap_start, self.visited, done = tel.sim_time, set(), True
        self.visited.add(n)
        self.prev = n
        return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", choices=["autopilot", "cnn"], default="autopilot")
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "cnn_fast_v1.pt"))
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--noise", action="store_true", help="steering perturbations (autopilot only)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--root", default=str(DATA_RAW_DIR))
    ap.add_argument("--cte-limit", type=float, default=3.0, help="|cte| above = track departure")
    ap.add_argument("--cooldown", type=float, default=1.0)
    ap.add_argument("--vmax", type=float, default=AutopilotSettings.v_max)
    ap.add_argument("--k-heading", type=float, default=AutopilotSettings.k_heading)
    ap.add_argument("--k-cte", type=float, default=AutopilotSettings.k_cte)
    ap.add_argument("--lookahead", type=int, default=AutopilotSettings.lookahead)
    ap.add_argument("--throttle-scale", type=float, default=1.0, help="CNN throttle multiplier")
    ap.add_argument("--steer-scale", type=float, default=1.0, help="CNN steering gain")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any AutopilotSettings field, e.g. --set v_min=3 max_throttle=1")
    ap.add_argument("--profile", default=None, help="JSON speed profile (tools/optimize_profile.py)")
    ap.add_argument("--laps", type=int, default=0, help="stop after this many laps (0 = time only)")
    ap.add_argument("--max-departures", type=int, default=0, help="stop after N departures (0 = never)")
    ap.add_argument("--json-out", default=None, help="append the summary (one JSON line) to this file")
    ap.add_argument("--label", default="", help="free text stored in the JSON summary")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print state every second")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = AutopilotSettings(k_heading=args.k_heading, k_cte=args.k_cte, lookahead=args.lookahead,
                                 v_max=args.vmax, v_min=min(AutopilotSettings.v_min, args.vmax))
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = type(getattr(settings, k))(float(v)) if k != "cte_sign" else float(v)
    settings = dataclasses.replace(settings, **overrides)
    noise = SteeringNoise() if args.noise else None
    cnn = None
    if args.driver == "cnn":
        from donkey.cnn import CnnDriver
        cnn = CnnDriver(args.model)

    laps = LapCounter()
    clean_laps = []          # lap times of laps completed with no departure/collision
    lap_ok = True
    stats = {"skipped": 0, "frames": 0, "departures": 0, "collisions": 0, "stuck": 0, "speed": [], "abs_cte": []}
    rec = None
    with DonkeyClient(SimConfig()) as client:
        info = client.start()
        first = client.get_frame(timeout=2.0)
        profile = None
        if args.profile:
            pj = json.loads(Path(args.profile).read_text())
            profile = pj["profile"]
            settings = dataclasses.replace(settings, **{k: v for k, v in pj.get("settings", {}).items()
                                                         if k not in overrides})
        teacher = TrackAutopilot(client.fetch_track(first.total_nodes), settings, profile)
        if args.record:
            rec = SessionRecorder(
                scene=client.sim_cfg.scene, tag=args.tag or f"{args.driver}{'_noise' if noise else ''}",
                root=Path(args.root), config=config_dict(client.sim_cfg, client.car_cfg, client.cam_cfg),
                hold_seconds=1.0,
                extra_meta={"controller": args.driver, "autopilot_settings": settings.__dict__,
                            "noise": bool(noise), "client_start": info,
                            "reset_cooldown_s": args.cooldown,
                            "label_note": "cmd_* = clean teacher command; applied = cmd + noise "
                                          "(see echo_steering)"})
        client.flush()
        last = client.get_frame(timeout=2.0)

        def do_reset(reason):
            nonlocal cooldown_until, lap_ok
            lap_ok = True
            client.reset_car()
            teacher.reset()
            laps.restart()
            if rec:
                rec.new_episode(reason, sim_time=last.sim_time)
            cooldown_until = last.sim_time + args.cooldown

        cooldown_until = None
        do_reset("start")
        slow_since = None
        t_end = time.perf_counter() + 60 * args.minutes
        while time.perf_counter() < t_end:
            # CNN: always act on the freshest image (stale frames are skipped, so a
            # slow loop never accumulates delay). Autopilot: every frame, in order.
            tel = client.wait_for_latest(timeout=1.0) if cnn else client.get_frame(timeout=1.0)
            if tel is None:
                if client.aborted:
                    raise SystemExit("connection lost")
                continue
            dt = tel.sim_time - last.sim_time if tel.sim_time > last.sim_time else 0.05
            stats["skipped"] += max(0, tel.frame_id - last.frame_id - 1)
            last = tel
            cooling = cooldown_until is not None and (tel.sim_time < cooldown_until or tel.hit != "none")
            if not cooling:
                cooldown_until = None

            steer_label, thr_label, brake_label = teacher(tel)
            if cnn is not None:
                steer, thr, brake = cnn(tel)
                steer = max(-1.0, min(1.0, steer * args.steer_scale))
                thr *= args.throttle_scale
            else:
                steer, thr, brake = steer_label, thr_label, brake_label
                if noise is not None and not cooling:
                    steer = max(-1.0, min(1.0, steer + noise(dt)))
            cmd = client.send_controls(steer, thr, brake)
            if args.verbose and tel.frame_id % 20 == 0:
                print(f"t={tel.sim_time:7.2f} node={tel.active_node:3d} cte={tel.cte:+6.2f} "
                      f"v={tel.speed:4.2f} st={steer:+.2f} th={thr:.2f} hit={tel.hit} cool={cooling}")
            if cooling:
                continue

            if rec:
                label = dict(cmd, steering=steer_label, throttle=thr_label, brake=brake_label)
                rec.record(tel, label, lap_count=laps.laps,
                           mode="recovery" if noise is not None and noise.active else "normal")
            stats["frames"] += 1
            stats["speed"].append(tel.speed)
            stats["abs_cte"].append(abs(tel.cte))
            if laps.update(tel):
                if lap_ok:
                    clean_laps.append(laps.lap_times[-1])
                lap_ok = True
                if not args.quiet:
                    print(f"LAP {laps.laps}: {laps.lap_times[-1]:.2f} s")
                if args.laps and laps.laps >= args.laps:
                    break
            if abs(tel.cte) > args.cte_limit or tel.hit != "none":
                lap_ok = False

            slow_since = (slow_since or tel.sim_time) if tel.speed < 0.3 else None
            if tel.hit != "none":
                stats["collisions"] += 1
                print(f"collision with {tel.hit} at node {tel.active_node}")
                do_reset("hit")
            elif abs(tel.cte) > args.cte_limit:
                stats["departures"] += 1
                print(f"track departure (cte {tel.cte:+.2f}) at node {tel.active_node}")
                do_reset("off_track")
                if args.max_departures and stats["departures"] >= args.max_departures:
                    break
            elif slow_since is not None and tel.sim_time - slow_since > 3.0:
                stats["stuck"] += 1
                print(f"stuck at node {tel.active_node}")
                do_reset("stuck")
                slow_since = None
        client.send_controls(0.0, 0.0, 1.0)

    summary = {
        "driver": args.driver, "minutes": args.minutes, "frames": stats["frames"],
        "laps": laps.laps, "lap_times_s": [round(t, 2) for t in laps.lap_times],
        "best_lap_s": round(min(laps.lap_times), 2) if laps.lap_times else None,
        "clean_laps_s": [round(t, 2) for t in clean_laps],
        "skipped_frames": stats["skipped"],
        "collisions": stats["collisions"], "departures": stats["departures"], "stuck": stats["stuck"],
        "mean_speed": round(statistics.fmean(stats["speed"]), 2) if stats["speed"] else None,
        "mean_abs_cte": round(statistics.fmean(stats["abs_cte"]), 3) if stats["abs_cte"] else None,
        "max_abs_cte": round(max(stats["abs_cte"]), 3) if stats["abs_cte"] else None,
    }
    summary = {"label": args.label, "model": args.model if cnn else None,
               "settings": dataclasses.asdict(settings) if not cnn else None,
               "profile": args.profile,
               "throttle_scale": args.throttle_scale, **summary}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")
    if rec:
        rec.close()
        report = validate_session(rec.dir)
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
