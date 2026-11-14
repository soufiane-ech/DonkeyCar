"""
Learn a per-node speed profile for the autopilot by trial and error
(iterative learning control, one update per lap):

  * after each lap, nodes where the car went far from the centre line
    (max |cte| > --c-hi) get a LOWER target speed on the W nodes before them
    (= brake earlier for that corner);
  * nodes where the car stayed well centred (|cte| < --c-lo over the next W
    nodes) get a slightly HIGHER target speed;
  * a track departure (|cte| > --cte-limit) lowers the speed more strongly
    and resets the car.

    python tools/optimize_profile.py --laps 30
    python tools/optimize_profile.py --init models/profiles/<file>.json --laps 20

Output: models/profiles/profile_<timestamp>.json  (best clean lap + final profile)
        docs/experiments/profile_<timestamp>.jsonl (one line per lap)
Use it: python scripts/run_driver.py --profile models/profiles/<file>.json
"""

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey.autopilot import AutopilotSettings, TrackAutopilot   # noqa: E402
from donkey.config import PROJECT_ROOT                            # noqa: E402
from donkey.sim_client import DonkeyClient                        # noqa: E402

BASE = AutopilotSettings(v_min=2.5, v_max=5.0, max_throttle=1.0, brake_gain=1.0, brake_margin=0.2,
                         k_speed=0.5, lookahead_per_speed=0.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=30, help="number of lap updates")
    ap.add_argument("--init", default=None, help="start from this profile JSON")
    ap.add_argument("--c-hi", type=float, default=1.8)
    ap.add_argument("--c-lo", type=float, default=1.0)
    ap.add_argument("--inc", type=float, default=0.12, help="speed increase per lap on safe nodes")
    ap.add_argument("--dec", type=float, default=0.08, help="relative decrease on risky nodes")
    ap.add_argument("--window", type=int, default=6, help="nodes before a risky node that slow down")
    ap.add_argument("--v-cap", type=float, default=8.0)
    ap.add_argument("--cte-limit", type=float, default=3.0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--max-minutes", type=float, default=30.0)
    args = ap.parse_args()

    settings = BASE
    init = json.loads(Path(args.init).read_text()) if args.init else None
    if init:
        settings = dataclasses.replace(settings, **init["settings"])
    settings = dataclasses.replace(settings, **{k: float(v) if "." in v else int(v)
                                                for k, v in (kv.split("=") for kv in args.set)})
    stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out = PROJECT_ROOT / "models" / "profiles" / f"profile_{stamp}.json"
    log = PROJECT_ROOT / "docs" / "experiments" / f"profile_{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    with DonkeyClient() as client:
        client.start()
        first = client.get_frame(timeout=2.0)
        n_nodes = first.total_nodes
        nodes = client.fetch_track(n_nodes)
        heuristic = TrackAutopilot(nodes, settings)
        prof = list(init["profile"]) if init else [heuristic.target_speed(i) for i in range(n_nodes)]
        pilot = TrackAutopilot(nodes, settings, prof)
        pilot.profile = prof                       # same list object: updates apply immediately

        best = None
        lap_idx, lap_start, prev_node, visited = 0, None, None, set()
        max_cte = [0.0] * n_nodes
        cooldown_until, departures, flying = None, 0, False
        t_stop = time.perf_counter() + 60 * args.max_minutes

        def reset(tel):
            nonlocal cooldown_until, lap_start, prev_node, visited, flying
            client.reset_car()
            cooldown_until = tel.sim_time + 1.0
            lap_start, prev_node, visited, flying = None, None, set(), False

        last = client.get_frame(timeout=2.0)
        prof_before, best_profile = list(prof), None
        reset(last)
        while lap_idx < args.laps and time.perf_counter() < t_stop:
            tel = client.get_frame(timeout=1.0)
            if tel is None:
                continue
            last = tel
            if cooldown_until is not None:
                client.send_controls(0.0, 0.0, 0.0)
                if tel.sim_time < cooldown_until or tel.hit != "none" or abs(tel.cte) > 1.0:
                    continue
                cooldown_until = None
            steer, thr, brake = pilot(tel)
            client.send_controls(steer, thr, brake)
            n = tel.active_node
            if lap_start is None:
                lap_start = tel.sim_time
            max_cte[n] = max(max_cte[n], abs(tel.cte))

            departed = abs(tel.cte) > args.cte_limit or tel.hit != "none"
            lap_done = (prev_node is not None and prev_node >= n_nodes - 6 and n <= 5
                        and len(visited) >= 0.8 * n_nodes)
            visited.add(n)
            prev_node = n
            if not (departed or lap_done):
                continue

            # ---- update the profile (iterative learning)
            lap_time = tel.sim_time - lap_start
            risky = [i for i in range(n_nodes) if max_cte[i] > args.c_hi]
            if departed:
                departures += 1
                for j in range(n - 10, n + 1):
                    prof[j % n_nodes] = max(1.5, prof[j % n_nodes] * 0.85)
            for i in risky:
                for j in range(i - args.window, i + 1):
                    prof[j % n_nodes] = max(1.5, prof[j % n_nodes] * (1 - args.dec))
            if lap_done:
                for i in range(n_nodes):
                    ahead = [max_cte[(i + k) % n_nodes] for k in range(args.window + 1)]
                    if max(ahead) < args.c_lo:
                        prof[i] = min(args.v_cap, prof[i] + args.inc)
            rec = {"lap": lap_idx, "event": "departure" if departed else "lap",
                   "flying": flying, "time_s": round(lap_time, 2), "at_node": n,
                   "max_cte": round(max(max_cte), 2), "risky_nodes": len(risky),
                   "mean_target": round(sum(prof) / n_nodes, 3)}
            print(json.dumps(rec), flush=True)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**rec, "profile": [round(v, 3) for v in prof]}) + "\n")
            clean = lap_done and flying and max(max_cte) < args.c_hi
            if lap_done and flying and max(max_cte) < args.cte_limit and (best is None or lap_time < best["time_s"]):
                best = {"time_s": round(lap_time, 2), "max_cte": round(max(max_cte), 2),
                        "clean_margin": clean, "lap": lap_idx}
                best_profile = list(prof_before)
            lap_idx += 1
            max_cte = [0.0] * n_nodes
            if departed:
                reset(tel)
            else:
                lap_start, visited, flying = tel.sim_time, set(), True
            prof_before = list(prof)
            pilot.profile = prof
        client.send_controls(0.0, 0.0, 1.0)

    result = {"settings": dataclasses.asdict(settings), "profile": [round(v, 3) for v in prof],
              "best": best, "best_profile": [round(v, 3) for v in best_profile] if best else None,
              "departures": departures, "laps": lap_idx, "log": str(log.name),
              "params": vars(args)}
    out.write_text(json.dumps(result, indent=1))
    if best:   # the profile that produced the fastest flying lap, directly usable by run_driver
        out.with_name(out.stem + "_best.json").write_text(json.dumps(
            {**result, "profile": result["best_profile"]}, indent=1))
    print("best flying lap:", best)
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
