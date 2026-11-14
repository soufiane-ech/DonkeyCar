"""
Tune the autopilot's global settings to minimise the flying-lap time
(coordinate search, one parameter at a time, +/- step).

An evaluation = reset, 1 standing lap + N flying laps. Score = mean flying-lap
time. Rejected if a departure/collision happens or max |cte| > --cte-max
(safety margin, so that a CNN imitating the teacher has room for error).

    python tools/tune_teacher.py --max-evals 30

Output: models/profiles/teacher_<timestamp>.json (best settings, usable with
        run_driver.py --profile ...) and docs/experiments/tune_<timestamp>.jsonl
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

START = AutopilotSettings(v_min=2.5, v_max=5.0, max_throttle=1.0, brake_gain=1.0, brake_margin=0.2,
                          k_speed=0.5, lookahead_per_speed=0.3)
STEPS = {"v_min": 0.25, "v_max": 0.5, "turn_full": 10.0, "lookahead": 1, "lookahead_per_speed": 0.1,
         "k_heading": 0.005, "brake_margin": 0.2, "speed_lookahead": 1, "k_cte": 0.05}
LIMITS = {"v_min": (1.5, 6), "v_max": (2.5, 9), "turn_full": (30, 120), "lookahead": (1, 6),
          "lookahead_per_speed": (0, 1), "k_heading": (0.01, 0.08), "brake_margin": (0, 1.5),
          "speed_lookahead": (2, 14), "k_cte": (0, 0.6)}


def evaluate(client, nodes, s: AutopilotSettings, flying_laps: int, cte_max: float, cte_limit=3.0):
    pilot = TrackAutopilot(nodes, s)
    n_nodes = len(nodes)
    client.reset_car()
    time.sleep(1.2)
    client.flush()
    laps, lap_start, prev, visited, max_cte, ok = [], None, None, set(), 0.0, True
    t_start = time.perf_counter()
    while len(laps) < 1 + flying_laps:
        if time.perf_counter() - t_start > 40 * (1 + flying_laps):
            return {"ok": False, "reason": "timeout", "laps": laps}
        tel = client.get_frame(timeout=1.0)
        if tel is None:
            continue
        if lap_start is None:
            if abs(tel.cte) > 1.0 or tel.hit != "none":      # stale values right after reset
                client.send_controls(0, 0, 0)
                continue
            lap_start = tel.sim_time
        st, th, br = pilot(tel)
        client.send_controls(st, th, br)
        if len(laps) >= 1:                       # flying laps only count for the safety margin
            max_cte = max(max_cte, abs(tel.cte))
        if abs(tel.cte) > cte_limit or tel.hit != "none":
            client.send_controls(0, 0, 1)
            return {"ok": False, "reason": f"departure at node {tel.active_node}", "laps": laps,
                    "max_cte": round(abs(tel.cte), 2)}
        n = tel.active_node
        if prev is not None and prev >= n_nodes - 6 and n <= 5 and len(visited) >= 0.8 * n_nodes:
            laps.append(round(tel.sim_time - lap_start, 2))
            lap_start, visited = tel.sim_time, set()
        visited.add(n)
        prev = n
    client.send_controls(0, 0, 1)
    flying = laps[1:]
    res = {"ok": max_cte <= cte_max, "laps": laps, "score": round(sum(flying) / len(flying), 3),
           "max_cte": round(max_cte, 2)}
    if not res["ok"]:
        res["reason"] = f"max |cte| {max_cte:.2f} > {cte_max}"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-evals", type=int, default=30)
    ap.add_argument("--flying-laps", type=int, default=2)
    ap.add_argument("--cte-max", type=float, default=2.2)
    ap.add_argument("--params", nargs="*", default=list(STEPS))
    ap.add_argument("--init", default=None, help="start from settings in this JSON")
    args = ap.parse_args()

    stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out = PROJECT_ROOT / "models" / "profiles" / f"teacher_{stamp}.json"
    log = PROJECT_ROOT / "docs" / "experiments" / f"tune_{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    best_s = START
    if args.init:
        best_s = dataclasses.replace(START, **json.loads(Path(args.init).read_text())["settings"])

    def save(best_s, best):
        out.write_text(json.dumps({"settings": dataclasses.asdict(best_s), "profile": None,
                                   "best": best, "tuned_params": args.params}, indent=1))

    with DonkeyClient() as client:
        client.start()
        first = client.get_frame(timeout=2.0)
        nodes = client.fetch_track(first.total_nodes)
        evals = 0

        def run(s, note):
            nonlocal evals
            evals += 1
            r = evaluate(client, nodes, s, args.flying_laps, args.cte_max)
            line = {"eval": evals, "note": note, **r, "settings": dataclasses.asdict(s)}
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
            print(f"[{evals:2d}] {note:<32} ok={r['ok']!s:<5} score={r.get('score')} "
                  f"laps={r['laps']} max_cte={r.get('max_cte')} {r.get('reason', '')}", flush=True)
            return r

        best = run(best_s, "start")
        if not best["ok"]:
            print("start settings not valid; stopping")
            return 1
        save(best_s, best)
        improved = True
        while improved and evals < args.max_evals:
            improved = False
            for p in args.params:
                for sign in (+1, -1):
                    if evals >= args.max_evals:
                        break
                    v = getattr(best_s, p) + sign * STEPS[p]
                    lo, hi = LIMITS[p]
                    if not lo <= v <= hi:
                        continue
                    v = type(getattr(START, p))(round(v, 4))
                    cand = dataclasses.replace(best_s, **{p: v})
                    r = run(cand, f"{p}={v}")
                    if r["ok"] and r["score"] < best["score"] - 0.05:
                        best_s, best, improved = cand, r, True
                        save(best_s, best)
                        print(f"     -> new best {best['score']} s", flush=True)
                        break          # keep moving along this parameter next round
    print("BEST", best, dataclasses.asdict(best_s))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
