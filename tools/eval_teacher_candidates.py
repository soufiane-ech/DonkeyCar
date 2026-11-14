"""
Evaluate a list of autopilot setting variants (same protocol as tune_teacher.py).

    python tools/eval_teacher_candidates.py --base models/profiles/teacher_fast_v1.json \
        --cand "lookahead=4 k_heading=0.0333" "lookahead=4 v_min=2.75"
"""
import argparse, dataclasses, json, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from donkey.autopilot import AutopilotSettings          # noqa: E402
from donkey.config import PROJECT_ROOT                  # noqa: E402
from donkey.sim_client import DonkeyClient              # noqa: E402
from tune_teacher import evaluate                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--cand", nargs="+", required=True)
ap.add_argument("--flying-laps", type=int, default=2)
ap.add_argument("--cte-max", type=float, default=2.2)
args = ap.parse_args()
base = AutopilotSettings(**json.loads(Path(args.base).read_text())["settings"])
log = PROJECT_ROOT / "docs" / "experiments" / f"candidates_{datetime.now():%Y-%m-%d_%H-%M-%S}.jsonl"
with DonkeyClient() as c:
    c.start()
    nodes = c.fetch_track(c.get_frame(timeout=2.0).total_nodes)
    for cand in args.cand:
        kv = dict(x.split("=") for x in cand.split())
        s = dataclasses.replace(base, **{k: type(getattr(base, k))(float(v)) for k, v in kv.items()})
        r = evaluate(c, nodes, s, args.flying_laps, args.cte_max)
        print(f"{cand:<45} ok={r['ok']!s:<5} score={r.get('score')} laps={r['laps']} "
              f"max_cte={r.get('max_cte')} {r.get('reason', '')}", flush=True)
        with open(log, "a") as fh:
            fh.write(json.dumps({"cand": cand, **r, "settings": dataclasses.asdict(s)}) + "\n")
