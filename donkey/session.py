"""
Reading and validating recorded sessions (no pandas needed).

    meta, rows = load_session(path)
    report = validate_session(path)      # report["ok"] is True if every check passed
"""

import csv
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from .recorder import COLUMNS

_INT_COLS = {"sample_id", "episode_id", "lap_count", "frame_id", "active_node", "total_nodes",
             "img_w", "img_h"}
_STR_COLS = {"image", "mode", "hit", "image_format"}
SIM_DT = 0.05   # simulator frame period (s)


def _parse(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k in _STR_COLS:
            out[k] = v
        elif k in _INT_COLS:
            out[k] = int(v) if v not in ("", "nan") else -1
        else:
            out[k] = float(v) if v != "" else math.nan
    return out


def load_session(path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    with open(path / "records.csv", newline="", encoding="utf-8") as fh:
        rows = [_parse(r) for r in csv.DictReader(fh)]
    return meta, rows


def validate_session(path, decode_every: int = 1, gap_factor: float = 1.5) -> Dict[str, Any]:
    """Structural checks of one session folder. `decode_every`=N decodes every N-th image."""
    path = Path(path)
    checks: Dict[str, Dict[str, Any]] = {}

    def check(name, ok, **detail):
        checks[name] = {"ok": bool(ok), **detail}

    meta, rows = load_session(path)
    n = len(rows)
    check("meta status complete", meta.get("status") == "complete", status=meta.get("status"))
    check("rows == meta written", n == meta.get("counts", {}).get("written"),
          rows=n, meta_written=meta.get("counts", {}).get("written"))

    with open(path / "records.csv", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    check("csv header", header == COLUMNS)

    ids = [r["sample_id"] for r in rows]
    check("sample_id contiguous", ids == list(range(n)))

    missing = [r["image"] for r in rows if not (path / r["image"]).is_file()]
    files = {p.name for p in (path / "images").iterdir()}
    orphans = sorted(files - {Path(r["image"]).name for r in rows})
    check("every row has its image", not missing, missing=missing[:10])
    check("no orphan image files", not orphans, orphans=orphans[:10])

    bad, sizes = [], set()
    for r in rows[::max(1, decode_every)]:
        try:
            with Image.open(BytesIO((path / r["image"]).read_bytes())) as im:
                im.load()
                sizes.add(im.size)
        except Exception as e:
            bad.append((r["image"], repr(e)))
    check("images decode", not bad, decoded=len(rows[::max(1, decode_every)]), errors=bad[:5])
    check("single image size", len(sizes) <= 1, sizes=sorted(sizes))
    cam = meta.get("config", {}).get("camera", {})
    if cam:
        expected = (cam.get("img_w"), cam.get("img_h"))
        check("image size == camera config", sizes <= {expected},
              expected=list(expected), got=sorted(sizes))

    sim_t = [r["sim_time"] for r in rows]
    frame_ids = [r["frame_id"] for r in rows]
    check("sim_time strictly increasing", all(b > a for a, b in zip(sim_t, sim_t[1:])))
    check("frame_id strictly increasing", all(b > a for a, b in zip(frame_ids, frame_ids[1:])))
    # Gaps are detected on the sim clock (frames missing from the recording). Their
    # wall-clock length can differ: the simulator sometimes runs slower than real
    # time (observed: down to ~0.7x for a few seconds, ~0.95x on average).
    gaps = [{"after_sample": rows[i]["sample_id"], "dt_s": round(sim_t[i + 1] - sim_t[i], 3),
             "wall_dt_s": round(rows[i + 1]["wall_time"] - rows[i]["wall_time"], 3)}
            for i in range(n - 1) if sim_t[i + 1] - sim_t[i] > gap_factor * SIM_DT]
    checks["time gaps (info)"] = {"ok": True, "count": len(gaps), "gaps": gaps[:20]}
    wall_span = rows[-1]["wall_time"] - rows[0]["wall_time"] if n > 1 else 0.0
    checks["sim speed vs real time (info)"] = {
        "ok": True, "ratio": round((sim_t[-1] - sim_t[0]) / wall_span, 3) if wall_span else None}

    nan_cmd = sum(any(math.isnan(r[c]) for c in ("cmd_steering", "cmd_throttle", "cmd_brake"))
                  for r in rows)
    out_of_range = sum(not (-1 <= r["cmd_steering"] <= 1 and -1 <= r["cmd_throttle"] <= 1
                            and 0 <= r["cmd_brake"] <= 1) for r in rows)
    check("commands valid", nan_cmd == 0 and out_of_range == 0,
          nan=nan_cmd, out_of_range=out_of_range)
    ep = [r["episode_id"] for r in rows]
    check("episode_id non-decreasing", all(b >= a for a, b in zip(ep, ep[1:])),
          episodes=sorted(set(ep)))

    # Sessions recorded with a reset cooldown: nothing may be recorded during it.
    # The cooldown runs on the sim clock, so it is checked on sim_time.
    cooldown = meta.get("reset_cooldown_s")
    if cooldown:
        resets = [e["sim_time"] for e in meta.get("episodes", []) if "sim_time" in e]
        inside = sum(any(t < r["sim_time"] < t + cooldown for t in resets) for r in rows)
        first_of_ep = {}
        for r in rows:
            first_of_ep.setdefault(r["episode_id"], r)
        stale_hit = [ep for ep, r in first_of_ep.items() if ep > 0 and r["hit"] != "none"]
        check("no samples in reset cooldown", inside == 0 and not stale_hit,
              resets=len(resets), samples_inside=inside, episodes_starting_with_hit=stale_hit)

    duration = sim_t[-1] - sim_t[0] if n > 1 else 0.0
    return {"session": path.name, "ok": all(c["ok"] for c in checks.values()),
            "samples": n, "duration_sim_s": round(duration, 2), "checks": checks}


def print_report(report: Dict[str, Any]) -> None:
    print(f"Session {report['session']}: {report['samples']} samples, "
          f"{report['duration_sim_s']} s of sim time")
    for name, c in report["checks"].items():
        detail = {k: v for k, v in c.items() if k != "ok"}
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {name:<30} {detail if detail else ''}")
    print("  =>", "VALID" if report["ok"] else "INVALID")
