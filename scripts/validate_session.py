"""
Validate recorded sessions and print quick statistics.

    python scripts/validate_session.py --latest          # most recent session in data/raw
    python scripts/validate_session.py --all             # every session in data/raw
    python scripts/validate_session.py data/raw/<session_folder>

Structural checks come from donkey.session.validate_session (rows <-> images,
timestamps, commands, cooldown...). Statistics show whether the session is
useful: steering distribution, recovery share, collisions, speed.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from donkey.config import DATA_RAW_DIR                                     # noqa: E402
from donkey.session import load_session, print_report, validate_session    # noqa: E402

BINS = [-1.0, -0.6, -0.3, -0.1, -0.02, 0.02, 0.1, 0.3, 0.6, 1.0001]


def stats(path: Path) -> None:
    meta, rows = load_session(path)
    if not rows:
        return
    cam = meta.get("config", {}).get("camera", {})
    print(f"  camera: {cam.get('img_w')}x{cam.get('img_h')} {cam.get('img_enc')} fov {cam.get('fov')}")
    steer = [r["cmd_steering"] for r in rows]
    thr = [r["cmd_throttle"] for r in rows]
    modes = Counter(r["mode"] for r in rows)
    hits = Counter(r["hit"] for r in rows if r["hit"] != "none")
    print(f"  notes: {meta.get('notes') or '-'}")
    print(f"  modes: {dict(modes)}   episodes: {len({r['episode_id'] for r in rows})}   "
          f"start-line crossings: {max(r['lap_count'] for r in rows)}   undo-discarded: "
          f"{meta.get('counts', {}).get('discarded', 0)}")
    print(f"  throttle mean {statistics.fmean(thr):.3f}   speed mean "
          f"{statistics.fmean(r['speed'] for r in rows):.2f}   brake frames "
          f"{sum(r['cmd_brake'] > 0 for r in rows)}")
    print(f"  frames with hit: {sum(hits.values())} {dict(hits) if hits else ''}")
    print("  steering distribution (cmd_steering):")
    n = len(steer)
    for lo, hi in zip(BINS, BINS[1:]):
        c = sum(lo <= s < hi for s in steer)
        print(f"    [{lo:+.2f}, {min(hi, 1.0):+.2f})  {c:6d} {100 * c / n:5.1f}%  " + "#" * int(60 * c / n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session folders")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", default=str(DATA_RAW_DIR))
    args = ap.parse_args()

    found = sorted(p for p in Path(args.root).glob("*") if (p / "meta.json").is_file())
    paths = [Path(p) for p in args.sessions]
    if args.all:
        paths += found
    elif args.latest and found:
        paths.append(found[-1])
    if not paths:
        print(f"No session given / found in {args.root}")
        return 1

    all_ok, total, cameras = True, 0, Counter()
    for p in paths:
        report = validate_session(p)
        print_report(report)
        stats(p)
        print()
        all_ok &= report["ok"]
        total += report["samples"]
        cam = load_session(p)[0].get("config", {}).get("camera", {})
        cameras[f"{cam.get('img_w')}x{cam.get('img_h')} fov{cam.get('fov')}"] += 1
    if len(paths) > 1:
        print(f"{len(paths)} sessions, {total} samples, all valid: {all_ok}")
        print(f"camera configs: {dict(cameras)}"
              + ("" if len(cameras) == 1 else "   WARNING: mixed camera configs, "
                 "do not merge these sessions into one dataset"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
