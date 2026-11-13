"""
Telemetry probe for the DonkeyCar simulator (Step 1 of the roadmap).

Goal: verify, against the *real* simulator, what the protocol provides before
building the recorder on top of it:
  * which keys a telemetry packet really contains (cte, steering_angle, time,
    activeNode, ...), with their types and value ranges;
  * the real telemetry rate and the frame-to-frame jitter;
  * whether `cam_config` is applied once we wait for `car_loaded`
    (the old test_client.py asked for 255x255 but got 160x120);
  * how the simulator echoes the commands we send (steering / throttle / brake);
  * what `reset_car` does to the position.

Nothing is written outside docs/telemetry_probe/<timestamp>/.
The car only moves gently (throttle 0.2 for 2 s) unless --no-drive is given.

Usage (simulator running, from the project root):
    conda activate donkeycar_env
    python scripts/probe_telemetry.py
    python scripts/probe_telemetry.py --scene roboracingleague_1 --no-drive
"""

import argparse
import base64
import json
import logging
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image
from gym_donkeycar.core.client import SDClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fields we want to confirm. "library" = parsed by gym_donkeycar 1.3.1,
# "unverified" = believed to be sent by the simulator but not in the library.
EXPECTED_FIELDS = {
    "library": ["image", "pos_x", "pos_y", "pos_z", "speed", "vel_x", "vel_y", "vel_z",
                "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z",
                "roll", "pitch", "yaw", "cte", "hit"],
    "unverified": ["steering_angle", "throttle", "time", "activeNode", "totalNodes"],
}
IMAGE_KEYS = ("image", "image_b", "imageb")
STEADY_PHASE_PREFIXES = ("idle", "steer", "drive", "brake")


def _truncate(value, limit=200):
    """Shorten long strings (e.g. base64 payloads) for logging."""
    if isinstance(value, str) and len(value) > limit:
        return f"<str len={len(value)}>"
    if isinstance(value, list) and len(value) > 20:
        return f"<list len={len(value)}>"
    return value


class ProbeClient(SDClient):
    """Records every incoming message. The socket-thread callback stays cheap:
    no pixel decoding, no disk I/O, no printing."""

    def __init__(self, host, port, poll_socket_sleep_time=0.01):
        # Attributes must exist before super().__init__, which starts the thread.
        self.lock = threading.Lock()
        self.phase = "connect"
        self.telemetry = []          # one dict per telemetry packet (image removed)
        self.other_msgs = []         # non-telemetry messages (truncated)
        self.msg_counts = Counter()
        self.sample_images = {}      # phase -> raw encoded image bytes (first frame)
        self.scene_names = None
        self.car_loaded = threading.Event()
        self.t0 = time.perf_counter()
        super().__init__(host, port, poll_socket_sleep_time=poll_socket_sleep_time)

    def on_msg_recv(self, j):
        t = time.perf_counter() - self.t0
        msg_type = j.get("msg_type")
        phase = self.phase

        if msg_type != "telemetry":
            with self.lock:
                self.msg_counts[msg_type] += 1
                self.other_msgs.append({"t": round(t, 4), "phase": phase,
                                        **{k: _truncate(v) for k, v in j.items()}})
            if msg_type == "car_loaded":
                self.car_loaded.set()
            elif msg_type == "scene_names":
                self.scene_names = j.get("scene_names")
            return

        rec = {"_t": t, "_phase": phase}
        raw = None
        for k, v in j.items():
            if k in IMAGE_KEYS:
                rec[f"{k}__b64len"] = len(v) if isinstance(v, str) else None
            elif k == "lidar":
                rec["lidar__len"] = len(v) if v is not None else None
            else:
                rec[k] = v

        if isinstance(j.get("image"), str):
            try:
                raw = base64.b64decode(j["image"])
                with Image.open(BytesIO(raw)) as im:  # reads the header only
                    rec["_img_w"], rec["_img_h"] = im.size
                    rec["_img_mode"], rec["_img_format"] = im.mode, im.format
                rec["_img_nbytes"] = len(raw)
            except Exception as e:  # corrupted frame: keep a trace, don't crash
                rec["_img_error"] = repr(e)
                raw = None

        with self.lock:
            self.msg_counts["telemetry"] += 1
            self.telemetry.append(rec)
            if raw is not None and phase not in self.sample_images:
                self.sample_images[phase] = raw

    # ---- helpers used from the main thread (all sends happen on this thread) ----
    def send_json(self, msg):
        self.send_now(json.dumps(msg))

    def send_control(self, steering, throttle, brake=0.0):
        self.send_json({"msg_type": "control", "steering": str(steering),
                        "throttle": str(throttle), "brake": str(brake)})

    def n_frames(self):
        with self.lock:
            return len(self.telemetry)

    def frames_since(self, idx):
        with self.lock:
            return self.telemetry[idx:]


def wait_until(predicate, timeout, step=0.02):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def cam_config_msg(w, h, enc, fov, msg_type="cam_config"):
    # Same field set as gym_donkeycar's send_cam_config; "0" = simulator default.
    return {"msg_type": msg_type, "fov": str(fov), "fish_eye_x": "0", "fish_eye_y": "0",
            "img_w": str(w), "img_h": str(h), "img_d": "3", "img_enc": enc,
            "offset_x": "0", "offset_y": "0", "offset_z": "0",
            "rot_x": "0", "rot_y": "0", "rot_z": "0"}


def hold(client, phases, name, steering, throttle, brake, seconds, hz=20.0):
    """Send a constant command at `hz` for `seconds`, labelling frames with `name`."""
    client.phase = name
    phases.append({"phase": name, "t_start": time.perf_counter() - client.t0,
                   "steering": steering, "throttle": throttle, "brake": brake,
                   "seconds": seconds})
    end = time.perf_counter() + seconds
    while time.perf_counter() < end and not client.aborted:
        client.send_control(steering, throttle, brake)
        time.sleep(1.0 / hz)


def test_cam_config(client, w, h, enc, fov, phase, timeout=5.0):
    client.phase = phase
    start_idx = client.n_frames()
    t_send = time.perf_counter()
    client.send_json(cam_config_msg(w, h, enc, fov))
    ok = wait_until(lambda: any(f.get("_img_w") == w and f.get("_img_h") == h
                                for f in client.frames_since(start_idx)), timeout)
    latency = time.perf_counter() - t_send if ok else None
    sizes = Counter((f.get("_img_w"), f.get("_img_h")) for f in client.frames_since(start_idx))
    return {"requested": [w, h], "applied": ok,
            "latency_s": round(latency, 3) if latency else None,
            "sizes_seen_after_send": {f"{k[0]}x{k[1]}": v for k, v in sizes.items()}}


# --------------------------------------------------------------------------- report

def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def describe_fields(frames):
    n = len(frames)
    info = {}
    keys = sorted({k for f in frames for k in f if not k.startswith("_")})
    for k in keys:
        vals = [f[k] for f in frames if k in f]
        types = Counter(type(v).__name__ for v in vals)
        entry = {"present_pct": round(100.0 * len(vals) / n, 1),
                 "types": dict(types), "first": vals[0], "last": vals[-1]}
        nums = [v for v in vals if numeric(v)]
        if nums:
            entry.update(min=min(nums), max=max(nums), mean=round(statistics.fmean(nums), 5))
        strs = [v for v in vals if isinstance(v, str)]
        if strs:
            entry["unique_strings"] = sorted(set(strs))[:20]
        info[k] = entry
    return info


def rate_stats(times):
    if len(times) < 3:
        return None
    dts = sorted(b - a for a, b in zip(times, times[1:]))
    p = lambda q: dts[min(len(dts) - 1, int(q * len(dts)))]
    span = times[-1] - times[0]
    return {"frames": len(times), "span_s": round(span, 3),
            "fps": round((len(times) - 1) / span, 2) if span > 0 else None,
            "dt_median_ms": round(1000 * p(0.5), 2), "dt_p95_ms": round(1000 * p(0.95), 2),
            "dt_max_ms": round(1000 * dts[-1], 2),
            "gaps_over_100ms": sum(d > 0.1 for d in dts)}


def phase_summary(frames, phases, settle=0.3):
    """Per commanded phase: mean of echoed fields, ignoring the first `settle` s."""
    out = []
    for i, ph in enumerate(phases):
        t_end = phases[i + 1]["t_start"] if i + 1 < len(phases) else float("inf")
        sel = [f for f in frames if ph["t_start"] + settle <= f["_t"] < t_end]
        row = {k: ph[k] for k in ("phase", "steering", "throttle", "brake")}
        row["frames"] = len(sel)
        for key in ("steering_angle", "throttle", "speed", "cte", "activeNode"):
            vals = [f[key] for f in sel if numeric(f.get(key))]
            if vals:
                row[f"echo_{key}_mean"] = round(statistics.fmean(vals), 4)
        pos = [(f["pos_x"], f["pos_z"]) for f in sel if numeric(f.get("pos_x"))]
        if len(pos) >= 2:
            row["pos_xz_start"] = [round(v, 3) for v in pos[0]]
            row["pos_xz_end"] = [round(v, 3) for v in pos[-1]]
        out.append(row)
    return out


def build_report(client, args, phases, connection, cam_tests):
    frames = client.frames_since(0)
    warnings = []
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "python": sys.version.split()[0],
        "connection": connection,
        "message_type_counts": dict(client.msg_counts),
        "telemetry_frames": len(frames),
    }
    if not frames:
        warnings.append("No telemetry received at all.")
        report["warnings"] = warnings
        return report

    fields = describe_fields(frames)
    report["telemetry_fields"] = fields

    present = set(fields) | {k.split("__")[0] for k in fields if "__" in k}
    report["expected_fields_check"] = {
        group: {k: (k in present) for k in keys} for group, keys in EXPECTED_FIELDS.items()}
    for group, keys in EXPECTED_FIELDS.items():
        missing = [k for k in keys if k not in present]
        if missing:
            warnings.append(f"{group} fields missing: {missing}")
    report["unexpected_fields"] = sorted(
        k for k in present - set(EXPECTED_FIELDS["library"]) - set(EXPECTED_FIELDS["unverified"])
        - {"msg_type", "lidar"} if "__" not in k)  # "__" = keys derived by this probe

    # Images per phase
    img = defaultdict(lambda: {"sizes": Counter(), "modes": Counter(), "formats": Counter(),
                               "nbytes": [], "errors": 0})
    for f in frames:
        d = img[f["_phase"]]
        if "_img_error" in f:
            d["errors"] += 1
            continue
        d["sizes"][f"{f.get('_img_w')}x{f.get('_img_h')}"] += 1
        d["modes"][f.get("_img_mode")] += 1
        d["formats"][f.get("_img_format")] += 1
        d["nbytes"].append(f.get("_img_nbytes", 0))
    report["images_by_phase"] = {
        ph: {"sizes": dict(d["sizes"]), "modes": dict(d["modes"]), "formats": dict(d["formats"]),
             "mean_bytes": round(statistics.fmean(d["nbytes"])) if d["nbytes"] else None,
             "decode_errors": d["errors"]}
        for ph, d in img.items()}
    report["cam_config_tests"] = cam_tests
    for name, res in cam_tests.items():
        if not res["applied"]:
            warnings.append(f"cam_config '{name}' {res['requested']} was NOT applied.")

    # Rates (steady phases only, config phases excluded)
    steady = [f for f in frames if f["_phase"].startswith(STEADY_PHASE_PREFIXES)]
    report["rate_local_clock"] = rate_stats([f["_t"] for f in steady])
    if steady and numeric(steady[0].get("time")):
        report["rate_sim_time_field"] = rate_stats([f["time"] for f in steady if numeric(f.get("time"))])

    report["phases"] = phase_summary(frames, phases)
    report["warnings"] = warnings
    return report


def print_summary(report, out_dir):
    print("\n" + "=" * 70)
    print("TELEMETRY PROBE SUMMARY")
    print("=" * 70)
    print("Connection   :", report["connection"])
    print("Messages     :", report["message_type_counts"])
    if "telemetry_fields" not in report:
        print("WARNINGS     :", report["warnings"])
        return
    print("\nTelemetry keys (type, range):")
    for k, e in report["telemetry_fields"].items():
        rng = f"[{e['min']} .. {e['max']}]" if "min" in e else ""
        extra = f" values={e['unique_strings']}" if "unique_strings" in e else ""
        print(f"  {k:<22} {list(e['types'])} {e['present_pct']:>5}%  {rng}{extra}")
    print("\nExpected fields check:")
    for group, d in report["expected_fields_check"].items():
        print(f"  {group:<10} present: {[k for k, v in d.items() if v]}")
        print(f"  {'':<10} MISSING: {[k for k, v in d.items() if not v]}")
    print("\nCamera config tests:")
    for name, r in report["cam_config_tests"].items():
        print(f"  {name:<8} requested {r['requested']} applied={r['applied']} "
              f"latency={r['latency_s']}s seen={r['sizes_seen_after_send']}")
    print("\nRate (local clock, steady phases):", report["rate_local_clock"])
    if "rate_sim_time_field" in report:
        print("Rate (sim 'time' field)          :", report["rate_sim_time_field"])
    print("\nPhases (commanded -> echoed):")
    for p in report["phases"]:
        print("  ", p)
    if report["warnings"]:
        print("\nWARNINGS:")
        for w in report["warnings"]:
            print("  -", w)
    print(f"\nFull report: {out_dir / 'report.json'}")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--scene", default="roboracingleague_1")
    ap.add_argument("--img-w", type=int, default=160, help="target camera width")
    ap.add_argument("--img-h", type=int, default=120, help="target camera height")
    ap.add_argument("--img-enc", default="PNG", choices=["PNG", "JPG", "TGA"])
    ap.add_argument("--fov", type=int, default=0, help="0 = simulator default")
    ap.add_argument("--test-w", type=int, default=128, help="temporary size used to prove cam_config works")
    ap.add_argument("--test-h", type=int, default=96)
    ap.add_argument("--no-drive", action="store_true", help="never apply throttle")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "telemetry_probe"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    out_dir = Path(args.out) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Connecting to {args.host}:{args.port} ...")
    try:
        client = ProbeClient(args.host, args.port)
    except Exception as e:
        print(f"ERROR: {e}\nIs the simulator running and listening on port {args.port}?")
        return 1

    phases, cam_tests, connection = [], {}, {}
    try:
        # 1) Scene: reuse it if a car already spawned, otherwise load it.
        already = wait_until(lambda: client.car_loaded.is_set() or client.n_frames() > 0, 2.0)
        connection["scene_already_loaded"] = already
        if not already:
            client.phase = "scene_select"
            client.send_json({"msg_type": "get_scene_names"})
            wait_until(lambda: client.scene_names is not None, 3.0)
            connection["scene_names"] = client.scene_names
            if client.scene_names and args.scene not in client.scene_names:
                print(f"ERROR: scene '{args.scene}' not in {client.scene_names}")
                return 1
            print(f"Loading scene '{args.scene}' ...")
            t = time.perf_counter()
            client.send_json({"msg_type": "load_scene", "scene_name": args.scene})
            if not client.car_loaded.wait(30.0):
                print("WARNING: no 'car_loaded' within 30 s, continuing anyway.")
            connection["car_loaded_after_s"] = round(time.perf_counter() - t, 2)
        connection["car_loaded_msg"] = client.car_loaded.is_set()

        # 2) Baseline frames with the simulator's default camera (before any config).
        client.phase = "default_cam"
        got = wait_until(lambda: client.n_frames() >= 20, 5.0)
        connection["telemetry_before_config"] = got
        print("Default camera frames received:", client.n_frames())

        # 3) Car / racer config, second camera and lidar disabled (as in test_client.py).
        client.phase = "config"
        client.send_json({"msg_type": "racer_info", "racer_name": "probe", "car_name": "probe",
                          "bio": "", "country": "France", "guid": "probe"})
        time.sleep(0.2)
        client.send_json({"msg_type": "car_config", "body_style": "car01", "body_r": "255",
                          "body_g": "0", "body_b": "255", "car_name": "probe", "font_size": "100"})
        time.sleep(0.5)
        client.send_json({"msg_type": "cam_config_b", "enabled": "false"})
        time.sleep(0.2)
        client.send_json({"msg_type": "lidar_config", "enabled": "false"})
        time.sleep(0.2)

        # 4) Prove cam_config works: request a distinct size, then the target size.
        print("Testing cam_config ...")
        cam_tests["test"] = test_cam_config(client, args.test_w, args.test_h, args.img_enc, args.fov, "cam_test")
        cam_tests["target"] = test_cam_config(client, args.img_w, args.img_h, args.img_enc, args.fov, "cam_target")

        # 5) Command / echo phases.
        print("Running command phases ...")
        hold(client, phases, "idle", 0.0, 0.0, 0.0, 2.0)
        for s in (-1.0, -0.5, 0.0, 0.5, 1.0):
            hold(client, phases, f"steer_{s:+.1f}", s, 0.0, 0.0, 1.0)
        if not args.no_drive:
            hold(client, phases, "drive_0.2", 0.0, 0.2, 0.0, 2.0)
            hold(client, phases, "brake_1.0", 0.0, 0.0, 1.0, 1.5)
        client.send_json({"msg_type": "reset_car"})
        hold(client, phases, "idle_after_reset", 0.0, 0.0, 0.0, 2.0)
        client.send_control(0.0, 0.0, 0.0)
        time.sleep(0.2)
    finally:
        client.stop()

    report = build_report(client, args, phases, connection, cam_tests)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    frames = client.frames_since(0)
    with open(out_dir / "packets_sample.jsonl", "w", encoding="utf-8") as fh:
        for f in frames[:5] + frames[-5:]:
            fh.write(json.dumps(f, default=str) + "\n")
        for m in client.other_msgs:
            fh.write(json.dumps(m, default=str) + "\n")
    for phase, raw in client.sample_images.items():
        ext = (Image.open(BytesIO(raw)).format or "bin").lower()
        (out_dir / f"sample_{phase}.{ext}").write_bytes(raw)

    print_summary(report, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
