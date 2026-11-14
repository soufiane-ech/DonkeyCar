"""
Camera probe: try several cam_config settings on the running simulator and
measure what is really received (resolution, encoding, frame rate, missing
frames, bytes/frame, decode + preview cost). One sample image per config is
saved for a visual comparison.

    python tools/probe_camera.py
    python tools/probe_camera.py --seconds 8 --drive 0.2

Output: docs/camera_probe/<timestamp>/report.json + <config>.png + contact_sheet.png
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                       # noqa: E402
from PIL import Image, ImageDraw                                         # noqa: E402

from donkey.config import PROJECT_ROOT, CameraConfig, CarConfig, SimConfig   # noqa: E402
from donkey.sim_client import DonkeyClient                               # noqa: E402

BASE = CameraConfig()
CANDIDATES = {
    "160x120_default":    replace(BASE),
    "160x120_jpg":        replace(BASE, img_enc="JPG"),
    "256x256":            replace(BASE, img_w=256, img_h=256),
    "320x240":            replace(BASE, img_w=320, img_h=240),
    "640x480":            replace(BASE, img_w=640, img_h=480),
    "320x240_fov90":      replace(BASE, img_w=320, img_h=240, fov=90),
    "320x240_fov120":     replace(BASE, img_w=320, img_h=240, fov=120),
    "320x240_up_tilt":    replace(BASE, img_w=320, img_h=240, offset_y=0.5, rot_x=15),
    "320x240_fov90_up_tilt": replace(BASE, img_w=320, img_h=240, fov=90, offset_y=0.5, rot_x=15),
    "320x240_tilt10":     replace(BASE, img_w=320, img_h=240, rot_x=10),
    "320x240_tilt20":     replace(BASE, img_w=320, img_h=240, rot_x=20),
    "320x240_fov110":     replace(BASE, img_w=320, img_h=240, fov=110),
    "320x240_fov110_tilt10": replace(BASE, img_w=320, img_h=240, fov=110, rot_x=10),
    "320x240_up02_tilt15": replace(BASE, img_w=320, img_h=240, offset_y=0.2, rot_x=15),
    "320x240_fwd05":      replace(BASE, img_w=320, img_h=240, offset_z=0.5),
}


def measure(client: DonkeyClient, cam: CameraConfig, seconds: float, throttle: float):
    client.reset_car()
    client.send_json(cam.to_msg())
    t0 = time.perf_counter()
    applied = None
    while time.perf_counter() - t0 < 5.0:
        tel = client.get_frame(timeout=1.0)
        if tel is not None and (tel.img_w, tel.img_h) == (cam.img_w, cam.img_h):
            applied = time.perf_counter() - t0
            break
    if applied is None:
        tel = client.latest()
        return {"applied": False, "got": (tel.img_w, tel.img_h) if tel else None}
    time.sleep(0.5)                 # let the reset settle
    client.flush()

    frames, dec_ms, prev_ms = [], [], []
    import pygame
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        tel = client.get_frame(timeout=1.0)
        if tel is None:
            continue
        client.send_controls(0.0, throttle, 0.0)
        a = time.perf_counter()
        img = tel.image()
        b = time.perf_counter()
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        pygame.transform.smoothscale(surf, (640, 480 if cam.img_h != cam.img_w else 640))
        c = time.perf_counter()
        dec_ms.append(1000 * (b - a))
        prev_ms.append(1000 * (c - b))
        frames.append(tel)
    client.send_controls(0.0, 0.0, 1.0)

    sim = [f.sim_time for f in frames]
    wall = [f.t_local for f in frames]
    ids = [f.frame_id for f in frames]
    sim_gaps = sum(1 for a, b in zip(sim, sim[1:]) if b - a > 0.075)
    return {
        "applied": True, "apply_s": round(applied, 2),
        "got": (frames[-1].img_w, frames[-1].img_h), "format": frames[-1].image_format,
        "frames": len(frames),
        "wall_fps": round((len(wall) - 1) / (wall[-1] - wall[0]), 2),
        "sim_dt_ms_median": round(1000 * float(np.median(np.diff(sim))), 2),
        "sim_speed_ratio": round((sim[-1] - sim[0]) / (wall[-1] - wall[0]), 3),
        "sim_time_gaps": sim_gaps,
        "client_id_skips": sum(1 for a, b in zip(ids, ids[1:]) if b != a + 1),
        "kb_per_frame": round(float(np.mean([len(f.image_bytes) for f in frames])) / 1024, 1),
        "decode_ms": round(float(np.mean(dec_ms)), 2),
        "preview_ms": round(float(np.mean(prev_ms)), 2),
        "mean_brightness": round(float(frames[-1].image().mean()), 1),
    }, frames[len(frames) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--drive", type=float, default=0.0, help="throttle during measurement")
    ap.add_argument("--only", nargs="*", help="subset of candidate names")
    args = ap.parse_args()

    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1, 1))

    out = PROJECT_ROOT / "docs" / "camera_probe" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out.mkdir(parents=True)
    report = {"seconds": args.seconds, "throttle": args.drive, "results": {}}
    samples = {}
    with DonkeyClient(SimConfig(), CarConfig(), BASE) as client:
        report["start"] = client.start()
        for name, cam in CANDIDATES.items():
            if args.only and name not in args.only:
                continue
            res = measure(client, cam, args.seconds, args.drive)
            if isinstance(res, tuple):
                res, tel = res
                (out / f"{name}.{tel.image_format}").write_bytes(tel.image_bytes)
                samples[name] = tel.image()
            res["config"] = asdict(cam)
            report["results"][name] = res
            print(f"{name:<24} " + json.dumps({k: v for k, v in res.items() if k != 'config'}))
        client.send_json(BASE.to_msg())          # leave the sim with the default camera
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    # contact sheet: every sample resized to 320 px wide, labelled
    tiles = []
    for name, img in samples.items():
        im = Image.fromarray(img)
        im = im.resize((320, round(320 * im.height / im.width)), Image.LANCZOS)
        canvas = Image.new("RGB", (320, im.height + 18), (20, 20, 20))
        canvas.paste(im, (0, 18))
        ImageDraw.Draw(canvas).text((4, 3), name, fill=(255, 255, 0))
        tiles.append(canvas)
    if tiles:
        cols = 3
        rows = math.ceil(len(tiles) / cols)
        th = max(t.height for t in tiles)
        sheet = Image.new("RGB", (cols * 320, rows * th), (0, 0, 0))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 320, (i // cols) * th))
        sheet.save(out / "contact_sheet.png")
    print("report:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
