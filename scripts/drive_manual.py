"""
Manual driving + data collection (Step 4 of the roadmap).

Drive the car with the keyboard in a pygame window that shows the car's camera
and the current command/telemetry. Press R to record: every frame is then saved
with the command you sent (SessionRecorder -> data/raw/<session>/).

    conda activate donkeycar_env
    python scripts/drive_manual.py
    python scripts/drive_manual.py --tag recovery --notes "edge recoveries, clockwise"
    python scripts/drive_manual.py --cruise 0.3 --reload

Click the pygame window so it has keyboard focus (not the simulator window).

Keys
    Left / A, Right / D    steer (smooth ramp; tap = small correction)
    Up / W                 accelerate (boost above cruise)
    Down / S               brake (also cancels cruise)
    C                      cruise ON/OFF (constant throttle, recommended)
    + / -                  cruise throttle +/- 0.05
    R                      recording ON/OFF
    T                      recovery tag ON/OFF (mode column = recovery)
    Backspace              undo: delete the last N seconds (default 5 s)
    Space                  reset car (new episode, 1 s cooldown not recorded)
    Esc                    quit (session is closed and validated)
"""

import argparse
import logging
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame  # noqa: E402

from donkey.config import (DATA_RAW_DIR, CameraConfig, CarConfig, SimConfig,  # noqa: E402
                           config_dict)
from donkey.controllers import KeyboardController, KeyboardSettings          # noqa: E402
from donkey.recorder import SessionRecorder                                   # noqa: E402
from donkey.session import print_report, validate_session                     # noqa: E402
from donkey.sim_client import DonkeyClient, SimConnectionError                # noqa: E402

PANEL_W = 320
WHITE, GREY, DARK = (235, 235, 235), (140, 140, 140), (25, 25, 30)
RED, GREEN, YELLOW, ORANGE, BLUE = (230, 60, 60), (80, 200, 120), (240, 200, 60), (255, 150, 40), (90, 160, 255)

HELP = ["Arrows/WASD steer/accel/brake", "C cruise   +/- cruise speed", "R record ON/OFF",
        "T recovery tag", "Backspace undo last {undo:g}s", "Space reset car", "Esc quit"]


class ManualSession:
    """State of the collection run (recording flag, cooldown, tags, messages)."""

    def __init__(self, args, client, start_info, settings):
        self.args, self.client, self.start_info, self.settings = args, client, start_info, settings
        self.recorder = None              # created on first R press (no empty sessions)
        self.recording = False
        self.mode = "normal"
        self.cooldown_until = None        # sim_time until which nothing is recorded
        self.toast, self.toast_until = "", 0.0

    # ---- events
    def message(self, text, seconds=2.5):
        self.toast, self.toast_until = text, time.perf_counter() + seconds
        print(text)

    def toggle_recording(self):
        if self.recorder is None:
            sim, car, cam = self.client.sim_cfg, self.client.car_cfg, self.client.cam_cfg
            self.recorder = SessionRecorder(
                scene=sim.scene, tag=self.args.tag, root=Path(self.args.root),
                config=config_dict(sim, car, cam), notes=self.args.notes,
                hold_seconds=self.args.undo + 1.0,
                extra_meta={"controller": "keyboard", "keyboard_settings": asdict(self.settings),
                            "client_start": self.start_info,
                            "reset_cooldown_s": self.args.cooldown, "undo_s": self.args.undo})
            self.message(f"Session: {self.recorder.session_id}", 4)
        self.recording = not self.recording
        self.message("Recording ON" if self.recording else "Recording PAUSED")

    def toggle_recovery(self):
        self.mode = "normal" if self.mode == "recovery" else "recovery"
        self.message(f"Mode: {self.mode.upper()}")

    def undo(self):
        if self.recorder is None:
            return self.message("Nothing recorded yet")
        n = self.recorder.discard_last(self.args.undo)
        self.message(f"Undo: removed {n} frames ({self.args.undo:g} s)")

    def reset(self, controller, last_sim_time):
        self.client.reset_car()
        controller.reset()
        if self.recorder is not None:
            self.recorder.new_episode("manual_reset", sim_time=last_sim_time)
        # Frames right after reset_car still carry the old hit / a meaningless cte
        # (observed in Step 3): skip them on the *sim* clock.
        self.cooldown_until = last_sim_time + self.args.cooldown
        self.message("Car reset")

    # ---- per frame
    def in_cooldown(self, tel) -> bool:
        if self.cooldown_until is None:
            return False
        if tel.sim_time >= self.cooldown_until and tel.hit == "none":
            self.cooldown_until = None
            return False
        return True

    def status(self, cooling):
        if not self.recording:
            return "PAUSED", GREY
        return ("COOLDOWN", YELLOW) if cooling else ("REC", RED)


def draw_bar(surf, x, y, w, h, value, lo, hi, color, centered=False):
    pygame.draw.rect(surf, (60, 60, 70), (x, y, w, h))
    if centered:
        mid = x + w // 2
        end = mid + int(value / max(abs(lo), hi) * (w // 2))
        pygame.draw.rect(surf, color, (min(mid, end), y, abs(end - mid), h))
        pygame.draw.line(surf, WHITE, (mid, y - 2), (mid, y + h + 2))
    else:
        pygame.draw.rect(surf, color, (x, y, int(w * (value - lo) / (hi - lo)), h))


def render(screen, fonts, tel, cmd, ctrl, sess, fps, scale):
    big, small = fonts
    cam_w, cam_h = round(tel.img_w * scale), round(tel.img_h * scale)
    img = tel.image()                                             # (H, W, 3) uint8
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    # Bilinear zoom (smoothscale) for display only: recorded images are untouched.
    screen.blit(pygame.transform.smoothscale(surf, (cam_w, cam_h)), (0, 0))

    cooling = sess.cooldown_until is not None
    label, color = sess.status(cooling)
    # status badge on the camera view
    pygame.draw.rect(screen, DARK, (8, 8, 150, 34), border_radius=6)
    pygame.draw.circle(screen, color, (26, 25), 9)
    screen.blit(big.render(label, True, color), (42, 12))
    if sess.mode == "recovery":
        pygame.draw.rect(screen, DARK, (166, 8, 130, 34), border_radius=6)
        screen.blit(big.render("RECOVERY", True, ORANGE), (174, 12))
    # steering bar along the bottom of the camera view
    draw_bar(screen, 20, cam_h - 26, cam_w - 40, 14, cmd["steering"], -1, 1, GREEN, centered=True)

    # side panel
    x0 = cam_w
    pygame.draw.rect(screen, DARK, (x0, 0, PANEL_W, screen.get_height()))
    y = 12

    def line(text, col=WHITE, font=small, dy=22):
        nonlocal y
        screen.blit(font.render(text, True, col), (x0 + 14, y))
        y += dy

    line(f"Steering {cmd['steering']:+.2f}", font=big, dy=26)
    draw_bar(screen, x0 + 14, y, PANEL_W - 28, 10, cmd["steering"], -1, 1, GREEN, centered=True)
    y += 20
    line(f"Throttle {cmd['throttle']:.2f}" + ("   BRAKE" if cmd["brake"] > 0 else ""),
         RED if cmd["brake"] > 0 else WHITE, font=big, dy=26)
    draw_bar(screen, x0 + 14, y, PANEL_W - 28, 10, cmd["throttle"], 0, 1, BLUE)
    y += 20
    line(f"Cruise {'ON ' if ctrl.cruise else 'OFF'} @ {ctrl.s.cruise_throttle:.2f}",
         GREEN if ctrl.cruise else GREY)
    line(f"Speed {tel.speed:5.2f}    CTE {tel.cte:+6.2f}")
    line(f"Hit: {tel.hit}", RED if tel.hit != "none" else GREY)
    # start-line crossings also count resets onto the line: not a reliable lap count
    line(f"Node {tel.active_node}/{tel.total_nodes}  Start-line x{len(sess.client.lap_events)}")
    line(f"Sim {fps:4.1f} fps   backlog {sess.client.backlog}", GREY)
    y += 6
    rec = sess.recorder
    if rec is None:
        line("No session yet (press R)", GREY)
    else:
        line(f"Episode {rec.episode_id}   mode {sess.mode}")
        line(f"Saved {rec.n_written}  pending {rec.n_pending}")
        line(f"Discarded {rec.n_discarded}", GREY)
        line(rec.session_id[:19] + " ...", GREY, dy=18)   # date_time part of the folder
    y += 8
    for h in HELP:
        line(h.format(undo=sess.args.undo), GREY, dy=19)

    if time.perf_counter() < sess.toast_until:
        t = big.render(sess.toast, True, WHITE)
        bg = pygame.Rect(0, 0, t.get_width() + 24, t.get_height() + 12)
        bg.center = (cam_w // 2, cam_h // 2)
        pygame.draw.rect(screen, DARK, bg, border_radius=8)
        screen.blit(t, (bg.x + 12, bg.y + 6))
    pygame.display.flip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="manual", help="session folder tag (e.g. manual, recovery)")
    ap.add_argument("--notes", default="", help="free text saved in meta.json")
    ap.add_argument("--cruise", type=float, default=KeyboardSettings.cruise_throttle)
    ap.add_argument("--undo", type=float, default=5.0, help="seconds removed by Backspace")
    ap.add_argument("--cooldown", type=float, default=1.0, help="sim seconds not recorded after a reset")
    ap.add_argument("--scale", type=float, default=2.5,
                    help="camera preview zoom (2.5 -> 800x600 for a 320x240 camera)")
    ap.add_argument("--root", default=str(DATA_RAW_DIR))
    ap.add_argument("--reload", action="store_true", help="exit and reload the scene first")
    ap.add_argument("--scene", default=SimConfig().scene)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("PIL").setLevel(logging.INFO)

    settings = KeyboardSettings(cruise_throttle=args.cruise)
    ctrl = KeyboardController(settings)
    sim, car, cam = SimConfig(scene=args.scene), CarConfig(), CameraConfig()

    pygame.init()
    screen = pygame.display.set_mode((round(cam.img_w * args.scale) + PANEL_W,
                                      max(round(cam.img_h * args.scale), 480)))
    pygame.display.set_caption("DonkeyCar - manual driving")
    fonts = (pygame.font.SysFont("consolas,dejavusansmono,monospace", 22, bold=True),
             pygame.font.SysFont("consolas,dejavusansmono,monospace", 16))

    sess, status = None, "interrupted"
    try:
        with DonkeyClient(sim, car, cam) as client:
            info = client.start(force_reload=args.reload)
            sess = ManualSession(args, client, info, settings)
            client.flush()
            last = client.get_frame(timeout=2.0)
            if last is None:
                raise SimConnectionError("no telemetry after start")
            sess.reset(ctrl, last.sim_time)           # known starting point
            sess.message("Ready - press R to record, C for cruise", 4)

            wall = deque(maxlen=20)
            running = True
            while running:
                tel = client.get_frame(timeout=1.0)
                if tel is None:
                    if client.aborted:
                        raise SimConnectionError("connection to the simulator lost")
                    pygame.event.pump()
                    continue
                dt = tel.sim_time - last.sim_time if tel.sim_time > last.sim_time else 0.05
                last = tel
                wall.append(tel.t_local)

                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN:
                        k = ev.key
                        if k == pygame.K_ESCAPE:
                            running = False
                        elif k == pygame.K_r:
                            sess.toggle_recording()
                        elif k == pygame.K_t:
                            sess.toggle_recovery()
                        elif k == pygame.K_BACKSPACE:
                            sess.undo()
                        elif k == pygame.K_SPACE:
                            sess.reset(ctrl, tel.sim_time)
                        elif k == pygame.K_c:
                            sess.message(f"Cruise {'ON' if ctrl.toggle_cruise() else 'OFF'}")
                        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                            sess.message(f"Cruise throttle {ctrl.adjust_cruise(+1):.2f}")
                        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            sess.message(f"Cruise throttle {ctrl.adjust_cruise(-1):.2f}")

                keys = pygame.key.get_pressed()
                steering, throttle, brake = ctrl.update(
                    left=keys[pygame.K_LEFT] or keys[pygame.K_a],
                    right=keys[pygame.K_RIGHT] or keys[pygame.K_d],
                    accel=keys[pygame.K_UP] or keys[pygame.K_w],
                    brake=keys[pygame.K_DOWN] or keys[pygame.K_s], dt=dt)
                cmd = client.send_controls(steering, throttle, brake)

                cooling = sess.in_cooldown(tel)
                if sess.recording and not cooling:
                    sess.recorder.record(tel, cmd, lap_count=len(client.lap_events), mode=sess.mode)
                if tel.hit != "none" and not cooling and sess.toast_until < time.perf_counter():
                    sess.message(f"HIT {tel.hit} - Backspace to undo, Space to reset")

                # Skip drawing when frames are waiting, so commands never fall behind.
                if client.backlog == 0:
                    fps = (len(wall) - 1) / (wall[-1] - wall[0]) if len(wall) > 1 and wall[-1] > wall[0] else 0.0
                    render(screen, fonts, tel, cmd, ctrl, sess, fps, args.scale)
            status = "complete"
    except SimConnectionError as e:
        print("ERROR:", e)
        return 1
    finally:
        if sess is not None and sess.recorder is not None:
            sess.recorder.close(status=status)   # data recorded so far is always kept
        pygame.quit()

    if sess is None or sess.recorder is None:
        print("No session recorded.")
        return 0
    print()
    report = validate_session(sess.recorder.dir)
    print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
