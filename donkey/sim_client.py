"""
DonkeyClient: a reliable client for the DonkeyCar simulator.

Refactored from test_client.SimpleClient. Fixes found in the Step 1 analysis:
  * the socket-thread callback only parses and stores the latest frame
    (no disk I/O, no printing, no sleeps, never raises);
  * configuration is sent once, from the main thread, *after* `car_loaded`,
    and the camera size is verified on the received frames;
  * every message is sent from the main thread with `send_now` (the SDClient
    single-slot `send()` silently drops messages and is never used);
  * every received frame goes into a FIFO buffer, so the main loop can process
    each frame and send one command per frame (image -> command pairing).
    Frames sometimes arrive in bursts of two (local jitter, the simulator
    clock itself is a steady 50 ms), so keeping only the latest frame would
    lose ~1-2% of them.

Two ways to consume frames:
  * get_frame()        : every frame, in order (data collection);
  * wait_for_latest()  : freshest frame only, older ones discarded (inference).

Typical use:
    with DonkeyClient() as client:
        client.start()                      # load scene, configure, verify camera
        while True:
            tel = client.get_frame(timeout=1.0)
            client.send_controls(steering, throttle)
"""

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from gym_donkeycar.core.client import SDClient

from .config import (BRAKE_RANGE, STEERING_RANGE, THROTTLE_RANGE, CameraConfig, CarConfig,
                     SimConfig)
from .telemetry import Telemetry

logger = logging.getLogger(__name__)


def _clip(v: float, lo_hi) -> float:
    lo, hi = lo_hi
    return max(lo, min(hi, float(v)))


class SimConnectionError(RuntimeError):
    pass


class DonkeyClient(SDClient):

    def __init__(self, sim: SimConfig = SimConfig(), car: CarConfig = CarConfig(),
                 cam: CameraConfig = CameraConfig(), buffer_size: int = 200):
        self.sim_cfg, self.car_cfg, self.cam_cfg = sim, car, cam

        # State shared with the socket thread. Must exist before super().__init__,
        # which connects and starts that thread immediately.
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._latest: Optional[Telemetry] = None
        self._buffer: deque = deque()          # unread frames, oldest first
        self._buffer_size = buffer_size        # ~10 s at 20 Hz before frames are dropped
        self.frames_dropped = 0                # frames lost because the buffer was full
        self._frame_counter = 0
        self.car_loaded = threading.Event()
        self.scene_selection_ready = threading.Event()
        self._scene_names_evt = threading.Event()
        self.scene_names: Optional[List[str]] = None
        self.need_car_config = False      # set if the sim asks for config again
        self.lap_events: List[Dict[str, Any]] = []   # start-line crossings
        self.callback_errors = 0
        self.last_command: Optional[Dict[str, float]] = None
        self.track_nodes: Dict[int, Dict[str, Any]] = {}   # node_position replies

        try:
            super().__init__(sim.host, sim.port, poll_socket_sleep_time=sim.poll_socket_sleep)
        except Exception as e:
            raise SimConnectionError(f"Cannot connect to {sim.host}:{sim.port}. "
                                     f"Is the simulator running? ({e})") from e

    # ================================================================ socket thread
    def on_msg_recv(self, j: Dict[str, Any]) -> None:
        # An exception here would make SDClient abort the whole connection.
        try:
            msg_type = j.get("msg_type")
            if msg_type == "telemetry":
                t_local, wall = time.perf_counter(), time.time()
                with self._cond:
                    tel = Telemetry.from_packet(j, self._frame_counter, t_local, wall)
                    self._frame_counter += 1
                    self._latest = tel
                    if len(self._buffer) >= self._buffer_size:
                        self._buffer.popleft()
                        self.frames_dropped += 1
                    self._buffer.append(tel)
                    self._cond.notify_all()
            elif msg_type == "car_loaded":
                self.car_loaded.set()
            elif msg_type == "scene_selection_ready":
                self.scene_selection_ready.set()
            elif msg_type == "scene_names":
                self.scene_names = j.get("scene_names")
                self._scene_names_evt.set()
            elif msg_type == "need_car_config":
                self.need_car_config = True
            elif msg_type == "collision_with_starting_line":
                self.lap_events.append({"t_local": time.perf_counter(),
                                        "sim_timestamp": j.get("timeStamp"),
                                        "starting_line_index": j.get("starting_line_index")})
            elif msg_type == "node_position":
                self.track_nodes[int(j["index"])] = j
            elif msg_type == "aborted":
                with self._cond:
                    self._cond.notify_all()   # wake up a waiting main loop
            else:
                logger.debug("unhandled message: %s", msg_type)
        except Exception:
            self.callback_errors += 1
            logger.exception("error while handling a simulator message")

    # ================================================================ sending (main thread)
    def send_json(self, msg: Dict[str, Any]) -> None:
        with self._send_lock:
            self.send_now(json.dumps(msg))

    def send_controls(self, steering: float, throttle: float, brake: float = 0.0) -> Dict[str, float]:
        """Send one control command (values are clipped). Returns what was really sent."""
        cmd = {"steering": _clip(steering, STEERING_RANGE),
               "throttle": _clip(throttle, THROTTLE_RANGE),
               "brake": _clip(brake, BRAKE_RANGE)}
        self.send_json({"msg_type": "control", **{k: str(v) for k, v in cmd.items()}})
        cmd["t_local"] = time.perf_counter()
        self.last_command = cmd
        return cmd

    def fetch_track(self, total_nodes: int, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Track centre-line nodes (pos_x/pos_z + orientation quaternion), via the
        sim's node_position message (verified 2026-11-13). Node orientation is in
        the same frame as telemetry yaw; node positions are NOT in the telemetry
        pos frame."""
        for i in range(total_nodes):
            if i not in self.track_nodes:
                self.send_json({"msg_type": "node_position", "index": str(i)})
                time.sleep(0.01)
        if not self._wait(lambda: len(self.track_nodes) >= total_nodes, timeout):
            raise SimConnectionError(f"only {len(self.track_nodes)}/{total_nodes} track nodes received")
        return [self.track_nodes[i] for i in range(total_nodes)]

    def reset_car(self) -> None:
        self.send_json({"msg_type": "reset_car"})

    def exit_scene(self) -> None:
        self.car_loaded.clear()
        self.scene_selection_ready.clear()
        self.send_json({"msg_type": "exit_scene"})

    # ================================================================ receiving
    @property
    def frames_received(self) -> int:
        return self._frame_counter

    @property
    def backlog(self) -> int:
        """Frames received but not yet consumed."""
        with self._cond:
            return len(self._buffer)

    def latest(self) -> Optional[Telemetry]:
        """Most recent frame, without consuming anything."""
        with self._cond:
            return self._latest

    def get_frame(self, timeout: float = 1.0) -> Optional[Telemetry]:
        """Next unread frame in arrival order (no frame is skipped).
        Returns None on timeout or if the connection aborted."""
        with self._cond:
            self._cond.wait_for(lambda: self.aborted or self._buffer, timeout)
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def wait_for_latest(self, timeout: float = 1.0) -> Optional[Telemetry]:
        """Freshest frame; older unread frames are discarded (for real-time driving)."""
        with self._cond:
            self._cond.wait_for(lambda: self.aborted or self._buffer, timeout)
            if not self._buffer:
                return None
            tel = self._buffer[-1]
            self._buffer.clear()
            return tel

    def flush(self) -> None:
        """Discard unread frames (e.g. after setup, before starting a recording)."""
        with self._cond:
            self._buffer.clear()

    # ================================================================ setup
    def start(self, force_reload: bool = False) -> Dict[str, Any]:
        """Make sure the scene is loaded, configure car + camera once, verify the camera.
        Returns a dict describing what happened (useful for session metadata)."""
        info: Dict[str, Any] = {"scene_requested": self.sim_cfg.scene}

        # Observed behaviour: on connection the sim sends 'car_loaded' (+ telemetry) if a
        # scene is running, or 'scene_selection_ready' if it is at the menu.
        self._wait(lambda: self.car_loaded.is_set() or self._latest is not None
                   or self.scene_selection_ready.is_set(), self.sim_cfg.detect_loaded_timeout)
        already = self.car_loaded.is_set() or self._latest is not None
        if already and force_reload:
            logger.info("exiting current scene")
            self.exit_scene()
            # The sim does NOT send 'scene_selection_ready' after exit_scene (observed):
            # wait until telemetry stops instead.
            if not self._wait_telemetry_stopped(quiet=1.0, timeout=10.0):
                raise SimConnectionError("telemetry still running 10 s after exit_scene")
            with self._cond:
                self._latest = None
                self._buffer.clear()
            already = False

        if already:
            # The telemetry does not say which scene is running: it cannot be verified.
            logger.warning("a car is already loaded: reusing the current scene "
                           "(cannot verify it is '%s'; use force_reload to be sure)",
                           self.sim_cfg.scene)
            info["scene_loaded_by_us"] = False
        else:
            self._load_scene(self.sim_cfg.scene)
            info["scene_loaded_by_us"] = True
            info["scene_names"] = self.scene_names

        self.configure()
        info["camera_verified"] = self.verify_camera()
        self.flush()   # frames from before/during configuration are not wanted
        return info

    def _load_scene(self, scene: str) -> None:
        # The sim only answers get_scene_names once it is at the menu: retry a few times.
        for _ in range(3):
            self._scene_names_evt.clear()
            self.send_json({"msg_type": "get_scene_names"})
            if self._scene_names_evt.wait(2.0):
                break
        if self.scene_names is None:
            logger.warning("no scene list received; trying to load '%s' anyway", scene)
        elif scene not in self.scene_names:
            raise SimConnectionError(f"scene '{scene}' not available: {self.scene_names}")
        logger.info("loading scene %s", scene)
        self.car_loaded.clear()
        self.send_json({"msg_type": "load_scene", "scene_name": scene})
        if not self.car_loaded.wait(self.sim_cfg.load_timeout):
            raise SimConnectionError(f"scene '{scene}': no 'car_loaded' within "
                                     f"{self.sim_cfg.load_timeout} s")

    def configure(self) -> None:
        """Racer, car body, second camera (off), lidar (off), main camera."""
        self.send_json(self.car_cfg.racer_msg())
        time.sleep(0.2)
        self.send_json(self.car_cfg.car_msg())
        time.sleep(0.5)
        self.send_json({"msg_type": "cam_config_b", "enabled": "false"})
        time.sleep(0.1)
        self.send_json({"msg_type": "lidar_config", "enabled": "false"})
        time.sleep(0.1)
        self.send_json(self.cam_cfg.to_msg())
        self.need_car_config = False

    def verify_camera(self) -> bool:
        """Wait for a frame with the configured size (~0.1 s in the probe)."""
        w, h = self.cam_cfg.img_w, self.cam_cfg.img_h
        ok = self._wait(lambda: self._latest is not None and
                        (self._latest.img_w, self._latest.img_h) == (w, h),
                        self.sim_cfg.camera_check_timeout)
        if not ok:
            got = (self._latest.img_w, self._latest.img_h) if self._latest else None
            raise SimConnectionError(f"camera config not applied: expected {w}x{h}, got {got}")
        return True

    def _wait_telemetry_stopped(self, quiet: float, timeout: float) -> bool:
        """True once no telemetry frame has arrived for `quiet` seconds (counted from
        now at the earliest, so a call right after exit_scene always waits)."""
        t_start = time.perf_counter()
        end = t_start + timeout
        while time.perf_counter() < end:
            last = self.latest()
            ref = max(t_start, last.t_local if last else t_start)
            if time.perf_counter() - ref > quiet:
                return True
            time.sleep(0.05)
        return False

    @staticmethod
    def _wait(predicate, timeout: float, step: float = 0.02) -> bool:
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            if predicate():
                return True
            time.sleep(step)
        return bool(predicate())

    # ================================================================ shutdown
    def close(self) -> None:
        """Stop the car (zero throttle + brake) and close the connection."""
        try:
            if not self.aborted:
                self.send_controls(0.0, 0.0, 1.0)
                time.sleep(0.1)
        except OSError:
            pass
        self.stop()

    def __enter__(self) -> "DonkeyClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
