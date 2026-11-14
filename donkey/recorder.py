"""
Session recorder: saves (image, command, telemetry) samples to disk.

One session = one folder, never overwritten:

    data/raw/2026-11-13_20-30-00_roboracingleague_1_manual/
        images/000000.png ...   encoded bytes exactly as sent by the simulator
        records.csv             one row per image, see COLUMNS
        meta.json               configs, versions, counts, column documentation

Label definition (the pairing used for training):
    row i = image of telemetry frame i  +  the command sent right after
    receiving that frame (cmd_*). The simulator applies a command ~2 frames
    (~100 ms) later (measured in Step 2); echo_* columns show what the
    simulator reported as applied *in this frame*.

Design:
  * the drive loop calls record() at 20 Hz: no disk I/O there, it only appends
    to an in-memory "pending" buffer;
  * samples older than `hold_seconds` leave the buffer and go to a writer
    thread, which writes the image *before* its CSV row (a row never points
    to a missing file, even after a crash);
  * discard_last(seconds) removes samples still in the buffer (bad driving);
  * sample_id is assigned when a sample is committed, so ids are contiguous.
"""

import csv
import json
import logging
import math
import platform
import queue
import threading
import time
from collections import deque
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DATA_RAW_DIR
from .telemetry import Telemetry

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
ECHO_DELAY_FRAMES = 2   # measured in Step 2 (distribution 1:40, 2:342, 3:12 over 394 frames)

# Column order of records.csv (+ short documentation written into meta.json).
COLUMN_DOC = {
    "sample_id": "contiguous index within the session (= image file number)",
    "image": "image path relative to the session folder",
    "episode_id": "increments at each car reset",
    "mode": "driving mode tag set by the collector (e.g. normal, recovery)",
    "lap_count": "start-line crossings so far (-1 if unknown)",
    "cmd_steering": "TARGET: steering sent after receiving this image [-1, 1]",
    "cmd_throttle": "TARGET: throttle sent after receiving this image [-1, 1]",
    "cmd_brake": "brake sent after receiving this image [0, 1]",
    "cmd_latency_ms": "time between frame reception and command sending",
    "frame_id": "client frame counter (gaps = frames not recorded)",
    "t_session": "local reception time since session start (s)",
    "wall_time": "local reception time, unix epoch (s)",
    "sim_time": "simulator clock (s), 50 ms steps - authoritative timestamp",
    "echo_steering": "steering applied by the sim in this frame (lags cmd by ~2 frames)",
    "echo_throttle": "throttle applied by the sim in this frame",
    "speed": "car speed (sim units)",
    "cte": "cross-track error to the track path",
    "hit": "object hit, 'none' otherwise",
    "active_node": "current track path node (progress)",
    "total_nodes": "number of path nodes",
    "pos_x": "", "pos_y": "", "pos_z": "",
    "vel_x": "", "vel_y": "", "vel_z": "",
    "gyro_x": "", "gyro_y": "", "gyro_z": "",
    "accel_x": "", "accel_y": "", "accel_z": "",
    "roll": "degrees [-180, 180)", "pitch": "degrees [-180, 180)", "yaw": "degrees [0, 360)",
    "odom_fl": "wheel odometry", "odom_fr": "", "odom_rl": "", "odom_rr": "",
    "img_w": "", "img_h": "", "image_format": "png | jpg | tga",
}
COLUMNS: List[str] = list(COLUMN_DOC)
_TELEMETRY_RENAME = {"steering_angle": "echo_steering", "throttle": "echo_throttle"}


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("gym_donkeycar", "numpy", "pillow"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else repr(v)
    return v


class SessionRecorder:

    def __init__(self, scene: str, tag: str = "manual", root: Path = DATA_RAW_DIR,
                 config: Optional[Dict[str, Any]] = None, notes: str = "",
                 extra_meta: Optional[Dict[str, Any]] = None, hold_seconds: float = 5.0,
                 queue_size: int = 1000):
        self.created = datetime.now()
        self.dir = self._unique_dir(Path(root), f"{self.created:%Y-%m-%d_%H-%M-%S}_{scene}_{tag}")
        self.images_dir = self.dir / "images"
        self.images_dir.mkdir(parents=True)
        self.session_id = self.dir.name

        self.hold_seconds = hold_seconds
        self.mode = "normal"
        self.episode_id = 0
        self._t0 = time.perf_counter()
        self._pending: deque = deque()          # (t_local, Telemetry, row) not yet committed
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._next_sample_id = 0
        self._writer_error: Optional[BaseException] = None
        self._closed = False

        # counters
        self.n_recorded = 0          # record() calls
        self.n_discarded = 0         # removed by discard_last()
        self.n_written = 0           # rows on disk (updated by writer thread)
        self.discard_events: List[Dict[str, Any]] = []
        self.episodes: List[Dict[str, Any]] = [{"episode_id": 0, "reason": "start"}]
        self._first_sim_time = math.nan
        self._last_sim_time = math.nan

        self.meta: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "session_id": self.session_id,
            "status": "recording",
            "created": self.created.isoformat(timespec="seconds"),
            "scene": scene,
            "tag": tag,
            "notes": notes,
            "config": config or {},
            "versions": _versions(),
            "label_definition": ("cmd_* = command sent right after receiving this row's image. "
                                 f"The sim applies it ~{ECHO_DELAY_FRAMES} frames later "
                                 "(see echo_* columns)."),
            "command_echo_delay_frames": ECHO_DELAY_FRAMES,
            "columns": COLUMN_DOC,
            **(extra_meta or {}),
        }
        self._write_meta()

        self._csv_file = open(self.dir / "records.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=COLUMNS)
        self._csv.writeheader()
        self._csv_file.flush()
        self._writer = threading.Thread(target=self._writer_loop, name="recorder-writer", daemon=True)
        self._writer.start()
        logger.info("recording session %s", self.dir)

    # ================================================================ public API
    def record(self, tel: Telemetry, command: Dict[str, float], lap_count: int = -1,
               mode: Optional[str] = None) -> None:
        """Add one sample: `tel` (image + telemetry) and the command sent in response
        (as returned by DonkeyClient.send_controls)."""
        self._raise_writer_error()
        if self._closed:
            raise RuntimeError("recorder is closed")
        row = {_TELEMETRY_RENAME.get(k, k): v for k, v in tel.to_record().items()}
        row.pop("t_local", None)
        row.update({
            "episode_id": self.episode_id,
            "mode": mode or self.mode,
            "lap_count": lap_count,
            "cmd_steering": command["steering"],
            "cmd_throttle": command["throttle"],
            "cmd_brake": command.get("brake", 0.0),
            "cmd_latency_ms": round(1000.0 * (command["t_local"] - tel.t_local), 3)
                              if "t_local" in command else math.nan,
            "t_session": tel.t_local - self._t0,
        })
        self._pending.append((tel.t_local, tel, row))
        self.n_recorded += 1
        if math.isnan(self._first_sim_time):
            self._first_sim_time = tel.sim_time
        self._last_sim_time = tel.sim_time
        self._commit_older_than(time.perf_counter() - self.hold_seconds)

    def discard_last(self, seconds: float) -> int:
        """Drop samples received during the last `seconds` (only those still in
        memory, i.e. up to `hold_seconds`). Returns the number dropped."""
        limit = time.perf_counter() - seconds
        n = 0
        while self._pending and self._pending[-1][0] >= limit:
            self._pending.pop()
            n += 1
        self.n_discarded += n
        self.discard_events.append({"t_session": round(time.perf_counter() - self._t0, 3),
                                    "seconds": seconds, "frames": n})
        logger.info("discarded %d frames (last %.1f s)", n, seconds)
        return n

    def new_episode(self, reason: str = "reset", sim_time: float = math.nan) -> int:
        """Call when the car is reset (or a new attempt starts). `sim_time` = simulator
        time of the last frame before the reset (used to check the reset cooldown)."""
        self.episode_id += 1
        event = {"episode_id": self.episode_id, "reason": reason,
                 "t_session": round(time.perf_counter() - self._t0, 3)}
        if not math.isnan(sim_time):
            event["sim_time"] = sim_time
        self.episodes.append(event)
        return self.episode_id

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    def close(self, status: str = "complete") -> Path:
        """Commit everything still pending, stop the writer, finalize meta.json."""
        if self._closed:
            return self.dir
        self._commit_older_than(float("inf"))
        self._closed = True
        self._queue.put(None)                  # sentinel
        self._writer.join()
        self._csv_file.close()
        if self._writer_error is not None:
            status = "error"
        self.meta.update({
            "status": status if self.n_written else "empty",
            "finished": datetime.now().isoformat(timespec="seconds"),
            "counts": {"recorded": self.n_recorded, "discarded": self.n_discarded,
                       "written": self.n_written},
            "duration_sim_s": round(self._last_sim_time - self._first_sim_time, 3)
                              if self.n_recorded else 0.0,
            "episodes": self.episodes,
            "discard_events": self.discard_events,
        })
        if self._writer_error is not None:
            self.meta["writer_error"] = repr(self._writer_error)
        self._write_meta()
        logger.info("session closed: %s (%d samples)", self.dir, self.n_written)
        self._raise_writer_error()
        return self.dir

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Keep what was recorded even if the drive loop crashed.
        self.close(status="complete" if exc_type is None else "interrupted")

    # ================================================================ internals
    @staticmethod
    def _unique_dir(root: Path, name: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        for i in range(1, 1000):
            candidate = root / (name if i == 1 else f"{name}_{i}")
            try:
                candidate.mkdir()        # atomic: fails if it already exists
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError(f"cannot create a unique session folder for {name}")

    def _commit_older_than(self, limit: float) -> None:
        while self._pending and self._pending[0][0] < limit:
            _, tel, row = self._pending.popleft()
            row["sample_id"] = self._next_sample_id
            row["image"] = f"images/{self._next_sample_id:06d}.{tel.image_format or 'bin'}"
            self._next_sample_id += 1
            self._queue.put((tel.image_bytes, row))   # blocks only if the disk is far behind

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._writer_error is not None:
                continue                              # drain the queue after an error
            image_bytes, row = item
            try:
                (self.dir / row["image"]).write_bytes(image_bytes)
                self._csv.writerow({k: _fmt(row.get(k, "")) for k in COLUMNS})
                self._csv_file.flush()
                self.n_written += 1
            except BaseException as e:               # surfaced in the main thread
                self._writer_error = e
                logger.exception("recorder writer failed")

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError(f"recorder writer failed: {self._writer_error!r}")

    def _write_meta(self) -> None:
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.dir / "meta.json")
