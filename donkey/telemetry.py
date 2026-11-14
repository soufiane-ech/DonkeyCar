"""
Typed representation of one simulator telemetry packet.

All fields below were observed in 100% of packets for roboracingleague_1
(tools/probe_telemetry.py, 2026-11-13). Missing numeric fields become NaN
(int fields: -1) so that downstream cleaning can detect them, and unknown keys
are kept in `extra` instead of being silently dropped.
"""

import base64
import math
from dataclasses import dataclass, field, fields
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

_IMAGE_KEYS = ("image", "image_b", "imageb")

# telemetry key -> dataclass attribute (floats)
_FLOAT_FIELDS = {
    "time": "sim_time",
    "steering_angle": "steering_angle", "throttle": "throttle", "speed": "speed", "cte": "cte",
    "pos_x": "pos_x", "pos_y": "pos_y", "pos_z": "pos_z",
    "vel_x": "vel_x", "vel_y": "vel_y", "vel_z": "vel_z",
    "gyro_x": "gyro_x", "gyro_y": "gyro_y", "gyro_z": "gyro_z",
    "accel_x": "accel_x", "accel_y": "accel_y", "accel_z": "accel_z",
    "roll": "roll", "pitch": "pitch", "yaw": "yaw",
    "front_left_odom": "odom_fl", "front_right_odom": "odom_fr",
    "rear_left_odom": "odom_rl", "rear_right_odom": "odom_rr",
}
_INT_FIELDS = {"activeNode": "active_node", "totalNodes": "total_nodes"}
_WRAPPED_ANGLES = ("roll", "pitch")   # arrive in [0, 360): 359.99 means -0.01 (probe)


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def wrap_180(deg: float) -> float:
    """Map an angle in degrees to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0 if math.isfinite(deg) else deg


def image_format(data: Optional[bytes]) -> Optional[str]:
    """Identify the encoding from magic bytes (used for file extensions)."""
    if not data:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpg"
    return "tga"  # TGA has no reliable magic number; it is the only other sim option


@dataclass
class Telemetry:
    frame_id: int            # increasing counter assigned by our client (0, 1, 2, ...)
    t_local: float           # time.perf_counter() at reception (for latency measurements)
    wall_time: float         # time.time() at reception (human-readable timestamps)
    image_bytes: bytes       # encoded image exactly as sent by the simulator
    img_w: int = -1
    img_h: int = -1
    sim_time: float = math.nan   # simulator clock (s); steps of 50.05 ms -> authoritative timestamp
    steering_angle: float = math.nan  # echo of the last steering command, same -1..1 scale
    throttle: float = math.nan        # echo of the last throttle command
    speed: float = math.nan
    cte: float = math.nan             # cross-track error to the track path
    hit: str = "none"                 # name of object hit, "none" otherwise
    active_node: int = -1             # index of current path node (track progress)
    total_nodes: int = -1
    pos_x: float = math.nan
    pos_y: float = math.nan
    pos_z: float = math.nan
    vel_x: float = math.nan
    vel_y: float = math.nan
    vel_z: float = math.nan
    gyro_x: float = math.nan
    gyro_y: float = math.nan
    gyro_z: float = math.nan
    accel_x: float = math.nan
    accel_y: float = math.nan
    accel_z: float = math.nan
    roll: float = math.nan            # degrees, wrapped to [-180, 180)
    pitch: float = math.nan           # degrees, wrapped to [-180, 180)
    yaw: float = math.nan             # degrees, [0, 360) as sent (heading)
    odom_fl: float = math.nan
    odom_fr: float = math.nan
    odom_rl: float = math.nan
    odom_rr: float = math.nan
    image_b_bytes: Optional[bytes] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_packet(cls, packet: Dict[str, Any], frame_id: int, t_local: float,
                    wall_time: float) -> "Telemetry":
        img = base64.b64decode(packet["image"]) if isinstance(packet.get("image"), str) else b""
        img_b_str = packet.get("image_b", packet.get("imageb"))
        tel = cls(frame_id=frame_id, t_local=t_local, wall_time=wall_time, image_bytes=img,
                  image_b_bytes=base64.b64decode(img_b_str) if isinstance(img_b_str, str) else None)
        if img:
            try:
                with Image.open(BytesIO(img)) as im:   # header only, no pixel decoding
                    tel.img_w, tel.img_h = im.size
            except Exception:
                pass   # stays -1: flagged as an invalid frame later
        for key, attr in _FLOAT_FIELDS.items():
            if key in packet:
                setattr(tel, attr, _to_float(packet[key]))
        for key, attr in _INT_FIELDS.items():
            if key in packet:
                setattr(tel, attr, _to_int(packet[key]))
        if "hit" in packet:
            tel.hit = str(packet["hit"])
        for a in _WRAPPED_ANGLES:
            setattr(tel, a, wrap_180(getattr(tel, a)))
        known = set(_FLOAT_FIELDS) | set(_INT_FIELDS) | set(_IMAGE_KEYS) | {"msg_type", "hit"}
        tel.extra = {k: v for k, v in packet.items() if k not in known}
        return tel

    # ------------------------------------------------------------------ helpers
    @property
    def image_format(self) -> Optional[str]:
        return image_format(self.image_bytes)

    def image(self) -> np.ndarray:
        """Decode the camera image to an RGB uint8 array of shape (H, W, 3)."""
        with Image.open(BytesIO(self.image_bytes)) as im:
            return np.asarray(im.convert("RGB"))

    def to_record(self) -> Dict[str, Any]:
        """Flat dict of all scalar fields (no image bytes), e.g. for a CSV row."""
        skip = {"image_bytes", "image_b_bytes", "extra"}
        rec = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}
        rec["image_format"] = self.image_format
        return rec
