"""
Central configuration. Every script imports its defaults from here, so a
session's settings live in one place and can be saved into its meta.json.

Values marked (probe) were verified with tools/probe_telemetry.py on
2026-11-13 against the running simulator.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Command ranges accepted by the simulator. Steering and throttle are echoed back
# unchanged in telemetry ("steering_angle", "throttle") on the same scale (probe).
STEERING_RANGE = (-1.0, 1.0)
THROTTLE_RANGE = (-1.0, 1.0)   # negative = reverse
BRAKE_RANGE = (0.0, 1.0)       # continuous hand brake

TELEMETRY_HZ = 20.0            # simulator 'time' advances by 50.05 ms per frame (probe)


@dataclass(frozen=True)
class SimConfig:
    host: str = "127.0.0.1"        # "trainmydonkey.com" for the virtual racing server
    port: int = 9091
    scene: str = "roboracingleague_1"
    poll_socket_sleep: float = 0.01   # SDClient socket-thread poll period (needed on Windows)
    detect_loaded_timeout: float = 2.0  # wait for an already-spawned car before loading a scene
    load_timeout: float = 30.0
    camera_check_timeout: float = 3.0


@dataclass(frozen=True)
class CarConfig:
    # body_style: donkey | bare | car01 | cybertruck | f1
    # Keep the same style/colour for every recording session: the car's hood is
    # visible at the bottom of the camera image (probe).
    body_style: str = "car01"
    body_rgb: tuple = (255, 0, 255)
    car_name: str = "Choupette 53"
    font_size: int = 100
    racer_name: str = "Reda"
    bio: str = ""
    country: str = "France"
    guid: str = "essai"

    def racer_msg(self) -> dict:
        return {"msg_type": "racer_info", "racer_name": self.racer_name, "car_name": self.car_name,
                "bio": self.bio, "country": self.country, "guid": self.guid}

    def car_msg(self) -> dict:
        r, g, b = self.body_rgb
        return {"msg_type": "car_config", "body_style": self.body_style,
                "body_r": str(r), "body_g": str(g), "body_b": str(b),
                "car_name": self.car_name, "font_size": str(self.font_size)}


@dataclass(frozen=True)
class CameraConfig:
    # 0 for a field = simulator default. Default encoding is JPEG; PNG is lossless (probe).
    # Resolution chosen with tools/probe_camera.py (2026-11-13):
    #   160x120  sim default size, too coarse to read the track at a distance;
    #   320x240  exact size received, 20.0 fps, no missing frames, ~110 KB/frame PNG,
    #            ~5 ms decode  -> CHOSEN (sharp, same field of view as 160x120);
    #   640x480  the simulator slows down to ~0.7x real time  -> rejected.
    # fov 0 = default = 90 deg (identical images). Camera offsets put the car body
    # in the view, rot_x tilt gave no usable gain -> keep the default camera pose.
    # Every session saves this config in meta.json; training/inference must use the
    # same one (preprocessing can still downscale later).
    img_w: int = 320
    img_h: int = 240
    img_d: int = 3
    img_enc: str = "PNG"           # PNG | JPG | TGA
    fov: int = 0
    fish_eye_x: float = 0.0
    fish_eye_y: float = 0.0
    offset_x: float = 0.0          # left / right
    offset_y: float = 0.0          # up / down
    offset_z: float = 0.0          # forward / back
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0

    def to_msg(self, msg_type: str = "cam_config") -> dict:
        # Same field set as gym_donkeycar's send_cam_config (all values as strings).
        return {"msg_type": msg_type, **{k: str(v) for k, v in asdict(self).items()}}


def config_dict(sim: SimConfig, car: CarConfig, cam: CameraConfig) -> dict:
    """Plain-dict snapshot, e.g. for a session's meta.json."""
    return {"sim": asdict(sim), "car": asdict(car), "camera": asdict(cam)}
