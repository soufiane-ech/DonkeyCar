"""
CNN driver: camera image (+ current speed) -> steering, throttle, brake.

preprocess() is shared by training and driving, so both always see exactly
the same input. The model never receives cte, position or track nodes.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 320x240 camera: rows above ~CROP_TOP are walls/ceiling, rows below CROP_BOTTOM
# show the car's own hood.
CROP_TOP, CROP_BOTTOM = 80, 215
IN_H, IN_W = 64, 160
SPEED_SCALE = 5.0
BRAKE_VALUE = 0.3          # brake applied when the CNN predicts "brake" (= autopilot's value)
BRAKE_THRESHOLD = 0.5


def preprocess(img: np.ndarray) -> np.ndarray:
    """(240, 320, 3) uint8 RGB -> (IN_H, IN_W, 3) uint8."""
    crop = img[CROP_TOP:CROP_BOTTOM]
    return np.array(Image.fromarray(crop).resize((IN_W, IN_H), Image.BILINEAR))


class DriveNet(nn.Module):
    """PilotNet-style CNN (~1M parameters), small enough to train on a CPU."""

    def __init__(self):
        super().__init__()

        def block(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, padding=k // 2), nn.BatchNorm2d(o), nn.ReLU())

        self.features = nn.Sequential(
            block(3, 24, 5, 2), block(24, 32, 5, 2), block(32, 48, 3, 2),
            block(48, 64, 3, 1), block(64, 64, 3, 1), nn.Flatten())
        n = 64 * (IN_H // 8) * (IN_W // 8)
        self.head = nn.Sequential(nn.Linear(n + 1, 100), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 3))

    def forward(self, x, speed):
        # x: (B, 3, H, W) float in [0, 1]; speed: (B, 1) scaled
        z = torch.cat([self.features(x), speed], dim=1)
        out = self.head(z)
        # steering [-1, 1], throttle [0, 1], brake logit (P(brake) = sigmoid)
        return torch.tanh(out[:, :1]), torch.sigmoid(out[:, 1:2]), out[:, 2:3]


def to_tensor(batch_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).float().div_(255.0)


class CnnDriver:
    def __init__(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model = DriveNet()
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        torch.set_num_threads(2)
        self.meta = ckpt.get("meta", {})
        # "binary": brake on/off at BRAKE_VALUE (models up to cnn.pt v3)
        # "regression": third output = brake amount in [0, 1] (sigmoid)
        self.brake_mode = self.meta.get("brake_mode", "binary")

    @torch.no_grad()
    def __call__(self, tel):
        x = to_tensor(preprocess(tel.image())[None])
        v = torch.tensor([[tel.speed / SPEED_SCALE]], dtype=torch.float32)
        steer, thr, brake_out = self.model(x, v)
        p = float(torch.sigmoid(brake_out)[0, 0])
        if self.brake_mode == "regression":
            brake = p if p > 0.05 else 0.0
        else:
            brake = BRAKE_VALUE if p > BRAKE_THRESHOLD else 0.0
        return float(steer[0, 0]), 0.0 if brake else float(thr[0, 0]), brake
