"""TrackNet architecture used by the pretrained ball and court-keypoint weights.

Ported from github.com/yastrebksv/TrackNet so the published state dicts load unchanged.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from .paths import WEIGHTS


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, input_channels: int = 3, out_channels: int = 14):
        super().__init__()
        self.conv1 = ConvBlock(input_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)

    def forward(self, x):
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv7(self.conv6(self.conv5(x))))
        x = self.ups1(self.conv10(self.conv9(self.conv8(x))))
        x = self.ups2(self.conv13(self.conv12(self.conv11(x))))
        x = self.ups3(self.conv15(self.conv14(x)))
        return self.conv18(self.conv17(self.conv16(x)))


PRETRAINED_BALL = WEIGHTS / "tracknet.pt"
FINETUNED_BALL = WEIGHTS / "tracknet_ft.pt"
COREML_DIR = WEIGHTS / "coreml"


class BallMask(nn.Module):
    """Ball model for the Neural Engine.

    Takes the current and two previous frames as separate (1, H, W, 3) BGR pixel arrays (0-255),
    so the CPU does no channel reordering, concatenation or scaling. Returns one channel that is
    positive exactly where the heatmap argmax exceeds 127, which is all the tracker thresholds on;
    returning the 256-class heatmap made the Neural Engine 4x slower (11 vs 44 fps on an M3 Pro).
    """

    def __init__(self, net: BallTrackerNet):
        super().__init__()
        self.net = net

    def forward(self, cur, prev1, prev2):
        x = torch.cat([cur, prev1, prev2], 3).permute(0, 3, 1, 2) / 255
        y = self.net(x)
        return y[:, 128:].amax(1) - y[:, :128].amax(1)


def coreml_ball(path: Path, height: int = 360, width: int = 512):
    """Core ML (Neural Engine) conversion of the given ball weights, cached by weights tag."""
    import coremltools as ct
    import numpy as np

    out = COREML_DIR / f"{weights_tag(path).replace(':', '_')}_{width}x{height}_hwc_mask.mlpackage"
    if not out.exists():
        example = [torch.zeros(1, height, width, 3)] * 3
        with torch.no_grad():
            traced = torch.jit.trace(BallMask(load("ball", torch.device("cpu"), path)).eval(), example)
        model = ct.convert(traced, inputs=[ct.TensorType(name=k, shape=example[0].shape, dtype=np.float16)
                                           for k in ("cur", "prev1", "prev2")],
                           outputs=[ct.TensorType(name="margin", dtype=np.float16)],
                           minimum_deployment_target=ct.target.macOS14, convert_to="mlprogram")
        COREML_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name("tmp_" + out.name)
        model.save(str(tmp))
        tmp.rename(out)
    return ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_AND_NE)


def coreml_available() -> bool:
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return torch.backends.mps.is_available()


def device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # fixed input shapes per chunk
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def yolo_device() -> str:
    """Device string in the form Ultralytics expects."""
    dev = device()
    return "0" if dev.type == "cuda" else dev.type


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def ball_weights(path: str | Path | None = None) -> Path:
    """Explicit path, else the US Open fine-tune when it exists, else the pretrained weights."""
    if path:
        return Path(path)
    return FINETUNED_BALL if FINETUNED_BALL.exists() else PRETRAINED_BALL


def weights_tag(path: Path) -> str:
    """Short content hash, stored with cached tracks so a weights change triggers re-tracking."""
    st = path.stat()
    return _tag(str(path), st.st_mtime_ns, st.st_size)


@lru_cache
def _tag(path: str, mtime_ns: int, size: int) -> str:
    return f"{Path(path).stem}:{hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]}"


def load(kind: str, dev: torch.device | None = None, weights: str | Path | None = None) -> BallTrackerNet:
    dev = dev or device()
    if kind == "ball":
        model, path = BallTrackerNet(9, 256), ball_weights(weights)
    elif kind == "court":
        model, path = BallTrackerNet(3, 15), Path(weights) if weights else WEIGHTS / "court.pt"
    else:
        raise ValueError(kind)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(dev).eval()
