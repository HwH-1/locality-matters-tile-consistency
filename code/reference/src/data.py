"""Test image loading and deterministic synthetic image generation.

Natural images are the public-domain / CC0 samples bundled with scikit-image.
Each image is resized so that the shorter side is 256 px and centre-cropped to
256 x 256.  Synthetic images cover orientation, edge, scale and flat-region
behaviour.  All randomness is seeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from skimage import data

SIZE = 256
SEED = 20261025


@dataclass(frozen=True)
class TestImage:
    name: str
    kind: str
    license_note: str
    image: np.ndarray  # float32 H x W x 3 in [0, 1]


def _to_rgb_float(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _resize_crop(arr: np.ndarray, size: int = SIZE) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return np.ascontiguousarray(arr[top : top + size, left : left + size])


def _add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img + rng.normal(0.0, sigma, img.shape).astype(np.float32), 0.0, 1.0)


def synth_wedges(size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx) / (size / 2.0)
    phi = np.arctan2(yy - cy, xx - cx)
    k = np.floor((phi + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    palette = np.array(
        [
            [0.86, 0.32, 0.24],
            [0.91, 0.62, 0.19],
            [0.85, 0.84, 0.25],
            [0.35, 0.70, 0.33],
            [0.22, 0.63, 0.66],
            [0.24, 0.42, 0.74],
            [0.52, 0.32, 0.70],
            [0.80, 0.30, 0.55],
        ],
        dtype=np.float32,
    )
    img = palette[k]
    rings = 0.5 + 0.5 * np.cos(10 * np.pi * r)
    img = img * (0.72 + 0.28 * rings[..., None])
    img[r > 0.98] = 0.95
    img[r < 0.06] = 0.98
    return _add_noise(np.clip(img, 0, 1), 0.015, rng)


def synth_checker(size: int = SIZE, cell: int = 32) -> np.ndarray:
    rng = np.random.default_rng(SEED + 2)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Anti-aliased square wave via a smooth cosine product.
    fx = np.cos(np.pi * xx / cell)
    fy = np.cos(np.pi * yy / cell)
    v = np.sign(fx * fy)
    v = 0.5 + 0.5 * np.tanh(v * 6.0)
    base = np.stack([v, v, v], axis=-1)
    tint = np.array([0.98, 0.94, 0.88], dtype=np.float32)
    img = base * tint + (1 - base) * np.array([0.08, 0.10, 0.14], dtype=np.float32)
    return _add_noise(img, 0.01, rng)


def synth_stripes(size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(SEED + 3)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    t = yy / (size - 1)
    period = 4.0 + 26.0 * t
    angle = (np.pi / 2.0) * (xx / (size - 1))
    coord = (xx - size / 2) * np.cos(angle) + (yy - size / 2) * np.sin(angle)
    s = 0.5 + 0.5 * np.sin(2 * np.pi * coord / period)
    s = (s**1.6).astype(np.float32)
    img = np.stack(
        [0.12 + 0.80 * s, 0.18 + 0.72 * s * 0.95, 0.25 + 0.60 * s * 0.85], axis=-1
    )
    return _add_noise(np.clip(img, 0, 1), 0.012, rng)


def synth_regions(size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(SEED + 4)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = 0.18 + 0.55 * xx / (size - 1)
    img[..., 1] = 0.20 + 0.45 * yy / (size - 1)
    img[..., 2] = 0.35 + 0.25 * (1 - xx / (size - 1))

    def disc(cx, cy, rad, colour):
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
        img[m] = colour

    disc(70, 80, 46, np.array([0.88, 0.80, 0.24], dtype=np.float32))
    disc(185, 175, 52, np.array([0.78, 0.28, 0.32], dtype=np.float32))
    disc(178, 72, 30, np.array([0.26, 0.62, 0.52], dtype=np.float32))
    img[150:235, 25:95] = np.array([0.24, 0.36, 0.72], dtype=np.float32)
    return _add_noise(np.clip(img, 0, 1), 0.035, rng)


NATURAL: list[tuple[str, Callable[[], np.ndarray], str]] = [
    ("astronaut", data.astronaut, "NASA, public domain"),
    ("camera", data.camera, "CC0, Lav Varshney"),
    ("coffee", data.coffee, "CC0, Rachel Michetti"),
    ("chelsea", data.chelsea, "CC0, Stefan van der Walt"),
    ("rocket", data.rocket, "SpaceX, public domain"),
    ("coins", data.coins, "No known copyright restrictions"),
    ("hubble", data.hubble_deep_field, "NASA, public domain"),
    ("brick", data.brick, "CC0Textures, CC0"),
]

SYNTHETIC: list[tuple[str, Callable[[], np.ndarray]]] = [
    ("synth_wedges", synth_wedges),
    ("synth_checker", synth_checker),
    ("synth_stripes", synth_stripes),
    ("synth_regions", synth_regions),
]


def load_all() -> list[TestImage]:
    out: list[TestImage] = []
    for name, fn, note in NATURAL:
        out.append(TestImage(name, "natural", note, _resize_crop(_to_rgb_float(fn()))))
    for name, fn in SYNTHETIC:
        out.append(TestImage(name, "synthetic", "Generated here (deterministic)", fn()))
    return out


def save_rgb(path, img: np.ndarray) -> None:
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
