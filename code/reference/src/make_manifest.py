"""Write a hash manifest for the deterministic test set."""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from src.data import SEED, load_all  # noqa: E402


def digest(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main() -> None:
    rows = []
    for im in load_all():
        arr = im.image
        rows.append({
            "name": im.name,
            "kind": im.kind,
            "source_or_licence": im.license_note,
            "shape": "x".join(map(str, arr.shape)),
            "dtype": str(arr.dtype),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "sha256_png": digest(arr),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "image_manifest.csv", index=False, float_format="%.6f")
    print(df[["name", "sha256_png"]].to_string(index=False))
    print("seed:", SEED, "rows:", len(df))


if __name__ == "__main__":
    main()
