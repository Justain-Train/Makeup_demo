"""Download the face-parsing weights (BiSeNet ResNet18, ONNX) and verify SHA-256.

Usage (from backend/):
    python scripts/download_face_parser.py                 # ResNet18 (default)
    python scripts/download_face_parser.py --arch resnet34 # benchmark only

LICENSE WARNING: the weights were trained on CelebAMask-HQ, which is
non-commercial research data. Not cleared for commercial use.

The upstream project publishes no checksums. The hashes below were computed
from the files downloaded on 2026-11-10 (trust on first use) and are pinned so
that any later change to the file is detected. The weights are gitignored.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights"
MODELS = {
    "resnet18": (
        "face_parser_resnet18.onnx",
        "resnet18.onnx",
        53_205_364,
        "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f",
    ),
    "resnet34": (
        "face_parser_resnet34.onnx",
        "resnet34.onnx",
        93_632_554,
        "5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e",
    ),
}
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(MODELS), default="resnet18")
    args = ap.parse_args()
    dest_name, remote_name, size, expected = MODELS[args.arch]
    dest = MODELS_DIR / dest_name

    if dest.exists() and sha256_of(dest) == expected:
        print(f"Weights already present and verified at {dest}")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{remote_name}"
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url}\n  -> {dest}")
    try:
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size != size:
            raise ValueError(f"unexpected size {tmp.stat().st_size} (expected {size})")
        digest = sha256_of(tmp)
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch: got {digest}, expected {expected}")
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001 - surface any failure clearly
        tmp.unlink(missing_ok=True)
        sys.exit(f"Download failed: {exc}")
    print(f"Done and verified ({dest.stat().st_size / 1e6:.1f} MB)")
    print("Note: CelebAMask-HQ-trained weights are non-commercial research use only.")


if __name__ == "__main__":
    main()
