"""Download the MediaPipe Face Landmarker model bundle.

Usage (from backend/):
    python scripts/download_model.py

The .task file bundles three models: a face detector, the 478-point face
mesh landmarker (468 mesh points + 10 iris points), and a blendshape model.
"""

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def main() -> None:
    if DEST.exists():
        print(f"Model already present at {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}\n  -> {DEST}")
    tmp = DEST.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(DEST)
    except Exception as exc:  # noqa: BLE001 - surface any download failure clearly
        tmp.unlink(missing_ok=True)
        sys.exit(f"Download failed: {exc}")
    print(f"Done ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
