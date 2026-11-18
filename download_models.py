"""One-time setup: downloads the YuNet face detector and SFace face
recognizer ONNX models from the OpenCV Zoo into ./models/.

Run this once before starting the Flask app:
    python download_models.py
"""
from pathlib import Path

import requests

MODELS_DIR = Path(__file__).parent / "models"
FILES = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in FILES.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"[skip] {filename} already downloaded")
            continue
        print(f"[download] {filename} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"[done] saved to {dest} ({len(resp.content) // 1024} KB)")


if __name__ == "__main__":
    main()
