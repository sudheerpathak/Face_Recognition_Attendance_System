"""Face detection + embedding, built on OpenCV's YuNet detector and SFace
recognizer (both ship as small ONNX models, no dlib/CMake build needed).

Pipeline this module implements:
    frame -> detect_faces() -> largest_face()/each face -> embed() -> 128-d vector
The vector is what gets stored per student (Enroll) and compared live
(Matching) via cosine_score().
"""
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# SFace's officially recommended cosine-similarity match threshold
# (score >= this means "same person").
COSINE_MATCH_THRESHOLD = 0.363


class FaceEngine:
    def __init__(self, score_threshold=0.9):
        if not DETECTOR_PATH.exists() or not RECOGNIZER_PATH.exists():
            raise RuntimeError(
                "Model files missing under models/. Run `python download_models.py` first."
            )
        self._detector = cv2.FaceDetectorYN_create(
            str(DETECTOR_PATH), "", (320, 320), score_threshold=score_threshold
        )
        self._recognizer = cv2.FaceRecognizerSF_create(str(RECOGNIZER_PATH), "")

    def detect_faces(self, frame):
        """Returns an array of detected faces; each row is
        [x, y, w, h, 5x(landmark_x, landmark_y), score]."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    @staticmethod
    def largest_face(faces):
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def embed(self, frame, face_row):
        """Aligns + crops the given detected face and returns its 1x128 embedding."""
        aligned = self._recognizer.alignCrop(frame, face_row)
        return self._recognizer.feature(aligned)

    def cosine_score(self, emb1, emb2):
        """Higher = more similar. Match if >= COSINE_MATCH_THRESHOLD."""
        return self._recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
