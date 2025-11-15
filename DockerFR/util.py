"""
Utility implementations for the face recognition project.

The code below stitches together a lightweight, classical computer vision
pipeline so the `/face-similarity` endpoint can operate without heavyweight
deep-learning dependencies. It is intentionally simple yet deterministic,
giving students a reference implementation they can extend with stronger
models later on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Reference five-point template (RetinaFace/ArcFace) for 112x112 crops.
RETINA_TEMPLATE: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]
    face: np.ndarray
    confidence: float


def _decode_image(image: Any) -> np.ndarray:
    """
    Convert raw bytes or numpy arrays into a BGR image understood by OpenCV.
    """
    if isinstance(image, np.ndarray):
        return image.copy()
    if isinstance(image, (bytes, bytearray)):
        array = np.frombuffer(image, dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Unable to decode the provided image bytes.")
        return decoded
    raise TypeError("Images must be raw bytes or numpy arrays.")


def detect_faces(image: Any) -> List[DetectedFace]:
    """
    Detect faces within the provided image using a Haar cascade detector.
    """
    frame = _decode_image(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    detections: List[DetectedFace] = []
    frame_area = float(frame.shape[0] * frame.shape[1])
    for (x, y, w, h) in faces:
        crop = frame[y : y + h, x : x + w]
        confidence = (w * h) / frame_area
        detections.append(
            DetectedFace(
                bbox=(int(x), int(y), int(w), int(h)),
                face=crop,
                confidence=confidence,
            )
        )
    detections.sort(key=lambda det: det.confidence, reverse=True)
    return detections


def compute_face_embedding(face_image: Any) -> np.ndarray:
    """
    Compute an embedding by running a Discrete Cosine Transform over
    a normalized grayscale crop and returning the first 512 coefficients.
    """
    face = _decode_image(face_image)
    resized = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dct = cv2.dct(gray)
    embedding = dct.flatten()[:512]
    if embedding.size < 512:
        embedding = np.pad(embedding, (0, 512 - embedding.size))
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding.astype(np.float32)


def detect_face_keypoints(face_image: Any) -> np.ndarray:
    """
    Produce five pseudo-landmarks positioned relative to the crop.

    This heuristic approximation supplies enough structure for alignment when
    no full landmark detector is available.
    """
    face = _decode_image(face_image)
    height, width = face.shape[:2]
    keypoints = np.array(
        [
            [0.3 * width, 0.35 * height],
            [0.7 * width, 0.35 * height],
            [0.5 * width, 0.52 * height],
            [0.32 * width, 0.75 * height],
            [0.68 * width, 0.75 * height],
        ],
        dtype=np.float32,
    )
    return keypoints


def _estimate_alignment_matrix(keypoints: np.ndarray) -> np.ndarray:
    """
    Estimate an affine transform that maps the detected keypoints to the
    RetinaFace template.
    """
    if keypoints.shape != (5, 2):
        raise ValueError("Expected five (x, y) keypoints for alignment.")
    matrix, _ = cv2.estimateAffinePartial2D(keypoints, RETINA_TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        LOGGER.warning("Failed to estimate alignment matrix, using identity transform.")
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return matrix.astype(np.float32)


def warp_face(image: Any, homography_matrix: Any) -> np.ndarray:
    """
    Warp the provided face image using the supplied homography/affine matrix.
    """
    face = _decode_image(image)
    matrix = np.asarray(homography_matrix, dtype=np.float32)
    if matrix.shape == (2, 3):
        return cv2.warpAffine(face, matrix, (112, 112))
    if matrix.shape == (3, 3):
        return cv2.warpPerspective(face, matrix, (112, 112))
    raise ValueError("Homography matrix must be 2x3 or 3x3.")


def antispoof_check(face_image: Any) -> float:
    """
    Perform a simple spoofing heuristic based on image sharpness. Blurry,
    low-texture faces generally receive a lower score than crisp, detailed
    real captures.
    """
    face = _decode_image(face_image)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = float(np.tanh(lap_var / 300.0))
    return max(0.0, min(1.0, score))


def _prepare_embedding(image_data: Any) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Helper that decodes the bytes, extracts the most confident face, aligns it,
    and returns the embedding along with useful diagnostics.
    """
    frame = _decode_image(image_data)
    detections = detect_faces(frame)
    if not detections:
        raise ValueError("No face detected in the provided image.")
    primary = detections[0]
    keypoints = detect_face_keypoints(primary.face)
    matrix = _estimate_alignment_matrix(keypoints)
    aligned = warp_face(primary.face, matrix)
    spoof_score = antispoof_check(aligned)
    embedding = compute_face_embedding(aligned)
    debug_info = {
        "bbox": primary.bbox,
        "confidence": primary.confidence,
        "spoof_score": spoof_score,
    }
    return embedding, spoof_score, debug_info


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns a similarity score between two faces by
    running detection -> alignment -> anti-spoofing -> embeddings and finally
    cosine similarity between the vectors.
    """
    emb_a, spoof_a, debug_a = _prepare_embedding(image_a)
    emb_b, spoof_b, debug_b = _prepare_embedding(image_b)

    if min(spoof_a, spoof_b) < 0.2:
        LOGGER.warning(
            "Potential spoof detected: scores %.3f / %.3f, boxes %s / %s",
            spoof_a,
            spoof_b,
            debug_a["bbox"],
            debug_b["bbox"],
        )

    similarity = float(np.dot(emb_a, emb_b) / ((np.linalg.norm(emb_a) + 1e-8) * (np.linalg.norm(emb_b) + 1e-8)))
    similarity = max(-1.0, min(1.0, similarity))
    return similarity
