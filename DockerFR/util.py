"""
Utility implementations for the face recognition project using InsightFace/RetinaFace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict

import cv2
import numpy as np
from insightface.app import FaceAnalysis

LOGGER = logging.getLogger(__name__)

# Initialize InsightFace app with RetinaFace detector and ArcFace recognizer
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]
    face: np.ndarray
    confidence: float
    keypoints: np.ndarray
    embedding: np.ndarray


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
    Detect faces using InsightFace RetinaFace detector.
    Returns faces with bounding boxes, keypoints, and embeddings.
    """
    frame = _decode_image(image)
    faces = face_app.get(frame)
    
    if not faces:
        return []
    
    detections: List[DetectedFace] = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        crop = frame[y1:y2, x1:x2]
        confidence = float(face.det_score)
        keypoints = face.kps  # 5 keypoints
        embedding = face.normed_embedding  # 512-dim ArcFace embedding
        
        detections.append(
            DetectedFace(
                bbox=(x1, y1, w, h),
                face=crop,
                confidence=confidence,
                keypoints=keypoints,
                embedding=embedding,
            )
        )
    
    # Sort by confidence
    detections.sort(key=lambda det: det.confidence, reverse=True)
    return detections


def compute_face_embedding(face_image: Any) -> np.ndarray:
    """
    Compute face embedding using ArcFace model from InsightFace.
    Returns a 512-dimensional normalized embedding.
    """
    frame = _decode_image(face_image)
    faces = face_app.get(frame)
    
    if not faces:
        raise ValueError("No face detected in the image")
    
    # Return the embedding of the most confident face
    return faces[0].normed_embedding


def detect_face_keypoints(face_image: Any) -> np.ndarray:
    """
    Detect 5 facial keypoints (landmarks) using RetinaFace.
    Returns: (5, 2) array of [x, y] coordinates for:
    - Left eye, Right eye, Nose tip, Left mouth corner, Right mouth corner
    """
    frame = _decode_image(face_image)
    faces = face_app.get(frame)
    
    if not faces:
        raise ValueError("No face detected in the image")
    
    # Return keypoints of the most confident face
    return faces[0].kps


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
    Simple anti-spoofing check based on image sharpness.
    Note: For production, use a dedicated anti-spoofing model.
    """
    face = _decode_image(face_image)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = float(np.tanh(lap_var / 300.0))
    return max(0.0, min(1.0, score))


def _prepare_embedding(image_data: Any) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Helper that extracts face, computes embedding, and returns diagnostics.
    """
    detections = detect_faces(image_data)
    
    if not detections:
        raise ValueError("No face detected in the provided image.")
    
    primary = detections[0]
    spoof_score = antispoof_check(primary.face)
    
    debug_info = {
        "bbox": primary.bbox,
        "confidence": primary.confidence,
        "spoof_score": spoof_score,
        "num_faces": len(detections),
    }
    
    return primary.embedding, spoof_score, debug_info


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns similarity score between two faces.
    Uses RetinaFace for detection and ArcFace for embeddings.
    """
    emb_a, spoof_a, debug_a = _prepare_embedding(image_a)
    emb_b, spoof_b, debug_b = _prepare_embedding(image_b)
    
    if min(spoof_a, spoof_b) < 0.2:
        LOGGER.warning(
            "Potential spoof detected: scores %.3f / %.3f",
            spoof_a,
            spoof_b,
        )
    
    # Cosine similarity (embeddings are already normalized)
    similarity = float(np.dot(emb_a, emb_b))
    similarity = max(-1.0, min(1.0, similarity))
    
    LOGGER.info(
        "Similarity: %.4f | Face A: %d faces (conf=%.3f) | Face B: %d faces (conf=%.3f)",
        similarity,
        debug_a["num_faces"],
        debug_a["confidence"],
        debug_b["num_faces"],
        debug_b["confidence"],
    )
    
    return similarity