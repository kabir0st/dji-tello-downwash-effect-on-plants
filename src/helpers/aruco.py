import cv2
import numpy as np
from cv2 import aruco

ARUCO_DICT = aruco.DICT_4X4_50
TAG_SIZE_MM = 50.0
TAG_IDS = {0}


def make_detector():
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    p = aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 63
    p.adaptiveThreshWinSizeStep = 6
    p.polygonalApproxAccuracyRate = 0.05
    p.minMarkerPerimeterRate = 0.08
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, p)


def detect_tags(bgr, detector):
    corners, ids, _ = detector.detectMarkers(bgr.max(axis=2))
    if ids is None:
        return []
    return [
        (int(i), c[0].astype(np.float64))
        for c, i in zip(corners, ids.ravel())
        if int(i) in TAG_IDS
    ]


def tag_width_px(corners):
    return float(
        np.mean([np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)])
    )


def focal_from_tag(width_px, distance_mm):
    """Pinhole: f = w * Z / S"""
    return width_px * distance_mm / TAG_SIZE_MM


def distance_from_tag(width_px, focal_px):
    """Pinhole: Z = f * S / w"""
    return focal_px * TAG_SIZE_MM / width_px
