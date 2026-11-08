from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "drone"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
TAG_SIZE_MM = 50.0
TAG_IDS = {0}

MIN_TAG_FRAMES = 10
TOF_MAX_STD_MM = 50.0
# the beam is then hopping between leaves and floor rather than tracking the drone

# A tape-measured calibration shot (Tello above a tag at known distance) would pin it down.
TELLO_FOCAL_PX = 840.0

# --- Masks, OpenCV HSV (H 0-180, S/V 0-255)
PLANT_HSV_LO = (
    20,
    50,
    40,
)
PLANT_HSV_HI = (90, 255, 255)
TARP_HSV_LO = (90, 40, 40)
TARP_HSV_HI = (130, 255, 255)
MIN_BLOB_PX = 400
