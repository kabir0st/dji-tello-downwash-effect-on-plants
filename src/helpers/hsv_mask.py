import cv2
import numpy as np

from helpers.configs import MIN_BLOB_PX, PLANT_HSV_HI, PLANT_HSV_LO

BORDER_PX, BG_ERODE_PX = 8, 15


def plant_mask_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, PLANT_HSV_LO, PLANT_HSV_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_PX]
    return np.isin(lab, keep)


def border_mask(shape):
    m = np.zeros(shape[:2], bool)
    m[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = True
    return m


def background_mask(plant):
    k = np.ones((BG_ERODE_PX, BG_ERODE_PX), np.uint8)
    return cv2.erode((~plant).astype(np.uint8), k) > 0


def masks_hsv(frame, i):
    m = plant_mask_hsv(frame)
    return m, m
