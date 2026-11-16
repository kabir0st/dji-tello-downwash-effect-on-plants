"""Plant motion and shape metrics from SAM masks (method: logs_justification.md §3-§5)."""

from collections import deque

import cv2
import numpy as np
import pandas as pd

from helpers.aruco import TAG_SIZE_MM, detect_tags, make_detector, tag_width_px
from helpers.configs import TELLO_FOCAL_PX
from helpers.masks import iter_masks

LAG_S = 0.1  # shape change is measured between masks 0.1 s apart, whatever the fps
FLOW_MAX_SIDE = 960  # flow at <= 960 px: Tello at full size, phone at half size
BG_ERODE = 9  # px gap between the plant edge and the background sample
BG_STEP = 8  # every 8th background pixel is used to fit the camera motion
RANSAC_PX = 1.0  # background points further than this from the fit are ignored (tarp flutter)
MIN_ZOOM_PCT = 1.0  # below this the camera did not move up/down enough to fit the canopy zoom
TAG_FRAMES = 30  # frames searched for the ArUco tag
COLUMNS = [
    "frame",
    "t_s",
    "area",
    "cx",
    "cy",
    "width",
    "height",
    "iou_lag",
    "flow_mean_px",
    "flow_p95_px",
    "ego_px",
    "bg_residual_px",
    "cam_scale",
]


def shape_metrics(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return {"area": 0}
    return {
        "area": len(ys),
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else np.nan


def camera_motion(flow, bg, xy):
    # shift + rotation + zoom of the background between two frames, as a 2x3 matrix
    sel = bg[::BG_STEP, ::BG_STEP]
    src = xy[::BG_STEP, ::BG_STEP][sel]
    dst = src + flow[::BG_STEP, ::BG_STEP][sel]
    M = None
    if len(src) >= 3:
        M, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX
        )
    return np.float64([[1, 0, 0], [0, 1, 0]]) if M is None else M


def flow_metrics(flow, xy, plant, bg, M):
    cam = xy @ M[:, :2].T + M[:, 2] - xy  # the flow the camera motion alone would cause
    net = np.linalg.norm(flow - cam, axis=2)
    row = {}
    if bg.any():
        row["bg_residual_px"] = float(np.median(net[bg]))
    if plant.any():
        row["flow_mean_px"] = float(net[plant].mean())
        row["flow_p95_px"] = float(np.percentile(net[plant], 95))
        row["ego_px"] = float(np.linalg.norm(cam[plant], axis=1).mean())
    return row


def frame_metrics(video_path, mask_file, fps):
    lag = max(1, round(LAG_S * fps))
    history = deque(maxlen=lag)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    kernel = np.ones((BG_ERODE, BG_ERODE), np.uint8)
    cap = cv2.VideoCapture(video_path)
    rows, prev = [], None
    to_first, scale = np.eye(3), 1.0  # current frame -> frame 0, and background zoom since frame 0
    try:
        for i, m in iter_masks(mask_file):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = m.shape
            s = min(1.0, FLOW_MAX_SIDE / max(h, w))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            plant = cv2.resize(m.astype(np.uint8), gray.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            row = {"frame": i, "t_s": i / fps}
            if prev is None:
                ys, xs = np.mgrid[: gray.shape[0], : gray.shape[1]]
                xy = np.dstack([xs, ys]).astype(np.float32) / s  # full-size pixel coordinates
            else:
                prev_gray, prev_plant = prev
                flow = dis.calc(prev_gray, gray, None) / s
                bg = cv2.erode((~prev_plant).astype(np.uint8), kernel) > 0
                M = camera_motion(flow, bg, xy)
                row |= flow_metrics(flow, xy, prev_plant, bg, M)
                to_first = to_first @ np.linalg.inv(np.vstack([M, [0, 0, 1]]))
                scale *= np.hypot(M[0, 0], M[1, 0])
            row["cam_scale"] = scale
            # undo the camera motion so shape is measured as if the camera never moved
            stable = cv2.warpAffine(m.astype(np.uint8), to_first[:2], (w, h), flags=cv2.INTER_NEAREST) > 0
            row |= shape_metrics(stable)
            if len(history) == lag:
                row["iou_lag"] = iou(stable, history[0])
            history.append(stable)
            prev = (gray, plant)
            rows.append(row)
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=COLUMNS)


def canopy_zoom(df):
    # the canopy is closer than the background, so it zooms more when the drone goes up/down.
    # fit how much extra: ln(area) = slope * ln(background zoom) + c
    seen = df[df["area"] > 0]
    log_area, log_scale = np.log(seen["area"]), np.log(seen["cam_scale"])
    zoom_pct = 100 * (df["cam_scale"].max() / df["cam_scale"].min() - 1)
    if zoom_pct < MIN_ZOOM_PCT or len(seen) < 3:
        return zoom_pct, 0.0, log_area
    slope = np.polyfit(log_scale, log_area, 1)[0]
    # the canopy is never farther than the background: a negative slope is not zoom, don't remove it
    return zoom_pct, slope, log_area - max(slope, 0.0) * log_scale


def tag_mm_per_px(video_path):
    detector = make_detector()
    cap = cv2.VideoCapture(video_path)
    widths = []
    for _ in range(TAG_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        widths += [tag_width_px(c) for _, c in detect_tags(frame, detector)]
    cap.release()
    return TAG_SIZE_MM / float(np.median(widths)) if widths else np.nan


def mm_per_px(run, cam):
    if cam == "phone":
        return tag_mm_per_px(run[cam])
    return run["height_cm"] * 10 / TELLO_FOCAL_PX


def summarize(df, fps, mm):
    size_px = np.sqrt(df["area"].median())
    flow = df["flow_mean_px"].mean()
    residual = df["bg_residual_px"].mean()
    zoom_pct, zoom_slope, log_area = canopy_zoom(df)
    return {
        "n_frames": len(df),
        "empty_frames": int((df["area"] == 0).sum()),
        "zoom_pct": zoom_pct,
        "depth_ratio": 1 + zoom_slope / 2 if zoom_slope else np.nan,
        "flow_mm_s": flow * fps * mm,
        "flow_p95_mm_s": df["flow_p95_px"].mean() * fps * mm,
        "ego_mm_s": df["ego_px"].mean() * fps * mm,
        "flow_rel_s": flow * fps / size_px,
        "signal_ratio": flow / residual if residual else np.nan,
        "shape_change": (1 - df["iou_lag"]).mean(),
        "area_cv": log_area.std(),
        "jitter_rel": np.hypot(df["cx"].std(), df["cy"].std()) / size_px,
        "height_mm": df["height"].median() * mm,
        "width_mm": df["width"].median() * mm,
    }
