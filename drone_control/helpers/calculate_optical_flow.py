import sys

import cv2
import numpy as np
from get_distance import focal_for_width, get_distance_mm

VIDEO_PATH = "stablized.mp4"
BASELINE_VIDEO_PATH = None
DISTANCE_IMAGE_PATH = "get_dist.png"  # frame with ArUco tag next to the plant
SHOW = True  # show video next to motion mask (press q to hide)
MOTION_THRESHOLD_PX = 1.0  # flow above this is drawn white


def show_motion(frame, mag):
    mask = (mag > MOTION_THRESHOLD_PX).astype(np.uint8) * 255
    view = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    cv2.imshow("frame | motion", cv2.resize(view, None, fx=0.4, fy=0.4))
    return cv2.waitKey(1) & 0xFF != ord("q")


def flow_magnitudes(video_path, show=False):
    """Per-frame mean and RMS optical flow magnitude (px/frame)."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    if not ok:
        raise FileNotFoundError(f"Cannot open {video_path}")
    width = frame.shape[1]
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    means, rms = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        means.append(mag.mean())
        rms.append(np.sqrt(np.mean(mag**2)))
        if show:
            show = show_motion(frame, mag)
        prev = gray
    cap.release()
    cv2.destroyAllWindows()
    return float(np.mean(means)), float(np.mean(rms)), width


def find_distance_mm(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    distance = get_distance_mm(frame) if ok else None
    return distance


def analyze(video_path, show=False):
    mean_px, rms_px, width = flow_magnitudes(video_path, show)
    distance_mm = find_distance_mm(video_path)
    if not distance_mm:
        distance_mm = get_distance_mm(cv2.imread(DISTANCE_IMAGE_PATH))
    # px/mm on the plant plane
    scale = focal_for_width(width) / distance_mm
    return {
        "distance_mm": distance_mm,
        "scale_px_per_mm": scale,
        "mean_px": mean_px,
        "rms_px": rms_px,
        "mean_mm": mean_px / scale,
        "rms_mm": rms_px / scale,
    }


def print_metrics(name, m):
    print(f"--- {name} ---")
    print(f"Distance to plant: {m['distance_mm']:.1f} mm")
    print(f"Scale:             {m['scale_px_per_mm']:.4f} px/mm")
    print(
        f"Mean flow:         {m['mean_px']:.4f} px/frame = {m['mean_mm']:.4f} mm/frame"
    )
    print(f"RMS flow:          {m['rms_px']:.4f} px/frame = {m['rms_mm']:.4f} mm/frame")


def percent_change(value, baseline):
    return (value - baseline) / baseline * 100


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    baseline = sys.argv[2] if len(sys.argv) > 2 else BASELINE_VIDEO_PATH

    drone = analyze(video, SHOW)
    print_metrics(video, drone)

    if baseline is None:
        print("No baseline video given; skipping change vs baseline.")
    else:
        base = analyze(baseline)
        print_metrics(baseline, base)
        print("--- Change relative to baseline ---")
        print(f"Mean: {percent_change(drone['mean_mm'], base['mean_mm']):+.1f} %")
        print(f"RMS:  {percent_change(drone['rms_mm'], base['rms_mm']):+.1f} %")
