import cv2


def clip_fps(run, cam):
    if cam == "video" and run["run_info"].get("measured_fps"):
        return float(run["run_info"]["measured_fps"])
    cap = cv2.VideoCapture(run[cam])
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)


def frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"Cannot read {path}")
    return frame


def iter_pairs(path, step=1):
    """Yield (i, frame_i, frame_{i+step}) for consecutive sampled frames."""
    cap = cv2.VideoCapture(path)
    prev, i = None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                if prev is not None:
                    yield prev[0], prev[1], frame
                prev = (i, frame)
            i += 1
    finally:
        cap.release()
