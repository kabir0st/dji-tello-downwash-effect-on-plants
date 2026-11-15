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
