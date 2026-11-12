import cv2


def clip_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)


def frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def iter_paris(path, step=1):
    cap = cv2.VideoCapture(path)
    prev, i = None, 0
    start = 0
    stop = None
    try:
        while stop is None or i < stop:
            if i < start:
                ok = cap.grab()
            else:
                ok, frame = cap.read()
                if ok and (i - start) % step == 0:
                    if prev is not None:
                        yield prev[0], prev[1], frame
                    prev = (i, frame)
            if not ok:
                break
            i += 1
    finally:
        cap.release()

