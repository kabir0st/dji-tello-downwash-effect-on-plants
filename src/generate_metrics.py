

from collections import deque

import cv2
import numpy as np
import pandas as pd

from helpers.clip import clip_fps, first_frame, frame_count
from helpers.configs import METRICS_DIR
from helpers.masks import check_masks, iter_masks, mask_path
from helpers.utils import iter_clips

FRAMES_DIR = METRICS_DIR / "frames"

def condition(run):
    pass

LAG_S = 1
BG_ERODE = 9
BG_STEP = 8
RANSAC_PX = 1.0
MIN_ZOOM_PCT = 1.0
TAG_FRAMES = 30

# tello 960x720 : 4:3
# phone 1920x1080 : 16:9
FLOW_MAX_SIDE = 960

def frame_metrics(clip, fps, masks):
    # converting lag per sec to frames
    lag = max(1, round(LAG_S * fps))
    history = deque(maxlen=lag)
    dis = cv2.DISOpticalFlow(
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    )
    kernel = np.ones(
        (BG_ERODE,BG_ERODE), np.uint8
    )
    print(f'lag : {lag}, history : {history}')
    cap = cv2.VideoCapture(
        clip
    )
    rows, prev = [],None
    to_first , scale = np.eye(3), 1.0
    try:
        for i, m in iter_masks(masks):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = m.shape
            s = min(1.0, FLOW_MAX_SIDE / max(h, w))
            gray = cv2.cvtColor()
            
    finally:
        cap.release()



def load_frames(run, cam, fps, masks):
    path = FRAMES_DIR / f"{run['clip_id']}_{cam}.json"
    
    if path.exists():
        print(f'{path} already there.')
        return pd.read_json(path, orient="records")
    df = frame_metrics(run[cam], fps, masks)
    
    # path.parent.mkdir(parents=True, exist_ok=True)
    # df.to_json(path, orient="records")
    # print(f"{path.name}: {len(df)} frames in {time.time() - t0:.0f} s")
    # return df


def main():
    rows = []
    for plant, run, cam in iter_clips():
        clip, clip_id = run[cam], run['clip_id']
        masks = mask_path(clip_id, cam)
        shape = first_frame(clip).shape[:2]
        ok, reason, _ = check_masks(
            masks, shape, frame_count(clip)
        )
        if not ok:
            print(f"skipped {clip_id} {cam}: masks {reason}")
            continue
        fps = clip_fps(run, cam)
        df = load_frames(run, cam, fps, masks)
        # print(df.head())
        break


if __name__ == "__main__":
    main()