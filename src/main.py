import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import time

import cv2
import numpy as np

from helpers.clip import first_frame
from helpers.sam import SamClicker, click_prompt, track
from helpers.utils import load_dataset

INFO_KEYS = ["clip_id", "plant_id", "plant_species", "run_type", "height_cm", "repeat"]
QA_HEAT_MAX_PX = 5.0

DATA_PATH = "./../data"
CLIP_KEYS = ["video", "phone"]


def eta(t0, done, total):
    if done == 0:
        return "--:--:--"
    return time.strftime(
        "%H:%M:%S", time.gmtime((time.time() - t0) / done * (total - done))
    )


def annotate(clip, clicker):
    prompt = click_prompt(first_frame(clip), clicker, clip)
    cv2.destroyAllWindows()
    print("prompt:", prompt)
    return prompt


def process_clip(clip, clicker):
    frame = first_frame(clip)
    s = min(1280 / frame.shape[1], 720 / frame.shape[0], 1.0)
    prompt = click_prompt(frame, clicker, clip)
    cv2.destroyAllWindows()
    if prompt is None or prompt == "quit":
        return prompt

    total = int(cv2.VideoCapture(clip).get(cv2.CAP_PROP_FRAME_COUNT))
    win, t0, first = "SAM track", time.time(), None

    for i, frm, m in track(clip, prompt, clicker):
        # area from the first frame / no. of pixels plant mask gives
        first = first or max(int(m.sum()), 1)
        vis = frm.copy()
        vis[m] = (0.5 * vis[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        cs, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, cs, -1, (255, 255, 255), 2)
        vis = cv2.resize(vis, None, fx=s, fy=s)
        txt = (
            f"frame {i + 1}/{total}  {(time.time() - t0) / (i + 1):.1f} s/frame  "
            f"ETA {eta(t0, i + 1, total)}  area {100 * m.sum() / first:.0f}%"
        )
        cv2.putText(
            vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.imshow(win, vis)
        key = cv2.waitKey(10) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    print("prompt:", prompt)
    return None


def process_plant(plant, clicker):
    for baseline in plant["baseline"]:
        has_wind = "_wind" in baseline["run_info"]["plant_id"]
        print(
            f"(proc) plant: { plant['plant_name'] } | wind: {has_wind} | height: {baseline['height_cm']} |  "
        )
        for key in CLIP_KEYS:
            if baseline[key]:
                print("Path : ", baseline[key])
                if process_clip(baseline[key], clicker) == "quit":
                    return "quit"
    return None


def main():
    clicker = SamClicker()
    plants = load_dataset("./../data")
    if not plants:
        raise ValueError("Cannot find any data.")
    for plant in plants:
        process_plant(plant, clicker)
        break


if __name__ == "__main__":
    main()
