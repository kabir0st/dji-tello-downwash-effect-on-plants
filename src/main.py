import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import time

import cv2
import numpy as np

from helpers.sam import SamClicker, track
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
    cap = cv2.VideoCapture(clip)

    ok, frame = cap.read()
    try:
        if not ok:
            raise OSError("Cannot read file on ", clip)
        clicker.set_image(frame)
        pts, lbl = [], []
        state = {"mask": None}
        s = min(1280 / frame.shape[1], 720 / frame.shape[0], 1.0)

        def redo():
            state["mask"] = clicker.predict(pts, lbl) if 1 in lbl else None

        def on_mouse(event, x, y, *_):
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                pts.append([x / s, y / s])  # window pixels -> frame pixels
                lbl.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
                redo()

        cv2.namedWindow("SAM2 Annotate", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("SAM2 Annotate", on_mouse)
        while True:  # 3. the display loop
            vis = frame.copy()
            if state["mask"] is not None:
                m = state["mask"]
                vis[m] = (0.5 * vis[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            for (x, y), l in zip(pts, lbl):
                cv2.circle(
                    vis, (int(x), int(y)), 8, (0, 255, 0) if l else (0, 0, 255), -1
                )
            cv2.imshow("SAM2 Annotate", cv2.resize(vis, None, fx=s, fy=s))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("z") and pts:
                pts.pop()
                lbl.pop()
                redo()
            elif key == ord("c"):
                pts.clear()
                lbl.clear()
                redo()
            elif key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        print("points:", pts, "labels:", lbl)

    finally:
        pass


def process_clip(clip, clicker):
    cap = cv2.VideoCapture(clip)
    ok, frame = cap.read()  # 1. one frame
    cap.release()
    if not ok:
        raise OSError(f"Cannot read {clip}")

    clicker.set_image(frame)
    pts, lbl = [], []
    state = {"mask": None}
    s = min(1280 / frame.shape[1], 720 / frame.shape[0], 1.0)

    def redo():
        state["mask"] = clicker.predict(pts, lbl) if 1 in lbl else None

    def on_mouse(event, x, y, *_):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            pts.append([x / s, y / s])
            lbl.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
            redo()

    win = "SAM Clicker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        if state["mask"] is not None:
            m = state["mask"]
            vis[m] = (0.5 * vis[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        for (x, y), l in zip(pts, lbl):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0) if l else (0, 0, 255), -1)
        cv2.imshow(win, cv2.resize(vis, None, fx=s, fy=s))
        key = cv2.waitKey(10) & 0xFF
        if key == ord("z") and pts:
            pts.pop()
            lbl.pop()
            redo()
        elif key == ord("c"):
            pts.clear()
            lbl.clear()
            redo()
        elif key in (ord("q"), 27):
            break
    total = int(cv2.VideoCapture(clip).get(cv2.CAP_PROP_FRAME_COUNT))
    win, t0, first = "SAM track", time.time(), None
    print(pts, lbl)
    for i, frm, m in track(clip, pts, lbl, clicker):
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
    print("points:", pts, "labels:", lbl)


def process_plant(plant, clicker):
    for baseline in plant["baseline"]:
        has_wind = "_wind" in baseline["run_info"]["plant_id"]
        print(
            f"(proc) plant: { plant['plant_name'] } | wind: {has_wind} | height: {baseline['height_cm']} |  "
        )
        for key in CLIP_KEYS:
            if baseline[key]:
                print("Path : ", baseline[key])
                process_clip(baseline[key], clicker)


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
