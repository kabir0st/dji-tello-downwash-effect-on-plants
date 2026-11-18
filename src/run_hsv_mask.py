import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np

from helpers.hsv_mask import background_mask, border_mask, plant_mask_hsv
from helpers.utils import load_dataset

INFO_KEYS = ["clip_id", "plant_id", "plant_species", "run_type", "height_cm", "repeat"]
QA_HEAT_MAX_PX = 5.0

CLIP_KEYS = ["video", "phone"]


def process_clip(clip):
    cap = cv2.VideoCapture(clip)

    w = 1280
    h = 720

    if not cap.isOpened():
        raise OSError(f"Cannot open video: {clip}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        plant = plant_mask_hsv(frame)
        bg = background_mask(plant) & border_mask(frame.shape)

        overlay = frame.copy()
        overlay[plant] = (0, 255, 0)
        overlay[bg] = (255, 0, 0)
        vis = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        combo = np.hstack((frame, vis))
        combo = cv2.resize(combo, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        cv2.namedWindow("Original | Plant Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original | Plant Mask", w, h)
        cv2.imshow("Original | Plant Mask", combo)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def process_plant(plant):
    for baseline in plant["baseline"]:
        has_wind = "_wind" in baseline["run_info"]["plant_id"]
        print(
            f"(proc) plant: { plant['plant_name'] } | wind: {has_wind} | height: {baseline['height_cm']} |  "
        )
        for key in CLIP_KEYS:
            if baseline[key]:
                print("Path : ", baseline[key])
                process_clip(baseline[key])


def main():
    plants = load_dataset()
    if not plants:
        raise ValueError("Cannot find any data.")
    for plant in plants:
        process_plant(plant)
        break
    # pprint(plants)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exp:
        print("(main)(err) : ", exp)
