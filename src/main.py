
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np

from helpers.mask import plant_mask_hsv
from helpers.utils import load_dataset

INFO_KEYS = ["clip_id", "plant_id", "plant_species", "run_type", "height_cm", "repeat"]
QA_HEAT_MAX_PX = 5.0

DATA_PATH = "./../data"
CLIP_KEYS = ["video", "phone"]



def process_clip(clip):
    cap = cv2.VideoCapture(clip)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

    if not cap.isOpened():
        raise OSError(f"Cannot open video: {clip}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask_bool = plant_mask_hsv(frame)
        mask_u8   = mask_bool.astype(np.uint8) * 255
        masked = cv2.bitwise_and(frame, frame, mask=mask_u8)
        combo = np.hstack((frame, masked))
        combo_resized = cv2.resize(combo, (w, h), interpolation=cv2.INTER_AREA)

        cv2.namedWindow("Original | Plant Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original | Plant Mask", w, h)
        cv2.imshow("Original | Plant Mask", combo_resized)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
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
                break
        break



def main():
    plants = load_dataset("./../data")
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
