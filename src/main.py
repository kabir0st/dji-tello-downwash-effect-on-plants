
import cv2

from helpers.aruco import make_detector
from helpers.utils import load_dataset

INFO_KEYS = ["clip_id", "plant_id", "plant_species", "run_type", "height_cm", "repeat"]
QA_HEAT_MAX_PX = 5.0

DATA_PATH = './../data'
CLIP_KEYS = [ 'video', 'phone']

def measure_frames(clip, show=False):
    detector = make_detector()
    rows, prev, grid, qa = [], None, None, None


def iter_frames(clip_path):
    cap = cv2.VideoCapture(
        clip_path
    )
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, i
            i += 1
    finally:
        cap.release()

def process_plant(plant):
    for baseline in plant['baseline']:
        has_wind = "_wind" in baseline['run_info']['plant_id']
        print(f"(proc) plant: { plant['plant_name'] } | wind: {has_wind} | height: {baseline['height_cm']} |  ")

        for clip_key in CLIP_KEYS:
            if baseline[clip_key]:
                print(plant['run_info'])
                # Clip(
                #     plant
                # )
        #         for i, frame in iter_frames(baseline[clip_key]):
        #             cv2.imshow("frame", frame)
        #             if cv2.waitKey(delay) & 0xFF == ord('q'):
        #                 break

        #         cap.release()
        #         cv2.destroyAllWindows()
        # # for i, frame in iter_frames()


def main():
    plants = load_dataset("./../data")
    if not plants:
        raise ValueError("Cannot find any data.")
    for plant in plants:
        process_plant(plant)
        break
    

if __name__ == "__main__":
    try:
        main()
    except ValueError as exp:
        print("(main)(err) : ",exp)