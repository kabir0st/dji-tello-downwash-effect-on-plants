from helpers.clip import clip_fps, frame_count
from helpers.utils import load_dataset

INFO_KEYS = ["clip_id", "plant_id", "plant_species", "run_type", "height_cm", "repeat"]
QA_HEAT_MAX_PX = 5.0

DATA_PATH = "./../data"
CLIP_KEYS = ["video", "phone"]


def process_plant(plant):
    for baseline in plant["baseline"]:
        has_wind = "_wind" in baseline["run_info"]["plant_id"]
        print(
            f"(proc) plant: { plant['plant_name'] } | wind: {has_wind} | height: {baseline['height_cm']} |  "
        )
        for key in CLIP_KEYS:
            if baseline[key]:
                print("Path : ", baseline[key])
                
        break


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
        print("(main)(err) : ", exp)
