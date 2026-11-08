from helpers.utils import find_clips

INFO_KEYS = ["clip_id", "plant_id", "plant_species", "run_type", "height_cm", "repeat"]
QA_HEAT_MAX_PX = 5.0


def main():
    drone_clips = find_clips("data/drone")
    print(drone_clips)


if __name__ == "__main__":
    main()
