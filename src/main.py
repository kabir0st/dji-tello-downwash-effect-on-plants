from helpers.utils import load_dataset

INFO_KEYS = ["clip_id", "plant_id", "plant_species", "run_type", "height_cm", "repeat"]
QA_HEAT_MAX_PX = 5.0


def main():
    plants = load_dataset("data")
    print(plants[0]["runs"][0]["telemetry"].head())
    print(plants[0]["baseline"][0]["phone"])


if __name__ == "__main__":
    main()
