import json
from pathlib import Path

import pandas as pd

from helpers.configs import DATA_DIR


def _media(path: Path) -> str | None:
    return str(path) if path.is_file() else None


def load_run(run_dir: Path) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text())
    return {
        "clip_id": info["clip_id"],
        "height_cm": info["height_cm"],
        "side_wind": run_dir.name.endswith("_r2"),
        "repeat": info["repeat"],
        "run_info": info,
        "run_dir": run_dir,
        "telemetry": pd.read_csv(run_dir / "telemetry.csv"),
        "video": _media(run_dir / "video.mp4"),
        "phone": _media(run_dir / "phone.mp4"),
    }


def _load_runs(folder: Path) -> list[dict]:
    if not folder.is_dir():
        return []
    runs = [
        load_run(d) for d in folder.glob("h*_r*") if (d / "run_info.json").is_file()
    ]
    return sorted(runs, key=lambda r: r["run_dir"].name)


def load_dataset(root: str | Path = DATA_DIR) -> list[dict]:
    plants = []
    for plant_dir in Path(root).glob("p*_*"):
        if not plant_dir.is_dir():
            continue
        baseline = _load_runs(plant_dir / "ReferenceRun")
        runs = _load_runs(plant_dir / "HoverRun")
        if not baseline + runs:
            continue
        plant_id, _, plant_name = plant_dir.name.partition("_")
        plants.append(
            {
                "plant_id": plant_id,
                "plant_name": plant_name,
                "baseline": baseline,
                "runs": runs,
            }
        )
    return sorted(plants, key=lambda p: int(p["plant_id"][1:]))


def iter_clips(cams=("phone", "video")):
    for plant in load_dataset():
        for run in plant["baseline"] + plant["runs"]:
            for cam in cams:
                if run[cam]:
                    yield plant, run, cam
