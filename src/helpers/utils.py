import json
from pathlib import Path

import pandas as pd


def _media(path: Path) -> str | None:
    return str(path) if path.is_file() else None


def load_run(run_dir: Path) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text())
    return {
        "clip_id": info["clip_id"],
        "height_cm": info["height_cm"],
        "repeat": info["repeat"],
        "run_info": info,
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
    return sorted(runs, key=lambda r: (r["height_cm"], r["repeat"]))


def _shared_info(runs: list[dict]) -> dict:
    first = runs[0]["run_info"]
    return {
        k: v for k, v in first.items() if all(r["run_info"].get(k) == v for r in runs)
    }


def load_dataset(root: str | Path = "data") -> list[dict]:
    plants = []
    for plant_dir in Path(root).glob("p*_*"):
        if not plant_dir.is_dir():
            continue
        baseline = _load_runs(plant_dir / "ReferenceRun")
        runs = _load_runs(plant_dir / "HoverRun")
        all_runs = baseline + runs
        if not all_runs:
            continue
        plant_id, _, plant_name = plant_dir.name.partition("_")
        plants.append(
            {
                "plant_id": plant_id,
                "plant_name": plant_name,
                "meta_data": _shared_info(all_runs),
                "baseline": baseline,
                "runs": runs,
            }
        )
    return sorted(plants, key=lambda p: int(p["plant_id"][1:]))
