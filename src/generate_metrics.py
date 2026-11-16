import time

import pandas as pd

from helpers.clip import clip_fps, first_frame, frame_count
from helpers.configs import METRICS_DIR
from helpers.masks import check_masks, mask_path
from helpers.metrics import frame_metrics, mm_per_px, summarize
from helpers.sam import load_prompts
from helpers.utils import iter_clips

FRAMES_DIR = METRICS_DIR / "frames"


def condition(run):
    if run["run_info"]["run_type"] == "HoverRun":
        return f"hover_{run['height_cm']}"
    return "wind_ref" if run["wind"] else "ref"


def load_frames(run, cam, masks, fps):
    """Per-frame table, cached as JSON. Recomputed when the masks are newer than the cache."""
    path = FRAMES_DIR / f"{run['clip_id']}_{cam}.json"
    if path.exists() and path.stat().st_mtime > masks.stat().st_mtime:
        return pd.read_json(path, orient="records")
    t0 = time.time()
    df = frame_metrics(run[cam], masks, fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records")
    print(f"{path.name}: {len(df)} frames in {time.time() - t0:.0f} s")
    return df


def main():
    prompts = load_prompts()
    rows = []
    for plant, run, cam in iter_clips():
        clip, clip_id = run[cam], run["clip_id"]
        masks = mask_path(clip_id, cam)
        shape = first_frame(clip).shape[:2]
        prompt = prompts.get(clip_id, {}).get(cam)
        ok, reason, _ = check_masks(masks, shape, frame_count(clip), prompt)
        if not ok:
            print(f"skipped {clip_id} {cam}: masks {reason}")
            continue
        fps = clip_fps(run, cam)
        df = load_frames(run, cam, masks, fps)
        mm = mm_per_px(run, cam)
        rows.append(
            {
                "plant_id": plant["plant_id"],
                "plant": plant["plant_name"],
                "clip_id": clip_id,
                "cam": cam,
                "condition": condition(run),
                "height_cm": run["height_cm"],
                "repeat": run["repeat"],
                "fps": fps,
                "mm_per_px": mm,
                "drone_h_cm": float(run["telemetry"]["h"].median()),
                **summarize(df, fps, mm),
            }
        )
    out = METRICS_DIR / "clips.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out, orient="records", indent=1)
    print(f"saved {len(rows)} clips to {out}")


if __name__ == "__main__":
    main()
