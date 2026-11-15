import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
from itertools import chain

import cv2
import numpy as np

from helpers.clip import first_frame, frame_count
from helpers.configs import OUT_DIR
from helpers.masks import mask_path, masks_done, pack_mask, save_masks
from helpers.sam import SamClicker, load_prompts, track
from helpers.utils import load_dataset

CLIP_KEYS = ["phone", "video"]
CLIPS_DIR = OUT_DIR / "masked_clips"
SHOW = True


def draw_mask(frame, mask):
    vis = frame.copy()
    vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    cs, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, cs, -1, (255, 255, 255), 2)
    return vis


def track_clip(clip, clip_id, clip_type, plant_name, prompt, sam):
    out_video = CLIPS_DIR / plant_name / f"{clip_type}_{clip_id}.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    total = frame_count(clip)
    h, w = first_frame(clip).shape[:2]
    s = min(1280 / w, 720 / h, 1.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    masks, first, t0 = {}, None, time.time()
    for i, frame, m in track(clip, prompt, sam):
        masks[i] = pack_mask(m)
        vis = draw_mask(frame, m)
        writer.write(vis)

        first = first or max(int(m.sum()), 1)
        per_frame = (time.time() - t0) / (i + 1)
        eta = time.strftime("%H:%M:%S", time.gmtime(per_frame * (total - i - 1)))
        text = (
            f"{clip_id} | {clip_type} | {i + 1}/{total} | {per_frame:.1f} s/frame | "
            f"ETA {eta} | area {100 * m.sum() / first:.0f}%"
        )
        print("\r" + text, end="", flush=True)

        if SHOW:
            small = cv2.resize(vis, None, fx=s, fy=s)
            cv2.putText(
                small, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.imshow("SAM track", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                writer.release()
                out_video.unlink()
                print("\nstopped, this clip was not saved")
                return "quit"
    print()
    writer.release()

    if not masks:
        out_video.unlink()
        print("could not read any frames:", clip)
        return None
    path = mask_path(clip_id, clip_type)
    save_masks(masks, (h, w), path, complete=True)
    print("saved", path, "and", out_video)
    return None


def main():
    plants = load_dataset("./../data")
    if not plants:
        raise ValueError("Cannot find any data.")
    prompts = load_prompts()
    sam = SamClicker()
    for plant in plants:
        for clips in chain(plant["baseline"], plant["runs"]):
            for clip_type in CLIP_KEYS:
                clip_id = clips["clip_id"]
                if not clips[clip_type]:
                    continue
                if clip_type not in prompts.get(clip_id, {}):
                    print(
                        "no annotation yet, run annotate.py first:", clip_id, clip_type
                    )
                    continue
                if masks_done(mask_path(clip_id, clip_type)):
                    continue
                prompt = prompts[clip_id][clip_type]
                res = track_clip(
                    clips[clip_type],
                    clip_id,
                    clip_type,
                    plant["plant_name"],
                    prompt,
                    sam,
                )
                if res == "quit":
                    cv2.destroyAllWindows()
                    return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
