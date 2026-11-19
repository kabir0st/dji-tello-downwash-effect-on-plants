import sys

SHARD, N_SHARDS = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, 1)

import os
from itertools import chain

os.environ["QT_QPA_PLATFORM"] = "xcb"

import time

import cv2
import numpy as np

from helpers.clip import first_frame, frame_count
from helpers.configs import OUT_DIR
from helpers.masks import check_masks, load_masks, mask_path, pack_mask, save_masks
from helpers.sam import SamClicker, load_prompts, track
from helpers.utils import load_dataset

CLIP_KEYS = ["phone", "video"]
CLIPS_DIR = OUT_DIR / "masked_clips"
SHOW = False


def overlay_path(plant_name, clip_type, clip_id):
    return CLIPS_DIR / plant_name / f"{clip_type}_{clip_id}.mp4"


def overlay_ok(path, n_frames):
    if not path.exists():
        return False
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n == n_frames


def open_writer(clip, out_video, size):
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_video), fourcc, fps, size)


def draw_mask(frame, mask):
    vis = frame.copy()
    vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    cs, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, cs, -1, (255, 255, 255), 2)
    return vis


def render_overlay(clip, mask_file, out_video, total):
    h, w = first_frame(clip).shape[:2]
    writer = open_writer(clip, out_video, (w, h))
    get = load_masks(mask_file)
    cap = cv2.VideoCapture(clip)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            m = get(i)
            writer.write(frame if m is None else draw_mask(frame, m))
            i += 1
            print(f"\rrebuilding overlay {out_video.name} | {i}/{total}", end="", flush=True)
    finally:
        cap.release()
        writer.release()
    print()
    print("saved", out_video)


def track_clip(clip, clip_id, clip_type, plant_name, prompt, sam):
    out_video = overlay_path(plant_name, clip_type, clip_id)
    total = frame_count(clip)
    h, w = first_frame(clip).shape[:2]
    s = min(1280 / w, 720 / h, 1.0)
    writer = open_writer(clip, out_video, (w, h))

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
    writer.release()
    if not masks:
        out_video.unlink()
        print("could not read any frames:", clip)
        return None
    path = mask_path(clip_id, clip_type)
    save_masks(masks, (h, w), path, complete=True, prompt=prompt)
    print("saved", path, "and", out_video)
    return None


def main():
    plants = load_dataset("./../data")
    if not plants:
        raise ValueError("Cannot find any data.")
    prompts = load_prompts()
    sam = SamClicker()
    k = 0
    for plant in plants:
        for clips in chain(plant["baseline"], plant["runs"]):
            k += 1
            if k % N_SHARDS != SHARD:
                continue
            for clip_type in CLIP_KEYS:
                clip_id = clips["clip_id"]
                clip = clips[clip_type]
                if not clip:
                    continue
                if clip_type not in prompts.get(clip_id, {}):
                    print(
                        "no annotation yet, run annotate.py first:", clip_id, clip_type
                    )
                    continue
                prompt = prompts[clip_id][clip_type]
                total = frame_count(clip)
                h, w = first_frame(clip).shape[:2]
                path = mask_path(clip_id, clip_type)

                ok, reason, n_empty = check_masks(path, (h, w), total)
                if ok:
                    if n_empty:
                        print(
                            f"warning: {clip_id} {clip_type}: {n_empty}/{total} "
                            "frames have an empty mask, consider re-annotating"
                        )
                    out_video = overlay_path(plant["plant_name"], clip_type, clip_id)
                    if not overlay_ok(out_video, total):
                        render_overlay(clip, path, out_video, total)
                    continue
                if reason != "missing":
                    print("invalid mask, regenerating:", clip_id, clip_type, "-", reason)
                res = track_clip(
                    clip, clip_id, clip_type, plant["plant_name"], prompt, sam
                )
                if res == "quit":
                    cv2.destroyAllWindows()
                    return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
