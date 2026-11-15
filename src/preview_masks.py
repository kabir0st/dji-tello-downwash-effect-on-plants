import argparse
import json
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

from helpers.clip import first_frame
from helpers.configs import DATA_DIR, ROOT
from helpers.sam import FONT, SamClicker
from helpers.utils import load_dataset

CLIP_KEYS = ["phone", "video"]
PROMPTS = ROOT / "logs" / "sam_prompts.json"
OUT = ROOT / "logs" / "previews"


def draw_mask(frame, mask):
    vis = frame.copy()
    vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    cs, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, cs, -1, (255, 255, 255), 2)
    return vis


def draw_points(vis, points, labels):
    s = vis.shape[1] / 1000
    r = max(5, int(7 * s))
    for k, ((x, y), l) in enumerate(zip(points, labels)):
        p = (int(x), int(y))
        cv2.circle(vis, p, r + 2, (0, 0, 0), -1)
        cv2.circle(vis, p, r, (0, 255, 0) if l else (0, 0, 255), -1)
        cv2.putText(vis, str(k), (p[0] + r, p[1] - r), FONT, 0.6 * s, (255, 255, 255), max(1, int(2 * s)))


def main():
    ap = argparse.ArgumentParser(description="Draw frame-0 SAM masks for prompts.")
    ap.add_argument("--prompts", type=Path, default=PROMPTS, help="prompts json")
    ap.add_argument("--only", default="", help="only clip_ids starting with this, e.g. p10_")
    args = ap.parse_args()

    prompts = json.loads(args.prompts.read_text())
    sam = SamClicker()
    OUT.mkdir(parents=True, exist_ok=True)
    for plant in load_dataset(DATA_DIR):
        for clip in chain(plant["baseline"], plant["runs"]):
            clip_id = clip["clip_id"]
            if not clip_id.startswith(args.only):
                continue
            for cam in CLIP_KEYS:
                prompt = prompts.get(clip_id, {}).get(cam)
                if not clip[cam] or not prompt:
                    continue
                frame = first_frame(clip[cam])
                pts_only = frame.copy()
                draw_points(pts_only, prompt["points"], prompt["labels"])
                cv2.imwrite(str(OUT / f"{clip_id}__{cam}.jpg"), pts_only)
                sam.set_image(frame)
                mask = sam.predict(prompt["points"], prompt["labels"])
                pct = 100 * mask.mean()
                vis = draw_mask(frame, mask)
                draw_points(vis, prompt["points"], prompt["labels"])
                name = f"{clip_id}__{cam}"
                s = vis.shape[1] / 1000
                text = f"{name} | mask {pct:.1f}% of frame"
                th = max(1, int(2 * s))
                (tw, tht), base = cv2.getTextSize(text, FONT, 0.9 * s, th)
                cv2.rectangle(vis, (0, 0), (tw + 20, tht + base + 20), (0, 0, 0), -1)
                cv2.putText(vis, text, (10, tht + 10), FONT, 0.9 * s, (255, 255, 255), th)
                cv2.imwrite(str(OUT / f"{name}__mask.jpg"), vis)
                print(f"{name:32s} mask {pct:5.1f}%")


if __name__ == "__main__":
    main()
