from itertools import chain

import cv2

from helpers.clip import first_frame
from helpers.masks import mask_path
from helpers.sam import SamClicker, click_prompt, load_prompts, save_prompt
from helpers.utils import load_dataset

CLIP_KEYS = ["phone", "video"]


def annotate_clip(clip, clip_id, clip_type, sam):
    title = f"{clip_id} | {clip_type}"
    res = click_prompt(first_frame(clip), sam, title)
    if res == "quit":
        return "quit"
    if res is None:
        print("skipped", title)
        return None
    pts, lbl = res
    save_prompt(clip_id, clip_type, {"points": pts, "labels": lbl})
    old = mask_path(clip_id, clip_type)
    if old.exists():
        old.unlink()
    print("saved", title)
    return None


def main():
    plants = load_dataset("./../data")
    if not plants:
        raise ValueError("Cannot find any data.")
    done = load_prompts()
    sam = SamClicker()
    for plant in plants:
        for clips in chain(plant["baseline"], plant["runs"]):
            for clip_type in CLIP_KEYS:
                if not clips[clip_type]:
                    continue
                if clip_type in done.get(clips["clip_id"], {}):
                    continue
                if (
                    annotate_clip(clips[clip_type], clips["clip_id"], clip_type, sam)
                    == "quit"
                ):
                    cv2.destroyAllWindows()
                    return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
