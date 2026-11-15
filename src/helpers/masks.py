import json
import zipfile

import numpy as np

from helpers.configs import (
    MASK_DIR,
    SAM_MODEL,
)

MASK_KEYS = ("frame_idx", "shape", "complete", "packed")


def mask_path(clip_id, cam):
    return MASK_DIR / SAM_MODEL.split("/")[-1] / f"{clip_id}_{cam}.npz"


def pack_mask(mask):
    return np.packbits(mask, axis=-1)


def prompt_key(prompt):
    return json.dumps(prompt, sort_keys=True)


def save_masks(masks, shape, path, complete, prompt=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = sorted(masks)
    np.savez_compressed(
        path,
        frame_idx=np.array(idx),
        shape=np.array(shape),
        complete=complete,
        packed=np.stack([masks[i] for i in idx]),
        prompt="" if prompt is None else prompt_key(prompt),
    )


def check_masks(path, shape, n_frames, prompt=None):
    """Return (ok, reason, n_empty) for the mask file at `path`.

    ok is True only if the file is a complete mask for a clip of `shape` (h, w)
    with `n_frames` frames, made from `prompt` (when the file stores one).
    n_empty counts frames whose mask has no pixels; it does not affect ok.
    """
    if not path.exists():
        return False, "missing", 0
    h, w = shape
    try:
        with np.load(path) as z:
            if any(k not in z for k in MASK_KEYS):
                return False, "missing keys", 0
            if not bool(z["complete"]):
                return False, "incomplete", 0
            saved_shape = tuple(int(v) for v in z["shape"])
            if saved_shape != (h, w):
                return False, f"shape {saved_shape} != {(h, w)}", 0
            idx = z["frame_idx"]
            if len(idx) != n_frames or not np.array_equal(idx, np.arange(n_frames)):
                return False, f"frame count {len(idx)} != {n_frames}", 0
            packed = z["packed"]
            if packed.shape != (n_frames, h, (w + 7) // 8):
                return False, f"packed array shape {packed.shape}", 0
            saved_prompt = str(z["prompt"]) if "prompt" in z else ""
            if prompt is not None and saved_prompt and saved_prompt != prompt_key(prompt):
                return False, "prompt changed", 0
            n_empty = int((~packed.reshape(n_frames, -1).any(axis=1)).sum())
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
        return False, f"unreadable ({type(e).__name__})", 0
    return True, "ok", n_empty


def load_masks(path):
    z = np.load(path)
    packed, w = z["packed"], int(z["shape"][1])
    row = {int(f): j for j, f in enumerate(z["frame_idx"])}

    def get(i):
        j = row.get(i)
        if j is None:
            return None
        return np.unpackbits(packed[j], axis=-1, count=w).astype(bool)

    return get
