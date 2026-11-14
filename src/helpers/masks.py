import numpy as np

from helpers.configs import (
    MASK_DIR,
    SAM_MODEL,
)


def mask_path(clip_id, cam):
    return MASK_DIR / SAM_MODEL.split("/")[-1] / f"{clip_id}_{cam}.npz"


def pack_mask(mask):
    return np.packbits(mask, axis=-1)
