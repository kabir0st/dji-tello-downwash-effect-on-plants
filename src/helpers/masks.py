import numpy as np

from helpers.configs import (
    MASK_DIR,
    SAM_MODEL,
)


def mask_path(clip_id, cam):
    return MASK_DIR / SAM_MODEL.split("/")[-1] / f"{clip_id}_{cam}.npz"


def pack_mask(mask):
    return np.packbits(mask, axis=-1)


def save_masks(masks, shape, path, complete):
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = sorted(masks)
    np.savez_compressed(
        path,
        frame_idx=np.array(idx),
        shape=np.array(shape),
        complete=complete,
        packed=np.stack([masks[i] for i in idx]),
    )


def masks_done(path):
    return path.exists() and bool(np.load(path)["complete"])


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
