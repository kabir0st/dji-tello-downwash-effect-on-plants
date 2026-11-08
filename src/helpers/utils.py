from pathlib import Path


def find_clips(root):
    return sorted(p.parent for p in Path(root).glob("*/*/h*_r*/run_info.json"))
