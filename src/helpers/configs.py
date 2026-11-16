from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ANNOT_DIR = ROOT / "annotations"
OUT_DIR = ROOT / "outputs"
MASK_DIR = OUT_DIR / "masks"
RUNS_DIR = OUT_DIR / "runs"
METRICS_DIR = OUT_DIR / "metrics"

# need to calibrate this one
TELLO_FOCAL_PX = 840.0

# SAM_MODEL = "facebook/sam2.1-hiera-tiny"
SAM_MODEL = "facebook/sam2.1-hiera-large"
