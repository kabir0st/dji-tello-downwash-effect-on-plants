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

# SAM_MODEL = "facebook/sam2.1-hiera-large"
SAM_MODEL = "facebook/sam3"

# HSV plant masking, OpenCV HSV (H 0-180, S/V 0-255)
PLANT_HSV_LO = (20, 50, 40)
PLANT_HSV_HI = (90, 255, 255)
TARP_HSV_LO = (90, 40, 40)
TARP_HSV_HI = (130, 255, 255)
MIN_BLOB_PX = 400
