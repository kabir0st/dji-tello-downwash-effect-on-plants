# Quantifying Rotor-Downwash-Induced Canopy Motion and Its Effect on Plant-Health Indices

A methods guide for this dataset: what the data supports, which motion-quantification
method to use (and why, with citations), and working code to implement the full pipeline.

---

## 1. TL;DR — the recommended method

**Use masked, ego-motion-compensated dense optical flow (OpenCV DIS) as the primary
motion metric, convert it to physical units (mm/s), and couple it with per-frame
RGB vegetation-index stability metrics on the same plant mask.**

- The **phone camera (static, 60 fps side view)** is the *physical truth* instrument:
  it measures how much the downwash actually moves the canopy.
- The **Tello nadir video** is the *sensor view* instrument: it measures what a
  mapping/VI camera would actually experience (leaf motion + its own platform wobble),
  which is exactly the thing that corrupts NDVI/NIR measurements.
- Report both. The paper's argument becomes: *downwash moves leaves by X mm/s
  (phone), and under that motion the per-frame vegetation index observed from the
  drone varies Y× more than over an undisturbed canopy (Tello).*

A pilot run on this dataset (p1 perilla, first 150 frames, this repo's venv) already
separates the conditions cleanly:

| Clip (Tello nadir) | Plant-mask mean flow | Background (ego-motion) | Net plant flow |
|---|---|---|---|
| HoverRun h030_r1 | 3.46 px/fr | 1.20 px/fr | **2.26 px/fr** |
| HoverRun h050_r1 | 3.43 px/fr | 1.22 px/fr | **2.20 px/fr** |
| ReferenceRun h030_r1 | 0.13 px/fr | 0.02 px/fr | **0.11 px/fr** |

| Clip (phone, static, half-res) | Plant mean | Background | Net |
|---|---|---|---|
| HoverRun h030_r1 | 0.65 px/fr | 0.15 px/fr | **0.51 px/fr** |
| ReferenceRun h030_r1 | 0.15 px/fr | 0.15 px/fr | **0.00 px/fr** |

And the downstream vegetation-index effect is already visible with the exact code
from §5 (p1, first 60 frames, Tello view, vector ego-compensation):

| Metric | HoverRun h030_r1 | ReferenceRun h030_r1 | Ratio |
|---|---|---|---|
| Net plant flow | 3.19 px/fr | 0.09 px/fr | ~34× |
| GLI temporal SD | 0.0020 | 0.0008 | 2.6× |
| ExG temporal SD | 0.0031 | 0.0012 | 2.6× |
| Canopy projected-area CV | 3.42% | 0.29% | ~12× |

Two things to notice: (1) the effect is huge and trivially detectable; (2) during
hover a large share of the raw plant-pixel flow in the Tello view is camera
ego-motion (ego ≈ 0.7–1.2 px/fr) — skipping the compensation step would inflate
the result and reviewers will catch it.

---

## 2. What is actually in the data (read this before analyzing)

Layout per run: `data/<plant>/<RunType>/h<height>_r<n>/` containing
`video.mp4` (Tello nadir, 960×720 @ 30 fps, 10–20 s), `phone.mp4` (side view,
1080×1920 @ ~60 fps, **variable length**, sometimes shorter than the Tello clip),
`telemetry.csv` (~9 Hz: `tof, h, baro, pitch, roll, yaw, vgx, vgy, vgz, bat, templ, temph`),
`frame_times.csv` (Tello frame index → seconds), `run_info.json`.

Plants: p1 perilla, p2 burdock, p3 lemonbalm, p4 calendula, p6 wild strawberry,
p8 avocado, p9 palledium, p10 asparagus. Conditions: HoverRun (drone hovering 30 or
50 cm above the canopy) vs ReferenceRun (drone on a static stand, motors off —
matched-geometry baseline).

### Quirks found by inspection — handle these explicitly

1. **Wind-on clips are disguised as repeats.** `p1_perilla/ReferenceRun/h030_r2`
   and `p2_burdock/ReferenceRun/h050_r2` are actually `p1_wind_ref_h030_r1` and
   `p2_wind_ref_h050_r1` (fan on). The **only** reliable marker is `"wind"` in
   `run_info.json["clip_id"]` — `wind_mean_ms` is 0 in every file, and the check in
   `src/main.py` (`"_wind" in run_info["plant_id"]`) never matches because
   `plant_id` is just `"p1"`. Parse `clip_id`. If you treat these as plain r2
   repeats, your no-wind baseline is contaminated.
2. **Reference-height coverage is asymmetric.** Hover h030+h050 exist for all 8
   plants, but motors-off references exist at h050 for p1–p9 (7 plants) and at h030
   only for p1 and p10. This is fine: with the motors off the *plant* is identical
   regardless of stand height; only the camera scale changes. Compare in **mm/s**
   (after scale conversion) and the cross-height pairing is defensible. For the VI
   stability comparison, prefer same-height pairs where they exist and say so.
3. **ArUco tags are unusable in the Tello nadir video** (detected in ~1–2% of
   frames: small, at frame edges, motion-blurred) but detected in ~100% of phone
   frames (~85 px wide). Consequence: scale the Tello view with the pinhole model +
   telemetry height (`TELLO_FOCAL_PX` in `src/helpers/configs.py`), and scale the
   phone view with the ArUco tags.
4. **Hover telemetry `tof` is noisy (std ≈ 9.5 cm)** — the ToF beam hops between
   leaf tops and the tarp. Use the median of `h` (barometric/fused height, std
   ≈ 2.5 cm) for the scale computation and keep `tof` only as a QA signal
   (`TOF_MAX_STD_MM` already exists in configs for this).
5. **Dappled sunlight moves across the tarp** during some clips. This rules out
   plain frame-differencing/background-subtraction metrics (they fire on
   illumination change) and is another reason the plant mask + flow approach wins.
6. **The fan and anemometer are in frame** (right side of the Tello view, side of
   the phone view). The HSV plant mask excludes them, but keep the morphological
   open/close steps so stray fan pixels don't leak in.
7. **Phone clips are trimmed differently per run** (e.g. 7.4 s vs 20 s). Never
   compare clip *sums*; always compare per-frame distributions or means, and use
   the overlapping stable window when you need synchronization.

---

## 3. Method choice: comparison table (for the paper)

Candidate ways to quantify wind/downwash-induced canopy motion from video:

| # | Method | Representative work | What it gives | Why not / why yes for this dataset |
|---|--------|---------------------|---------------|------------------------------------|
| 1 | **Dense optical flow (classical: Farnebäck, DIS)** | Farnebäck 2003 (SCIA); Kroeger et al. 2016 (ECCV) "Fast Optical Flow using Dense Inverse Search"; applied to vegetation: [Kulkarni 2026, *Meteorological Applications*](https://rmets.onlinelibrary.wiley.com/doi/10.1002/met.70156) tracks tree motion with optical flow to recover 3-s gust speeds | Per-pixel displacement field every frame → mean/percentile magnitude, direction, spectra | **Chosen (primary).** Dense (works on low-texture leaves where sparse features die), illumination-robust compared to differencing, trivially masked to plant pixels, converts to mm/s. DIS preset MEDIUM runs 35–50 fps on this data (3–4× Farnebäck, near-identical values — see §1 pilot). |
| 2 | **Deep optical flow (RAFT)** | [Teed & Deng 2020, ECCV](https://arxiv.org/abs/2003.12039) "RAFT: Recurrent All-Pairs Field Transforms" | State-of-the-art dense flow, robust to blur and large motions | **Chosen (validation subset).** Overkill as the primary metric (GPU, ~1–2 fps), but running it on ~5% of frames and reporting agreement with DIS pre-empts the "classical flow is unreliable on deforming foliage" review comment. |
| 3 | **Image-correlation velocimetry (PIV-style)** | [Py, de Langre & Moulia 2005, *Agric. Forest Meteorology*](https://www.sciencedirect.com/science/article/abs/pii/S0168192305000808) — the classic crop-canopy motion paper; [Barbacci et al. 2014](https://www.sciencedirect.com/science/article/abs/pii/S0168192313002712) for trees | Dense velocity field via patch cross-correlation | Scientifically the direct ancestor of our choice — cite it as precedent that "standard video is sufficient to measure canopy motion velocimetry." Modern dense optical flow is the same idea with better regularization; no reason to hand-roll PIV. |
| 4 | **Frame differencing / motion energy** | [Durand et al. 2026, *New Phytologist*](https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.70975) quantify wheat-canopy "quantity of motion" by frame differencing | Scalar motion activity per frame | Cheap, but confounded by the moving sun patches in our clips, gives no displacement magnitude or direction, and cannot be converted to mm/s. Keep only as a cross-check. |
| 5 | **Sparse feature tracking (KLT)** | Lucas & Kanade 1981; Shi & Tomasi 1994 | Trajectories of a few hundred corners | Leaves are self-similar and deform; tracks drift and die mid-clip, and the sampling is biased toward leaf edges. Rejected. |
| 6 | **Marker-based videogrammetry** (physical dots on leaves) | Barbacci et al. 2014 discuss marker vs marker-less | mm-accurate point trajectories | Invasive (markers add mass and change leaf aerodynamics — the thing we measure), sparse, and impossible retroactively: our clips have no leaf markers. Rejected. |
| 7 | **Motion magnification (Eulerian/phase-based)** | Wu et al. 2012, SIGGRAPH "Eulerian Video Magnification" | Amplified visualization of sub-pixel motion | Qualitative: amplifies but does not measure. Useful for a figure/video supplement, not for quantification. |
| 8 | **Visual anemometry / learned wind inference** | [Cardona, Howland & Dabiri 2019, NeurIPS](https://arxiv.org/abs/1905.13290) "Seeing the Wind"; [Dabiri et al. 2023 review](https://arxiv.org/abs/2304.04728) | Wind *speed* inferred from vegetation motion | Solves the inverse problem (motion → wind); we need the forward one (known source → motion → index error). Cite as evidence that vegetation motion in video is a quantitative wind signal. |
| 9 | **Airflow-side quantification (anemometry/CFD)** | [Yang et al. 2022, *Pest Manag. Sci.*](https://scijournals.onlinelibrary.wiley.com/doi/abs/10.1002/ps.6790) six-rotor downwash field; [Agronomy 2025 fruit-tree downwash model](https://doi.org/10.3390/agronomy15081867); CFD: [Zhang et al. 2022, *Comput. Electron. Agric.*](https://www.sciencedirect.com/science/article/abs/pii/S0168169922005981) | The air velocity field, not the plant response | Complementary, not competing: it characterizes the forcing, we measure the canopy's optical response. Our in-frame anemometer readings connect the two. |

**Why the downwash→index link matters (motivation citation):**
[Yao et al. 2019, *Sensors* 19(4):816](https://pmc.ncbi.nlm.nih.gov/articles/PMC6412810/)
showed that rotor downwash "disrupts the Lambertian reflection characteristics of the
crop canopy," that a dual-band sensor whose footprint lay inside the disturbed zone
dropped from R² = 0.81 to R² = 0.63 against a ground spectrometer, and that at 1 m
hover the disturbed zone was an ellipse of roughly 0.45 × 0.40 m. That paper
*avoided* the problem with a 0.6 m boom; **our study measures the problem itself at
canopy level, per plant, with a motors-off control — which is the gap in the
literature** (no 2022–2025 paper we found quantifies canopy disturbance from
downwash via optical flow; the closest is downwash-induced flow used for terrain
classification: [IEEE 2016](https://ieeexplore.ieee.org/document/7404458)).

One-line justification for the paper:

> We quantify canopy disturbance with masked dense optical flow, following
> video-velocimetry precedent for wind-driven canopies (Py et al., 2005; Barbacci
> et al., 2014) and recent optical-flow tracking of vegetation motion (Kulkarni,
> 2026), using the DIS algorithm (Kroeger et al., 2016) with RAFT (Teed & Deng,
> 2020) as an accuracy check, and relate it to the stability of per-frame
> vegetation indices, whose corruption by downwash was demonstrated by Yao et
> al. (2019).

---

## 4. The pipeline

```
                        ┌────────────────────────────────────────────┐
 video.mp4 (Tello) ──►  │ 1. plant mask (HSV, configs.py)            │
 phone.mp4 (static) ──► │ 2. DIS dense flow between frame pairs      │
                        │ 3. ego-motion compensation:                │
                        │    subtract median background flow VECTOR  │
                        │ 4. per-frame metrics on plant mask:        │
                        │    mean|u|, P95|u|, motion energy          │
                        │ 5. px → mm:  pinhole+telemetry (Tello),    │
                        │              ArUco (phone)                 │
                        └──────────────┬─────────────────────────────┘
                                       │ per-frame motion table
                        ┌──────────────▼─────────────────────────────┐
                        │ 6. per-frame RGB VIs on the same mask:     │
                        │    ExG, GLI, VARI, NGRDI                   │
                        │    + canopy projected-area + specular frac │
                        │ 7. stability metrics: temporal SD, CV      │
                        └──────────────┬─────────────────────────────┘
                                       │ per-run summary rows
                        ┌──────────────▼─────────────────────────────┐
                        │ 8. paired stats: hover vs reference        │
                        │    per plant (n=8), Wilcoxon signed-rank   │
                        │    + height dose-response + wind clips     │
                        └────────────────────────────────────────────┘
```

### 4.1 Motion metrics (per frame, plant mask only)

For flow field **u**(x,y) between consecutive frames, with plant mask *M* and
background mask *B* (everything not plant, eroded):

- ego-compensated flow: **û** = **u** − median_B(**u**)   (vector median per component —
  removes hover translation/jitter; the tarp is planar and dominates *B*, so this
  is a cheap, robust stand-in for full homography stabilization)
- `flow_mean` = mean over M of |**û**|  → multiply by fps and mm/px → **mm/s**
- `flow_p95` = 95th percentile over M of |**û**| (captures the violently flapping
  leaf tips that drive index error; means underweight them)
- `motion_energy` = mean over M of |**û**|²
- optional per-run: Welch PSD of the `flow_mean` time series → dominant flutter
  frequency (Hz). Py et al. (2005) report coherent canopy modes; a downwash-driven
  spectrum vs the near-zero reference spectrum is a strong, physical figure.

### 4.2 Scale (px → mm)

- **Tello view:** mm/px at canopy level = Z_canopy / f, with f = `TELLO_FOCAL_PX`
  (≈ 840 px, see configs; re-calibrate with a tape-measured tag shot if possible)
  and Z_canopy = median(`h`) − plant-top offset. With h030: Z ≈ 300 mm → ≈ 0.36 mm/px.
  Report the calibration uncertainty; it is a scale factor, identical for hover and
  reference, so **ratios between conditions are unaffected by it.**
- **Phone view:** mm/px = `TAG_SIZE_MM` / median tag width in px (≈ 50/85 ≈ 0.59 mm/px),
  computed per clip from `helpers/aruco.py` detections. Tags lie on the tarp plane;
  state the small depth-offset approximation (plant ≈ tag depth) in limitations.

### 4.3 Vegetation-index impact (the NDVI story without an NIR camera)

The Tello is RGB-only, so compute RGB plant-health indices whose corruption
mechanism (leaf reorientation → viewing/illumination geometry and specular
changes → reflectance change) is the same one that corrupts NDVI (Yao et al. 2019):

- ExG = 2g − r − b (Woebbecke et al. 1995), with r,g,b chromatic coordinates
- GLI = (2G − R − B)/(2G + R + B) (Louhaichi et al. 2001)
- VARI = (G − R)/(G + R − B) (Gitelson et al. 2002)
- NGRDI = (G − R)/(G + R)

Per frame, over the plant mask, record: canopy-mean of each index, canopy
projected area (mask pixel count × (mm/px)² — leaves tilting edge-on shrink it),
and specular fraction (pixels with V > 240 ∧ S < 60 in HSV — the Lambertian-
breakdown signature).

**The headline metrics** per run: temporal SD and CV of each canopy-mean index,
and the hover/reference ratio of those SDs. Expected result shape: "downwash
inflated the frame-to-frame SD of GLI by a factor of k, meaning a single-frame VI
snapshot taken during hover carries k× the sampling error of an undisturbed one."
Then link: regress per-frame |index − run median| against per-frame `flow_mean`
— a positive slope is the direct motion→index-error link, which is the paper's
novel quantitative claim.

### 4.4 Statistics

- Unit of analysis: run → aggregate repeats → one value per plant × condition ×
  height. Wind clips (p1, p2) are excluded from the main contrast and reported
  separately as a "natural wind vs downwash" comparison.
- Main test: **paired Wilcoxon signed-rank** across plants (n = 8) on net flow
  (mm/s) and on index-SD, hover vs reference. With n = 8 and ~20× effects,
  power is not a concern; report the matched-pairs rank-biserial correlation
  (or Cliff's δ) as effect size, not just p.
- Height effect: paired comparison h030 vs h050 within HoverRun (all 8 plants
  have both). Note: similar *pixel* flow at both heights becomes a real
  difference in mm/s after scale conversion — compare only in mm/s.
- Species heterogeneity: one figure, per-plant paired lines (reference→hover).
  With one plant per species, you cannot test species effects — describe only.

---

## 5. Implementation

Dependencies already in the venv: `opencv-python`, `numpy`, `pandas`, `matplotlib`.
For the stats add scipy: `uv add scipy`.

Suggested layout (drop these into `src/analysis/`):

### 5.1 `src/analysis/flow_metrics.py`

```python
"""Masked, ego-compensated dense optical flow metrics for one clip."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from helpers.configs import MIN_BLOB_PX, PLANT_HSV_HI, PLANT_HSV_LO

BG_ERODE_PX = 15          # keep background sample away from leaf boundaries
MIN_MASK_PX = 500         # skip frames where the mask collapsed


def plant_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, PLANT_HSV_LO, PLANT_HSV_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # drop blobs smaller than MIN_BLOB_PX (fan grille, anemometer glints)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.isin(lab, [i for i in range(1, n) if stats[i, 4] >= MIN_BLOB_PX])
    return keep


def flow_table(video_path: str | Path, step: int = 1) -> pd.DataFrame:
    """Per-frame-pair motion metrics. step>1 subsamples frame pairs."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok:
        raise FileNotFoundError(video_path)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, idx = [], 0
    while True:
        for _ in range(step):
            ok, frame = cap.read()
            idx += 1
            if not ok:
                break
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = dis.calc(prev, gray, None)
        pm = plant_mask(frame)
        bg = cv2.erode((~pm).astype(np.uint8), np.ones((BG_ERODE_PX,) * 2, np.uint8)) > 0
        if pm.sum() < MIN_MASK_PX or bg.sum() < MIN_MASK_PX:
            prev = gray
            continue
        # ego-motion: median background flow VECTOR, subtracted component-wise
        ego = np.median(flow[bg], axis=0)
        mag = np.linalg.norm(flow - ego, axis=2)
        rows.append({
            "frame_idx": idx,
            "flow_mean_px": float(mag[pm].mean()),
            "flow_p95_px": float(np.percentile(mag[pm], 95)),
            "motion_energy_px2": float((mag[pm] ** 2).mean()),
            "ego_px": float(np.linalg.norm(ego)),
            "bg_residual_px": float(np.median(mag[bg])),   # QA: should be ~0
            "mask_px": int(pm.sum()),
        })
        prev = gray
    cap.release()
    return pd.DataFrame(rows)
```

### 5.2 `src/analysis/indices.py`

```python
"""Per-frame RGB vegetation indices + disturbance signatures on the plant mask."""
import cv2
import numpy as np
import pandas as pd

from analysis.flow_metrics import plant_mask

EPS = 1e-6


def frame_indices(bgr: np.ndarray) -> dict | None:
    pm = plant_mask(bgr)
    if pm.sum() < 500:
        return None
    b, g, r = [bgr[..., i][pm].astype(np.float64) for i in range(3)]
    tot = r + g + b + EPS
    rn, gn, bn = r / tot, g / tot, b / tot          # chromatic coordinates
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    specular = (hsv[..., 2] > 240) & (hsv[..., 1] < 60) & pm
    return {
        "exg": float((2 * gn - rn - bn).mean()),
        "gli": float(((2 * g - r - b) / (2 * g + r + b + EPS)).mean()),
        "vari": float(((g - r) / (g + r - b + EPS)).mean()),
        "ngrdi": float(((g - r) / (g + r + EPS)).mean()),
        "area_px": int(pm.sum()),
        "specular_frac": float(specular.sum() / pm.sum()),
    }


def index_table(video_path: str) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    rows, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        d = frame_indices(frame)
        if d is not None:
            rows.append({"frame_idx": idx, **d})
        idx += 1
    cap.release()
    return pd.DataFrame(rows)
```

### 5.3 `src/analysis/scale.py`

```python
"""px -> mm conversion: pinhole+telemetry for the Tello, ArUco for the phone."""
import cv2
import numpy as np
import pandas as pd

from helpers.aruco import make_detector, tag_width_px
from helpers.configs import TAG_SIZE_MM, TELLO_FOCAL_PX


def tello_mm_per_px(telemetry: pd.DataFrame, plant_height_cm: float) -> float:
    """Distance camera->canopy-top over focal length."""
    drone_h_mm = float(np.median(telemetry["h"])) * 10.0      # h is cm
    z_canopy = drone_h_mm - plant_height_cm * 10.0            # above-plant gap...
    # ...is already what run_info 'height_cm' encodes for HoverRun; for
    # ReferenceRun (h==0) fall back to the nominal stand height:
    if z_canopy <= 0:
        z_canopy = None
    return z_canopy / TELLO_FOCAL_PX if z_canopy else None


def phone_mm_per_px(video_path: str, n_probe: int = 60) -> float | None:
    det, widths = make_detector(), []
    cap = cv2.VideoCapture(video_path)
    for _ in range(n_probe):
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            widths += [tag_width_px(c[0]) for c in corners]
    cap.release()
    return TAG_SIZE_MM / float(np.median(widths)) if widths else None
```

For the ReferenceRun Tello scale, use the *nominal* condition height from
`run_info.json["height_cm"]` (the stand puts the camera at the same geometry);
for HoverRun use telemetry as above and `height_cm` as the fallback. Because the
scale enters both conditions as a constant factor, your hover/reference *ratios*
are insensitive to its exact value — say this in the methods section.

### 5.4 `src/analysis/run_analysis.py`

```python
"""Walk the dataset -> per-frame tables + one tidy per-run summary CSV."""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flow_metrics import flow_table
from analysis.indices import index_table
from analysis.scale import phone_mm_per_px, tello_mm_per_px
from helpers.utils import load_dataset

OUT = Path("../outputs/analysis")
STABLE_WINDOW_S = (2.0, 18.0)   # skip spin-up/landing transients in Tello clips


def summarize_run(run: dict, plant: dict) -> dict:
    info = run["run_info"]
    is_wind = "wind" in info["clip_id"]            # NOT wind_mean_ms, NOT plant_id
    fps = info.get("measured_fps") or 30.0
    row = {
        "clip_id": info["clip_id"], "plant": plant["plant_name"],
        "run_type": info["run_type"], "height_cm": info["height_cm"],
        "repeat": info["repeat"], "wind": is_wind,
    }
    for cam, scale in (
        ("video", tello_mm_per_px(run["telemetry"], info.get("plant_height_cm", 0))
                  or info["height_cm"] * 10.0 / 840.0),
        ("phone", phone_mm_per_px(run["phone"]) if run["phone"] else None),
    ):
        if not run[cam] or not scale:
            continue
        ft = flow_table(run[cam])
        lo, hi = (np.array(STABLE_WINDOW_S) * fps).astype(int)
        ft = ft[(ft.frame_idx >= lo) & (ft.frame_idx <= hi)] if cam == "video" else ft
        it = index_table(run[cam])
        row |= {
            f"{cam}_flow_mean_mm_s": ft.flow_mean_px.mean() * scale * fps,
            f"{cam}_flow_p95_mm_s": ft.flow_p95_px.mean() * scale * fps,
            f"{cam}_bg_residual_px": ft.bg_residual_px.mean(),       # QA
            f"{cam}_gli_sd": it.gli.std(), f"{cam}_exg_sd": it.exg.std(),
            f"{cam}_vari_sd": it.vari.std(),
            f"{cam}_area_cv": it.area_px.std() / it.area_px.mean(),
            f"{cam}_specular_mean": it.specular_frac.mean(),
        }
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(r, p)
            for p in load_dataset("../data")
            for r in p["baseline"] + p["runs"]]
    pd.DataFrame(rows).to_csv(OUT / "per_run_summary.csv", index=False)


if __name__ == "__main__":
    main()
```

### 5.5 Paired statistics (after `uv add scipy`)

```python
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("../outputs/analysis/per_run_summary.csv")
df = df[~df.wind]
per_plant = (df.groupby(["plant", "run_type"])["video_flow_mean_mm_s"]
               .mean().unstack())                 # repeats+heights aggregated
pairs = per_plant.dropna()
stat, p = wilcoxon(pairs["HoverRun"], pairs["ReferenceRun"])
ratio = (pairs["HoverRun"] / pairs["ReferenceRun"])
print(f"n={len(pairs)} plants, Wilcoxon W={stat:.1f}, p={p:.4f}, "
      f"median hover/ref ratio={ratio.median():.1f}x")
```

Repeat the same block for `video_gli_sd`, `video_exg_sd`, `video_area_cv`,
`phone_flow_mean_mm_s`. For the motion→index link:

```python
# per-frame coupling within each hover clip
merged = ft.merge(it, on="frame_idx")            # from one HoverRun clip
dev = (merged.gli - merged.gli.median()).abs()
print(merged.flow_mean_px.corr(dev, method="spearman"))
```

### 5.6 Optional: RAFT validation subset

Run `torchvision.models.optical_flow.raft_small` (or princeton-vl/RAFT) on every
20th frame pair of two clips per condition; report the correlation between
DIS and RAFT plant-mean magnitudes (expect r > 0.95). One table row in the
supplement neutralizes the method-choice objection.

### 5.7 Optional but cheap wins

- **Flutter spectrum figure:** `scipy.signal.welch` on the per-frame
  `flow_mean_px` series (fs = 30 Hz). Hover should show a broad 1–8 Hz bump
  (rotor-forced flutter); reference should be flat near zero; the fan clips
  something in between — one figure that tells the whole story.
- **Motion-magnified supplement video** (Eulerian magnification on a reference
  clip) to show even the "still" condition micro-motions — makes the SNR of the
  main result vivid.

---

## 6. QA checklist before trusting a number

- [ ] `bg_residual_px` after ego-compensation: < 0.05 px/fr for reference clips
      (measured ≈ 0.02). For hover clips expect ≈ 0.3–0.5 px/fr — **the tarp itself
      flutters under downwash**, so don't use a hard absolute threshold; instead
      require net plant flow ≥ 5× `bg_residual_px`, and mention tarp flutter as the
      residual's source in the methods (it is downwash signal, not error — but on
      the wrong surface).
- [ ] Plant mask fraction is stable within a clip (CV < 20%) and the mask overlay
      visually excludes fan, anemometer, stand legs, and pot rim for ≥3 spot frames
      per plant (mask area itself is a *result* — leaves tilting — so verify
      exclusions, don't demand constancy between conditions).
- [ ] Wind clips (`"wind" in clip_id`) excluded from the main contrast.
- [ ] Tello `h`-based scale within ±20% of the nominal `height_cm` geometry.
- [ ] Phone ArUco scale found for every phone clip actually used.
- [ ] Stable-window trim applied to Tello hover clips (first/last 2 s).
- [ ] Repeats aggregated before any test (no pseudo-replication: n = plants, not clips).

## 7. Limitations to state in the paper

1. RGB indices proxy the NDVI disturbance mechanism (geometry + specular), but
   NIR behavior is not directly measured — frame as mechanism demonstration, citing
   Yao et al. (2019) for the NIR-band consequence, and/or add one NIR-camera run
   in future work.
2. Single small quadrotor (Tello, ~87 g); downwash of ag-class UAVs is far
   stronger (Yang et al. 2022) — our effect sizes are conservative lower bounds.
3. One plant per species; indoor, controlled illumination not guaranteed
   (dappled sunlight noted) — mitigated by the paired motors-off design.
4. Tello scale rests on a nominal focal length; affects absolute mm/s only,
   not condition ratios.
