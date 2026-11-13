import cv2
import numpy as np

FONT, WHITE = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255)
COLORS = {"HoverRun": "#2a78d6", "ReferenceRun": "#eb6834"}


def fit(img, w):
    return cv2.resize(img, (w, round(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)


def label(img, text):
    bar = np.zeros((32, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 22), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def grid(panels, cols=3, w=480):
    tiles = []
    for title, img in panels:
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        tiles.append(label(fit(img, w), title))
    tiles += [np.zeros((1, w, 3), np.uint8)] * (-len(tiles) % cols)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        h = max(t.shape[0] for t in row)
        rows.append(np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                               for t in row]))
    return np.vstack(rows)


def overlay(bgr, mask, color=(0, 255, 0), alpha=0.45):
    """Tint the mask pixels with a colour."""
    out = bgr.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.array(color)).astype(np.uint8)
    return out


def dim_outside(bgr, mask, keep=0.2):
    out = (bgr * keep).astype(np.uint8)
    out[mask] = bgr[mask]
    return out


def flow_to_bgr(flow, max_px):
    dx, dy = flow[..., 0], flow[..., 1]
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (np.degrees(np.arctan2(dy, dx)) % 360 / 2).astype(np.uint8)  # OpenCV hue 0-179
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(np.hypot(dx, dy) / max_px, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(bgr, flow, step=32, scale=4.0, mask=None):
    out = bgr.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if mask is not None and not mask[y, x]:
                continue
            dx, dy = flow[y, x]
            cv2.arrowedLine(out, (x, y), (int(x + dx * scale), int(y + dy * scale)),
                            WHITE, 1, cv2.LINE_AA, tipLength=0.3)
    return out


def _hex_bgr(h):
    return np.array([int(h[i:i + 2], 16) for i in (5, 3, 1)], float)


RAMP = np.array([(1 - t) * _hex_bgr("#0d2b52") + t * _hex_bgr("#d6e8ff")
                 for t in np.linspace(0, 1, 256)], np.uint8)

def heat(bgr, values, mask, vmax, unit=""):
    out = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) // 3, cv2.COLOR_GRAY2BGR)
    idx = (np.clip(values / vmax, 0, 1) * 255).astype(np.uint8)
    out[mask] = RAMP[idx[mask]]
    h, w = out.shape[:2]
    fs, th, bw = w / 1000, max(1, round(w / 500)), max(20, w // 40)  
    y0, y1, x0 = 50, h - 50, w - bw - 20
    out[y0:y1, x0:x0 + bw] = RAMP[np.linspace(255, 0, y1 - y0).astype(np.uint8)][:, None]
    top = f"{vmax:.3g} {unit}"
    tw, txh = cv2.getTextSize(top, FONT, fs, th)[0]
    cv2.putText(out, top, (x0 - tw - 8, y0 + txh), FONT, fs, WHITE, th, cv2.LINE_AA)
    cv2.putText(out, "0", (x0 - cv2.getTextSize("0", FONT, fs, th)[0][0] - 8, y1), FONT, fs, WHITE, th,
                cv2.LINE_AA)
    return out
