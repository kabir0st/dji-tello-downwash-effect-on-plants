import json

import cv2
import numpy as np
import torch
from transformers import (
    Sam3TrackerModel,
    Sam3TrackerProcessor,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)

from helpers.configs import ANNOT_DIR, SAM_MODEL

PROMPTS = ANNOT_DIR / "sam_prompts.json"
SAM_LOOKBACK = 16
FONT = cv2.FONT_HERSHEY_SIMPLEX
HELP = (
    "L-drag box around plant | L-click plant | R-click NOT plant | "
    "z undo | c clear | Enter save | Esc skip | q quit"
)
# SAM3's tracker degrades sharply with dense clicks (20+ points -> fragmented masks).
# Prompt with a box around the whole plant plus a few corrective clicks instead.
MAX_GOOD_POINTS = 10
DRAG_PX = 8

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def pick_dtype():
    # bf16 only on NVIDIA Ampere/Ada (A5000, RTX 4000 Ada); fp32 on Turing, your AMD card, CPU
    if torch.cuda.is_available() and torch.version.cuda and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float32

DTYPE = pick_dtype()

def load_prompts():
    return json.loads(PROMPTS.read_text()) if PROMPTS.exists() else {}


def save_prompt(clip_id, cam, prompt):
    data = load_prompts()
    data.setdefault(clip_id, {})[cam] = prompt
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS.write_text(json.dumps(data, indent=1))


def load_sam_video(device):
    model = Sam3TrackerVideoModel.from_pretrained(SAM_MODEL).to(device).eval()
    proc = Sam3TrackerVideoProcessor.from_pretrained(SAM_MODEL)
    return model, proc


class SamClicker:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3TrackerModel.from_pretrained(SAM_MODEL).to(self.device).eval()
        self.proc = Sam3TrackerProcessor.from_pretrained(SAM_MODEL)
        self.vmodel, self.vproc = load_sam_video(self.device)

    def set_image(self, bgr):
        self.rgb = np.ascontiguousarray(bgr[..., ::-1])
        inp = self.proc(images=self.rgb, return_tensors="pt")
        with torch.inference_mode():
            self.emb = self.model.get_image_embeddings(
                inp["pixel_values"].to(self.device)
            )

    def predict(self, points, labels, box=None):
        kw = {}
        if points:
            kw.update(input_points=[[points]], input_labels=[[labels]])
        if box is not None:
            kw.update(input_boxes=[[box]])
        inp = self.proc(images=self.rgb, return_tensors="pt", **kw)
        args = {
            k: inp[k].to(self.device)
            for k in ("input_points", "input_labels", "input_boxes")
            if k in inp
        }
        with torch.inference_mode():
            out = self.model(
                image_embeddings=self.emb, multimask_output=False, **args
            )
        m = self.proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        return np.asarray(m).reshape(-1, *self.rgb.shape[:2])[0].astype(bool)


def track(path, prompt, sam):
    sess = sam.vproc.init_video_session(
        inference_device=sam.device, dtype=torch.float32
    )
    cap = cv2.VideoCapture(path)
    i = 0
    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = np.ascontiguousarray(frame[..., ::-1])
                inp = sam.vproc(images=rgb, device=sam.device, return_tensors="pt")
                if i == 0:
                    kw = {}
                    if prompt.get("points"):
                        kw.update(
                            input_points=[[prompt["points"]]],
                            input_labels=[[prompt["labels"]]],
                        )
                    if prompt.get("box"):
                        kw.update(input_boxes=[[prompt["box"]]])
                    sam.vproc.add_inputs_to_inference_session(
                        inference_session=sess,
                        frame_idx=0,
                        obj_ids=1,
                        original_size=inp.original_sizes[0],
                        **kw,
                    )
                out = sam.vmodel(
                    inference_session=sess, frame=inp.pixel_values[0].to(sam.device)
                )
                m = sam.vproc.post_process_masks(
                    [out.pred_masks.float()],
                    original_sizes=inp.original_sizes,
                    binarize=True,
                )[0]
                old = i - SAM_LOOKBACK
                if old > 0:
                    sess.processed_frames[old] = None
                    for o in sess.output_dict_per_obj.values():
                        o["non_cond_frame_outputs"].pop(old, None)
                yield i, frame, np.asarray(m.cpu()).reshape(-1, *frame.shape[:2])[
                    0
                ].astype(bool)
                i += 1
    finally:
        cap.release()


def make_prompt(pts, lbl, box=None):
    prompt = {"points": [[round(x, 1), round(y, 1)] for x, y in pts], "labels": list(lbl)}
    if box is not None:
        prompt = {"box": [round(v, 1) for v in box], **prompt}
    return prompt


def click_prompt(frame, clicker, title, init=None):
    """Returns a prompt dict {"box"?, "points", "labels"}, None (skip) or "quit".

    init: an existing prompt to start from (e.g. the saved one), so it can be edited.
    """
    clicker.set_image(frame)
    init = init or {}
    pts = [list(p) for p in init.get("points", [])]
    lbl = list(init.get("labels", []))
    box = list(init["box"]) if init.get("box") else None
    # history of actions for undo: "pt" or ("box", previous_box)
    hist = ([("box", None)] if box else []) + ["pt"] * len(pts)
    state = {"mask": None, "box": box, "drag": None, "cur": None, "hist": hist}
    s = min(1280 / frame.shape[1], 720 / frame.shape[0], 1.0)

    def redo():
        ok = 1 in lbl or state["box"] is not None
        state["mask"] = clicker.predict(pts, lbl, state["box"]) if ok else None

    redo()

    def on_mouse(event, x, y, flags, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"], state["cur"] = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            state["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"] is not None:
            (x0, y0), state["drag"] = state["drag"], None
            if max(abs(x - x0), abs(y - y0)) >= DRAG_PX:
                h, w = frame.shape[:2]
                state["hist"].append(("box", state["box"]))
                state["box"] = [
                    max(0.0, min(x0, x) / s), max(0.0, min(y0, y) / s),
                    min(w - 1.0, max(x0, x) / s), min(h - 1.0, max(y0, y) / s),
                ]
            else:
                pts.append([x0 / s, y0 / s])
                lbl.append(1)
                state["hist"].append("pt")
            redo()
        elif event == cv2.EVENT_RBUTTONDOWN:
            pts.append([x / s, y / s])
            lbl.append(0)
            state["hist"].append("pt")
            redo()

    win = "SAM prompt"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        if state["mask"] is not None:
            m = state["mask"]
            vis[m] = (0.5 * vis[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        if state["box"] is not None:
            b = [int(v) for v in state["box"]]
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        for (x, y), l in zip(pts, lbl):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0) if l else (0, 0, 255), -1)
        vis = cv2.resize(vis, None, fx=s, fy=s)
        if state["drag"] is not None:
            cv2.rectangle(vis, state["drag"], state["cur"], (0, 255, 255), 1)
        cv2.putText(vis, title, (10, 25), FONT, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, HELP, (10, 50), FONT, 0.5, (255, 255, 255), 1)
        if len(pts) > MAX_GOOD_POINTS:
            cv2.putText(
                vis, f"{len(pts)} points: SAM3 works best with a box + <= {MAX_GOOD_POINTS}",
                (10, 75), FONT, 0.5, (0, 0, 255), 2,
            )
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and (1 in lbl or state["box"] is not None):
            return make_prompt(pts, lbl, state["box"])
        elif key == ord("z") and state["hist"]:
            last = state["hist"].pop()
            if last == "pt":
                pts.pop()
                lbl.pop()
            else:
                state["box"] = last[1]
            redo()
        elif key == ord("c"):
            pts.clear()
            lbl.clear()
            state["box"] = None
            state["hist"].clear()
            redo()
        elif key == 27:
            return None
        elif key == ord("q"):
            return "quit"
