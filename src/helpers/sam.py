import json

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

from helpers.configs import ANNOT_DIR, SAM_MODEL

PROMPTS = ANNOT_DIR / "sam_prompts.json"
SAM_LOOKBACK = 16
FONT = cv2.FONT_HERSHEY_SIMPLEX
HELP = "L-click plant | R-click NOT plant | z undo | c clear | Enter save | Esc skip | q quit"


def load_prompts():
    return json.loads(PROMPTS.read_text()) if PROMPTS.exists() else {}


def save_prompt(clip_id, cam, prompt):
    data = load_prompts()
    data.setdefault(clip_id, {})[cam] = prompt
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS.write_text(json.dumps(data, indent=1))


def load_sam_video(device):
    model = Sam2VideoModel.from_pretrained(SAM_MODEL).to(device).eval()
    proc = Sam2VideoProcessor.from_pretrained(SAM_MODEL)
    return model, proc


class SamClicker:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam2Model.from_pretrained(SAM_MODEL).to(self.device).eval()
        self.proc = Sam2Processor.from_pretrained(SAM_MODEL)
        self.vmodel, self.vproc = load_sam_video(self.device)

    def set_image(self, bgr):
        self.rgb = np.ascontiguousarray(bgr[..., ::-1])
        inp = self.proc(images=self.rgb, return_tensors="pt")
        with torch.inference_mode():
            self.emb = self.model.get_image_embeddings(
                inp["pixel_values"].to(self.device)
            )

    def predict(self, points, labels):
        inp = self.proc(
            images=self.rgb,
            input_points=[[points]],
            input_labels=[[labels]],
            return_tensors="pt",
        )
        with torch.inference_mode():
            out = self.model(
                input_points=inp["input_points"].to(self.device),
                input_labels=inp["input_labels"].to(self.device),
                image_embeddings=self.emb,
                multimask_output=False,
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
                    sam.vproc.add_inputs_to_inference_session(
                        inference_session=sess,
                        frame_idx=0,
                        obj_ids=1,
                        input_points=[[prompt["points"]]],
                        input_labels=[[prompt["labels"]]],
                        original_size=inp.original_sizes[0],
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


def click_prompt(frame, clicker, title):
    clicker.set_image(frame)
    pts, lbl = [], []
    state = {"mask": None}
    s = min(1280 / frame.shape[1], 720 / frame.shape[0], 1.0)

    def redo():
        state["mask"] = clicker.predict(pts, lbl) if 1 in lbl else None

    def on_mouse(event, x, y, *_):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            pts.append([x / s, y / s])
            lbl.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
            redo()

    win = "SAM prompt"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        if state["mask"] is not None:
            m = state["mask"]
            vis[m] = (0.5 * vis[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        for (x, y), l in zip(pts, lbl):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0) if l else (0, 0, 255), -1)
        vis = cv2.resize(vis, None, fx=s, fy=s)
        cv2.putText(vis, title, (10, 25), FONT, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, HELP, (10, 50), FONT, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and 1 in lbl:
            return pts, lbl
        elif key == ord("z") and pts:
            pts.pop()
            lbl.pop()
            redo()
        elif key == ord("c"):
            pts.clear()
            lbl.clear()
            redo()
        elif key == 27:
            return None
        elif key == ord("q"):
            return "quit"
