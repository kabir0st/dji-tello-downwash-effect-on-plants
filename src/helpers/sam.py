import json

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

from helpers.configs import ANNOT_DIR, SAM_MODEL

PROMPTS = ANNOT_DIR / "sam_prompts.json"
SAM_LOOKBACK = 16


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


def track(path, points, labels, clicker):
    """Follow the plant clicked on frame 0. Yields (i, frame, bool mask)."""
    sess = clicker.vproc.init_video_session(
        inference_device=clicker.device, dtype=torch.float32
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
                inp = clicker.vproc(
                    images=rgb, device=clicker.device, return_tensors="pt"
                )
                if i == 0:
                    clicker.vproc.add_inputs_to_inference_session(
                        inference_session=sess,
                        frame_idx=0,
                        obj_ids=1,
                        input_points=[[points]],
                        input_labels=[[labels]],
                        original_size=inp.original_sizes[0],
                    )
                out = clicker.vmodel(
                    inference_session=sess, frame=inp.pixel_values[0].to(clicker.device)
                )
                m = clicker.vproc.post_process_masks(
                    [out.pred_masks.float()],
                    original_sizes=inp.original_sizes,
                    binarize=True,
                )[0]
                # SAM 2 only looks back ~16 frames; free older ones or RAM grows ~20 MB/frame
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
