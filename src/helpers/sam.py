import json

import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

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


class SamClicker:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam2Model.from_pretrained(SAM_MODEL).to(self.device).eval()
        self.proc = Sam2Processor.from_pretrained(SAM_MODEL)

        print(self.device)
        print(self.model)

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
        m = self.proc.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"]
        )[0]
        return np.asarray(m).reshape(
            -1, *self.rgb.shape[:2]
        )[0].astype(bool)
