"""
esp32_recorder.py - record the ESP32-CAM stream on its own laptop.

Keys (click the video window first):
    R = start recording      X = stop + save      Q = quit

Folders:
    esp32_dataset/
    ├── recordings_log.csv                 <- one row per recording (all plants)
    └── <plant>/                           <- same folder for the same plant
        └── <run_id>/                      <- a different folder for each run
            ├── rec_01/  video.mp4, frame_times.csv, rec_info.json
            └── rec_02/  ...
"""
import os
import csv
import json
import time
import threading
from datetime import datetime

import cv2

# ---------------- settings ----------------
ESP32_URL = "http://10.249.169.179:81/stream"   # your ESP32-CAM stream address
ROOT = "esp32_dataset"
WINDOW = "ESP32-CAM"
VIDEO_FPS_HINT = 15        # playback speed hint only - real times are in frame_times.csv

LOG_FIELDS = ["plant", "run_id", "rec", "start_time", "start_epoch_s",
              "duration_s", "n_frames", "measured_fps", "rec_path"]


# ---------------- camera ----------------
class Esp32Stream:
    """Reads the MJPEG stream in a background thread and keeps the latest frame
    (avoids OpenCV's internal buffer adding delay)."""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open ESP32 stream: {url}")
        self.frame, self.running = None, True
        threading.Thread(target=self._reader, daemon=True).start()
        print(f"ESP32-CAM connected: {url}")

    def _reader(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f          # new array each time -> 'is not' detects new frames

    def close(self):
        self.running = False
        self.cap.release()


# ---------------- one recording ----------------
class Recorder:
    """Writes video.mp4 + frame_times.csv, and rec_info.json on stop."""

    def __init__(self, rec_dir, first_frame):
        h, w = first_frame.shape[:2]
        self.dir = rec_dir
        self.vw = cv2.VideoWriter(os.path.join(rec_dir, "video.mp4"),
                                  cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS_HINT, (w, h))
        self.ff = open(os.path.join(rec_dir, "frame_times.csv"), "w", newline="")
        self.fw = csv.writer(self.ff)
        self.fw.writerow(["frame_idx", "t_s", "epoch_s"])   # epoch = laptop wall clock
        self.t0 = time.monotonic()
        self.epoch0 = time.time()
        self.n = 0

    def write(self, frame):
        t = time.monotonic() - self.t0
        self.vw.write(frame)
        self.fw.writerow([self.n, f"{t:.4f}", f"{self.epoch0 + t:.4f}"])
        self.n += 1
        return t

    def stop(self, info):
        duration = time.monotonic() - self.t0
        self.vw.release()
        self.ff.close()
        info.update({
            "start_time": datetime.fromtimestamp(self.epoch0).isoformat(timespec="milliseconds"),
            "start_epoch_s": round(self.epoch0, 4),
            "duration_s": round(duration, 2),
            "n_frames": self.n,
            "measured_fps": round(self.n / duration, 2) if duration > 0 else 0,
        })
        with open(os.path.join(self.dir, "rec_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        return info


# ---------------- helpers ----------------
def clean(text):
    """'Plant 01 Lettuce ' -> 'plant_01_lettuce' (safe folder name)."""
    return text.strip().lower().replace(" ", "_")


def next_rec_dir(run_dir):
    """run_dir/rec_01, rec_02, ... (never overwrites)."""
    n = 1
    while os.path.exists(os.path.join(run_dir, f"rec_{n:02d}")):
        n += 1
    path = os.path.join(run_dir, f"rec_{n:02d}")
    os.makedirs(path)
    return path


def append_log(info):
    path = os.path.join(ROOT, "recordings_log.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore", restval="")
        if new_file:
            w.writeheader()
        w.writerow(info)


def save(rec, plant, run_id, rec_dir):
    info = rec.stop({"plant": plant, "run_id": run_id,
                     "rec": os.path.basename(rec_dir),
                     "rec_path": os.path.relpath(rec_dir, ROOT)})
    append_log(info)
    print(f"Saved {rec_dir}: {info['n_frames']} frames, "
          f"{info['duration_s']} s, {info['measured_fps']} fps")


# ---------------- main ----------------
def main():
    plant = clean(input("Plant name (e.g. plant01_lettuce): "))
    run_id = clean(input("Run ID (e.g. run01): "))
    run_dir = os.path.join(ROOT, plant, run_id)
    os.makedirs(run_dir, exist_ok=True)

    cam = Esp32Stream(ESP32_URL)
    rec, rec_dir, last = None, None, None
    print("Click the video window.  R = record   X = stop + save   Q = quit")

    try:
        while True:
            frame = cam.frame
            if frame is not None and frame is not last:          # a NEW frame
                last = frame
                shown = frame.copy()
                if rec:
                    t = rec.write(frame)
                    cv2.putText(shown, f"REC {t:5.1f} s   X = stop + save", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(shown, f"{plant} / {run_id}   R = record   Q = quit", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WINDOW, shown)

            key = cv2.waitKey(1) & 0xFF
            k = chr(key).lower() if key != 255 else ""

            if k == "r" and rec is None and last is not None:      # start
                rec_dir = next_rec_dir(run_dir)
                rec = Recorder(rec_dir, last)
                print(f"Recording -> {rec_dir}")
            elif k == "x" and rec is not None:                     # stop + save
                save(rec, plant, run_id, rec_dir)
                rec = None
            elif k == "q":                                         # quit
                break
    finally:
        if rec is not None:              # quit or crash while recording -> still save
            save(rec, plant, run_id, rec_dir)
        cam.close()
        cv2.destroyAllWindows()
        print("Closed.")


if __name__ == "__main__":
    main()