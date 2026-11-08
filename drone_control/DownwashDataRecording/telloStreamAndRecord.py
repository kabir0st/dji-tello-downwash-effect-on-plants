# Code Reference by Shiva (SHiTU23)
# Link: https://github.com/SHiTU23/Tello_drone

import cv2
from djitellopy import Tello
import os
import time
import csv
from datetime import datetime
import json

FRAME_IS_RGB = True
CLIP_DURATION_SECOND = 20
VIDEO_TEXT_DISPLAY = "Down Wash Test"
SAMPLE_RATE_SECOND = 10


STATE_KEYS = ["tof", "h", "baro", "pitch", "roll", "yaw",
              "vgx", "vgy", "vgz", "bat", "templ", "temph"]
METADATA_FIELDS = ["clip_id", "plant_id", "plant_species", "run_type",
                   "height_cm", "repeat", "date_time", "duration_s",
                   "n_frames", "measured_fps", "battery_start_pct",
                   "temp_start_c", "plant_height_cm", "origin_h_cm",
                   "origin_tof_cm", "origin_baro_cm", "light_lux",
                   "wind_mean_ms", "notes", "clip_path"]



def to_bgr(frame):
    """OpenCV expects BGR. Convert only if the Tello frame is RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if FRAME_IS_RGB else frame

# Class for tello stream and record function
class TelloStream:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon() # encoded camera frame 
        self.reader = self.tello.get_frame_read()  # read frame and decode
        self.batteryStat = self.tello.get_battery()
        print(f'Tello Stream Connected: Battery Status: {self.batteryStat}%')

    def raw_frame(self):
        return self.reader.frame

    # actual bgr frame, ready for display or saving:
    def frame(self):
        f = self.reader.frame
        return None if f is None else to_bgr(f)

    def state(self):
        """Current telemetry values, in STATE_KEYS order."""
        s = self.tello.get_current_state()
        return [s.get(k, "") for k in STATE_KEYS]

    def battery(self):
        return self.tello.get_battery()

    def temperature(self):
        return self.tello.get_temperature()

    def close(self):
        self.tello.streamoff()
        self.tello.end()
        cv2.destroyAllWindows()
        print("Connection closed.")

def preview(stream, text=""):
    """Show one live frame with an optional message. Returns the key pressed."""
    f = stream.frame()
    if f is not None:
        f = f.copy()
        if text:
            cv2.putText(f, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        cv2.imshow(VIDEO_TEXT_DISPLAY, f)
    return cv2.waitKey(1) & 0xFF

def stableCountdown(stream, STABLE_PERIOD):
    t_end = time.monotonic() + STABLE_PERIOD
    while (left := t_end - time.monotonic()) > 0:
        preview(stream, f"Recording starts in {left:.0f} s")



# ---------------- folders ----------------
def test_dir_for(root, plant_id, species):
    """dataset/plant01_lettuce"""
    path = os.path.join(root, f"{plant_id}_{species}")
    os.makedirs(path, exist_ok=True)
    return path


def next_clip_dir(test_dir, run_type, height_cm):
    """test_dir/run_type/h030_r1, h030_r2, ... (repeat number auto-increments)."""
    base = os.path.join(test_dir, run_type)
    os.makedirs(base, exist_ok=True)
    rep = 1
    while os.path.exists(os.path.join(base, f"h{height_cm:03d}_r{rep}")):
        rep += 1
    path = os.path.join(base, f"h{height_cm:03d}_r{rep}")
    os.makedirs(path)
    return path, rep

def timestamp():
    return datetime.now().isoformat(timespec="seconds")

def record_clip(stream, clip_dir, duration = CLIP_DURATION_SECOND, show = True, tick = None ):
    first_stream = stream.frame()
    while first_stream is None:
        time.sleep(0.5)
        first_stream = stream.frame()

    # Setup folder and files:
    h, w = first_stream.shape[:2]
    vw = cv2.VideoWriter(os.path.join(clip_dir, "video.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    ff = open(os.path.join(clip_dir, "frame_times.csv"), "w", newline="")
    tf = open(os.path.join(clip_dir, "telemetry.csv"), "w", newline="")

    # run record video
    try:
        fw, tw = csv.writer(ff), csv.writer(tf)
        fw.writerow(["frame_idx", "t_s"])
        tw.writerow(["t_s"] + STATE_KEYS)

        t0 = time.monotonic()
        last_raw, last_tel, n = None, -1.0, 0   # n - frame se so far
        while (now := time.monotonic() - t0) < duration:
            raw = stream.raw_frame()
            if raw is not None and raw is not last_raw:   # save only NEW frames
                frame = to_bgr(raw)
                vw.write(frame)
                fw.writerow([n, f"{now:.4f}"])
                n += 1
                last_raw = raw
                if show:
                    shown = frame.copy()
                    cv2.putText(shown, f"REC {now:4.1f} / {duration} s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(VIDEO_TEXT_DISPLAY, shown)
                    cv2.waitKey(1)
            if now - last_tel >= 1.0 / SAMPLE_RATE_SECOND:      # telemetry at 10 Hz
                tw.writerow([f"{now:.4f}"] + stream.state())
                last_tel = now
            if tick is not None:                           
                tick(now)
            time.sleep(0.002)
    finally:
        vw.release()
        ff.close()
        tf.close()
    return {"duration_s": duration, "n_frames": n,
            "measured_fps": round(n / duration, 2)}


# ---------------- logs ----------------
def save_run_info(clip_dir, info):
    with open(os.path.join(clip_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

#   Add one row per clip to dataset/metadata.csv (header written once).
def append_metadata(root, info):
    path = os.path.join(root, "metadata.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDS,
                           extrasaction="ignore", restval="")
        if new_file:
            w.writeheader()
        w.writerow(info)

