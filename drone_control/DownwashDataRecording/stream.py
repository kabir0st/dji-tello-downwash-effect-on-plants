import telloStreamAndRecord as tsr 
import os 
import keyboard
import cv2 
import time

ROOT = "dataset"
RUN_TYPES = {1: "ReferenceRun",2: "HoverRun",3: "PassByRun"}
RUN_PREFIX = {"ReferenceRun": "ref", "HoverRun": "hov", "PassByRun": "pass"}
STABLE_PERIOD = 5

#Tello Manual Control Sign and Command

MANUAL_SPEED = 30 
KEY_HOLD_S = 0.15 
KEEP_ALIVE_S = 0.05  # to maintain constant communincation 

# Source : tello points thread: tellopilots.com/threads/identifying-movement-commands-in-sdk.3034/
 #        : key: (left/right, fwd/back, up/down, yaw)
KEYMAP = {
    "left":      (-1, 0, 0, 0), "right":     (1, 0, 0, 0),
    "up":        (0, 1, 0, 0),  "down":      (0, -1, 0, 0),
    "page up":   (0, 0, 1, 0),  "page down": (0, 0, -1, 0),
    "a":         (0, 0, 0, -1), "d":         (0, 0, 0, 1),
}

def keys_to_rc():
    """Add up every key held RIGHT NOW -> one rc command."""
    lr = fb = ud = yaw = 0
    for name, (a, b, c, d) in KEYMAP.items():
        if keyboard.is_pressed(name):
            lr += a; fb += b; ud += c; yaw += d
    return tuple(max(-1, min(1, v)) * MANUAL_SPEED for v in (lr, fb, ud, yaw))


def draw_battery(f, battery):
    """Battery % in the top-right corner: green > 50, yellow 30-50, red < 30."""
    if battery > 50:
        colour = (0, 255, 0)        # green  (BGR)
    elif battery >= 30:
        colour = (0, 255, 255)      # yellow
    else:
        colour = (0, 0, 255)        # red
    x = f.shape[1] - 160            # 160 px from the right edge (works for any width)
    cv2.putText(f, f"Bat {battery}%", (x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

def manual_mode(stream):

    # 1. Reassign tello 
    tello = stream.tello 
    last_send = 0.0

    # 2. Manual control loop 
    while True:
        # 1. check height 
        h = tello.get_height()

        # 2. get frame and display contantly 
        f = stream.frame()

        # 3. display the height information in the screen
        if f is not None:
            f = f.copy()
            cv2.putText(f, f"Tello h (display only): {h} cm above take-off point",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(f, "Arrows move  PgUp/PgDn up/down  A/D turn  L lock+record  X land",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            draw_battery(f, tello.get_battery())          
            cv2.imshow(tsr.VIDEO_TEXT_DISPLAY, f)

        # 4. keep the window alive (needed for redraw)
        cv2.waitKey(1)
        now = time.monotonic()   #  for keep alive and locked position maintained

        # 5. lock/land
        if keyboard.is_pressed("l") or keyboard.is_pressed("x"):
            tello.send_rc_control(0, 0, 0, 0)
            action = "lock" if keyboard.is_pressed("l") else "land"
            while keyboard.is_pressed("l") or keyboard.is_pressed("x"):
                cv2.waitKey(1)                 # wait for release (no double trigger)
            return action

        # 6.  move with held keys (none held -> 0,0,0,0 = hold position)
        # sent at 20 Hz -> also the keep-alive signal
        if now - last_send >= KEEP_ALIVE_S:
            tello.send_rc_control(*keys_to_rc())
            last_send = now


# Once locked, hold the position until stream recorded
def hold_tick(stream):
    last = [-1.0]
    def tick(now):
        if now - last[0] >= 0.5:
            stream.tello.send_rc_control(0, 0, 0, 0)
            last[0] = now
    return tick



def referenceRun(stream, clip_dir):
    print("Stream and Recording Activated....!!")
    tsr.stableCountdown(stream, STABLE_PERIOD)   
    status = tsr.record_clip(stream, clip_dir)
    return status

def hoverRun(stream, clip_dir):
    print('Hover run activated.....!!')
    tsr.stableCountdown(stream, STABLE_PERIOD)
    status = tsr.record_clip(stream, clip_dir, tick=hold_tick(stream))
    return status


def passByRun(stream, clip_dir):
    print('Pass by run activated')


RUN_FUNCTIONS = {"ReferenceRun": referenceRun,
                 "HoverRun": hoverRun,
                 "PassByRun": passByRun}


def choose_run_type():
    while True: 
        run_type_index = input("Enter run type (1 = Reference Run, 2 = Hover Run, 3 = By Pass Run) :").strip()
        if int(run_type_index) in RUN_TYPES:
            return RUN_TYPES[int(run_type_index)]
        else:
            print('Enter valid run type...!')

# After L (lock): type the height in the VIDEO window while the drone keeps
# hovering. input() can't be used in the air: it blocks rc commands -> the Tello
# auto-lands after 15 s and the window freezes.
def type_height_in_window(stream, flying=True):
    """Digits = type, Enter = record, Backspace = delete, Esc = cancel.
    Returns the typed height (int) or None if cancelled."""
    tello = stream.tello
    typed, last_send = "", 0.0
    esc_text = "back to manual" if flying else "finish session"
    while True:
        f = stream.frame()
        if f is not None:
            f = f.copy()
            cv2.putText(f, f"Type height above canopy: {typed}_ cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(f, f"Enter = record   Backspace = delete   Esc = {esc_text}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            draw_battery(f, tello.get_battery())
            cv2.imshow(tsr.VIDEO_TEXT_DISPLAY, f)
        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()
        if flying and now - last_send >= KEEP_ALIVE_S:   # only when airborne
            tello.send_rc_control(0, 0, 0, 0)
            last_send = now

        if ord('0') <= key <= ord('9') and len(typed) < 3:
            typed += chr(key)
        elif key == 8 and typed:                     # Backspace
            typed = typed[:-1]
        elif key in (13, 10) and typed:              # Enter
            return int(typed)
        elif key == 27:                              # Esc
            return None

 
def main():

    # ask plant info 
    run_type = choose_run_type()
    plant_id = input("Plant ID (e.g. plant01): ").strip()
    species = input("Plant species (e.g. lettuce): ").strip().lower()
    test_dir = tsr.test_dir_for(ROOT, plant_id, species)
    run_function = RUN_FUNCTIONS[run_type]
    airborne = (run_type == "HoverRun")  

    # Tello initialized 
    stream = tsr.TelloStream()
    tello = stream.tello

    # Run strem and record
    try: 
        if airborne:
            print("Taking off. Arrows/PgUp/PgDn/A/D to fly, L = lock, X = land.")
            if (tello.get_battery() > 20):
                tello.takeoff()
            else:
                print("Battery Low.......! ")
        while True:    
            print("\nPress 'Q' to terminate or Enter height for next run:...!!")
                # Check keyboard pressed 
            if keyboard.is_pressed('Q'):
                print('Test Terminated: User interviene..!')
                break
            if airborne:
                manual_return = manual_mode(stream)
                # 2.1. if land mode
                if manual_return == "land":
                    # break 
                    print("Test terminated: Landing...!")
                    break
                # ---- lock: this position is the origin ----
                # 2.1. if the mode is locked 
                height = type_height_in_window(stream)  
                s = tello.get_current_state()
                if height is None:                         # Esc -> back to manual
                    continue
                print(f"  Locked: {height} cm above canopy (Tello h = {s.get('h')} cm)")
            else:
                # get height info
                # height = int(input("Camera height above the plant (In cm): ").strip())
                height = type_height_in_window(stream, flying=False)
                if height is None:                    # Esc -> finish session
                    print("Test terminated by user.")
                    break
                s = tello.get_current_state()

            # Record
            clip_dir, rep = tsr.next_clip_dir(test_dir, run_type, height)

            # Wait frame before recording: for drone stablzation 
            batteryStat = stream.battery()
            temperatureStat = stream.temperature()
            print(f"{run_type} at {height} cm: recording...")

            # Execute respective run mode
            stats = run_function(stream, clip_dir)


            info = {
                "clip_id": f"{plant_id}_{RUN_PREFIX[run_type]}_h{height:03d}_r{rep}",
                "plant_id": plant_id,
                "plant_species": species,
                "run_type": run_type,
                "height_cm": height,                  
                "repeat": rep,
                "date_time": tsr.timestamp(),
                "battery_start_pct": batteryStat,
                "temp_start_c": temperatureStat,
                "plant_height_cm": height,
                "origin_h_cm": s.get("h", ""),        # Tello estimate, logged only
                "origin_tof_cm": s.get("tof", ""),    # logged only
                "origin_baro_cm": round(float(s.get("baro", 0)) * 100, 1),
                "wind_mean_ms": 0,
                "clip_path": os.path.relpath(clip_dir, ROOT),
                **stats,
            }
            tsr.save_run_info(clip_dir, info)
            tsr.append_metadata(ROOT, info)
            print(f"Saved {clip_dir}: {stats['n_frames']} frames, "
                  f"{stats['measured_fps']} fps.")

            if not airborne:
                cv2.destroyWindow(tsr.VIDEO_TEXT_DISPLAY)   # close it while we wait for input
                cv2.waitKey(1)    
    finally:
        if airborne:                                  
            tello.send_rc_control(0, 0, 0, 0)
            try:
                tello.land()
            except Exception as e:
                print(f"  Land command not needed: {e}")
        stream.close()

if __name__ == "__main__":
    main()