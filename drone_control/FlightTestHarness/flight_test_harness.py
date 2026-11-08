"""
E1-01 Flight Test Harness

Purpose:
Test the Tello drone with different payload weights
and collect flight data for later analysis.
"""
from datetime import datetime
import time
import csv
from djitellopy import Tello, TelloException
import keyboard
import json
from pathlib import Path    # to manage files and folder
import logging
import math
import matplotlib.pyplot as plt

PAYLOAD_WEIGHTS = [0, 20, 40, 60]   # In gram
SAMPLE_INTERVAL = 1.0 # Record one telemetry sample every second
MIN_BATTERY_THRESHOLD = 20  # Minimum battery percentage for the real flight test
KEEP_ALIVE_INTERVAL = 5.0  # Interval to send keepalive signal to Tello (must stay well below ~15 s timeout)
PRE_FLIGHT_DEADLINE_THRESHOLD = 5.0
VPS_STATUS = None

LOG_DIR = Path(__file__).resolve().parent / "logs"

LOGGER = logging.getLogger("payload_test")
LOGGER.setLevel(logging.INFO)


#Define required inputs, outputs and measurement data
TELEMETRY_FIELDS = [
    "timestamp",
    "elapsed_seconds",
    "payload_weight_g",
    "battery_pct",
    "vps_status",
    "speed_x_cm_s",
    "speed_y_cm_s",
    "speed_z_cm_s",
    "speed_3d_cm_s",
    "altitude_cm",
    "tof_distance_cm",
    "temperature_low_c",
    "temperature_high_c",
    "temperature_avg_c",
    "estimated_distance_cm",
]


def local_timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def safe_get(getter):
    # djitellopy getters raise TelloException when a field is missing
    try:
        return getter()
    except TelloException:
        return None


def choose_payload():
    print("Available payload weights:", PAYLOAD_WEIGHTS)

    try:
        payload = int(input("Enter payload weight in grams: "))
    except ValueError:
        print("Enter one of the listed payload weights.")
        return None

    if payload not in PAYLOAD_WEIGHTS:
        print("Enter one of the listed payload weights.")
        return None

    return payload


def configure_logging(log_path: Path):
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.captureWarnings(True)
    return [file_handler, console_handler]



def plot_telemetry(telemetry_path: Path, output_path: Path):
    # setup files 
    with telemetry_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) < 2:
        LOGGER.warning("Not enough telemetry samples to plot.")
        return

    # get column data 
    def column_data(name: str):
        data_val = []
        for row in rows:
            try:
                data_val.append(float(row[name]))
            except (TypeError, ValueError):
                data_val.append(float('nan'))
        return data_val 

    # get timestamps 
    timestamp = column_data("elapsed_seconds")
    fig, ((ax_speed, ax_battery), (ax_alt, ax_temp)) = plt.subplots(
        2, 2, figsize=(12, 8), sharex=True
    )
    
    ax_speed.plot(timestamp, column_data("speed_x_cm_s"), label="x", linewidth=1)
    ax_speed.plot(timestamp, column_data("speed_y_cm_s"), label="y", linewidth=1)
    ax_speed.plot(timestamp, column_data("speed_z_cm_s"), label="z", linewidth=1)
    ax_speed.plot(timestamp, column_data("speed_3d_cm_s"), label="3D", linewidth=2, color="black")
    ax_speed.set_title("Speed")
    ax_speed.set_ylabel("cm/s")
    ax_speed.legend()

    ax_battery.plot(timestamp, column_data("battery_pct"), color="green")
    ax_battery.axhline(MIN_BATTERY_THRESHOLD, color="red", linestyle="--",
                       label=f"landing threshold ({MIN_BATTERY_THRESHOLD}%)")
    ax_battery.set_title("Battery")
    ax_battery.set_ylabel("%")
    ax_battery.legend()

    ax_alt.plot(timestamp, column_data("altitude_cm"), label="altitude (h)")
    ax_alt.plot(timestamp, column_data("tof_distance_cm"), label="ToF distance", linestyle="--")
    ax_alt.set_title("Altitude")
    ax_alt.set_ylabel("cm")
    ax_alt.legend()

    ax_temp.plot(timestamp, column_data("temperature_low_c"), label="low")
    ax_temp.plot(timestamp, column_data("temperature_high_c"), label="high")
    ax_temp.plot(timestamp, column_data("temperature_avg_c"), label="avg", color="black")
    ax_temp.set_title("Temperature")
    ax_temp.set_ylabel("deg C")
    ax_temp.legend()

    for ax in (ax_alt, ax_temp):
        ax.set_xlabel("elapsed time since takeoff (s)")
    for ax in (ax_speed, ax_battery, ax_alt, ax_temp):
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Flight test - payload {rows[0]['payload_weight_g']} g")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Flight plots saved to: {output_path}")

def run_payload_test(payload_weight_g : int):

    # Setup the log files and folder
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = LOG_DIR / f"flightTest_{payload_weight_g}g_{run_id}"# separate folder for each test run
    run_directory.mkdir()
    telemetry_path = run_directory / "telemetry.csv"
    events_path = run_directory / "warnings_errors.log"
    summary_path = run_directory / "summary.json"
    log_handler = configure_logging(events_path)

    # Setup local variables
    tello: Tello | None = None
    connected = False
    airborne = False
    flight_start: float | None = None
    flight_start_timestamp: str | None = None
    flight_end: float | None = None
    battery_start: int | None = None
    battery_start_timestamp: str | None = None
    battery_at_land: int | None = None
    battery_at_land_timestamp: str | None = None
    landing_command_succeeded = False
    landing_reason: str | None = None
    cleanup_landing_reason = "test shutdown"

    max_speed_cm_s: float | None = None
    max_altitude_cm: float | None = None
    max_temperature_c: float | None = None
    temperature_sum_c = 0.0
    temperature_sample_count = 0
    estimated_distance_cm = 0.0
    previous_speed_cm_s: float | None = None
    previous_sample_time: float | None = None

    movement_distance_cm = 30 
    rotation_degree = 10

    # local setup for common logs
    def land_handler(reason : str):
        nonlocal airborne, flight_end, battery_at_land, battery_at_land_timestamp
        nonlocal landing_command_succeeded, landing_reason

        if not airborne or tello is None:
            return
        landing_reason = reason
        LOGGER.info("Landing requested: %s.", reason)
        print(f"Landing: {reason}")
        tello.land()
        airborne = False
        landing_command_succeeded = True
        flight_end = time.monotonic()
        battery_at_land_timestamp = local_timestamp()
        battery_at_land = safe_get(tello.get_battery)
        if battery_at_land is None:
            LOGGER.warning("Battery-at-landing telemetry was unavailable.")

    # collect real telemetry data from the Tello drone
    # returns one CSV row (dict) and updates the running summary values

    def get_tello_telemetry(tello, payload_weight_g):
        nonlocal max_speed_cm_s, previous_speed_cm_s, previous_sample_time
        nonlocal estimated_distance_cm, max_altitude_cm
        nonlocal temperature_sum_c, temperature_sample_count, max_temperature_c

        now = time.monotonic()
        battery_pct = safe_get(tello.get_battery)
        speed_x_cm_s = safe_get(tello.get_speed_x)
        speed_y_cm_s = safe_get(tello.get_speed_y)
        speed_z_cm_s = safe_get(tello.get_speed_z)
        speed_3d = None
        if speed_x_cm_s is not None and speed_y_cm_s is not None and speed_z_cm_s is not None:
            speed_3d = math.sqrt(speed_x_cm_s**2 + speed_y_cm_s**2 + speed_z_cm_s**2)
            max_speed_cm_s = (
                speed_3d if max_speed_cm_s is None
                else max(max_speed_cm_s, speed_3d)
            )
            if previous_speed_cm_s is not None and previous_sample_time is not None:
                elapsed = now - previous_sample_time
                estimated_distance_cm += (
                    (previous_speed_cm_s + speed_3d) / 2 * elapsed
                )
            previous_speed_cm_s = speed_3d
            previous_sample_time = now
        else:
            previous_speed_cm_s = None
            previous_sample_time = None
        altitude_cm = safe_get(tello.get_height)    #reference from the point of take off
        if altitude_cm is not None:
            max_altitude_cm = (
                altitude_cm if max_altitude_cm is None
                else max(max_altitude_cm, altitude_cm)
            )
        tof_distance_cm = safe_get(tello.get_distance_tof)  #instance height of drone measure by sensor
        temperature_low = safe_get(tello.get_lowest_temperature)
        temperature_high = safe_get(tello.get_highest_temperature)

        temperature_avg = None
        if temperature_low is not None and temperature_high is not None:
            temperature_avg = (temperature_low + temperature_high) / 2
            temperature_sum_c += temperature_avg
            temperature_sample_count += 1
        if temperature_high is not None:
            max_temperature_c = (
                temperature_high if max_temperature_c is None
                else max(max_temperature_c, temperature_high)
            )

        elapsed_seconds = now - flight_start if flight_start is not None else 0.0

        # csv writes None as an empty cell
        return {
            "timestamp": local_timestamp(),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "payload_weight_g": payload_weight_g,
            "battery_pct": battery_pct,
            "vps_status": VPS_STATUS,
            "speed_x_cm_s": speed_x_cm_s,
            "speed_y_cm_s": speed_y_cm_s,
            "speed_z_cm_s": speed_z_cm_s,
            "speed_3d_cm_s": round(speed_3d, 2) if speed_3d is not None else None,
            "altitude_cm": altitude_cm,
            "tof_distance_cm": tof_distance_cm,
            "temperature_low_c": temperature_low,
            "temperature_high_c": temperature_high,
            "temperature_avg_c": (
                round(temperature_avg, 2) if temperature_avg is not None else None
            ),
            "estimated_distance_cm": round(estimated_distance_cm, 2),
        }


    #  Run the test cycle
    # Open file
    try:
        with telemetry_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=TELEMETRY_FIELDS)
            writer.writeheader()
            csv_file.flush()
            print(f"This run's files will be saved in: {run_directory}")

            # Connect tello
            try:
                tello = Tello()
                print("Connecting to Tello...")
                tello.connect()
                connected = True
                print("Tello connected.")

                # Read the battery status
                # if battery percentage is missing
                    # keep retrying until the pre-flight deadline
                preflight_deadline = time.monotonic() + PRE_FLIGHT_DEADLINE_THRESHOLD
                battery_status = safe_get(tello.get_battery)
                while battery_status is None:
                    if time.monotonic() >= preflight_deadline:
                        raise RuntimeError("No valid battery telemetry received; refusing takeoff.")
                    time.sleep(0.1)
                    battery_status = safe_get(tello.get_battery)

                if battery_status < MIN_BATTERY_THRESHOLD:
                    print("Battery too low for flight test.")
                    return  # the outer finally closes the Tello connection

                print("Battery:", battery_status, "%")
                print("Ready for flight test...")
                print('Press PGUP for take off or Q to abort test...!')

                # Run the pre-flight loop until take off or abort
                while True:
                    if keyboard.is_pressed("q"):
                        print("Test cancelled before takeoff.")
                        return
                    # if take off signal received
                    if keyboard.is_pressed('pgup'):
                        # 1. check for battery percentage
                        battery_start = safe_get(tello.get_battery)
                        battery_start_timestamp = local_timestamp()
                        if battery_start is None:
                            raise RuntimeError("Battery telemetry became unavailable before takeoff.")
                        if battery_start < MIN_BATTERY_THRESHOLD:
                            raise RuntimeError(
                                f"Battery is {battery_start}%; refusing takeoff below "
                                f"{MIN_BATTERY_THRESHOLD}%."
                            )
                        # 2. take off
                        takeoff_started = time.monotonic()
                        takeoff_timestamp = local_timestamp()
                        print("Taking off...")
                        tello.takeoff()

                        # 3. change flight status and note take off time
                        airborne = True
                        print(" Status : Airborne (Hovering)...!")
                        flight_start = takeoff_started
                        flight_start_timestamp = takeoff_timestamp
                        break
                    time.sleep(0.05)

                # Run the post flight loop at sample interval

                next_sample = time.monotonic()
                next_keepalive = time.monotonic() + KEEP_ALIVE_INTERVAL
                while airborne:
                    time.sleep(0.1)
                    #1. check for battery
                    battery_status = safe_get(tello.get_battery)
                    if battery_status is None:
                        LOGGER.error(
                            "Battery telemetry missing in flight; initiating failsafe landing."
                        )
                        land_handler("battery telemetry unavailable")
                        break
                    if battery_status < MIN_BATTERY_THRESHOLD:
                        LOGGER.warning(
                            "Battery fell to %s%% (below %s%%); initiating automatic landing.",
                            battery_status,
                            MIN_BATTERY_THRESHOLD,
                        )
                        land_handler(f"battery at {battery_status}%")
                        break

                    # 2. check for flight abort instruction
                    if keyboard.is_pressed("page down") or keyboard.is_pressed("q"):
                        land_handler("operator command")
                        break

                    # 2.1 Check for manual control
                    try:
                        if keyboard.is_pressed('up'):
                            tello.move_forward(movement_distance_cm)
                        elif keyboard.is_pressed('down'):
                            tello.move_back(movement_distance_cm)
                        elif keyboard.is_pressed('left'):
                            tello.move_left(movement_distance_cm)
                        elif keyboard.is_pressed('right'):
                            tello.move_right(movement_distance_cm)
                        elif keyboard.is_pressed('w'):
                            tello.move_up(movement_distance_cm)
                        elif keyboard.is_pressed('s'):
                            tello.move_down(movement_distance_cm)
                        elif keyboard.is_pressed('d'):
                            tello.rotate_clockwise(movement_distance_cm)
                        elif keyboard.is_pressed('a'):
                            tello.rotate_counter_clockwise(movement_distance_cm)
                    except TelloException:
                        LOGGER.warning("Manual command failed; continuing flight.", exc_info=True)
                    

                    # 3. Keep alive signal at certain interval
                        # Resets the Tello's ~15 s no-command auto-land timer.
                        # Zero-velocity RC = "hold position"
                    now = time.monotonic()
                    if now >= next_keepalive:
                        try:
                            tello.send_rc_control(0, 0, 0, 0)
                        except Exception:
                            # Do not abort the test on a single missed keepalive;
                            # log it so a later auto-landing can be explained.
                            LOGGER.warning(
                                "Keepalive command failed; the drone may auto-land "
                                "if this keeps happening.",
                                exc_info=True,
                            )
                        next_keepalive = time.monotonic() + KEEP_ALIVE_INTERVAL
                    now = time.monotonic()
                    if now < next_sample:
                        continue
                    next_sample = now + SAMPLE_INTERVAL

                    # 4. Collect one telemetry sample and write it to the CSV
                    writer.writerow(get_tello_telemetry(tello, payload_weight_g))
                    csv_file.flush()

            except KeyboardInterrupt:
                cleanup_landing_reason = "operator interrupt"
                LOGGER.warning("Operator interrupted the test; initiating landing if airborne.")
            except Exception:
                cleanup_landing_reason = "exception cleanup"
                LOGGER.exception("Payload test failed.")
                raise
            finally:
                if airborne:
                    try:
                        land_handler(cleanup_landing_reason)
                    except Exception:
                        LOGGER.exception("Landing command failed.")

                if flight_start is not None:
                    end_time = flight_end if flight_end is not None else time.monotonic()
                    duration_seconds = end_time - flight_start
                    summary = {
                        "flight_start_timestamp": flight_start_timestamp,
                        "flight_end_timestamp": (
                            local_timestamp() if flight_end is not None else None
                        ),
                        "payload_weight_g": payload_weight_g,
                        "tof_second": round(duration_seconds, 2),
                        "battery_start_pct": battery_start,
                        "battery_at_landing_pct": battery_at_land,
                        "vps_status": VPS_STATUS,
                        "max_speed_cm_s": (
                            round(max_speed_cm_s, 2) if max_speed_cm_s is not None else None
                        ),
                        "max_altitude_cm": (
                            round(max_altitude_cm, 2) if max_altitude_cm is not None else None
                        ),
                        "estimated_distance_cm": round(estimated_distance_cm, 2),
                        "max_temperature_c": (
                            round(max_temperature_c, 2)
                            if max_temperature_c is not None
                            else None
                        ),
                        "average_temperature_c": (
                            round(temperature_sum_c / temperature_sample_count, 2)
                            if temperature_sample_count
                            else None
                        ),
                        "landing_command_succeeded": landing_command_succeeded,
                        "landing_reason": landing_reason,
                        "telemetry_csv": str(telemetry_path),
                        "warnings_errors_log": str(events_path),
                        "distance_note": (
                            "Estimated by integrating Tello velocity telemetry; "
                            "drift and sensor error are possible."
                        ),
                    }
                    with summary_path.open("w", encoding="utf-8") as summary_file:
                        json.dump(summary, summary_file, indent=2)
                        summary_file.write("\n")
                    print(f"Flight summary saved to: {summary_path}")

                    # Plot telemetry
                    try:
                        plot_telemetry(telemetry_path, run_directory/"flight_plot.png")
                    except Exception:
                        LOGGER.exception("Could not generate flight plots.")

    finally:
        if tello is not None and connected:
            try:
                tello.end()
            except Exception:
                LOGGER.exception("Error while closing the Tello connection.")
        root_logger = logging.getLogger()
        for handler in log_handler:
            root_logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    while True:
        payload_weight_g = choose_payload()
        if payload_weight_g is not None:
            break
    run_payload_test(payload_weight_g)