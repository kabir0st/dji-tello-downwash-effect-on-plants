import csv
import matplotlib.pyplot as plt
import os

PAYLOAD_WEIGHTS = [0, 20, 40, 60]

print("Available payload weights:", PAYLOAD_WEIGHTS)
payload = int(input("Enter payload weight to visualize: "))

print("\nSelect test type:")
print("1 - Synthetic")
print("2 - Real")
test_mode = input("Enter 1 or 2: ")

if test_mode == "1":
    test_type = "synthetic"
else:
    test_type = "real"


if payload not in PAYLOAD_WEIGHTS:
    print("Invalid payload weight.")
    exit()

LOG_DIR = "logs"

run_prefix = f"payload_{payload}g_{test_type}_"

run_folders = sorted(
    folder for folder in os.listdir(LOG_DIR)
    if folder.startswith(run_prefix)
    and os.path.isdir(os.path.join(LOG_DIR, folder))
)

if not run_folders:
    print("No matching flight test runs found.")
    exit()

latest_run = run_folders[-1]
RUN_DIR = os.path.join(LOG_DIR, latest_run)

CSV_FILE = os.path.join(RUN_DIR, "telemetry.csv")

print("Using flight test run:", latest_run)

altitudes = []
batteries = []
temperatures = []

with open(CSV_FILE, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        altitude = int(row["altitude_cm"])
        altitudes.append(altitude)

        battery = int(row["battery_pct"])
        batteries.append(battery)

        temperature = float(row["temperature_c"])
        temperatures.append(temperature)

print("Altitude data:", altitudes)
print("Battery data:", batteries)
print("Temperature data:", temperatures)

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

axes[0].plot(altitudes)
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Altitude (cm)")
axes[0].set_title(f"Altitude during flight test - {payload}g payload")

axes[1].plot(batteries)
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Battery (%)")
axes[1].set_title(f"Battery during flight test - {payload}g payload")

axes[2].plot(temperatures)
axes[2].set_xlabel("Sample")
axes[2].set_ylabel("Temperature (°C)")
axes[2].set_title(f"Temperature during flight test - {payload}g payload")

plt.tight_layout(h_pad=2.5)
plt.savefig(
    os.path.join(
        RUN_DIR,
        "visual_log.png"
    )
)
plt.show()