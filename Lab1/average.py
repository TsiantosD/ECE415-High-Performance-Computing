#!/usr/bin/env python3
import os
import re
import statistics
import matplotlib.pyplot as plt

# Path to normal metrics
BASE_DIR = os.path.join("metrics", "normal")

# Regex to extract runtime
TIME_PATTERN = re.compile(r"Total time\s*=\s*([\d.]+)\s*seconds")

# Data structure: {exe_name: [times]}
runtimes = {}

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith("_output.log"):
            path = os.path.join(root, file)
            exe_name = file.replace("_output.log", "")
            with open(path, "r") as f:
                content = f.read()
                match = TIME_PATTERN.search(content)
                if match:
                    time = float(match.group(1))
                    runtimes.setdefault(exe_name, []).append(time)

# --- Compute stats ---
stats = []
for exe_name, times in sorted(runtimes.items(), key=lambda x: int(x[0].split("_")[0])):
    n = len(times)
    if n > 2:
        sorted_times = sorted(times)
        filtered = sorted_times[1:-1]  # exclude min/max
    else:
        filtered = times
    avg = statistics.mean(filtered)
    stdev = statistics.stdev(filtered) if len(filtered) > 1 else 0.0
    stats.append((exe_name, avg, stdev))

# --- Plotting ---
labels = [' '.join(s[0].split('_')[2:]).capitalize() for s in stats]
averages = [s[1] for s in stats]
stdevs = [s[2] for s in stats]

fig, ax = plt.subplots(figsize=(10, 5))

# Parameters for bar thickness
bar_height = 0.6
overlay_height = 0.6

y_pos = range(len(labels))

# Draw main bars (average runtime)
ax.barh(y_pos, averages, color="#0096FF", edgecolor="white", height=bar_height, label="Average runtime")

# Draw smaller bars on top for standard deviation
ax.barh([y for y in y_pos], stdevs, left=0, color="#C04000", edgecolor="white", height=overlay_height, label="Std Dev")

# Labels and formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Time (seconds)")
ax.set_title("Average Runtime")
ax.legend()
# plt.figtext(0.5, 0.01, "No compiler optimizations", wrap=True, fontsize=8, ha='center')
plt.tight_layout()

plt.savefig("metrics_summary.png", dpi=300)
