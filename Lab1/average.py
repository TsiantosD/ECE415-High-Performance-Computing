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
        if file.endswith(".log"):
            path = os.path.join(root, file)
            # Extract executable name (remove suffixes like _output or _runX)
            exe_name = re.sub(r'(_output|_run\d+)?\.log$', '', file)
            with open(path, "r") as f:
                content = f.read()
                match = TIME_PATTERN.search(content)
                if match:
                    time = float(match.group(1))
                    runtimes.setdefault(exe_name, []).append(time)

# --- Markdown table ---
exe_col_width = 60
runs_col_width = 6
avg_col_width = 12
stdev_col_width = 12

# Header
print(f"| {'Executable':<{exe_col_width}} | {'#Runs':>{runs_col_width}} | {'Average (s)':>{avg_col_width}} | {'Std Dev (s)':>{stdev_col_width}} | Min / Max outliers  |")
print(f"| {'-'*exe_col_width} | {'-'*runs_col_width} | {'-'*avg_col_width} | {'-'*stdev_col_width} | ------------------- |")

# --- Compute stats ---
stats = []
for exe_name, times in sorted(runtimes.items(), key=lambda x: int(x[0].split("_")[0])):
    n = len(times)
    if n > 2:
        sorted_times = sorted(times)
        min_outlier = sorted_times[0]
        max_outlier = sorted_times[-1]
        filtered = sorted_times[1:-1]  # exclude min/max
    else:
        min_outlier = "-"
        max_outlier = "-"
        filtered = times

    avg = statistics.mean(filtered)
    stdev = statistics.stdev(filtered) if len(filtered) > 1 else 0.0

    min_str = f"{min_outlier:.6f}" if isinstance(min_outlier, float) else min_outlier
    max_str = f"{max_outlier:.6f}" if isinstance(max_outlier, float) else max_outlier

    print(f"| {exe_name:<{exe_col_width}} | {n:>{runs_col_width}} | {avg:>{avg_col_width}.5f} | {stdev:>{stdev_col_width}.5f} | {min_str} / {max_str} |")

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

plt.savefig("average.png", dpi=300)
