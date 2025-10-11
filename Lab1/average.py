#!/usr/bin/env python3
import os
import re
import statistics

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

# --- Markdown table ---
exe_col_width = 30
runs_col_width = 6
avg_col_width = 12
stdev_col_width = 12

# Header
print(f"| {'Executable':<{exe_col_width}} | {'#Runs':>{runs_col_width}} | {'Average (s)':>{avg_col_width}} | {'Std Dev (s)':>{stdev_col_width}} | Min / Max outliers  |")
print(f"| {'-'*exe_col_width} | {'-'*runs_col_width} | {'-'*avg_col_width} | {'-'*stdev_col_width} | ------------------- |")

# Rows
for exe_name, times in sorted(runtimes.items(), key=lambda x: int(x[0].split("_")[0])):
    n = len(times)
    if n > 2:
        sorted_times = sorted(times)
        min_outlier = sorted_times[0]
        max_outlier = sorted_times[-1]
        filtered_times = sorted_times[1:-1]
        avg = statistics.mean(filtered_times)
        stdev = statistics.stdev(filtered_times) if len(filtered_times) > 1 else 0.0
    else:
        min_outlier = "-"
        max_outlier = "-"
        avg = statistics.mean(times)
        stdev = statistics.stdev(times) if n > 1 else 0.0

    # Format numbers with fixed decimals
    min_outlier_str = f"{min_outlier:.6f}" if isinstance(min_outlier, float) else min_outlier
    max_outlier_str = f"{max_outlier:.6f}" if isinstance(max_outlier, float) else max_outlier

    print(f"| {exe_name:<{exe_col_width}} | {n:>{runs_col_width}} | {avg:>{avg_col_width}.5f} | {stdev:>{stdev_col_width}.5f} | {min_outlier_str} / {max_outlier_str} |")

print(f"\nProcessed {sum(len(v) for v in runtimes.values())} runs.")
