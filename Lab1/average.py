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
            # Extract executable name (e.g., 1_sobel_orig)
            exe_name = file.replace("_output.log", "")
            # Read and extract time
            with open(path, "r") as f:
                content = f.read()
                match = TIME_PATTERN.search(content)
                if match:
                    time = float(match.group(1))
                    runtimes.setdefault(exe_name, []).append(time)

# Compute stats
print(f"{'Executable':30} {'Average (s)':>15} {'Std Dev (s)':>15}")
print("-" * 60)

for exe_name, times in sorted(runtimes.items()):
    avg = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"{exe_name:30} {avg:15.5f} {stdev:15.5f}")

print("\nProcessed", sum(len(v) for v in runtimes.values()), "runs.")
