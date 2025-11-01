import os
import re
import matplotlib.pyplot as plt
import numpy as np

METRICS_DIR = "metrics"
THREAD_DIR_PREFIX = "par"
SEQ_DIR = "seq"

# --- Helper function to extract computation time from a log file ---
def extract_runtime(logfile):
    with open(logfile, "r") as f:
        for line in f:
            match = re.search(r"Computation timing\s*=\s*([\d.]+)\s*sec", line)
            if match:
                return float(match.group(1))
    return None

# --- Gather runtimes ---
runtimes = {}

# Sequential run
seq_logs_dir = os.path.join(METRICS_DIR, SEQ_DIR)
if os.path.exists(seq_logs_dir):
    seq_times = []
    for fname in os.listdir(seq_logs_dir):
        if fname.endswith(".log"):
            t = extract_runtime(os.path.join(seq_logs_dir, fname))
            if t is not None:
                seq_times.append(t)
    if seq_times:
        runtimes["seq"] = np.mean(seq_times)

# Parallel runs
metrics_par_dir = os.path.join(METRICS_DIR, THREAD_DIR_PREFIX)
if os.path.exists(metrics_par_dir):
    for thread_folder in os.listdir(metrics_par_dir):
        thread_path = os.path.join(metrics_par_dir, thread_folder)
        if os.path.isdir(thread_path):
            # Parse number of threads from folder name, e.g., "4-threads"
            match = re.match(r"(\d+)-threads", thread_folder)
            if match:
                threads = int(match.group(1))
                times = []
                for fname in os.listdir(thread_path):
                    if fname.endswith(".log"):
                        t = extract_runtime(os.path.join(thread_path, fname))
                        if t is not None:
                            times.append(t)
                if times:
                    runtimes[threads] = np.mean(times)

# --- Prepare data for plotting ---
sorted_keys = ["seq"] + sorted([k for k in runtimes.keys() if k != "seq"])
x_labels = [str(k) for k in sorted_keys]
y_values = [runtimes[k] for k in sorted_keys]

# --- Plot ---
plt.figure(figsize=(10,6))
plt.plot(x_labels, y_values, marker='o', linestyle='-', color='b')
plt.xlabel("Threads")
plt.ylabel("Average Runtime (s)")
plt.title("Average Runtime vs Threads for K-Means")
plt.grid(True)
plt.tight_layout()
plt.show()
