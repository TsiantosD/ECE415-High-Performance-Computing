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
runtimes = {}  # store {threads: {"mean": value, "std": value}}

# Sequential run
seq_logs_dir = os.path.join(METRICS_DIR, SEQ_DIR)
if os.path.exists(seq_logs_dir):
    seq_times = []
    for fname in os.listdir(seq_logs_dir):
        if fname.endswith(".log"):
            t = extract_runtime(os.path.join(seq_logs_dir, fname))
            if t is not None:
                seq_times.append(t)
    if len(seq_times) >= 3:
        seq_times.remove(max(seq_times))
        seq_times.remove(min(seq_times))
    if seq_times:
        runtimes["seq"] = {
            "mean": np.mean(seq_times),
            "std": np.std(seq_times),
        }

# Parallel runs
metrics_par_dir = os.path.join(METRICS_DIR, THREAD_DIR_PREFIX)
if os.path.exists(metrics_par_dir):
    for thread_folder in os.listdir(metrics_par_dir):
        thread_path = os.path.join(metrics_par_dir, thread_folder)
        if os.path.isdir(thread_path):
            match = re.match(r"(\d+)-threads", thread_folder)
            if match:
                threads = int(match.group(1))
                times = []
                for fname in os.listdir(thread_path):
                    if fname.endswith(".log"):
                        t = extract_runtime(os.path.join(thread_path, fname))
                        if t is not None:
                            times.append(t)
                if len(times) >= 3:
                    times.remove(max(times))
                    times.remove(min(times))
                if times:
                    runtimes[threads] = {
                        "mean": np.mean(times),
                        "std": np.std(times),
                    }

# --- Prepare data for plotting ---
sorted_keys = ["seq"] + sorted([k for k in runtimes.keys() if k != "seq"])
x_labels = [str(k) for k in sorted_keys]
means = [runtimes[k]["mean"] for k in sorted_keys]
stds = [runtimes[k]["std"] for k in sorted_keys]

# --- Compute speedup compared to sequential ---
seq_time = runtimes["seq"]["mean"] if "seq" in runtimes else None
speedups = [seq_time / m if seq_time and m > 0 else 1.0 for m in means]

# --- Plot ---
plt.figure(figsize=(10,6))
plt.errorbar(x_labels, means, yerr=stds, fmt='o-', capsize=5, color='b', ecolor='red')
plt.xlabel("#Threads")
plt.ylabel("Runtime (s)")
plt.title("Average Runtime per set of Threads")
plt.grid(True)
plt.margins(x=0.1, y=0.2)

# --- Annotate each point ---
for i, (x, y, s, sd) in enumerate(zip(x_labels, means, speedups, stds)):
    # Runtime
    runtime = f"{y:.3f}s ±{sd:.3f}\n"
    plt.text(i, y + sd * 1.5, runtime, ha='center', va='bottom', fontsize=10, color='black')

    # Speedup
    speedup = f"{s:.2f}×"
    plt.text(i, y - sd * 3, speedup, ha='center', va='top', fontsize=12, color='black')

plt.tight_layout()
plt.savefig('runtimes.png')
plt.show()

