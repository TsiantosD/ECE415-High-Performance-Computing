import os
import re
import pandas as pd
import math

# ====== Configuration ======
OUTPUT_BASE_DIR = "output"
TARGET_KERNEL_LENGTH = 33
OUTPUT_CSV = "results.csv"

# Filename format: out_<kernel>_<image>_repX
FILENAME_REGEX = re.compile(r'out_(\d+)_(\d+)_rep(\d+)$')

# Log parsing
TIME_GPU_REGEX = re.compile(r'Time in GPU: ([\d\.]+)')
TIME_CPU_REGEX = re.compile(r'Time in CPU: ([\d\.]+)')


def parse_log_file(filepath):
    """Reads a log file and extracts GPU and CPU time."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        gpu_match = TIME_GPU_REGEX.search(content)
        cpu_match = TIME_CPU_REGEX.search(content)

        if not gpu_match or not cpu_match:
            return None, None

        return float(gpu_match.group(1)), float(cpu_match.group(1))

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None


def collect_all_data():
    """Walks through output/ and collects all runs."""
    rows = []

    for step in os.listdir(OUTPUT_BASE_DIR):
        step_path = os.path.join(OUTPUT_BASE_DIR, step)
        if not os.path.isdir(step_path):
            continue

        # Example:
        # output/step6/20251124_021628/out_33_1024_rep1
        for timestamp_folder in os.listdir(step_path):
            timestamp_path = os.path.join(step_path, timestamp_folder)
            if not os.path.isdir(timestamp_path):
                continue

            for filename in os.listdir(timestamp_path):
                match = FILENAME_REGEX.match(filename)
                if not match:
                    continue

                kernel = int(match.group(1))
                image = int(match.group(2))
                rep = int(match.group(3))

                if kernel != TARGET_KERNEL_LENGTH:
                    continue

                filepath = os.path.join(timestamp_path, filename)
                gpu_time, cpu_time = parse_log_file(filepath)

                if gpu_time is None or cpu_time is None:
                    continue

                rows.append({
                    "step": step,
                    "kernel_length": kernel,
                    "image_size": image,
                    "rep": rep,
                    "gpu_time": gpu_time,
                    "cpu_time": cpu_time
                })

    return pd.DataFrame(rows)


def compute_statistics(df):
    """Aggregates mean/std and computes speedup."""
    df_agg = (
        df.groupby(["step", "kernel_length", "image_size"])
          .agg(
              gpu_mean=("gpu_time", "mean"),
              gpu_std=("gpu_time", "std"),
              cpu_mean=("cpu_time", "mean"),
              cpu_std=("cpu_time", "std"),
              count=("gpu_time", "count")
          )
          .reset_index()
    )

    # Replace NaN std (only 1 sample)
    df_agg["gpu_std"] = df_agg["gpu_std"].fillna(0)
    df_agg["cpu_std"] = df_agg["cpu_std"].fillna(0)

    # Speedup = CPU / GPU
    df_agg["speedup_mean"] = df_agg["cpu_mean"] / df_agg["gpu_mean"]

    # Error propagation for division:
    # σ_s = s * sqrt((σ_cpu/cpu)^2 + (σ_gpu/gpu)^2)
    df_agg["speedup_std"] = (
        df_agg["speedup_mean"]
        * ((df_agg["cpu_std"] / df_agg["cpu_mean"])**2
           + (df_agg["gpu_std"] / df_agg["gpu_mean"])**2)**0.5
    )

    df_agg["speedup_std"] = df_agg["speedup_std"].fillna(0)

    return df_agg


def main():
    df = collect_all_data()

    if df.empty:
        print("No data found. Check directory structure.")
        return

    df_stats = compute_statistics(df)

    df_stats.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV → {OUTPUT_CSV}")
    print(df_stats)


if __name__ == "__main__":
    main()
