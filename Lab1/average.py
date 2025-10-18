#!/usr/bin/env python3
import os
import re
import statistics
import matplotlib.pyplot as plt

# --- Regex to extract runtime ---
TIME_PATTERN = re.compile(r"Total time\s*=\s*([\d.]+)\s*seconds")

# --- Base directory ---
BASE_DIR = "metrics"

# --- Data structure ---
# runtimes[(size, opt_level, method, exe_name)] = [times...]
runtimes = {}

# --- Walk through metrics ---
for size_dir in sorted(os.listdir(BASE_DIR)):
    size_path = os.path.join(BASE_DIR, size_dir)
    if not os.path.isdir(size_path):
        continue

    for opt_dir in sorted(os.listdir(size_path)):
        opt_path = os.path.join(size_path, opt_dir)
        if not os.path.isdir(opt_path):
            continue

        for method_dir in sorted(os.listdir(opt_path)):
            method_path = os.path.join(opt_path, method_dir)
            if not os.path.isdir(method_path):
                continue

            for root, _, files in os.walk(method_path):
                for file in files:
                    if not file.endswith(".log"):
                        continue

                    path = os.path.join(root, file)
                    exe_name = re.sub(r'(_output|_run\d+)?\.log$', '', file)

                    with open(path, "r") as f:
                        content = f.read()
                        match = TIME_PATTERN.search(content)
                        if match:
                            time = float(match.group(1))
                            key = (size_dir, opt_dir, method_dir, exe_name)
                            runtimes.setdefault(key, []).append(time)

# --- Markdown Header ---
exe_col_width = 50
runs_col_width = 6
avg_col_width = 12
stdev_col_width = 12

for (size, opt, method, _) in sorted({(s, o, m, '') for (s, o, m, _) in runtimes}):
    print(f"\n## 📊 Results for {size} | {opt} | {method}\n")
    print(f"| {'Executable':<{exe_col_width}} | {'#Runs':>{runs_col_width}} | {'Average (s)':>{avg_col_width}} | {'Std Dev (s)':>{stdev_col_width}} | Min / Max outliers  |")
    print(f"| {'-'*exe_col_width} | {'-'*runs_col_width} | {'-'*avg_col_width} | {'-'*stdev_col_width} | ------------------- |")

    stats = []
    for (s, o, m, exe_name), times in sorted(runtimes.items()):
        if (s, o, m) != (size, opt, method):
            continue

        n = len(times)
        if n > 2:
            sorted_times = sorted(times)
            min_outlier = sorted_times[0]
            max_outlier = sorted_times[-1]
            filtered = sorted_times[1:-1]
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

    # --- Plot ---
    if not stats:
        continue

    labels = [s[0] for s in stats]
    averages = [s[1] for s in stats]
    stdevs = [s[2] for s in stats]

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.6)))

    y_pos = range(len(labels))

    bars = ax.barh(y_pos, averages, color="#0096FF", edgecolor="white", height=0.6, label="Average runtime")
    ax.barh(y_pos, stdevs, left=0, color="#C04000", edgecolor="white", height=0.6, alpha=0.6, label="Std Dev")

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(averages) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{averages[i]:.3f}s ± {stdevs[i]:.3f}",
                va='center', ha='left', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Average Runtime — {size}x{size} (-{opt}) ")
    ax.legend()
    plt.tight_layout()

    # Save one plot per combination
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/average_{size}_{opt}_{method}.png", dpi=300)
    plt.close(fig)
