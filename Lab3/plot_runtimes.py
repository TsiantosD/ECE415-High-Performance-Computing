import os
import re
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Default Configuration ---
DEFAULT_OUTPUT_BASE_DIR = 'metrics'
PLOTS_DIR = 'plots'
CSV_DIR = os.path.join(PLOTS_DIR, 'csv')
TARGET_KERNEL_LENGTH = 33  # actual kernel length (2*r + 1)

# Regex for parsing logs
FILENAME_REGEX = re.compile(r'.*-(?:run_\d+)\.log$')
TIME_GPU_REGEX = re.compile(r'Time in GPU: ([\d\.]+)')
TIME_CPU_REGEX = re.compile(r'Time in CPU: ([\d\.]+)')


# ---------------------------------------------------------
#   PARSING FUNCTIONS – EXPORT TO CSV ONLY
# ---------------------------------------------------------
def parse_log_file(filepath):
    """Extracts GPU and CPU times from a log file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        gpu_time_match = TIME_GPU_REGEX.search(content)
        cpu_time_match = TIME_CPU_REGEX.search(content)
        gpu_time = float(gpu_time_match.group(1)) if gpu_time_match else None
        cpu_time = float(cpu_time_match.group(1)) if cpu_time_match else None
        return gpu_time, cpu_time
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None


def collect_data(output_base_dir):
    """Collects parsed log entries into a flat list."""
    data = []

    if not os.path.isdir(output_base_dir):
        print(f"Error: Output base directory '{output_base_dir}' does not exist.")
        return data

    subdirs = [d for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))]

    # User selects run folder
    if not subdirs:
        selected_dir = ""
        print(f"No subdirectories found in '{output_base_dir}'. Proceeding with base directory.")
    elif len(subdirs) == 1:
        selected_dir = subdirs[0]
    else:
        print("Multiple metrics subdirectories found. Select one to parse:")
        for idx, d in enumerate(subdirs, start=1):
            print(f"{idx}) {d}")
        while True:
            try:
                choice = int(input("Enter number: "))
                if 1 <= choice <= len(subdirs):
                    selected_dir = subdirs[choice - 1]
                    break
            except ValueError:
                pass
            print("Invalid selection. Try again.")

    if selected_dir:
        output_base_dir = os.path.join(output_base_dir, selected_dir)

    print(f"Selected metrics folder: {output_base_dir}")

    # Iterate configs
    for config_dir in os.listdir(output_base_dir):
        config_path = os.path.join(output_base_dir, config_dir)

        if not os.path.isdir(config_path):
            continue

        match = re.match(r'(\d+)_(\d+)(-nofmad)?(-dbl)?', config_dir)
        if not match:
            continue

        filter_radius = int(match.group(1))
        image_size = int(match.group(2))
        nofmad_flag = bool(match.group(3))
        dbl_flag = bool(match.group(4))

        src_folder_placeholder = os.path.basename(output_base_dir)

        for logfile in os.listdir(config_path):
            if not FILENAME_REGEX.match(logfile):
                continue

            filepath = os.path.join(config_path, logfile)
            gpu_time, cpu_time = parse_log_file(filepath)

            if gpu_time is not None and cpu_time is not None:
                data.append({
                    'src_folder': src_folder_placeholder,
                    'filter_radius': filter_radius,
                    'image_size': image_size,
                    'gpu_time': gpu_time,
                    'cpu_time': cpu_time,
                    'nofmad': nofmad_flag,
                    'dbl': dbl_flag,
                    'logfile': logfile
                })
    return data


def export_to_csv(df):
    """Exports aggregated CSVs per config label."""
    os.makedirs(CSV_DIR, exist_ok=True)

    df["label"] = df.apply(
        lambda r: f"{r['src_folder']}{'-nofmad' if r['nofmad'] else ''}{'-dbl' if r['dbl'] else ''}", axis=1
    )

    for label, group in df.groupby("label"):
        agg = (
            group.groupby(["filter_radius", "image_size"])
            .agg(
                gpu_time_mean=("gpu_time", "mean"),
                cpu_time_mean=("cpu_time", "mean"),
                gpu_time_std=("gpu_time", "std"),
                cpu_time_std=("cpu_time", "std"),
                count=("gpu_time", "count"),
            )
            .reset_index()
        )

        # Compute speedup
        agg["speedup"] = agg["cpu_time_mean"] / agg["gpu_time_mean"]
        agg["speedup_std"] = agg["speedup"] * np.sqrt(
            (agg["cpu_time_std"] / agg["cpu_time_mean"]) ** 2 +
            (agg["gpu_time_std"] / agg["gpu_time_mean"]) ** 2
        )

        csv_file = os.path.join(CSV_DIR, f"{label}_runtime_k{TARGET_KERNEL_LENGTH}.csv")
        agg.to_csv(csv_file, index=False)
        print(f"[CSV exported] {csv_file}")


# ---------------------------------------------------------
#   PLOTTING – *READS FROM CSV FILES ONLY*
# ---------------------------------------------------------
def plot_from_csv():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.5,
        "lines.markersize": 10
    })

    # Only consider runtime CSV files
    csv_files = [
        f for f in os.listdir(CSV_DIR)
        if f.endswith(".csv") and "_runtime_" in f
    ]

    if not csv_files:
        print("No runtime CSV files found in plots/csv/. Nothing to plot.")
        return

    required_cols = {
        "filter_radius", "image_size",
        "gpu_time_mean", "cpu_time_mean",
        "gpu_time_std", "cpu_time_std",
        "speedup", "speedup_std"
    }

    for csv_file in csv_files:
        path = os.path.join(CSV_DIR, csv_file)
        df = pd.read_csv(path)

        # Validate columns
        if not required_cols.issubset(df.columns):
            print(f"[Skipping] {csv_file} (missing required runtime columns)")
            continue

        if df.empty:
            print(f"[Skipping] {csv_file} (empty file)")
            continue

        label = csv_file.replace(".csv", "")

        # Two y-scale styles
        y_scales = [
            ("linear", "linear"),
            ("log", "log2"),
        ]

        for mpl_scale, suffix in y_scales:
            fig, ax = plt.subplots(figsize=(12, 7))

            # --- Plotting CPU/GPU runtime curves ---
            ax.plot(df["image_size"], df["gpu_time_mean"],
                    marker="o", linestyle="-", label="GPU Time")
            ax.plot(df["image_size"], df["cpu_time_mean"],
                    marker="x", linestyle="--", label="CPU Time")

            filter_radius_val = df["filter_radius"].iloc[0]
            title_prefix = "Doubles" if "dbl" in label else "Floats"

            ax.set_title(f"{title_prefix} – CPU/GPU Runtime (Filter radius: {filter_radius_val})")
            ax.set_xlabel("Image Size (N)")
            ax.set_ylabel("Runtime (seconds)")

            ax.set_xscale("log", base=2)
            if mpl_scale == "log":
                ax.set_yscale("log", base=2)

            ax.grid(True, linestyle="--", axis="x")
            ax.grid(True, linestyle=":", axis="y", which="both")

            ticks = sorted(df["image_size"].unique())
            ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter("%d"))

            ax.legend(loc="upper left")
            plt.tight_layout()

            out = os.path.join(PLOTS_DIR, f"{label}_{suffix}.png")
            plt.savefig(out)
            plt.close(fig)
            print(f"[Plot saved] {out}")


# ---------------------------------------------------------
#   MAIN MENU
# ---------------------------------------------------------
if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_BASE_DIR

    print("\nChoose an operation:")
    print("1) Parse logs → Export CSVs only")
    print("2) Plot from CSV files only")
    print("3) Both: Parse + Export CSV + Plot")
    choice = input("Enter choice [1-3]: ").strip()

    if choice not in {"1", "2", "3"}:
        print("Invalid choice.")
        sys.exit(1)

    # PARSE ONLY or BOTH
    if choice in {"1", "3"}:
        print(f"\n[Parsing logs] Base directory: {base_dir}")
        rows = collect_data(base_dir)
        if not rows:
            print("No data found.")
            sys.exit(1)

        df = pd.DataFrame(rows)
        export_to_csv(df)

    # PLOT ONLY or BOTH
    if choice in {"2", "3"}:
        print("\n[Plotting from CSV files]")
        plot_from_csv()

    print("\nDone.\n")
