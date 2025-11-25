import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Configuration ---
DEFAULT_OUTPUT_BASE_DIR = 'metrics'
PLOTS_DIR = 'plots'
CSV_DIR = os.path.join(PLOTS_DIR, 'csv')

# Regex for log files: timestamp-run_X.log
LOGFILE_REGEX = re.compile(r'.*-(?:run_\d+)\.log$')

# Regex to extract max difference from log content
MAX_DIFF_REGEX = re.compile(r'Max difference:\s*([0-9.eE+-]+)')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

################################################################################
# ---------------------------- PARSING FUNCTIONS ----------------------------- #
################################################################################

def parse_log_file(path):
    """Extract max difference from a log file."""
    try:
        with open(path, 'r') as f:
            content = f.read()
        m = MAX_DIFF_REGEX.search(content)
        if not m:
            return None
        return float(m.group(1))
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def collect_data(base_dir):
    """
    Collect max difference data from metrics and return a DataFrame.
    """
    records = []

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return pd.DataFrame()

    # Pick subdirectory (precision, etc.)
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not subdirs:
        print(f"No subdirectories found in '{base_dir}'. Exiting.")
        return pd.DataFrame()

    if len(subdirs) == 1:
        selected_subdir = subdirs[0]
        print(f"Only one subdirectory: {selected_subdir}. Selected automatically.")
    else:
        print("Multiple metric categories found:")
        for i, d in enumerate(subdirs, 1):
            print(f"  {i}) {d}")

        while True:
            try:
                idx = int(input("Select: ")) - 1
                if 0 <= idx < len(subdirs):
                    selected_subdir = subdirs[idx]
                    break
            except:
                pass
            print("Invalid choice.")

    metrics_path = os.path.join(base_dir, selected_subdir)
    print(f"Using metrics path: {metrics_path}")

    src_folder_placeholder = selected_subdir

    # Iterate configs (e.g., 128_1024-dbl)
    for config_dir in os.listdir(metrics_path):
        config_path = os.path.join(metrics_path, config_dir)
        if not os.path.isdir(config_path):
            continue

        match = re.match(r'(\d+)_(\d+)(-nofmad)?(-dbl)?', config_dir)
        if not match:
            continue

        filter_radius = int(match.group(1))
        image_size = int(match.group(2))
        nofmad_flag = bool(match.group(3))
        dbl_flag = bool(match.group(4))

        # Iterate over logs
        for logfile in os.listdir(config_path):
            if not LOGFILE_REGEX.match(logfile):
                continue

            full_path = os.path.join(config_path, logfile)
            max_diff = parse_log_file(full_path)
            if max_diff is None:
                continue

            records.append({
                "src_folder": src_folder_placeholder,
                "filter_radius": filter_radius,
                "image_size": image_size,
                "nofmad": nofmad_flag,
                "dbl": dbl_flag,
                "logfile": logfile,
                "max_diff": max_diff
            })

    return pd.DataFrame(records)


def export_to_csv(df):
    """Export the full parsed dataset and per-label CSVs."""
    if df.empty:
        print("No data to export.")
        return

    # Main CSV
    main_csv_path = os.path.join(CSV_DIR, "all_data.csv")
    df.to_csv(main_csv_path, index=False)
    print(f"Exported: {main_csv_path}")

    # Add label column
    df["label"] = df.apply(
        lambda r: f"{r['src_folder']}{'-nofmad' if r['nofmad'] else ''}{'-dbl' if r['dbl'] else ''}",
        axis=1
    )

    # Per-group CSVs
    for label, group in df.groupby("label"):
        out = os.path.join(CSV_DIR, f"{label}.csv")
        group.to_csv(out, index=False)
        print(f"Exported group CSV: {out}")


################################################################################
# ---------------------------- PLOTTING FUNCTIONS ---------------------------- #
################################################################################

def plot_from_csv():
    """
    Read CSV files from CSV_DIR and plot them.
    """

    # Only select CSVs that contain max_diff
    csv_files = []
    for f in os.listdir(CSV_DIR):
        if not f.endswith(".csv") or f == "all_data.csv":
            continue
        try:
            df_tmp = pd.read_csv(os.path.join(CSV_DIR, f), nrows=1)
            if "max_diff" in df_tmp.columns:
                csv_files.append(f)
        except Exception:
            continue

    if not csv_files:
        print("No precision CSV files found in /csv.")
        return

    for csv_file in csv_files:
        path = os.path.join(CSV_DIR, csv_file)
        df = pd.read_csv(path)

        if "max_diff" not in df.columns:
            print(f"Skipping non-precision CSV: {csv_file}")
            continue

        if df.empty:
            continue

        label = csv_file.replace(".csv", "")

        # Aggregate (though for 1 run STD is zero)
        agg = df.groupby("filter_radius").agg(
            max_diff_mean=("max_diff", "mean"),
            max_diff_std=("max_diff", "std")
        ).reset_index()
        agg["max_diff_std"] = agg["max_diff_std"].fillna(0)

        # Plot
        plt.figure(figsize=(9, 6))
        plt.errorbar(
            agg['filter_radius'],
            agg['max_diff_mean'],
            yerr=agg['max_diff_std'],
            fmt='o-', capsize=5, linewidth=2
        )

        plt.title(f"{label} - Max Difference vs Filter Radius")
        plt.xlabel("Filter Radius")
        plt.ylabel("Max Difference")
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)

        out_path = os.path.join(PLOTS_DIR, f"{label}_plot.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved plot: {out_path}")


################################################################################
# ---------------------------------- MAIN ------------------------------------ #
################################################################################

if __name__ == "__main__":
    output_base_dir = (
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_BASE_DIR
    )
    print(f"Using metrics directory: {output_base_dir}")

    print("\nChoose an action:")
    print("  1) Parse data only (export CSV)")
    print("  2) Plot data only (read CSV)")
    print("  3) Both parse and plot")
    choice = input("Enter choice [1-3]: ").strip()

    if choice == "1" or choice == "3":
        df = collect_data(output_base_dir)
        export_to_csv(df)

    if choice == "2" or choice == "3":
        plot_from_csv()

    print("Done.")
