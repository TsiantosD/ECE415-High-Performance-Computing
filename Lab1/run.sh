#!/bin/bash

CPU_CORE=2

PROJECT_DIR=$(pwd)
GOLDEN_FILE="$PROJECT_DIR/src/golden.grey"
OUTPUT_FILE="$PROJECT_DIR/src/output_sobel.grey"

set -e

# --- Build ---
cd ./src || { echo "src folder not found!"; exit 1; }
echo "==> Running make in ./src ..."
make || { echo "Make failed!"; exit 1; }

# --- Find executables ---
echo
echo "==> Listing available executables..."
EXES=()
while IFS= read -r exe; do
    EXES+=("$exe")
done < <(find . -maxdepth 1 -type f -executable -name "sobel_*" ! -name "*.c" | sort)

if [ ${#EXES[@]} -eq 0 ]; then
    echo "No executables found!"
    exit 1
fi

# --- Execution methods ---
METHODS=("normal" "hotspots" "performance-snapshot" "uarch-exploration" "memory-access")
METHOD_FLAGS=("" "hotspots" "performance-snapshot" "uarch-exploration" "memory-access")
METHOD_NAMES=("Normal run" "VTune - Hotspots" "VTune - Performance Snapshot" "VTune - Microarchitecture Exploration" "VTune - Memory Access")

echo
echo "Select execution method:"
for i in "${!METHODS[@]}"; do
    echo "[$((i+1))] ${METHOD_NAMES[$i]}"
done
echo "[a] Run all methods"

read -rp "Enter choice (1-${#METHODS[@]} or 'a' for all): " exec_mode
echo

if [[ "$exec_mode" == "a" ]]; then
    selected_indices=("${!METHODS[@]}")  # All methods
elif [[ "$exec_mode" =~ ^[0-9]+$ ]] && (( exec_mode >= 1 && exec_mode <= ${#METHODS[@]} )); then
    selected_indices=($((exec_mode-1)))
else
    echo "Invalid choice!"
    exit 1
fi

# --- Choose executable ---
echo "Available executables:"
i=1
for exe in "${EXES[@]}"; do
    echo "[$i] ${exe#./}"
    i=$((i+1))
done
echo "[a] Run all"

read -rp "Select executable to run (1-${#EXES[@]} or 'a' for all): " choice
echo

if [[ "$choice" == "a" ]]; then
    selected_exes=("${EXES[@]}")
elif [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#EXES[@]} )); then
    selected_exes=("${EXES[$((choice-1))]}")
else
    echo "Invalid choice!"
    exit 1
fi

# --- Function to run executable and diff ---
run_and_diff() {
    local exe="$1"
    local method="$2"
    local method_flag="$3"

    local exe_name="${exe#./}"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local metrics_dir="../metrics/${method}/${timestamp}_${exe_name}"
    mkdir -p "$metrics_dir"

    echo "==> Running $exe_name with method: $method ..."
    if [ "$method" = "normal" ]; then
        "./$exe_name"
    else
        taskset -c $CPU_CORE vtune -collect "$method_flag" -result-dir "$metrics_dir" -- "./$exe_name"
    fi

    echo "==> Comparing output.grey with golden.grey ..."
    if ! diff "$GOLDEN_FILE" "$OUTPUT_FILE" > /dev/null; then
        echo "⚠️ Difference found for $exe_name ($method)!"
    else
        echo "✅ Output matches golden file for $exe_name ($method)."
    fi
    echo "-------------------------------------"
}

# --- Run all combinations ---
for exe in "${selected_exes[@]}"; do
    for idx in "${selected_indices[@]}"; do
        method="${METHODS[$idx]}"
        method_flag="${METHOD_FLAGS[$idx]}"
        run_and_diff "$exe" "$method" "$method_flag"
    done
done
