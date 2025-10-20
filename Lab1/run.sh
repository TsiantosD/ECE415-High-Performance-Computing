#!/bin/bash

CPU_CORE=2
PROJECT_DIR=$(pwd)
AVERAGE_SCRIPT="$PROJECT_DIR/average.py"

set -e

# --- Parse arguments ---
EXEC_METHOD=""
EXECUTABLE=""
RUN_TIMES=1
CALCULATE_AVERAGE=false
IMAGE_FILE=""
OPTIMIZATION=O0

print_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --help                          Show this help message"
    echo "  --execution-method=<method>     Choose execution method"
    echo "                                  Options: normal, hotspots, performance-snapshot, uarch-exploration, memory-access, all"
    echo "  --executable=<name>             Choose executable (e.g. 1_sobel_orig or all)"
    echo "  --image=<filename>              Choose input image from ./src/input (e.g. 4096-timescapes.grey)"
    echo "  --times=<N>                     Run each executable N times (default: 1)"
    echo "  --calculate-average=<boolean>   Run the average.py script to calculate the average of all saved runs"
    echo "  --optimization=<O0|O1|O2|O3>    Choose compiler optimizations. Default: O0"
    echo "                                  Options: true, false"
    echo
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            print_help
            ;;
        --execution-method=*)
            EXEC_METHOD="${1#*=}"
            ;;
        --executable=*)
            EXECUTABLE="${1#*=}"
            ;;
        --image=*)
            IMAGE_FILE="${1#*=}"
            ;;
        --times=*)
            RUN_TIMES="${1#*=}"
            ;;
        --calculate-average=*)
            CALCULATE_AVERAGE="${1#*=}"
            ;;
        --optimization=*)
            OPTIMIZATION="${1#*=}"
            if [[ ! "$OPTIMIZATION" =~ ^O[0-3]$ ]]; then
                echo "Invalid optimization level: $OPTIMIZATION"
                echo "Valid options: O0, O1, O2, O3"
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
    shift
done

# --- Image selection ---
INPUT_DIR="$PROJECT_DIR/src/input"
GOLDEN_DIR="$PROJECT_DIR/src/golden"
OUTPUT_DIR="$PROJECT_DIR/src/output"

if [[ -z "$IMAGE_FILE" ]]; then
    echo
    echo "Available input images:"
    mapfile -t IMAGES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.grey" -printf "%f\n" | sort)
    if [[ ${#IMAGES[@]} -eq 0 ]]; then
        echo "No input images found in $INPUT_DIR!"
        exit 1
    fi

    i=1
    for img in "${IMAGES[@]}"; do
        echo "[$i] $img"
        ((i++))
    done
    read -rp "Select an image (1-${#IMAGES[@]}): " img_choice
    echo

    if [[ "$img_choice" =~ ^[0-9]+$ ]] && (( img_choice >= 1 && img_choice <= ${#IMAGES[@]} )); then
        IMAGE_FILE="${IMAGES[$((img_choice-1))]}"
    else
        echo "Invalid choice!"
        exit 1
    fi
fi

# --- Parse SIZE from filename ---
if [[ "$IMAGE_FILE" =~ ^([0-9]+)-(.+)\.grey$ ]]; then
    SIZE="${BASH_REMATCH[1]}"
    IMAGE_NAME="${BASH_REMATCH[2]}"
else
    echo "Invalid image filename format! Expected <SIZE>-<NAME>.grey"
    exit 1
fi

INPUT_PATH="$INPUT_DIR/$IMAGE_FILE"
GOLDEN_FILE="$GOLDEN_DIR/${IMAGE_NAME}.grey"
OUTPUT_FILE="$OUTPUT_DIR/${IMAGE_NAME}.grey"

# --- Clean & build---
cd ./src || { echo "src folder not found!"; exit 1; }
echo "> Running make clean in ./src ..."
make clean || { echo "Make clean!"; exit 1; }
echo
echo "> Running make in ./src with size: $SIZE and optimization: $OPTIMIZATION ..."
make SIZE="$SIZE" IMAGE_NAME="$IMAGE_NAME" OPT_LEVEL="$OPTIMIZATION" || { echo "Make failed!"; exit 1; }
echo

# --- Find executables ---
EXES=()
while IFS= read -r exe; do
    EXES+=("$exe")
done < <(find . -maxdepth 1 -type f -executable -name "*_sobel_*" ! -name "*.c" | sort)

if [ ${#EXES[@]} -eq 0 ]; then
    echo "No executables found!"
    exit 1
fi

# --- Execution methods ---
# --- Execution methods ---
METHODS=("normal" "hpc-performance" "memory-access" "hotspots" "performance-snapshot" "uarch-exploration" "memory-access" "perf")
METHOD_FLAGS=("" "hpc-performance" "memory-access" "hotspots" "performance-snapshot" "uarch-exploration" "memory-access" "")
METHOD_NAMES=("Normal run" "VTune - HPC Performance" "VTune - Memory Access" "VTune - Hotspots" "VTune - Performance Snapshot" "VTune - Microarchitecture Exploration" "VTune - Memory Access" "Perf stat")

# --- Determine selected methods ---
if [[ -n "$EXEC_METHOD" ]]; then
    if [[ "$EXEC_METHOD" == "all" ]]; then
        selected_indices=("${!METHODS[@]}")
    else
        found=false
        for i in "${!METHODS[@]}"; do
            if [[ "${METHODS[$i]}" == "$EXEC_METHOD" ]]; then
                selected_indices=($i)
                found=true
                break
            fi
        done
        if [[ "$found" == false ]]; then
            echo "Invalid method: $EXEC_METHOD"
            echo "Valid methods: ${METHODS[*]} or 'all'"
            exit 1
        fi
    fi
else
    echo
    echo "Select execution method:"
    for i in "${!METHODS[@]}"; do
        echo "[$((i+1))] ${METHOD_NAMES[$i]}"
    done
    echo "[a] Run all methods"
    read -rp "Enter choice (1-${#METHODS[@]} or 'a' for all): " exec_mode
    echo

    if [[ "$exec_mode" == "a" ]]; then
        selected_indices=("${!METHODS[@]}")
    elif [[ "$exec_mode" =~ ^[0-9]+$ ]] && (( exec_mode >= 1 && exec_mode <= ${#METHODS[@]} )); then
        selected_indices=($((exec_mode-1)))
    else
        echo "Invalid choice!"
        exit 1
    fi
fi

# --- Determine selected executables ---
if [[ -n "$EXECUTABLE" ]]; then
    if [[ "$EXECUTABLE" == "all" ]]; then
        selected_exes=("${EXES[@]}")
    else
        match_found=false
        for exe in "${EXES[@]}"; do
            if [[ "${exe#./}" == "$EXECUTABLE" ]]; then
                selected_exes=("$exe")
                match_found=true
                break
            fi
        done
        if [[ "$match_found" == false ]]; then
            echo "Executable not found: $EXECUTABLE"
            echo "Available: ${EXES[*]}"
            exit 1
        fi
    fi
else
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
fi

# --- Function to run executable and diff ---
run_and_diff() {
    local exe_name="${exe#./}"
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")

    local metrics_dir="../metrics/${SIZE}/${OPTIMIZATION}/${method}/${timestamp}_${exe_name}"
    mkdir -p "$metrics_dir"

    for ((run=1; run<=RUN_TIMES; run++)); do
        local log_file="$metrics_dir/${exe_name}_run${run}.log"
        echo "> [${run}/${RUN_TIMES}] Running $exe_name | Method: $method | Opt: $OPTIMIZATION | Size: $SIZE ..."

        if [[ "$method" == "normal" ]]; then
            taskset -c "$CPU_CORE" "./$exe_name" > "$log_file" 2>&1
        elif [[ "$method" == "perf" ]]; then
            taskset -c "$CPU_CORE" perf stat -e cpu-cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses "./$exe_name" > "$log_file" 2>&1
        else
            taskset -c "$CPU_CORE" vtune -collect "$method_flag" -result-dir "$metrics_dir/run_${run}" -- "./$exe_name" > "$log_file" 2>&1
        fi

        echo "> Comparing output.grey with golden.grey ..."
        if ! diff "$GOLDEN_FILE" "$OUTPUT_FILE" > /dev/null; then
            echo "⚠️ Difference found for $exe_name ($method, run $run, $OPTIMIZATION, size $SIZE)!"
        else
            echo "✅ Output matches golden file for $exe_name ($method, run $run, $OPTIMIZATION, size $SIZE)."
        fi
        echo
    done
}

# --- Run all combinations ---
for exe in "${selected_exes[@]}"; do
    for idx in "${selected_indices[@]}"; do
        method="${METHODS[$idx]}"
        method_flag="${METHOD_FLAGS[$idx]}"
        run_and_diff "$exe" "$method" "$method_flag"
    done
done

# --- After all runs, calculate averages (if requested) ---
if [ "$CALCULATE_AVERAGE" = true ]; then
    echo "> Calculating averages across all normal runs..."
    source $PROJECT_DIR/venv/bin/activate
    pip install -r $PROJECT_DIR/requirements.txt > /dev/null
    cd "$PROJECT_DIR" || exit 1
    python3 "$AVERAGE_SCRIPT"
fi
