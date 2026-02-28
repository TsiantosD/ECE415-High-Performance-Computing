#!/bin/bash

# ==========================================
# Configuration Defaults
# ==========================================
ITERATIONS=1
CHECK_OUTPUT_VAL=1
CU_FILE_SELECTED=""
INPUT_ARG=""
BLOCK_SIZE=256
GPU_MAX=4
DEBUG_MODE=0  # 0 = Release (Default), 1 = Debug

SEL_CPU_MODE=0
SEL_GPU_MODE=0

ONLY_CPU=0
CPU_COMPILER_FLAG="seq"

# ==========================================
# Help / Usage Function
# ==========================================
usage() {
    echo "Usage:"
    echo "  $0 [options]"
    echo ""
    echo "Options:"
    echo "  -n, --iterations=N       Number of runs (default: 1)"
    echo "  --cpu=MODE               CPU Mode (none, seq, omp)"
    echo "  --gpu=MODE               GPU Mode (on, off)"
    echo "  -f, --file=FILE|IDX      CUDA .cu file name (e.g. 2 or 02_nbody_cuda.cu)"
    echo "  -i, --input=FILE         Input file from Inputs/ directory"
    echo "  -b, --block-size=N       CUDA BLOCK_SIZE macro (default: 32)"
    echo "  -g, --gpu-max=N          GPU_MAX macro (default: 4)"
    echo "  -d, --debug              Compile in DEBUG mode"
    echo "  -h, --help               Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0 --iterations=10 --file=2 --input=galaxy_data.bin"
    echo "  $0 -n 5 -s -d"
    exit 1
}

# ==========================================
# 1. Parse Flags
# ==========================================
PARSED_OPTS=$(getopt \
    -o n:f:i:b:g:dh \
    --long iterations:,cpu:,gpu:,file:,input:,block-size:,gpu-max:,debug,help \
    -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    usage
fi

eval set -- "$PARSED_OPTS"

while true; do
    case "$1" in
        -n|--iterations)
            ITERATIONS="$2"; shift 2 ;;
        --cpu)
            case "$2" in
                none) SEL_CPU_MODE=1 ;;
                seq)  SEL_CPU_MODE=2 ;;
                omp)  SEL_CPU_MODE=3 ;;
                *) echo "Invalid CPU mode: $2"; exit 1 ;;
            esac
            shift 2 ;;
        --gpu)
             case "$2" in
                on)  SEL_GPU_MODE=1 ;;
                off) SEL_GPU_MODE=2 ;;
                *) echo "Invalid GPU mode: $2"; exit 1 ;;
            esac
            shift 2 ;;
        -f|--file)
            CU_FILE_SELECTED="$2"; shift 2 ;;
        -i|--input)
            INPUT_ARG="$2"; shift 2 ;;
        -b|--block-size)
            BLOCK_SIZE="$2"; shift 2 ;;
        -g|--gpu-max)
            GPU_MAX="$2"; shift 2 ;;
        -d|--debug)
            DEBUG_MODE=1; shift ;;
        -h|--help)
            usage ;;
        --)
            shift; break ;;
        *)
            usage ;;
    esac
done

# ==========================================
# 2. Interactive Selection
# ==========================================

# Check ITERATIONS (Must be a positive integer >= 1)
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    echo "Error: Iterations (-n) must be a positive integer. Received: '$ITERATIONS'"
    exit 1
fi


# 2.1 Select CPU Mode
if [ "$SEL_CPU_MODE" -eq 0 ]; then
    echo "----------------------------------------"
    echo "Select CPU Mode:"
    echo "  [1] None"
    echo "  [2] Sequential"
    echo "  [3] OpenMP"
    echo "----------------------------------------"
    read -p "Choice (Default 3): " choice
    [ -z "$choice" ] && choice=3
    SEL_CPU_MODE=$choice
fi

# 2.2 Select GPU Mode
if [ "$SEL_GPU_MODE" -eq 0 ]; then
    echo ""
    echo "----------------------------------------"
    echo "Select GPU Mode:"
    echo "  [1] On"
    echo "  [2] Off"
    echo "----------------------------------------"
    read -p "Choice (Default 1): " choice
    [ -z "$choice" ] && choice=1
    SEL_GPU_MODE=$choice
fi

# ==========================================
# 3. Logic Validation & Compilation Setup
# ==========================================

# Check Invalid Combo: CPU None + GPU Off
if [ "$SEL_CPU_MODE" -eq 1 ] && [ "$SEL_GPU_MODE" -eq 2 ]; then
    echo ""
    echo "ERROR: You cannot select CPU: None and GPU: Off."
    echo "       At least one computational unit must be active."
    exit 1
fi

# -- Configure Compilation Variables based on User Choice --

# Handle CPU Choices
if [ "$SEL_CPU_MODE" -eq 1 ]; then
    # CPU: None
    CPU_COMPILER_FLAG="seq" # Default compiler flag (needed for host code)
    CHECK_OUTPUT_VAL=0      # Force disable check (nothing to compare against)
    echo ">> CPU set to None: Disabling Output Check (CHECK_OUTPUT=0)."
elif [ "$SEL_CPU_MODE" -eq 2 ]; then
    # CPU: Sequential
    CPU_COMPILER_FLAG="seq"
elif [ "$SEL_CPU_MODE" -eq 3 ]; then
    # CPU: OpenMP
    CPU_COMPILER_FLAG="omp"
else
    echo "Invalid CPU selection."
    exit 1
fi

# Handle GPU Choices
if [ "$SEL_GPU_MODE" -eq 1 ]; then
    # GPU: On
    ONLY_CPU=0
else
    # GPU: Off
    ONLY_CPU=1
fi

# ==========================================
# 2. Select input (If not specified via flag)
# ==========================================
if [ -z "$INPUT_ARG" ]; then
    echo ""
    echo "No input specified. Searching in 'Inputs/'..."
    
    # Check if directory exists
    if [ ! -d "Inputs" ]; then
        echo "Error: 'Inputs' directory not found."
        exit 1
    fi

    # Create array of files in Inputs/
    # We use a glob to find files, checking if they exist
    shopt -s nullglob
    INPUT_FILES=(Inputs/*)
    shopt -u nullglob

    if [ ${#INPUT_FILES[@]} -eq 0 ]; then
        echo "Error: No files found in 'Inputs/'."
        exit 1
    fi

    echo "----------------------------------------"
    echo "Available Inputs:"
    i=1
    for file in "${INPUT_FILES[@]}"; do
        # Display only the filename, not the full path
        filename=$(basename "$file")
        echo "  [$i] $filename"
        ((i++))
    done
    echo "----------------------------------------"

    read -p "Select an input number (Default 1): " selection
    echo ""

    # Default to 1
    if [ -z "$selection" ]; then
        selection=1
    fi

    # Validate selection
    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#INPUT_FILES[@]}" ]; then
        FULL_PATH=${INPUT_FILES[$((selection-1))]}
        INPUT_ARG=$(basename "$FULL_PATH")
    else
        echo "Invalid selection. Exiting."
        exit 1
    fi
fi

# ==========================================
# 3. Construct Input/Output Paths
# ==========================================

INPUT_PATH="Inputs/$INPUT_ARG"

# Final check to ensure file exists (in case user typed a bad name via flag)
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file '$INPUT_PATH' not found!"
    exit 1
fi

# Extract filename without extension (e.g. test.bin -> test)
BASENAME="${INPUT_ARG%.*}"
# Extract extension (e.g. test.bin -> bin)
EXTENSION="${INPUT_ARG##*.}"
OUTPUT_PATH="Outputs/${BASENAME}_out.${EXTENSION}"

mkdir -p Outputs/

# ==========================================
# 4. Select Source File
# ==========================================

if [ "$ONLY_CPU" -eq 0 ]; then
    CU_FILES=()
    while IFS= read -r f; do
        CU_FILES+=("$f")
    done < <(find src -maxdepth 1 -name "*.cu" | sort)

    if [ ${#CU_FILES[@]} -eq 0 ]; then
        echo "No CUDA files found."
        exit 1
    fi

    # 4.2 Check if user provided an index via flag (e.g. -f 1)
    if [[ "$CU_FILE_SELECTED" =~ ^[0-9]+$ ]]; then
        # Convert user index (1-based) to array index (0-based)
        IDX=$((CU_FILE_SELECTED - 1))
        
        # Validate range
        if [ "$IDX" -ge 0 ] && [ "$IDX" -lt "${#CU_FILES[@]}" ]; then
            CU_FILE_SELECTED="$(basename "${CU_FILES[${IDX}]}")"
            echo "Selected CUDA File via index: $CU_FILE_SELECTED"
        else
            echo "Error: Invalid CUDA file index '$CU_FILE_SELECTED'."
            echo "Available files:"
            i=1
            for f in "${CU_FILES[@]}"; do
                echo "  [$i] $(basename "$f")"
                ((i++))
            done
            exit 1
        fi

    # 4.3 If no flag provided, run Interactive Mode
    elif [ -z "$CU_FILE_SELECTED" ]; then
        echo "----------------------------------------"
        echo "Available CUDA kernels:"
        i=1
        for f in "${CU_FILES[@]}"; do
            echo "  [$i] $(basename "$f")"
            ((i++))
        done
        echo "----------------------------------------"
        read -p "Select kernel (Default 1): " selection
        echo ""

        [ -z "$selection" ] && selection=1
        
        # Verify input is numeric
        if [[ "$selection" =~ ^[0-9]+$ ]]; then
            IDX=$((selection - 1))
            if [ "$IDX" -ge 0 ] && [ "$IDX" -lt "${#CU_FILES[@]}" ]; then
                CU_FILE_SELECTED="$(basename "${CU_FILES[$IDX]}")"
            else
                echo "Invalid selection index."
                exit 1
            fi
        else
            echo "Invalid selection input."
            exit 1
        fi
    fi
fi

# ==========================================
# 5. Compile
# ==========================================
case "$SEL_CPU_MODE" in
    1) CPU_DESC="None (No Host Calculation)" ;;
    2) CPU_DESC="Sequential" ;;
    3) CPU_DESC="OpenMP (Parallel)" ;;
    *) CPU_DESC="Unknown ($SEL_CPU_MODE)" ;;
esac

case "$SEL_GPU_MODE" in
    1) GPU_DESC="On (CUDA Enabled)" ;;
    2) GPU_DESC="Off (CPU Only)" ;;
    *) GPU_DESC="Unknown ($SEL_GPU_MODE)" ;;
esac

echo "========================================"
echo " Configuration Summary"
echo "========================================"
echo " Mode:          $([ "$DEBUG_MODE" -eq 1 ] && echo "DEBUG" || echo "RELEASE")"
echo " CPU Mode:      $CPU_DESC"
echo " GPU Mode:      $GPU_DESC"
echo ""
echo " Input:         $INPUT_ARG"
echo " CUDA Kernel:   $CU_FILE_SELECTED"
echo " Iterations:    $ITERATIONS"
echo "========================================"
echo "Compiling..."

# Clean first
make -C src clean CUDA_SRC="$CU_FILE_SELECTED" > /dev/null

# Construct Flags
if [ "$CPU_COMPILER_FLAG" == "seq" ]; then
    SEQ_CPU=1
else
    SEQ_CPU=0
fi

# --- FIX: Use a separate variable for compilation vs logging ---
MAKE_CUDA_SRC="$CU_FILE_SELECTED"
if [ "$ONLY_CPU" -eq 1 ]; then
    MAKE_CUDA_SRC="" # Pass empty to makefile if GPU is off
fi

make -C src clean CUDA_SRC="$MAKE_CUDA_SRC" > /dev/null

FLAGS="-DCHECK_OUTPUT=$CHECK_OUTPUT_VAL -DONLY_CPU=$ONLY_CPU -DSEQ_CPU=$SEQ_CPU -DBLOCK_SIZE=$BLOCK_SIZE -DGPU_MAX=$GPU_MAX"

if [ "$DEBUG_MODE" -eq 1 ]; then
    make -C src debug CPU_MODE="$CPU_COMPILER_FLAG" ONLY_CPU="$ONLY_CPU" CUDA_SRC="$MAKE_CUDA_SRC" USER_FLAGS="$FLAGS"
else
    make -C src CPU_MODE="$CPU_COMPILER_FLAG" ONLY_CPU="$ONLY_CPU" CUDA_SRC="$MAKE_CUDA_SRC" USER_FLAGS="$FLAGS"
fi

if [ $? -ne 0 ]; then
    echo "Compilation Failed!"
    exit 1
fi

# ==========================================
# 6. Execution Loop
# ==========================================

CU_FILE_BASENAME="${CU_FILE_SELECTED%.*}"
LOG_DIR=""
if [ "$ITERATIONS" -gt 1 ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S_%3N")
    LOG_DIR="results/$CU_FILE_BASENAME/$BASENAME/$TIMESTAMP"
    
    echo "Running $ITERATIONS times."
    echo "Logs stored in: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

# Determine simulation parameters
# We'll use 20 iterations and 0.05 dt for a more visible simulation if not specified
SIM_ITERS=${SIM_ITERS:-20}
SIM_DT=${SIM_DT:-0.05}

echo ""
echo "Starting Execution..."
echo "--------------------------------------"
for (( i=1; i<=ITERATIONS; i++ ))
do
    if [ "$ITERATIONS" -gt 1 ]; then
        echo "Run #$i..."
        ./src/main "$INPUT_PATH" "$SIM_ITERS" "$SIM_DT" "$OUTPUT_PATH" > "$LOG_DIR/run_$i.log"
    else
        ./src/main "$INPUT_PATH" "$SIM_ITERS" "$SIM_DT" "$OUTPUT_PATH" 
    fi
done
echo "--------------------------------------"
echo "Done."

TEMP_VAR="${LOG_DIR#results/}"
REPORT_FILE="${TEMP_VAR//\//-}.csv"
if [ "$ITERATIONS" -gt 1 ]; then
    echo "Creating csv from data..."
    python3 create_csv.py "$LOG_DIR" "results/$REPORT_FILE"
fi
