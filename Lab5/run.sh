#!/bin/bash

# ==========================================
# Configuration Defaults
# ==========================================
ITERATIONS=1
CHECK_OUTPUT_VAL=1
CU_FILE_SELECTED=""
INPUT_ARG=""
BLOCK_SIZE=32
GPU_MAX=4
CPU_MODE="omp"
DEBUG_MODE=0  # 0 = Release (Default), 1 = Debug
ONLY_CPU=0

# ==========================================
# Help / Usage Function
# ==========================================
usage() {
    echo "Usage:"
    echo "  $0 [options]"
    echo ""
    echo "Options:"
    echo "  -n, --iterations=N       Number of runs (default: 1)"
    echo "  -c, --check-output=VAL   Value for CHECK_OUTPUT macro (default: 1)"
    echo "  -f, --file=FILE|INDEX    CUDA .cu file name or index (e.g. 02_nbody_cuda.cu or 2)"
    echo "  -i, --input=FILE         Input file from Inputs/ directory"
    echo "  -b, --block-size=N       CUDA BLOCK_SIZE macro (default: 32)"
    echo "  -g, --gpu-max=N          GPU_MAX macro (default: 4)"
    echo "  -s, --sequential         Run CPU version sequentially (disable OpenMP)"
    echo "  -d, --debug              Compile in DEBUG mode"
    echo "  -o, --only-cpu           Run CPU version only (no GPU)"
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
    -o n:c:f:i:b:g:sdoh \
    --long iterations:,check-output:,file:,input:,block-size:,gpu-max:,sequential,debug,only-cpu,help \
    -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    usage
fi

eval set -- "$PARSED_OPTS"

while true; do
    case "$1" in
        -n|--iterations)
            ITERATIONS="$2"; shift 2 ;;
        -c|--check-output)
            CHECK_OUTPUT_VAL="$2"; shift 2 ;;
        -f|--file)
            CU_FILE_SELECTED="$2"; shift 2 ;;
        -i|--input)
            INPUT_ARG="$2"; shift 2 ;;
        -b|--block-size)
            BLOCK_SIZE="$2"; shift 2 ;;
        -g|--gpu-max)
            GPU_MAX="$2"; shift 2 ;;
        -s|--sequential)
            CPU_MODE="seq"; shift ;;
        -d|--debug)
            DEBUG_MODE=1; shift ;;
        -o|--only-cpu)
            ONLY_CPU=1; shift ;;
        -h|--help)
            usage ;;
        --)
            shift; break ;;
        *)
            usage ;;
    esac
done

# ==========================================
# 1.1 Validate Integers
# ==========================================

# Check ITERATIONS (Must be a positive integer >= 1)
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    echo "Error: Iterations (-n) must be a positive integer. Received: '$ITERATIONS'"
    exit 1
fi

# Check CHECK_OUTPUT_VAL (Must be an integer, 0 or 1 usually, or just any int)
# The regex ^-?[0-9]+$ allows for negative numbers if your logic supports them.
# If it must be strictly 0 or 1: regex is ^[01]$
if ! [[ "$CHECK_OUTPUT_VAL" =~ ^-?[0-9]+$ ]]; then
    echo "Error: Check Output Value (-c) must be an integer. Received: '$CHECK_OUTPUT_VAL'"
    exit 1
fi

# ==========================================
# 2. Select input (If not specified via flag)
# ==========================================
if [ -z "$INPUT_ARG" ]; then
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
# 4. Select Execution Mode & Source File
# ==========================================

if [ -z "$CU_FILE_SELECTED" ] && [ "$ONLY_CPU" -eq 0 ]; then
    echo "Select execution mode:"
    echo "  [1] CPU Sequential"
    echo "  [2] CPU OpenMP"
    echo "  [3] GPU"
    echo "  [4] GPU + OpenMP"
    echo "----------------------------------------"
    read -p "Choice (Default 3): " mode_choice
    echo ""

    [ -z "$mode_choice" ] && mode_choice=3

    case "$mode_choice" in
        1)
            ONLY_CPU=1
            CPU_MODE="seq"
            FILE_PATTERN="*.c"
            EXCLUDE_PATTERN="_omp.c"
            EXCLUDE_FILES=("main.c" "results_check.c")
            ;;
        2)
            ONLY_CPU=1
            CPU_MODE="omp"
            FILE_PATTERN="*_omp.c"
            EXCLUDE_FILES=()
            ;;
        3)
            ONLY_CPU=0
            CPU_MODE="seq"
            FILE_PATTERN="*.cu"
            EXCLUDE_FILES=()
            ;;
        4)
            ONLY_CPU=0
            CPU_MODE="omp"
            FILE_PATTERN="*.cu"
            EXCLUDE_FILES=()
            ;;
        *)
            echo "Invalid selection."
            exit 1
            ;;
    esac

    if [ "$ONLY_CPU" -eq 1 ]; then
        SRC_FILES=()
        while IFS= read -r f; do
            fname=$(basename "$f")
            skip=0
            # Exclude specified files
            for excl in "${EXCLUDE_FILES[@]}"; do
                if [ "$fname" == "$excl" ]; then
                    skip=1
                    break
                fi
            done
            # Exclude pattern (_omp.c) if needed
            if [ -n "${EXCLUDE_PATTERN:-}" ] && [[ "$fname" =~ $EXCLUDE_PATTERN ]]; then
                skip=1
            fi
            [ $skip -eq 0 ] && SRC_FILES+=("$f")
        done < <(find src -maxdepth 1 -name "$FILE_PATTERN")

        if [ ${#SRC_FILES[@]} -eq 0 ]; then
            echo "No matching CPU source files found."
            exit 1
        fi

        echo "----------------------------------------"
        echo "Available CPU sources:"
        i=1
        for f in "${SRC_FILES[@]}"; do
            echo "  [$i] $(basename "$f")"
            ((i++))
        done
        echo "----------------------------------------"
        read -p "Select file (Default 1): " selection
        echo ""

        [ -z "$selection" ] && selection=1
        CU_FILE_SELECTED="$(basename "${SRC_FILES[$((selection-1))]}")"

    else
        CU_FILES=()
        while IFS= read -r f; do
            CU_FILES+=("$f")
        done < <(find src -maxdepth 1 -name "*.cu" | sort)

        if [ ${#CU_FILES[@]} -eq 0 ]; then
            echo "No CUDA files found."
            exit 1
        fi

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
        CU_FILE_SELECTED="$(basename "${CU_FILES[$((selection-1))]}")"
    fi
fi


# ==========================================
# 5. Compile
# ==========================================
echo "========================================"
echo " Configuration Summary"
echo "========================================"
echo " Mode:          $([ "$DEBUG_MODE" -eq 1 ] && echo "DEBUG" || echo "RELEASE")"
echo " CPU Mode:      $CPU_MODE"
echo " Input:         $INPUT_ARG"
echo " CUDA Kernel:   $CU_FILE_SELECTED"
echo " Iterations:    $ITERATIONS"
echo " Check Output:  $CHECK_OUTPUT_VAL"
echo " Only CPU:  $ONLY_CPU"
echo "========================================"
echo "Compiling..."

# Clean first
make -C src clean CUDA_SRC="$CU_FILE_SELECTED" > /dev/null

# Construct Flags
if [ "$CPU_MODE" == "seq" ]; then
    SEQ_CPU=1
else
    SEQ_CPU=0
fi

if [ "$ONLY_CPU" -eq 1 ]; then
    CU_FILE_SELECTED=""
fi

if [ "$DEBUG_MODE" -eq 1 ]; then
    make -C src debug CPU_MODE="$CPU_MODE" ONLY_CPU="$ONLY_CPU" CUDA_SRC="$CU_FILE_SELECTED" USER_FLAGS="-DCHECK_OUTPUT=$CHECK_OUTPUT_VAL -DONLY_CPU=$ONLY_CPU -DSEQ_CPU=$SEQ_CPU -DBLOCK_SIZE=$BLOCK_SIZE -DGPU_MAX=$GPU_MAX"
else
    make -C src CPU_MODE="$CPU_MODE" ONLY_CPU="$ONLY_CPU" CUDA_SRC="$CU_FILE_SELECTED" USER_FLAGS="-DCHECK_OUTPUT=$CHECK_OUTPUT_VAL -DONLY_CPU=$ONLY_CPU -DSEQ_CPU=$SEQ_CPU -DBLOCK_SIZE=$BLOCK_SIZE -DGPU_MAX=$GPU_MAX"
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

echo ""
echo "Starting Execution..."
echo "--------------------------------------"
for (( i=1; i<=ITERATIONS; i++ ))
do
    if [ "$ITERATIONS" -gt 1 ]; then
        echo "Run #$i..."
        ./src/main "$INPUT_PATH" "$OUTPUT_PATH" > "$LOG_DIR/run_$i.log"
    else
        ./src/main "$INPUT_PATH" "$OUTPUT_PATH" 
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
