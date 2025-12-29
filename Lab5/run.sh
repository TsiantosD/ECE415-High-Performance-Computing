#!/bin/bash

# ==========================================
# Configuration Defaults
# ==========================================
ITERATIONS=1
CHECK_OUTPUT_VAL=1
CU_FILE_SELECTED=""
INPUT_ARG=""
CPU_MODE="omp"
DEBUG_MODE=0  # 0 = Release (Default), 1 = Debug
ONLY_CPU=0

# ==========================================
# Help / Usage Function
# ==========================================
usage() {
    echo "Usage: $0 [-n iterations] [-c check_output_val] [-f cuda_file.cu] [-i input_name] [-sdo]"
    echo ""
    echo "Flags:"
    echo "  -n: Number of times to run (Default: 1)"
    echo "  -c: Value for CHECK_OUTPUT (Default: 1)"
    echo "  -f: Specific .cu file (e.g., 'clahe.cu') OR a number ('1' for 1st file)"
    echo "  -i: Input filename (e.g., 'galaxy_data.bin'). Checks 'Inputs/'."
    echo "  -d: Compile in DEBUG mode (Target: debug). Default is Release."
    echo "  -s: Run CPU version in sequential mode."
    echo "  -o: Run only CPU version and write output to file."
    exit 1
}

# ==========================================
# 1. Parse Flags
# ==========================================
# Added 'd' to the option string (no colon after d because it takes no argument)
while getopts "n:c:f:i:sdo" opt; do
    case $opt in
        n) ITERATIONS=$OPTARG ;;
        c) CHECK_OUTPUT_VAL=$OPTARG ;;
        f) CU_FILE_SELECTED=$OPTARG ;;
        i) INPUT_ARG=$OPTARG ;;
        s) CPU_MODE="seq" ;;
        d) DEBUG_MODE=1 ;;
        o) ONLY_CPU=1 ;;
        *) usage ;;
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
# 4. Select CUDA File (If not specified)
# ==========================================
if [ ! -d "src" ]; then
    echo "Error: 'src' directory not found."
    exit 1
fi

# Safer way to read files into an array (handles spaces/newlines correctly)
CU_FILES=()
while IFS= read -r line; do
    CU_FILES+=("$line")
done < <(find src -maxdepth 1 -name "*.cu" | sort)

NUM_FILES=${#CU_FILES[@]}

# Sanity check to prevent the "integer expression" error
if [ -z "$NUM_FILES" ] || [ "$NUM_FILES" -eq 0 ]; then
    echo "Error: No .cu files found in src/."
    exit 1
fi

if [ -n "$CU_FILE_SELECTED" ]; then
    # Check if user provided a NUMBER (Integer)
    if [[ "$CU_FILE_SELECTED" =~ ^[0-9]+$ ]]; then
        INDEX=$CU_FILE_SELECTED
        
        # Validate Range
        if [ "$INDEX" -ge 1 ] && [ "$INDEX" -le "$NUM_FILES" ]; then
            FULL_CU_PATH=${CU_FILES[$((INDEX-1))]}
            CU_FILE_SELECTED=$(basename "$FULL_CU_PATH")
        else
            echo "Error: Index $INDEX is out of range (1-$NUM_FILES)."
            exit 1
        fi
    else
        # User provided a filename string
        CU_FILE_SELECTED=$(basename "$CU_FILE_SELECTED")
    fi
else
    # Interactive Mode
    echo "No .cu file specified. Searching in 'src/'..."
    echo "----------------------------------------"
    echo "Available Kernels:"
    i=1
    for file in "${CU_FILES[@]}"; do
        echo "  [$i] $(basename "$file")"
        ((i++))
    done
    echo "----------------------------------------"
    read -p "Select kernel # (Default 1): " selection
    echo ""
    [ -z "$selection" ] && selection=1

    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "$NUM_FILES" ]; then
        FULL_CU_PATH=${CU_FILES[$((selection-1))]}
        CU_FILE_SELECTED=$(basename "$FULL_CU_PATH")
    else
        echo "Invalid selection."
        exit 1
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

if [ "$DEBUG_MODE" -eq 1 ]; then
    make -C src debug CPU_MODE="$CPU_MODE" CUDA_SRC="$CU_FILE_SELECTED" USER_FLAGS="-DCHECK_OUTPUT=$CHECK_OUTPUT_VAL -DONLY_CPU=$ONLY_CPU -DSEQ_CPU=$SEQ_CPU"
else
    make -C src CPU_MODE="$CPU_MODE" CUDA_SRC="$CU_FILE_SELECTED" USER_FLAGS="-DCHECK_OUTPUT=$CHECK_OUTPUT_VAL -DONLY_CPU=$ONLY_CPU -DSEQ_CPU=$SEQ_CPU"
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
