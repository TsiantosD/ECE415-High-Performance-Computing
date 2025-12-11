#!/bin/bash

# ==========================================
# Configuration Defaults
# ==========================================
ITERATIONS=1
CHECK_OUTPUT_VAL=1
CU_FILE_SELECTED=""
IMAGE_ARG=""
DEBUG_MODE=0  # 0 = Release (Default), 1 = Debug

# ==========================================
# Help / Usage Function
# ==========================================
usage() {
    echo "Usage: $0 [-n iterations] [-c check_output_val] [-f cuda_file.cu] [-i image_name] [-d]"
    echo ""
    echo "Flags:"
    echo "  -n: Number of times to run (Default: 1)"
    echo "  -c: Value for CHECK_OUTPUT (Default: 1)"
    echo "  -f: Specific .cu file (e.g., 'clahe.cu')"
    echo "  -i: Image filename (e.g., 'test.pgm'). Checks 'Images/'."
    echo "  -d: Compile in DEBUG mode (Target: debug). Default is Release."
    exit 1
}

# ==========================================
# 1. Parse Flags
# ==========================================
# Added 'd' to the option string (no colon after d because it takes no argument)
while getopts "n:c:f:i:d" opt; do
    case $opt in
        n) ITERATIONS=$OPTARG ;;
        c) CHECK_OUTPUT_VAL=$OPTARG ;;
        f) CU_FILE_SELECTED=$OPTARG ;;
        i) IMAGE_ARG=$OPTARG ;;
        d) DEBUG_MODE=1 ;;
        *) usage ;;
    esac
done

# ==========================================
# 2. Select Image (If not specified via flag)
# ==========================================
if [ -z "$IMAGE_ARG" ]; then
    echo "----------------------------------------"
    echo "No image specified. Searching in 'Images/'..."
    
    # Check if directory exists
    if [ ! -d "Images" ]; then
        echo "Error: 'Images' directory not found."
        exit 1
    fi

    # Create array of files in Images/
    # We use a glob to find files, checking if they exist
    shopt -s nullglob
    IMG_FILES=(Images/*)
    shopt -u nullglob

    if [ ${#IMG_FILES[@]} -eq 0 ]; then
        echo "Error: No files found in 'Images/'."
        exit 1
    fi

    echo "Available Images:"
    i=1
    for file in "${IMG_FILES[@]}"; do
        # Display only the filename, not the full path
        filename=$(basename "$file")
        echo "  [$i] $filename"
        ((i++))
    done
    echo "----------------------------------------"

    read -p "Select an image number (Default 1): " selection

    # Default to 1
    if [ -z "$selection" ]; then
        selection=1
    fi

    # Validate selection
    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#IMG_FILES[@]}" ]; then
        FULL_PATH=${IMG_FILES[$((selection-1))]}
        IMAGE_ARG=$(basename "$FULL_PATH")
    else
        echo "Invalid selection. Exiting."
        exit 1
    fi
fi

# ==========================================
# 3. Construct Input/Output Paths
# ==========================================

INPUT_PATH="Images/$IMAGE_ARG"

# Final check to ensure file exists (in case user typed a bad name via flag)
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file '$INPUT_PATH' not found!"
    exit 1
fi

# Extract filename without extension (e.g. test.pgm -> test)
BASENAME="${IMAGE_ARG%.*}"
# Extract extension (e.g. test.pgm -> pgm)
EXTENSION="${IMAGE_ARG##*.}"
# Construct new output name (Output/test_out.pgm)
OUTPUT_PATH="Output/${BASENAME}_out.${EXTENSION}"

# Ensure Output directory exists
mkdir -p Output

echo "----------------------------------------"
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "----------------------------------------"

# ==========================================
# 4. Select CUDA File (If not specified)
# ==========================================
if [ -z "$CU_FILE_SELECTED" ]; then
    echo "Searching for .cu files in 'src/'..."
    
    # Find files in src/
    CU_FILES=($(find src -maxdepth 1 -name "*.cu" | sort))

    if [ ${#CU_FILES[@]} -eq 0 ]; then
        echo "Error: No .cu files found in src/."
        exit 1
    fi

    echo "Available Kernels:"
    i=1
    for file in "${CU_FILES[@]}"; do
        # Show full path "src/filename.cu" to user
        echo "  [$i] $file"
        ((i++))
    done
    echo "----------------------------------------"

    read -p "Select kernel # (Default 1): " selection
    [ -z "$selection" ] && selection=1

    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#CU_FILES[@]}" ]; then
        # This gets "src/filename.cu"
        FULL_CU_PATH=${CU_FILES[$((selection-1))]}
        # We need just "filename.cu" for the Makefile inside src
        CU_FILE_SELECTED=$(basename "$FULL_CU_PATH")
    else
        echo "Invalid selection."
        exit 1
    fi
else
    # If user provided a path manually (e.g. "src/test.cu" or "test.cu")
    # ensure we pass just the filename to the Makefile
    CU_FILE_SELECTED=$(basename "$CU_FILE_SELECTED")
fi

echo "Using Kernel: $CU_FILE_SELECTED"

# ==========================================
# 5. Compile
# ==========================================
echo "Compiling..."

# Clean first
make -C src clean CU_FILE="$CU_FILE_SELECTED" > /dev/null

# Construct Flags

if [ "$DEBUG_MODE" -eq 1 ]; then
    echo ">> Building DEBUG Target (slow, symbols enabled)"
    make -C src debug CU_FILE="$CU_FILE_SELECTED" USER_FLAGS="-DCHECK_OUTPUT=$CHECK_OUTPUT_VAL"
else
    echo ">> Building RELEASE Target (optimized)"
    make -C src CU_FILE="$CU_FILE_SELECTED" USER_FLAGS="-DCHECK_OUTPUT=$CHECK_OUTPUT_VAL"
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

echo "Starting Execution..."

for (( i=1; i<=ITERATIONS; i++ ))
do
    if [ "$ITERATIONS" -gt 1 ]; then
        echo "Run #$i..."
        ./src/main "$INPUT_PATH" "$OUTPUT_PATH" > "$LOG_DIR/run_$i.log"
    else
        ./src/main "$INPUT_PATH" "$OUTPUT_PATH"
    fi
done

echo "Done."

if [ "$ITERATIONS" -gt 1 ]; then
    echo "Creating csv from data..."
    python3 create_csv.py "$LOG_DIR" "$LOG_DIR/report.csv"
fi