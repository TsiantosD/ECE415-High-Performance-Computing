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
    echo "  -f: Specific .cu file (e.g., 'clahe.cu') OR a number ('1' for 1st file)"
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
            echo "Selected via Index [$INDEX]: $CU_FILE_SELECTED"
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
    echo "Searching for .cu files in 'src/'..."
    echo "Available Kernels:"
    i=1
    for file in "${CU_FILES[@]}"; do
        echo "  [$i] $(basename "$file")"
        ((i++))
    done
    echo "----------------------------------------"
    read -p "Select kernel # (Default 1): " selection
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