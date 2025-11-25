#!/bin/bash

# ================= CONFIGURATION =================
PROGRAM_NAME="convolution2d"
DEFAULT_IMAGE_SIZE=32

# Variables
SRC_DIR=""
USE_DOUBLES=false
DISABLE_FMAD=false
IMAGE_SIZE_INPUT=""
FILTER_RADIUS=""
REPEAT_COUNT=1
OUTPUT_DIR=""

# Directory constants
SOURCE_DIR="src"
OUTPUT_BASE_DIR="metrics"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message and exit"
    echo "  --src-dir <dir_name>    Directory under 'src/' containing the kernel"
    echo "  --use-doubles           Compile with USE_DOUBLES=1"
    echo "  --disable-fmad          Compile with FMAD=false"
    echo "  --image-size <N>        Image size (must be power of two)"
    echo "  --filter-radius <N>     Filter radius (kernel length = 2*N+1)"
    echo "  --repeat <N>            Repeat the same configuration N times"
    echo "  --output-dir <name>     Override the SRC_DIR folder name in metrics/"
    exit 0
}

is_power_of_two() {
    local n=$1
    if [ "$n" -le 0 ]; then return 1; fi
    (( n & (n - 1) )) && return 1 || return 0
}

# ---------------------------
# Parse Arguments
# ---------------------------
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help) usage ;;
        --src-dir) SRC_DIR="$2"; shift 2 ;;
        --use-doubles) USE_DOUBLES=true; shift ;;
        --disable-fmad) DISABLE_FMAD=true; shift ;;
        --image-size) IMAGE_SIZE_INPUT="$2"; shift 2 ;;
        --filter-radius) FILTER_RADIUS="$2"; shift 2 ;;
        --repeat) REPEAT_COUNT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------
# Prompt for missing inputs
# ---------------------------
if [ -z "$SRC_DIR" ] && [ -z "$OUTPUT_DIR" ]; then
    OPTIONS=($(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort))
    echo "Multiple source dirs found in '$SOURCE_DIR/'. Select one:"
    PS3="Enter number: "
    select opt in "${OPTIONS[@]}"; do
        [[ -n "$opt" ]] && { SRC_DIR=$opt; break; }
        echo "Invalid selection."
    done
fi

if [ -z "$IMAGE_SIZE_INPUT" ]; then
    while true; do
        read -r -p "Enter image size (power of two, default $DEFAULT_IMAGE_SIZE): " INPUT
        INPUT=${INPUT:-$DEFAULT_IMAGE_SIZE}
        if ! [[ "$INPUT" =~ ^[0-9]+$ ]]; then
            echo "Invalid input. Must be a number."
        elif ! is_power_of_two "$INPUT"; then
            echo "Image size must be a power of two."
        else
            IMAGE_SIZE_INPUT=$INPUT
            break
        fi
    done
fi

if [ -z "$FILTER_RADIUS" ]; then
    while true; do
        read -r -p "Enter filter radius: " INPUT
        [[ "$INPUT" =~ ^[0-9]+$ ]] && { FILTER_RADIUS=$INPUT; break; }
        echo "Invalid input. Must be a number."
    done
fi

# ---------------------------
# Setup paths
# ---------------------------
FULL_SRC_DIR="$SOURCE_DIR/$SRC_DIR"
TARGET_EXEC="./$FULL_SRC_DIR/$PROGRAM_NAME"
FILTER_LEN=$((2 * FILTER_RADIUS + 1))

# Construct folder suffix for double/nofmad
OPTION_SUFFIX=""
[ "$DISABLE_FMAD" = true ] && OPTION_SUFFIX="${OPTION_SUFFIX}-nofmad"
[ "$USE_DOUBLES" = true ] && OPTION_SUFFIX="${OPTION_SUFFIX}-dbl"

# Use OUTPUT_DIR if provided, else SRC_DIR
TARGET_FOLDER_NAME=${OUTPUT_DIR:-$SRC_DIR}
FILTER_IMAGE_DIR="${OUTPUT_BASE_DIR}/${TARGET_FOLDER_NAME}/${FILTER_RADIUS}_${IMAGE_SIZE_INPUT}${OPTION_SUFFIX}"

# ---------------------------
# Compile
# ---------------------------
echo "----------------------- COMPILATION -----------------------"
MAKE_CMD="make -C $FULL_SRC_DIR"
$MAKE_CMD clean
$MAKE_CMD $( [ "$USE_DOUBLES" = true ] && echo "USE_DOUBLES=1" ) $( [ "$DISABLE_FMAD" = true ] && echo "FMAD=false" )

if [ $? -ne 0 ] || [ ! -f "$TARGET_EXEC" ]; then
    echo "Compilation failed."
    exit 1
fi

# ---------------------------
# Execution
# ---------------------------
mkdir -p "$FILTER_IMAGE_DIR"

echo "------------------------ EXECUTION ------------------------"
echo "Filter radius: $FILTER_RADIUS → Kernel size: $FILTER_LEN"
echo "Image size:    $IMAGE_SIZE_INPUT"
echo "Repeat count:  $REPEAT_COUNT"
echo "Output folder: $FILTER_IMAGE_DIR"

for (( j=1; j<=REPEAT_COUNT; j++ )); do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S_%3N") # milliseconds included
    FILEPATH="${FILTER_IMAGE_DIR}/${TIMESTAMP}-run_${j}.log"
    echo "$FILTER_LEN $IMAGE_SIZE_INPUT" | "$TARGET_EXEC" > "$FILEPATH" 2>&1
    echo "Run $j/$REPEAT_COUNT → $(basename $FILEPATH)"
    sleep 0.01 # ensure timestamp difference if runs are very fast
done

echo "-----------------------------------------------------------"
echo "Execution complete. All output saved to:"
echo "  $FILTER_IMAGE_DIR"
echo "-----------------------------------------------------------"
