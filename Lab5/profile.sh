#!/bin/bash

# 1. Define Usage Function
usage() {
    echo "Usage: $0 [--gui] [--compute] [-m mode] [-c count] [-o output_name] <executable> [args...]"
    echo ""
    echo "Flags:"
    echo "  --gui:       Auto-open the result in the appropriate viewer"
    echo "  --compute:   Use Nsight Compute (ncu) for kernel analysis (Default: Nsight Systems)"
    echo "  -m, --mode:  Profiling detail level (for --compute only):"
    echo "               light  -> SpeedOfLight only (~2 passes)"
    echo "               basic  -> Common metrics (~10-15 passes)"
    echo "               full   -> All metrics (30-50+ passes, Default)"
    echo "  -c, --count: Number of kernels to profile (default: unlimited)"
    echo "  -o:          Custom profile name (Default: executable name)"
    echo "  -h:          Show this help message"
    exit 1
}

OUTPUT_NAME=""
OPEN_GUI=false
USE_COMPUTE=false
COMPUTE_MODE="full"
LAUNCH_COUNT=""

# 2. Parse Arguments
while [[ "$1" =~ ^- ]]; do
    case "$1" in
        -o)          OUTPUT_NAME="$2"; shift 2 ;;
        -m|--mode)   COMPUTE_MODE="$2"; shift 2 ;;
        -c|--count)  LAUNCH_COUNT="$2"; shift 2 ;;
        --gui)       OPEN_GUI=true; shift ;;
        --compute)   USE_COMPUTE=true; shift ;;
        -h|--help)   usage ;;
        *) break ;;
    esac
done

# 3. Validation
if [ -z "$1" ]; then
    usage
fi

# 4. Determine NCU Flags
# ADDED: --import-source yes to ensure code is visible in the GUI
if [ "$COMPUTE_MODE" == "light" ]; then
    NCU_FLAGS="--section SpeedOfLight --import-source yes"
elif [ "$COMPUTE_MODE" == "basic" ]; then
    NCU_FLAGS="--set detailed --import-source yes"
else 
    NCU_FLAGS="--set full --import-source yes"
fi

if [ -n "$LAUNCH_COUNT" ]; then
    NCU_FLAGS="$NCU_FLAGS --launch-count $LAUNCH_COUNT"
fi

# 5. Set Defaults & Extensions
if [ -z "$OUTPUT_NAME" ]; then
    OUTPUT_NAME="$(basename "$1")_profile"
fi

if [ "$USE_COMPUTE" = true ]; then
    TOOL_NAME="Nsight Compute ($COMPUTE_MODE)"
    FILE_EXT=".ncu-rep"
else
    TOOL_NAME="Nsight Systems"
    FILE_EXT=".nsys-rep"
fi

FULL_FILE="${OUTPUT_NAME}${FILE_EXT}"

echo "-----------------------------------------------------"
echo "Tool:      $TOOL_NAME"
if [ -n "$LAUNCH_COUNT" ]; then
    echo "Limit:     First $LAUNCH_COUNT kernels only"
fi
echo "Profiling: $@"
echo "Output:    $FULL_FILE"
echo "-----------------------------------------------------"

# 6. Run Profiler
if [ "$USE_COMPUTE" = true ]; then
    # --- Run Nsight Compute (ncu) ---
    
    NCU_PATH=$(which ncu)
    
    if [ -z "$NCU_PATH" ]; then
        if [ -f "/usr/local/cuda/bin/ncu" ]; then
            NCU_PATH="/usr/local/cuda/bin/ncu"
        else
            echo "Error: 'ncu' not found in PATH or default locations."
            exit 1
        fi
    fi

    sudo "$NCU_PATH" \
      $NCU_FLAGS \
      --force-overwrite \
      -o "${OUTPUT_NAME}" \
      "$@"
else
    # --- Run Nsight Systems (nsys) ---
    sudo nsys profile \
      --trace=cuda,nvtx,osrt \
      --force-overwrite=true \
      --output="${OUTPUT_NAME}" \
      "$@"
fi

# 7. Launch GUI
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ] && [ "$OPEN_GUI" = true ] && [ -f "$FULL_FILE" ]; then
    echo "Opening GUI..."
    sudo chown $USER "$FULL_FILE"
    sudo chown $USER "${OUTPUT_NAME}.sqlite" 2>/dev/null

    if [ "$USE_COMPUTE" = true ]; then
        if command -v ncu-ui &> /dev/null; then
            ncu-ui "$FULL_FILE"
        else
            echo "Error: Could not find 'ncu-ui' in PATH."
        fi
    else
        if command -v nsys-ui &> /dev/null; then
            nsys-ui "$FULL_FILE"
        elif command -v nsight-sys &> /dev/null; then
            nsight-sys "$FULL_FILE"
        else
            echo "Error: Could not find 'nsys-ui' in PATH."
        fi
    fi
fi