#!/bin/bash

# 1. Define Usage Function
usage() {
    echo "Usage: $0 [--gui] [-o output_name] <executable> [args...]"
    echo ""
    echo "Flags:"
    echo "  --gui: Auto-open the result in Nsight Systems"
    echo "  -o:    Custom profile name (Default: executable name)"
    echo "  -h:    Show this help message"
    exit 1
}

OUTPUT_NAME=""
OPEN_GUI=false

# 2. Parse Arguments
while [[ "$1" =~ ^- ]]; do
    case "$1" in
        -o)    OUTPUT_NAME="$2"; shift 2 ;;
        --gui) OPEN_GUI=true; shift ;;
        -h|--help) usage ;;
        *) break ;;
    esac
done

# 3. Validation
if [ -z "$1" ]; then
    usage
fi

# 4. Set Defaults
[ -z "$OUTPUT_NAME" ] && OUTPUT_NAME="$(basename "$1")_profile"
FULL_FILE="${OUTPUT_NAME}.nsys-rep"

echo "-----------------------------------------------------"
echo "Profiling: $@"
echo "Output:    $FULL_FILE"
echo "-----------------------------------------------------"

# 5. Run Profiler
sudo nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output="${OUTPUT_NAME}" \
  "$@"

# 6. Launch GUI (if requested & successful)
if [ $? -eq 0 ] && [ "$OPEN_GUI" = true ] && [ -f "$FULL_FILE" ]; then
    echo "Opening Nsight Systems..."
    
    # Fix ownership (sudo made it root) so current user can open it
    sudo chown $USER "$FULL_FILE"
    sudo chown $USER "${OUTPUT_NAME}.sqlite" 2>/dev/null
    
    # Check which GUI command is available (prefer nsys-ui)
    if command -v nsys-ui &> /dev/null; then
        nsys-ui "$FULL_FILE"
    elif command -v nsight-sys &> /dev/null; then
        nsight-sys "$FULL_FILE"
    else
        echo "Error: Could not find 'nsys-ui' or 'nsight-sys' in PATH."
    fi
fi