#!/bin/bash

# ================= CONFIGURATION =================
SIZES=(64 128 256 512 1024 2048 4096 8192 16384)
FILTER_RADIUS=16
REPEAT=10

# Prompt user for type
echo "Choose data type to run:"
echo "1) float"
echo "2) double"
echo "3) both"
read -p "Enter choice [1-3]: " CHOICE

RUN_FLOATS=false
RUN_DOUBLES=false

case $CHOICE in
    1) RUN_FLOATS=true ;;
    2) RUN_DOUBLES=true ;;
    3) RUN_FLOATS=true; RUN_DOUBLES=true ;;
    *) echo "Invalid choice."; exit 1 ;;
esac

# ---------------------------
# Run float commands
# ---------------------------
if [ "$RUN_FLOATS" = true ]; then
    echo "Running with floats..."
    for SIZE in "${SIZES[@]}"; do
        ./run.sh \
            --src-dir step6 \
            --filter-radius "$FILTER_RADIUS" \
            --image-size "$SIZE" \
            --repeat "$REPEAT" \
            --use-doubles false \
            --output-dir 5b-runtimes
    done
fi

# ---------------------------
# Run double commands
# ---------------------------
if [ "$RUN_DOUBLES" = true ]; then
    echo "Running with doubles..."
    for SIZE in "${SIZES[@]}"; do
        ./run.sh \
            --src-dir step6 \
            --filter-radius "$FILTER_RADIUS" \
            --image-size "$SIZE" \
            --repeat "$REPEAT" \
            --use-doubles true \
            --output-dir 6b-runtimes
    done
fi
