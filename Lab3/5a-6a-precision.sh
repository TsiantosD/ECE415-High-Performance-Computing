#!/bin/bash

# ================= CONFIGURATION =================
IMAGE_SIZE=1024
REPEAT=10

# Power-of-two kernel radii
RADII=(1 2 4 8 16 32 64 128 256)

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
    for R in "${RADII[@]}"; do
        ./run.sh \
            --src-dir step6 \
            --filter-radius "$R" \
            --image-size "$IMAGE_SIZE" \
            --repeat "$REPEAT" \
            --use-doubles false \
            --output-dir 5a-precision
    done
fi

# ---------------------------
# Run double commands
# ---------------------------
if [ "$RUN_DOUBLES" = true ]; then
    echo "Running with doubles..."
    for R in "${RADII[@]}"; do
        ./run.sh \
            --src-dir step6 \
            --filter-radius "$R" \
            --image-size "$IMAGE_SIZE" \
            --repeat "$REPEAT" \
            --use-doubles true \
            --output-dir 5b-precision
    done
fi
