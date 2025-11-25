#!/bin/bash

# ================= CONFIGURATION =================
SIZES=(64 128 256 512 1024 2048 4096 8192 16384)
FILTER_RADIUS=16
REPEAT=10

# ---------------------------
# Run float commands
# ---------------------------
echo "Running with floats..."
for SIZE in "${SIZES[@]}"; do
    ./run.sh \
        --src-dir step8 \
        --filter-radius "$FILTER_RADIUS" \
        --image-size "$SIZE" \
        --repeat "$REPEAT" \
        --use-doubles false \
        --output-dir 8-runtimes
done