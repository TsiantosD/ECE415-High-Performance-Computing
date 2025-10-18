#!/usr/bin/env bash
set -euo pipefail

GOLDEN_VERSION=1_sobel_orig

# Usage:
# ./generate-golden.sh SIZE-IMAGE_NAME.png
# Example:
# ./generate-golden.sh 9999-shibuya.png

if [ $# -ne 1 ]; then
    echo "Usage: $0 SIZE-IMAGE_NAME.png"
    exit 1
fi

INPUT_PNG="$1"

if [ ! -f "$INPUT_PNG" ]; then
    echo "Error: File '$INPUT_PNG' not found!"
    exit 1
fi

# Extract SIZE and IMAGE_NAME from the filename
# Expecting: SIZE-IMAGE_NAME.png
BASENAME=$(basename "$INPUT_PNG" .png)
SIZE="${BASENAME%%-*}"
IMAGE_NAME="${BASENAME#*-}"

if [[ -z "$SIZE" || -z "$IMAGE_NAME" ]]; then
    echo "Error: Input file name must be in format SIZE-IMAGE_NAME.png"
    exit 1
fi

echo "Processing image: $INPUT_PNG"
echo "SIZE: $SIZE, IMAGE_NAME: $IMAGE_NAME"

# Step 1: Convert PNG → .grey using png-to-grey.sh
./png-to-grey.sh "$INPUT_PNG"

GREY_FILE="${SIZE}-${IMAGE_NAME}.grey"

if [ ! -f "$GREY_FILE" ]; then
    echo "Error: Grey file '$GREY_FILE' not created by png-to-grey.sh"
    exit 1
fi

# Step 2: Move grey image to ./src/input/
mkdir -p ./src/input
mv "$GREY_FILE" "./src/input/$GREY_FILE"

# Step 3: Create empty placeholder in ./src/golden/
mkdir -p ./src/golden
PLACEHOLDER="./src/golden/${IMAGE_NAME}.grey"
touch "$PLACEHOLDER"

# Step 4: Run the main script
./run.sh --execution-method=normal --executable=$GOLDEN_VERSION --times=1 --calculate-average=false --image="$GREY_FILE"

# Step 5: Move generated output to ./src/golden/, overwriting placeholder
OUTPUT_FILE="./src/output/${IMAGE_NAME}.grey"
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Expected output file '$OUTPUT_FILE' not found after run.sh"
    exit 1
fi

mv "$OUTPUT_FILE" "$PLACEHOLDER"

echo "✅ Golden image generated: $PLACEHOLDER"
