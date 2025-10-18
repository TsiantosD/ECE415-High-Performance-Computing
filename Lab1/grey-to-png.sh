#!/usr/bin/env bash
set -euo pipefail

# grey-to-png-robust.sh
# Usage: ./grey-to-png-robust.sh input.grey width height [output.png]

if [ $# -lt 3 ]; then
    echo "Usage: $0 <input.grey> <width> <height> [output.png]"
    echo "Example: $0 ./src/input/9999-shibuya.grey 9999 9999"
    exit 1
fi

INPUT="$1"
WIDTH="$2"
HEIGHT="$3"
OUTPUT="${4:-$(basename "$INPUT" .grey).png}"

if [ ! -f "$INPUT" ]; then
    echo "Error: File '$INPUT' not found!"
    exit 1
fi

# Select a safe temp directory
for d in /var/tmp "$HOME/tmp" /tmp; do
    if [ -d "$d" ] && [ -w "$d" ]; then
        TMPDIR="$d/$(basename "$INPUT")-imtmp"
        mkdir -p "$TMPDIR"
        break
    fi
done
: "${TMPDIR:=$(pwd)/.imtmp_$(basename "$INPUT")}"
mkdir -p "$TMPDIR"

echo "Using temporary dir: $TMPDIR"
df -h "$TMPDIR" || true

# Configure resource limits
MEM_LIMIT="1GiB"
MAP_LIMIT="1GiB"
DISK_LIMIT="3GiB"

# --- Primary attempt ---
echo "Attempting convert with -limit ..."
if convert -limit memory "$MEM_LIMIT" -limit map "$MAP_LIMIT" -limit disk "$DISK_LIMIT" \
           -define registry:temporary-path="$TMPDIR" \
           -depth 8 -size "${WIDTH}x${HEIGHT}" "GRAY:${INPUT}" "$OUTPUT"
then
    echo "✅ Successfully created $OUTPUT"
    rm -rf "$TMPDIR"
    exit 0
else
    echo "⚠️ Primary convert attempt failed, retrying with environment overrides..."
fi

# --- Retry with environment overrides ---
export MAGICK_TMPDIR="$TMPDIR"
export MAGICK_TEMPORARY_PATH="$TMPDIR"
export MAGICK_MEMORY_LIMIT="$MEM_LIMIT"
export MAGICK_MAP_LIMIT="$MAP_LIMIT"
export MAGICK_DISK_LIMIT="$DISK_LIMIT"

if convert -limit memory "$MEM_LIMIT" -limit map "$MAP_LIMIT" -limit disk "$DISK_LIMIT" \
           -depth 8 -size "${WIDTH}x${HEIGHT}" "GRAY:${INPUT}" "$OUTPUT"
then
    echo "✅ Successfully created $OUTPUT (after retry)"
    rm -rf "$TMPDIR"
    exit 0
else
    echo "⚠️ Env-var retry failed."
fi

# --- Fallback to GraphicsMagick (if installed) ---
if command -v gm >/dev/null 2>&1; then
    echo "Trying GraphicsMagick fallback..."
    if gm convert -depth 8 -size "${WIDTH}x${HEIGHT}" "GRAY:${INPUT}" "$OUTPUT"; then
        echo "✅ Successfully created $OUTPUT using gm"
        rm -rf "$TMPDIR"
        exit 0
    fi
fi

# --- Optional: convert via PNM pipeline ---
if command -v rawtopgm >/dev/null 2>&1 && command -v pnmtopng >/dev/null 2>&1; then
    echo "Trying NetPBM fallback (rawtopgm → pnmtopng)..."
    TMPPGM="$TMPDIR/tmp.pgm"
    rawtopgm "$WIDTH" "$HEIGHT" "$INPUT" > "$TMPPGM"
    pnmtopng "$TMPPGM" > "$OUTPUT"
    if [ -s "$OUTPUT" ]; then
        echo "✅ Created $OUTPUT via NetPBM fallback."
        rm -rf "$TMPDIR"
        exit 0
    fi
fi

# --- If everything failed ---
echo
echo "❌ Conversion failed for $INPUT"
echo "Possible causes and fixes:"
echo "  1) Not enough RAM or /tmp space (check with: df -h $TMPDIR)"
echo "  2) Try increasing limits at the top of this script:"
echo "       MEM_LIMIT, MAP_LIMIT, DISK_LIMIT"
echo "  3) Install GraphicsMagick (gm):"
echo "       sudo apt install graphicsmagick"
echo "  4) Resize the input before conversion:"
echo "       convert -depth 8 -size ${WIDTH}x${HEIGHT} GRAY:$INPUT -resize 50% output.png"
echo "  5) Upgrade to ImageMagick 7 ('magick' binary)."
echo
echo "For diagnostics, please run:"
echo "  free -h"
echo "  df -h $TMPDIR"
echo "  convert --version"
rm -rf "$TMPDIR"
exit 2
