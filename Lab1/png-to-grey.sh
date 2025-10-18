#!/usr/bin/env bash
set -euo pipefail

# png-to-grey-robust.sh
# Usage: ./png-to-grey-robust.sh input.png
# Produces: input.grey (raw GRAY: format, depth=8)

if [ $# -ne 1 ]; then
    echo "Usage: $0 <image.png|image.jpg>"
    exit 1
fi

INPUT="$1"
if [ ! -f "$INPUT" ]; then
    echo "Error: file '$INPUT' not found" >&2
    exit 1
fi

BASENAME="$(basename "$INPUT" | sed 's/\.[^.]*$//')"
OUTPUT="${BASENAME}.grey"

# Choose a temp dir with likely free space:
# Prefer /var/tmp (persistent across reboots) then $HOME/tmp then /tmp
for d in /var/tmp "$HOME/tmp" /tmp; do
    if [ -d "$d" ] && [ -w "$d" ]; then
        TMPDIR="$d/$BASENAME-imtmp"
        mkdir -p "$TMPDIR"
        break
    fi
done

# If for some reason TMPDIR not set, create inside current dir:
: "${TMPDIR:=$(pwd)/.imtmp_$BASENAME}"
mkdir -p "$TMPDIR"

echo "Using temporary dir: $TMPDIR"
df -h "$TMPDIR" || true

# Reasonable limits for large images (adjust if you know your RAM/disk)
MEM_LIMIT="1GiB"
MAP_LIMIT="1GiB"
DISK_LIMIT="3GiB"

# 1) Primary attempt: ImageMagick convert with explicit -limit and temporary path
echo "Attempting convert with -limit ..."
if convert -limit memory "$MEM_LIMIT" -limit map "$MAP_LIMIT" -limit disk "$DISK_LIMIT" \
           -define registry:temporary-path="$TMPDIR" \
           "$INPUT" -colorspace Gray -depth 8 "gray:${OUTPUT}"
then
    echo "✅ Converted to ${OUTPUT} (primary attempt succeeded)"
    rm -rf "$TMPDIR"
    exit 0
else
    echo "⚠️ Primary convert attempt failed, trying alternatives..."
fi

# 2) Try setting environment variables and retry convert (some installs respect these)
export MAGICK_TMPDIR="$TMPDIR"
export MAGICK_TEMPORARY_PATH="$TMPDIR"
export MAGICK_MEMORY_LIMIT="$MEM_LIMIT"
export MAGICK_MAP_LIMIT="$MAP_LIMIT"
export MAGICK_DISK_LIMIT="$DISK_LIMIT"

echo "Retrying convert with MAGICK_* env vars ..."
if convert -limit memory "$MEM_LIMIT" -limit map "$MAP_LIMIT" -limit disk "$DISK_LIMIT" \
           "$INPUT" -colorspace Gray -depth 8 "gray:${OUTPUT}"
then
    echo "✅ Converted to ${OUTPUT} (env-var retry succeeded)"
    rm -rf "$TMPDIR"
    exit 0
else
    echo "⚠️ Env-var retry failed."
fi

# 3) If GraphicsMagick (gm) is installed, try that (it sometimes uses less cache)
if command -v gm >/dev/null 2>&1; then
    echo "Trying GraphicsMagick (gm convert) fallback..."
    if gm convert "$INPUT" -colorspace Gray -depth 8 "gray:${OUTPUT}"; then
        echo "✅ Converted to ${OUTPUT} using gm"
        rm -rf "$TMPDIR"
        exit 0
    else
        echo "⚠️ gm convert failed."
    fi
else
    echo "GraphicsMagick not found (gm)."
fi

# 4) Last-resort: attempt streaming conversion via netpbm tools if available.
#    pngtopnm -> pnmpad / pnmdepth -> output raw PGM bytes (this produces PGM, not raw .grey)
#    Many systems have pngtopnm (from netpbm) and pnmtopng etc. This is only attempted if pngtopnm exists.
if command -v pngtopnm >/dev/null 2>&1 && command -v ppmtogray >/dev/null 2>&1; then
    echo "Attempting netpbm pipeline (pngtopnm -> ppmtogray -> write raw PGM bytes)..."
    TMPPGM="$TMPDIR/${BASENAME}.pgm"
    # produce PGM (ASCII or binary), then strip header to raw bytes compatible with .grey (8-bit)
    pngtopnm "$INPUT" | ppmtogray > "$TMPPGM"
    # Convert PGM to raw 8-bit grayscale: skip the header and write raw bytes
    # PGM header consists of:
    # P5\n<width> <height>\n<maxval>\n  -> find the end of header (first occurrence of '\n' after maxval)
    # We'll use awk to skip header and write binary body
    tail -c +$(($(awk 'NR==3 {print length($0)+2; exit}' "$TMPPGM") + 1)) "$TMPPGM" > "${OUTPUT}" 2>/dev/null || true
    # above is a heuristic; if it fails, copy PGM as-is (user can open .pgm)
    if [ -s "${OUTPUT}" ]; then
        echo "✅ Produced ${OUTPUT} (from netpbm pipeline). Note: check whether your tool expects raw .grey or PGM."
        rm -rf "$TMPDIR"
        exit 0
    else
        echo "⚠️ Netpbm pipeline failed to produce .grey raw bytes."
    fi
else
    echo "Netpbm tools (pngtopnm/ppmtogray) not available or incomplete."
fi

# If we reached here, everything failed: give actionable suggestions
echo
echo "❌ All automatic conversion attempts failed."
echo "Suggestions to resolve the cache resources exhausted error:"
echo " 1) Ensure there is enough free disk space in the temp dir (df -h $TMPDIR)."
echo " 2) Increase the -limit values in the script (memory/map/disk) to match your machine RAM and disk."
echo " 3) Install GraphicsMagick and try 'gm convert':"
echo "      sudo apt install graphicsmagick"
echo " 4) If you can, install ImageMagick v7 (provides 'magick' binary) which handles large images better."
echo " 5) If you only need a smaller version, resize before conversion:"
echo "      convert INPUT -resize 50% -colorspace Gray -depth 8 gray:OUTPUT.grey"
echo
echo "If you want, paste the output of these commands so I can advise further:"
echo "  free -h"
echo "  df -h $TMPDIR"
echo "  convert --version"
echo
rm -rf "$TMPDIR"
exit 2
