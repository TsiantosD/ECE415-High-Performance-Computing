#!/bin/bash

# ================= CONFIGURATION =================
# Name of the executable file *inside* the step folder
# e.g., if full path is "step1/solver", set this to "solver"
PROGRAM_NAME="convolution2d" 

FIXED_SECOND_INPUT=32
DEFAULT_START=0
DEFAULT_LIMIT=15
# =================================================

# Initialize variables
STEP_DIR=""
USE_DOUBLES=false
DISABLE_FMAD=false
INPUTS_ARRAY=()

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message and exit"
    echo "  --step <dir_name>     Specify the step directory (e.g., step1). If omitted, scans and prompts."
    echo "  --use-doubles         Compile with USE_DOUBLES=1 (Adds '_dbl' to output folder)"
    echo "  --disable-fmad        Compile with FMAD=false (Adds '_nofmad' to output folder)"
    echo "  --inputs \"1 5 10\"     Standard Mode: Space-separated string of inputs. Defaults to range $DEFAULT_START-$DEFAULT_LIMIT."
    echo "                        Defaults to range $DEFAULT_START-$DEFAULT_LIMIT."
    echo "  --repeat <N>          Repeat Mode: Run a specific input N times."
    echo "  --target <input>      Repeat Mode: The specific input base to run."
    exit 0
}

# 1. PARSE ARGUMENTS
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            usage
            ;;
        --step)
            STEP_DIR="$2"
            shift 2
            ;;
        --use-doubles)
            USE_DOUBLES=true
            shift
            ;;
        --disable-fmad)
            DISABLE_FMAD=true
            shift
            ;;
        --inputs)
            # Read space-separated string into array
            IFS=' ' read -r -a INPUTS_ARRAY <<< "$2"
            shift 2
            ;;
        --repeat)
            REPEAT_COUNT="$2"
            shift 2
            ;;
        --target)
            TARGET_INPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ "$REPEAT_COUNT" -gt 0 ]; then
    # --- Repeat Mode ---
    if [ -z "$TARGET_INPUT" ]; then
        echo "Error: You must specify --target <input> when using --repeat."
        exit 1
    fi
fi

# 2. HANDLE STEP SELECTION (Prompt if missing)
if [ -z "$STEP_DIR" ]; then
    # Find directories starting with "step"
    # The 'find' command looks for folders matching "step*" in current dir
    OPTIONS=($(find . -maxdepth 1 -type d -name "step*" -printf "%f\n" | sort))
    
    if [ ${#OPTIONS[@]} -eq 0 ]; then
        echo "Error: No directories starting with 'step' found."
        exit 1
    fi
    
    if [ ${#OPTIONS[@]} -eq 1 ]; then
        STEP_DIR=${OPTIONS[0]}
        echo "Auto-selected only available step: $STEP_DIR"
    else
        echo "Multiple step directories found. Please select one:"
        PS3="Enter number: "
        select opt in "${OPTIONS[@]}"; do
            if [ -n "$opt" ]; then
                STEP_DIR=$opt
                break
            else
                echo "Invalid selection."
            fi
        done
    fi
fi

# Verify the executable path logic
TARGET_EXEC="./$STEP_DIR/$PROGRAM_NAME"

# 3. CONSTRUCT SUFFIX BASED ON OPTIONS
SUFFIX="${STEP_DIR}"

if [ "$USE_DOUBLES" = true ]; then
    SUFFIX="${SUFFIX}_dbl"
fi

if [ "$DISABLE_FMAD" = true ]; then
    SUFFIX="${SUFFIX}_nofmad"
fi

OUTPUT_DIR="output/${SUFFIX}"

# Generate Batch Timestamp ONCE for the folder
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${BATCH_TIMESTAMP}"

# 4. COMPILE (RUN MAKE)
echo "----------------------- COMPILATION -----------------------"
echo "Preparing '$STEP_DIR'..."
printf "Configuration: Doubles=$USE_DOUBLES, FMAD_Disabled=$DISABLE_FMAD\n"

# Construct Make command
MAKE_CMD="make -C $STEP_DIR"

if [ "$USE_DOUBLES" = true ]; then
    MAKE_CMD="$MAKE_CMD USE_DOUBLES=1"
fi

if [ "$DISABLE_FMAD" = true ]; then
    MAKE_CMD="$MAKE_CMD FMAD=false"
fi

printf "\nRunning: make -C $STEP_DIR clean\n"
make -C $STEP_DIR clean
printf "\nRunning: $MAKE_CMD\n"
$MAKE_CMD

# Check if make was successful
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi

if [ ! -f "$TARGET_EXEC" ]; then
    echo "Error: Executable '$TARGET_EXEC' not found after make."
    exit 1
fi

# 5. EXECUTION LOOP

# Create output directory
if [ ! -d "$RUN_OUTPUT_DIR" ]; then
    mkdir -p "$RUN_OUTPUT_DIR"
fi

echo "-----------------------------------------------------------"
printf "\n\n------------------------ EXECUTION ------------------------\n"
echo "Starting execution. Output folder: $OUTPUT_DIR"



# BRANCH: REPEAT MODE vs STANDARD MODE
if [ "$REPEAT_COUNT" -gt 0 ]; then

    # Calculate actual input: 2 * i + 1
    INPUT_VAL=$((2 * TARGET_INPUT + 1))
    
    echo "Mode: Repeating filter radius $TARGET_INPUT (Length: $INPUT_VAL) for $REPEAT_COUNT times."

    for (( j=1; j<=REPEAT_COUNT; j++ )); do
        # Filename includes repetition index to avoid overwrite
        FILENAME="out_${INPUT_VAL}_${FIXED_SECOND_INPUT}_rep${j}"
        FILEPATH="${RUN_OUTPUT_DIR}/${FILENAME}"
        
        echo "$INPUT_VAL $FIXED_SECOND_INPUT" | $TARGET_EXEC > "$FILEPATH" 2>&1
        
        echo -ne "Run $j/$REPEAT_COUNT -> $FILENAME\r"
    done

else
    # --- Standard Mode ---
    
    # If no custom inputs provided, use default 0 to LIMIT-1
    if [ ${#INPUTS_ARRAY[@]} -eq 0 ]; then
        echo "No inputs provided, using default range ($DEFAULT_START to $DEFAULT_LIMIT)."
        INPUTS_ARRAY=($(seq $DEFAULT_START $DEFAULT_LIMIT))
    fi

    for i in "${INPUTS_ARRAY[@]}"; do
        # Calculate actual input: 2 * i + 1
        INPUT_VAL=$((2 * i + 1))

        # Generate timestamp
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        
        FILENAME="out_${INPUT_VAL}_${FIXED_SECOND_INPUT}"
        FILEPATH="${RUN_OUTPUT_DIR}/${FILENAME}"
        
        # Run program with inputs piped to stdin
        # Note: We are running the compiled executable now, not python
        echo "$INPUT_VAL $FIXED_SECOND_INPUT" | $TARGET_EXEC > "$FILEPATH" 2>&1
        
        # Simple progress indicator
        echo -ne "Processed filter radius $i (Length: $INPUT_VAL) -> $FILENAME\r"
    done
fi

echo -e "\nExecution complete."
echo "-----------------------------------------------------------"