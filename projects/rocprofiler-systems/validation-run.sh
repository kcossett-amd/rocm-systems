#!/bin/bash

set -e  
cd "$(dirname "$0")"

export ROCPROFSYS_USE_OMPT=1
export OMP_OFFLOAD=mandatory

BUILD_FOLDER="rocprof-sys-build"
PROGRAM=""
DELETE_FLAG=false

# Function to show help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p PROGRAM    Specify the program to use. Options: gpu, mutex"
    echo "  -d            Clean up folders used during script and exit"
    echo "  -h            Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -p gpu"
    echo "  $0 -p mutex"
    echo "  $0 -d         # Cleanup only"
}

# WALL CLOCK STUFF
# export ROCPROFSYS_PROFILE=ON
# export ROCPROFSYS_USE_SAMPLING=ON
# export ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock

# Parse command line arguments
while getopts "p:dh" opt; do
    case $opt in
        p)
            PROGRAM="$OPTARG"
            ;;
        d)
            DELETE_FLAG=true
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            show_help
            exit 1
            ;;
    esac
done

if [[ "$DELETE_FLAG" == true ]]; then
    echo "Cleaning up directories..."
    BIN_DIR="validation-run-binaries"
    LOG_DIR="validation-run-logs"
    TRACE_OUTPUT_DIR="$PWD/validation-run-traces"
    
    rm -rf "$BIN_DIR" "$LOG_DIR" "$TRACE_OUTPUT_DIR"
    
    # Remove instrumented binary directories
    rm -rf rocprofsys-gpu.inst*
    rm -rf rocprofsys-mutex.inst*
    
    echo "Cleanup completed - removed $BIN_DIR, $LOG_DIR, $TRACE_OUTPUT_DIR, and instrumented binary directories"
    echo "Exiting..."
    exit 0
fi

# Validate PROGRAM argument
if [[ -z "$PROGRAM" ]]; then
    echo "Error: Program must be specified with -p option"
    show_help
    exit 1
fi

if [[ "$PROGRAM" != "gpu" && "$PROGRAM" != "mutex" ]]; then
    echo "Error: Program must be either 'gpu' or 'mutex'"
    show_help
    exit 1
fi

# Check if BUILD_FOLDER exists
if [[ ! -d "$BUILD_FOLDER" ]]; then
    echo "Error: Build folder '$BUILD_FOLDER' does not exist."
    echo "Please build the project first, or update BUILD_FOLDER variable to reflect the correct build directory name."
    exit 1
fi

export ROCPROFSYS_BIN_DIR="$PWD/$BUILD_FOLDER/bin"

if [[ ! -f "$ROCPROFSYS_BIN_DIR/rocprof-sys-instrument" ]] || [[ ! -f "$ROCPROFSYS_BIN_DIR/rocprof-sys-run" ]]; then
    echo "Error: rocprof-sys binaries not found in $ROCPROFSYS_BIN_DIR"
    echo "Please ensure the project is built correctly."
    exit 1
fi

# Display GPU-specific message
if [[ "$PROGRAM" == "gpu" ]]; then
    echo 'If using gpu program, ensure openmp lib is in library path (export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH)'
fi

SRC="$PROGRAM.f90"
BIN_DIR="validation-run-binaries"
LOG_DIR="validation-run-logs"
BIN="$BIN_DIR/$PROGRAM"
INSTR="$BIN_DIR/$PROGRAM.inst"
TRACE_OUTPUT_DIR="$PWD/validation-run-traces"

# Check if source file exists
if [[ ! -f "$SRC" ]]; then
    echo "Error: Source file '$SRC' not found."
    exit 1
fi

mkdir -p "$BIN_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$TRACE_OUTPUT_DIR"

echo "Compiling with amdflang (v20 required)..."
amdflang -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa "$SRC" -o "$BIN"
echo "Compiling Completed"

echo "Instrumenting..."
"$ROCPROFSYS_BIN_DIR/rocprof-sys-instrument" -o "$INSTR" -- "$BIN" > "$LOG_DIR/instrument.log" 2>&1
echo "Instrumenting Completed"

echo "Running instrumented binary..."
"$ROCPROFSYS_BIN_DIR/rocprof-sys-run" --debug -o "$TRACE_OUTPUT_DIR" -- "$INSTR" > "$LOG_DIR/run.log" 2>&1
echo "Running Completed"

echo "Saving output to $LOG_DIR/run.log"
echo "Trace output saved to $TRACE_OUTPUT_DIR"

echo "Script completed successfully!"

echo ""
echo ""
echo "Check $LOG_DIR/run.log for lines with \"[ROCPROFILER-SDK]\". You can verify your changes there"
echo ""
echo ""