#!/bin/bash

# Parse arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <exec> <csv_file> [qemu] [qemu_flags]"
    echo "      <exec>: path to the GEMMVBench executable"
    echo "      <csv_file>: path to the CSV file containing GEMM configurations."
    exit 1
fi

# Override QEMU and QEMU_FLAGS if provided as command-line arguments
EXEC=$1
CSV_FILE=$2

# Read CSV header
IFS=, read -r -a HEADER < "$CSV_FILE"

# Initialize counters
success_count=0
failure_count=0

# Process CSV entries
while IFS=, read -r M N K ALPHA BETA; do #"${HEADER[@]}"; do

    echo "$M $N $K $ALPHA $BETA"
    if ./$EXEC $M $N $K $ALPHA $BETA; then
        echo -e "\tsuccess"
        success_count=$((success_count + 1))
    else
        echo "$M $N $K $ALPHA $BETA"
        echo -e "\tFailure"
        failure_count=$((failure_count + 1))
    fi
done < <(tail -n +2 "$CSV_FILE")

# Print summary
echo "Total Success: $success_count"
echo "Total Failure: $failure_count"
