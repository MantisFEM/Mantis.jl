#!/bin/bash

# Script to find and run all runbenchmarks.jl files from the tools folder,
# cd'ing into the benchmarks directory first.

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$SCRIPT_DIR/../benchmarks"

# Change to the benchmarks directory
cd "$BENCHMARKS_DIR" || { echo "Could not cd into $BENCHMARKS_DIR"; exit 1; }

# Find and run all runbenchmarks.jl files
find . -type f -name 'runbenchmarks.jl' | while read -r benchmark_file; do
    echo "Running $benchmark_file"
    julia --project=. "$benchmark_file"
    if [ $? -ne 0 ]; then
        echo "Error running $benchmark_file"
        exit 1
    fi
done

echo "All benchmarks completed."
