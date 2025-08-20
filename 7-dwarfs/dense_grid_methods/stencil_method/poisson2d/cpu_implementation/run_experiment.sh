#!/bin/bash

# Parameter sets: "size_x,size_y,iterations"
parameter_sets=(
    "30,30,60000,4"
    "60,60,60000,4"
    "100,100,60000,4"
)

# Path to the executable
EXECUTABLE=./poisson2d

# Check if the executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not found. Please compile the code first." >&2
    exit 1
fi

# Run the experiments
for params in "${parameter_sets[@]}"; do
    IFS=',' read -r size_x size_y iterations batch<<< "$params"
    echo "Running with size_x=$size_x, size_y=$size_y, iterations=$iterations"
    $EXECUTABLE -sizex="$size_x" -sizey="$size_y" -iters="$iterations" -batch="$batch"
    echo "------------------------------------"
done
