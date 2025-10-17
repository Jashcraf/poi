#!/bin/bash

# List of numbers to loop over
numbers=(10.5 11 11.5)

# Loop through each number
for num in "${numbers[@]}"; do
    echo "Processing throughput: 10^-$num"
    python gen_dst2_aplc.py "$num"
done
