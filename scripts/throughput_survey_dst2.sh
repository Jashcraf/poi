#!/bin/bash

# List of numbers to loop over
numbers=(15 15.5 16 16.5 17 17.5 18 18.5 19 19.5 20)

# Loop through each number
for num in "${numbers[@]}"; do
    echo "Processing throughput: 10^-$num"
    python gen_dst2_aplc.py "$num"
done
