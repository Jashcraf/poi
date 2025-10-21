#!/bin/bash

# List of numbers to loop over
numbers=(2.5 3 3.5 4 4.5 5 5.5 6)

# Loop through each number
for num in "${numbers[@]}"; do
    echo "Processing IWA: $num"
    python iwa_survey_luvoirb.py "$num"
done
