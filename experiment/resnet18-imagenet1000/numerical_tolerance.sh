#!/bin/bash

for n_bit in 8 16 32; do
    for es in 0 1 2 3; do
        python numerical_tolerance.py --n-bit=$n_bit --es=$es --n-sample=8 > log${n_bit}_${es}_tolerance_result 2>&1 &
    done
done

wait