#!/bin/bash

for n_bit in 8 16 32; do
    for es in 0 1 2 3; do
        python run_tolerance.py --n-bit=$n_bit --es=$es > tolerance_log${n_bit}_${es} 2>&1 &
    done
done

wait