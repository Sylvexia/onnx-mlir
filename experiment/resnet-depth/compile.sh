#!/bin/bash

for n_bit in 8 16 32; do
    for es in 0 1 2 3; do
        python compile.py --n-bit=$n_bit --es=$es
    done
done

wait