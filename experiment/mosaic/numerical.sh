#!/bin/bash

for n_bit in 8 16; do
    for es in 0 1 2 3; do
        python numerical.py --n-bit=$n_bit --es=$es --n-sample=2 > log${n_bit}_${es}_result 2>&1 &
    done
done

wait