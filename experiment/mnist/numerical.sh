#!/bin/bash

for n_bit in 8 16 32; do
    for es in 0 1 2 3; do
        python numerical.py --n-bit=$n_bit --es=$es --n-sample=250 > log${n_bit}_${es}_result 2>&1 &
        python runfp16.py -m /home/sylvex/onnx-mlir/experiment/mnist/model/model.onnx --n-bit=$n_bit --es=$es --n-sample=250 > log${n_bit}_${es}_fp16 2>&1 &
    done
done

wait