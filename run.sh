#!/bin/sh -e

# Example: ./run.sh mnist_mlp


g++ -O3 -o $(dirname $0)/lite-model -std=c++14 $(dirname $0)/lite-model.cc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn

LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH exec $(dirname $0)/lite-model "$@"
