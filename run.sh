#!/bin/sh

g++ -O3 -o ./lite-model -std=c++14 lite-model.cc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcblas -lcudnn

exec ./lite-model "$@"
