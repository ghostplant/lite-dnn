#!/bin/sh -e

${CC:-/usr/local/rocm/bin/c++} -O2 -o lite-model -std=c++14 $(dirname $0)/lite-model.cc \
    -I $(dirname $0)/include -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
    -lcuda -lcudart -lcublas -lcudnn -lpthread \
    -lopencv_core -lopencv_highgui -lopencv_imgproc

./lite-model "$@" 2>/dev/null
