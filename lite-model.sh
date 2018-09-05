#!/bin/sh -e

g++ -O3 -o /tmp/lite-model -std=c++14 $(dirname $0)/lite-model.cc \
    -I $(dirname $0)/include -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
    -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lcuda -lcudart -lcublas -lcudnn -lpthread

LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH exec /tmp/lite-model "$@" 2>/dev/null
