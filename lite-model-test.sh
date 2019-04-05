#!/bin/sh -e

${CC:-/usr/bin/g++} -O2 -o lite-model -std=c++14 $(dirname $0)/lite-model.cc -DTEST_ONLY \
    -I $(dirname $0)/include -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
    -lcuda -lcudart -lcublas -lcudnn -lpthread \

./lite-model "$@" 2>/dev/null
