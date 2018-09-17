#!/bin/sh -e

mpic++ -O3 -o lite-model -std=c++14 $(dirname $0)/lite-model.cc \
    -I $(dirname $0)/include -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
    -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lcuda -lcudart -lcublas -lcudnn -lpthread -lnccl

HOSTS="${HOSTS:-localhost}"
MPI_HOST_NUM=$(echo "${HOSTS}" | tr \, \\n | wc -l)

IFNAME=${IFNAME:-enp216s0}

LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH exec mpiexec --allow-run-as-root --bind-to none \
    --mca oob_tcp_if_include ${IFNAME} --mca btl_tcp_if_include ${IFNAME} -x NCCL_SOCKET_IFNAME=${IFNAME} \
    --map-by slot -np ${MPI_HOST_NUM} --host ${HOSTS} ./lite-model "$@" 2>/dev/null
