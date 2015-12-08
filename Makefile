SHELL=/bin/bash

CC=gcc
CFLAGS=-c -Wall

CCFLAGS = -O1 -lm
MKFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR = $(dir $(MKFILE_PATH))
PAPI_DIR = $(MKFILE_DIR)papi
PAPI_INCLUDE = -I$(PAPI_DIR)/include
PAPI_LIB = $(PAPI_DIR)/lib/libpapi.a

all: serial_decomp parallel_decomp parallel_decomp_cyclic

papi: papi_cyclic papi_block

papi_cyclic: parallel_decomp_cyclic.c support.h
	mpiicc $(PAPI_INCLUDE) -O0 -g -openmp $< $(PAPI_LIB) -o parallel_decomp_cyclic -lm
	
papi_block: parallel_decomp.c support.h
	mpiicc $(PAPI_INCLUDE) -O0 -g -openmp $< $(PAPI_LIB) -o parallel_decomp -lm

parallel_decomp_cyclic: parallel_decomp_cyclic.c
	mpiicc -O0 -g -openmp parallel_decomp_cyclic.c -o parallel_decomp_cyclic -lm

parallel_decomp: parallel_decomp.c
	mpiicc -O0 -g -openmp parallel_decomp.c -o parallel_decomp -lm

serial_decomp: serial_decomp.o
	$(CC) serial_decomp.o -o serial_decomp

parallel_decomp.o: parallel_decomp.c
	mpicc -c parallel_decomp.c
	
parallel_decomp_cyclic.o: parallel_decomp_cyclic.c
	mpicc -c parallel_decomp_cyclic.c


serial_decomp.o: serial_decomp.c
	$(CC) $(CFLAGS) serial_decomp.c

clean:
	rm serial_decomp.o serial_decomp parallel_decomp.o parallel_decomp parallel_decomp_cyclic.o parallel_decomp_cyclic
