CC=gcc
CFLAGS=-c -Wall

all: serial_decomp parallel_decomp

parallel_decomp: parallel_decomp.c
	mpiicc -O0 -g -openmp parallel_decomp.c -o parallel_decomp -lm

serial_decomp: serial_decomp.o
	$(CC) serial_decomp.o -o serial_decomp

parallel_decomp.o: parallel_decomp.c
	mpicc -c parallel_decomp.c

serial_decomp.o: serial_decomp.c
	$(CC) $(CFLAGS) serial_decomp.c

clean:
	rm serial_decomp.o serial_decomp parallel_decomp.o parallel_decomp 
