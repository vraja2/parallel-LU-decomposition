CC=gcc
CFLAGS=-c -Wall

all: serial_decomp

serial_decomp: serial_decomp.o
	$(CC) serial_decomp.o -o serial_decomp

serial_decomp.o: serial_decomp.c
	$(CC) $(CFLAGS) serial_decomp.c

clean:
	rm serial_decomp.o serial_decomp 
