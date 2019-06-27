
CC = gcc
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall

all: pi integral

pi: pi.c utility.h
	$(CC) $(CFLAGS) $< -o $@

integral: integral.c function.o utility.h
	$(CC) $(CFLAGS) $< -o $@
	
clean:
	rm -f pi integral function.o
