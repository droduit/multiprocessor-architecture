
NVCC = nvcc
CFLAGS = -Wno-deprecated-gpu-targets -c -O3
LFLAGS = -Wno-deprecated-gpu-targets -lcudart -lcuda

all: assignment4.o implementation.o
	$(NVCC) $(LFLAGS) assignment4.o implementation.o -o assignment4

assignment4.o: assignment4.cu utility.h implementation.o
	$(NVCC) $(CFLAGS) $< -o $@

implementation.o: implementation.cu
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o assignment4


