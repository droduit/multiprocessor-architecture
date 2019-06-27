/* 
 *
 *  CUDA Example
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include <cuda_runtime.h>

using namespace std;

void incrementArrayOnHost(float *a, int N)
{
	int i;
	for (i=0; i < N; i++) a[i] = a[i]+5.f;
}

__global__ void incrementArrayOnDevice(float *a, int N)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	a[idx] = a[idx]+5.f;
}

int main(void)
{
	float *a_h, *b_h;           // pointers to host memory
	float *a_d;                 // pointer to device memory
	int i, N = 1024;
	size_t size = N*sizeof(float);

	// allocate arrays on host
	a_h = (float *)malloc(size);
	b_h = (float *)malloc(size);

	cudaSetDevice(0);

	// allocate array on device 
	if (cudaMalloc((void **) &a_d, size) != cudaSuccess)
		cout << "error in cudaMalloc" << endl;

	// initialization of host data
	for (i=0; i<N; i++) a_h[i] = (float)i;

	// copy data from host to device
	if (cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "error in cudaMemcpy" << endl;

	// do calculation on host
	incrementArrayOnHost(a_h, N);

	// do calculation on device:
	incrementArrayOnDevice <<< 2, N/2 >>> (a_d, N);
	cudaThreadSynchronize();

	// Retrieve result from device and store in b_h
	if (cudaMemcpy(b_h, a_d, size, cudaMemcpyDeviceToHost) != cudaSuccess)
		cout << "error in cudaMemcpy" << endl;

	// check results
	for (i=0; i<N; i++) assert(a_h[i] == b_h[i]);

	// cleanup
	free(a_h); 
	free(b_h);
	cudaFree(a_d); 

	return 0;
}
