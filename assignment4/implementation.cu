/*
============================================================================
Filename    : algorithm.c
Author      : Dominique Roduit
SCIPER      : 234868
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for(int n=0; n<(int) iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                            input[(i-1)*(length)+(j)]   +
                                            input[(i-1)*(length)+(j+1)] +
                                            input[(i)*(length)+(j-1)]   +
                                            input[(i)*(length)+(j)]     +
                                            input[(i)*(length)+(j+1)]   +
                                            input[(i+1)*(length)+(j-1)] +
                                            input[(i+1)*(length)+(j)]   +
                                            input[(i+1)*(length)+(j+1)] ) / 9;

            }
        }
        output[(length/2-1)*length+(length/2-1)] = 1000;
        output[(length/2)*length+(length/2-1)]   = 1000;
        output[(length/2-1)*length+(length/2)]   = 1000;
        output[(length/2)*length+(length/2)]     = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}

__global__ void processOnDevice(double *input, double *output, int length)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	int idx = (i)*(length)+(j);
	
	int isHotCore = (idx == (length/2-1)*length+(length/2-1)) ||
					 (idx == (length/2)*length+(length/2-1)) ||
					 (idx == (length/2-1)*length+(length/2)) ||
					 (idx == (length/2)*length+(length/2));
	
	if(i >= 1 && i < length-1 && j >= 1 && j < length-1 && !isHotCore) {
		output[idx] = (input[(i-1)*(length)+(j-1)] +
						input[(i-1)*(length)+(j)]   +
						input[(i-1)*(length)+(j+1)] +
						input[(i)*(length)+(j-1)]   +
						input[(i)*(length)+(j)]     +
						input[(i)*(length)+(j+1)]   +
						input[(i+1)*(length)+(j-1)] +
						input[(i+1)*(length)+(j)]   +
						input[(i+1)*(length)+(j+1)] ) / 9;
	}
	
}

/*
15 SMs, max 64 warps per SM 
2048 threads max per SM
16 thread blocks max per SM
960 concurrently scheduled warps/GPU. You can launch more, but won’t start until others finish
*/

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    /* Preprocessing goes here ------------------------------- */
    int thrsPerBlock = 8; 
    int nBlks =  ceil((double)length/thrsPerBlock);
    
    // We organize the thread blocks into 2D arrays of threads.
    dim3 gridSize(nBlks, nBlks);
    dim3 blockSize(thrsPerBlock,thrsPerBlock);
  
    double *input_d, *output_d; // pointers to device memory
    size_t size = sizeof(double)*length*length;
    
    // allocate input on device
    if(cudaMalloc((void**) &input_d, size) != cudaSuccess)
		cout << "error in input cudaMalloc" << endl;

	// allocate output on device
	if(cudaMalloc((void**) &output_d, size) != cudaSuccess)
		cout << "error in output cudaMalloc" << endl;
	// ---------------------------------------------------------- 
	
	/* Copying array from H to D  ---------------------------- */
    cudaEventRecord(cpy_H2D_start);

    if(cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "error in input cudaMemcpy H -> D" << endl;
	
	if(cudaMemcpy(output_d, output, size, cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "error in output cudaMemcpy H -> D" << endl;
		
    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);
	// ----------------------------------------------------------
    
    /* GPU calculations -------------------------------------- */
    cudaEventRecord(comp_start);
    double *temp_d;
    
    for(int n=0; n < (int)iterations; n++) {	
		processOnDevice <<< gridSize, blockSize >>> (input_d, output_d, length);
		if(n != iterations-1) {
			temp_d = input_d;
			input_d = output_d;
			output_d = temp_d;
		}
	}
	
    
    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);
    // ----------------------------------------------------------

	/* Copying array from D to H  ---------------------------- */
    cudaEventRecord(cpy_D2H_start);
    
    if(cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost) != cudaSuccess)
		cout << "error in output cudaMemcpy D -> H" << endl;
    
    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);
    // ----------------------------------------------------------

    /* Postprocessing goes here -------------------------------- */
	cudaFree(input_d);
	cudaFree(output_d);
	// ----------------------------------------------------------
	
    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}
