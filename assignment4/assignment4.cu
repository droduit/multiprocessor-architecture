/*
 ============================================================================
 Filename    : assignment4.c
 Author      : Arash Pourhabibi, Hussein Kassir
 ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;
#include "utility.h"

void array_process(double *input, double *output, int length, int iterations);
void GPU_array_process(double *input, double *output, int length, int iterations);

int main (int argc, const char *argv[]) {

    int length, iterations;
    double time;

    if (argc != 3) {
		cout<<"Invalid input!"<<endl<<"Usage: ./assignment4 <length> <iterations>"<<endl;
		return 1;
	} else {
        length      = atoi(argv[1]);
        iterations  = atoi(argv[2]);
        if(length%2!=0)
        {
            cout<<"Invalid input!"<<endl<<"Array length must be even"<<endl;
            return 1;
        }
	}


    //Allocate arrays
    double *input   = new double[length*length];
    double *output  = new double[length*length];

    //Reset Device
    cudaDeviceReset();

    //Initialize the arrays
    init(input, length);
    init(output, length);

    //Start timer
    set_clock();

    /*Use either the CPU or the GPU functions*/

    //CPU Baseline
    //Uncomment the block to use the baseline
    array_process(input, output, length, iterations);
    if(iterations%2==0)
    {
        double *temp;
        temp = input;
        input = output;
        output = temp;
    }

    //GPU function
    //GPU_array_process(input, output,  length, iterations);

    //Stop timer
    time = elapsed_time();

    //Report time required for n iterations
    cout<<"Running the algorithm on "<<length<<" by "<<length<<" array for "<<iterations<<" iteration takes "<<setprecision(4)<<time<<"s"<<endl;

    //Save array in filelength
    save(output, length);

    //Free allocated memory
    delete[] input;
    delete[] output;

    return 0;
}
