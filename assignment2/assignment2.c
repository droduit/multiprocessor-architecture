/*
 ============================================================================
 Filename    : assignment2.c
 Author      : Arash Pourhabibi, Hussein Kassir
 ============================================================================
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include "algorithm.c"

int main (int argc, const char *argv[]) {

    int threads, length, iterations;
    double time;
    
    if (argc != 5) {
		printf("Invalid input! \nUsage: ./assignment2 <threads> <length> <iterations> <output_filname>\n");
		return 1;
	} else {
        threads     = atoi(argv[1]);
        length      = atoi(argv[2]);
        iterations  = atoi(argv[3]);
        if(length%2!=0)
        {
            printf("Invalid input! Array length must be even\n");
            return 1;
        }
	}
    
    //Allocate a two-dimensional array
    double *input  = malloc(sizeof(double)*length*length);
    double *output = malloc(sizeof(double)*length*length);
    //Initialize the array
    init(input, length);
    init(output, length);
    
    //Start timer
    set_clock();
    
    //Optimize the following function
    simulate(input, output, threads, length, iterations);

    //Stop timer
    time = elapsed_time();
    
    //Report time required for n iterations
    printf("Running the algorithm with %d threads on %d by %d array for %d iteration takes %.4gs seconds \n", threads, length, length, iterations, time);
    
    //Save array in filelength
    save(output, length, argv[4]);
    
    //Free allocated memory
    free(input);
    free(output);

    return 0;
}
