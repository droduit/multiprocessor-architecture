/*
============================================================================
Filename    : integral.c
Author      : Dominique Roduit
SCIPER		: 234868 
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include "function.c"

double integrate (int num_threads, int samples, int a, int b, double (*f)(double));
double getVal(int samples, int a, int b, double (*f)(double));

int main (int argc, const char *argv[]) {

    int num_threads, num_samples, a, b;
    double integral;

    if (argc != 5) {
		printf("Invalid input! Usage: ./integral <num_threads> <num_samples> <a> <b>\n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        a = atoi(argv[3]);
        b = atoi(argv[4]);
	}

    set_clock();

    /* You can use your self-defined funtions by replacing identity_f. */
    integral = integrate (num_threads, num_samples, a, b, identity_f);

    printf("- Using %d threads: integral on [%d,%d] = %.15g computed in %.4gs.\n", num_threads, a, b, integral, elapsed_time());

    return 0;
}


double integrate (int num_threads, int samples, int a, int b, double (*f)(double)) {
    int chunk = samples / num_threads;
    omp_set_num_threads(num_threads);
     
    double c = 0;
    #pragma omp parallel for shared(num_threads, chunk) reduction(+:c)
    for(int i = 0; i < num_threads; ++i) {
		c += getVal(chunk, a, b, f);
	}
    
    return (double)(b - a) / (double)samples * c;
}

double getVal(int samples, int a, int b, double (*f)(double)) {
	double c = 0;
	rand_gen gen = init_rand();
	
	for(int i = 0; i < samples; ++i) {
		double xi = a + (b-a) * next_rand(gen);
		c += f(xi);
	}
	
	return c;
}
