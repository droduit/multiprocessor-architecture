/*
============================================================================
Filename    : pi.c
Author      : Dominique Roduit
SCIPER      : 234868
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"

double calculate_pi (int num_threads, int samples);
int inside_circle(const int samples);

int main (int argc, const char *argv[]) {

    int num_threads, num_samples;
    double pi;

    if (argc != 3) {
		printf("Invalid input! Usage: ./pi <num_threads> <num_samples> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
	}

    set_clock();
    pi = calculate_pi (num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}


double calculate_pi (int num_threads, int samples) {
    int circleArea = 0;
    int chunk = samples / num_threads;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for shared(num_threads, chunk) reduction(+:circleArea)
    for(int i = 0; i < num_threads; ++i) {
        circleArea += inside_circle(chunk);
    }

    return 4.0 * ((double)circleArea) / ((double) samples);
}

int inside_circle(const int samples) {
    int c = 0;
    rand_gen gen = init_rand();

    for(int i = 0; i < samples; ++i) {
        double x = next_rand(gen);
        double y = next_rand(gen);

        if(x*x + y*y < 1.0) {
           c++;
        }
    }
    return c;
}

