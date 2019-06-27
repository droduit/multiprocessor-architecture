/*
 ============================================================================
 Filename    : utility.h
 Author      : Arash Pourhabibi
 ============================================================================
 */

#include <omp.h>
#include <time.h>
#include <sys/time.h>

struct timeval start, end;

typedef struct rand_gen {
	unsigned short * seed;
	double (*rand_func) (struct rand_gen);
	
} rand_gen;

void set_clock(){
	gettimeofday(&start, NULL);
}

double elapsed_time(){
	gettimeofday(&end, NULL);
	
	double elapsed = (end.tv_sec - start.tv_sec); 
	elapsed += (double)(end.tv_usec - start.tv_usec) / 1000000.0;
	return elapsed;
}

// Gives a random number between 0 and 1
double next_rand(rand_gen gen){	
	return erand48(gen.seed);
}	

rand_gen init_rand(){
	unsigned short * seed_array = malloc (sizeof (unsigned short) * 3);
	seed_array[0] = 5;
	seed_array[1] = 10;
	seed_array[2] = omp_get_thread_num();

	rand_gen gen;
	gen.seed = seed_array;
	gen.rand_func = next_rand;

	return gen;
}
