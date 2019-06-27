/*
============================================================================
Filename    : algorithm.c
Author      : Dominique Roduit
SCIPER		: 234868
============================================================================
*/
#define input(i,j) input[(i)*length+(j)]
#define output(i,j) output[(i)*length+(j)]

void copyOutputInInput(int row, int length, double*, double*);
void updateGrid(int row, int length, double*, double*);

void simulate(double *input, double *output, int threads, int length, int iterations) {
	if(iterations <= 0 || threads <= 0) return;

	omp_set_num_threads(threads);	
	
	int row = 0;
	#pragma omp parallel private (row) shared (input, output) if(length > 24) 
	{	
		for(int iter = 0; iter < iterations; iter++) {
			#pragma omp for 
			for(row = 1; row < length-1; ++row) 
				updateGrid(row, length, input, output);
			
			#pragma omp for
			for(row = 1; row < length-1; ++row) 
				copyOutputInInput(row, length, input, output);
		}
	}
}

void updateGrid(int row, int length, double* input, double* output) {
	for(int col = 1; col < length-1; ++col) {
		if(!(input(row, col) >= INIT_VALUE)) {
			double sum = 
			input(row-1, col-1) + input(row, col-1) + input(row+1, col-1) +
			input(row-1, col) 	+ input(row, col) 	+ input(row+1, col)	  +
			input(row-1, col+1) + input(row, col+1) + input(row+1, col+1);
						
			output(row,col) = sum / 9.0;
		}
	}
}

void copyOutputInInput(int row, int length, double* input, double* output) {
	for(int col = 1; col < length-1; ++col) 
		input(row,col) = output(row, col);
}
