/*
 ============================================================================
 Filename    : utility.h
 Author      : Arash Pourhabibi, Hussein Kassir
 ============================================================================
 */
#ifndef UTILITYH
#define UTILITYH

struct timeval start, end;

void set_clock()
{
    gettimeofday(&start, NULL);
}

double elapsed_time()
{
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec); 
    elapsed += (double)(end.tv_usec - start.tv_usec) / 1000000.0;
    return elapsed;
}

//Initialize the two-dimensional array
void init(double *x, int length)
{
    memset(x, 0, sizeof(double)*length*length);
    x[(length/2-1)*length+(length/2-1)] = 1000;
    x[(length/2)*length+(length/2-1)]   = 1000;
    x[(length/2-1)*length+(length/2)]   = 1000;
    x[(length/2)*length+(length/2)]     = 1000;
}

//Save the two-dimensional array in a csv file
void save(double *x, int length)
{
    ofstream output_file;
    output_file.open("outputmatrix.csv");
    for(int i=0; i<length; i++)
    {
        for(int j=0; j<length-1; j++)
            output_file<<setprecision(4)<<x[i*length+j]<<";";
        output_file<<setprecision(4)<<x[i*length+length-1]<<";"<<endl;
    }
    output_file.close();
}

#endif
