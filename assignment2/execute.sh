#!/bin/bash
#SBATCH --workdir /path/to/working/directory
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --reservation cs307cpu

threads=4
length=10000
iterations=500
output="output.csv"

./assignment2 $threads $length $iterations $output
