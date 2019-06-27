#!/bin/bash
#SBATCH --workdir /scratch/droduit
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --reservation cs307cpu

echo STARTING AT `date`

./integral 8 2000000000 5 9

echo FINISHED at `date`
