#!/bin/bash -l
#SBATCH --workdir /home/kassir/assignment4
#SBATCH --reservation cs307gpu
#SBATCH --time 01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1

length=50
iterations=1

module load gcc cuda

make

./assignment4 $length $iterations
