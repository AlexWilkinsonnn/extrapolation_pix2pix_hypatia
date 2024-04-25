#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c4
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=compute-gpu-0-3
#SBATCH --mail-user=alexander.wilkinson.20@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

sleep 120

