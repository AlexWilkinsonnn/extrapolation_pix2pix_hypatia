#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c4
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=compute-gpu-0-3
#SBATCH --mail-user=alexander.wilkinson.20@ucl.ac.uk
#SBATCH --mail-type=END,FAIL


echo run_training.sh
cd /home/awilkins/extrapolation_pix2pix/jobs
module load gcc/8.3.0
source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh
conda activate 
conda activate pytorch-CycleGAN-and-pix2pix
python $1
