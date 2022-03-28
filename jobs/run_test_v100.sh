#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c4
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=alexander.wilkinson.20@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

echo run_test_v100.sh
cd /home/awilkins/extrapolation_pix2pix
module load gcc/8.3.0
source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh
conda activate pytorch-CycleGAN-and-pix2pix
python test_new.py

