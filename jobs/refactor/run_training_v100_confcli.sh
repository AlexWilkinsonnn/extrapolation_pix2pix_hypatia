#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c4
#SBATCH --gres=gpu:1
#SBATCH --error=/home/awilkins/extrapolation_pix2pix/jobs/logs/err/job%j.err
#SBATCH --output=/home/awilkins/extrapolation_pix2pix/jobs/logs/out/job%j.out

cd /home/awilkins/extrapolation_pix2pix
module load gcc/8.3.0
source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh
conda activate 
conda activate pytorch-CycleGAN-and-pix2pix
python train.py $1
