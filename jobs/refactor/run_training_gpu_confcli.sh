#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c8
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=compute-gpu-0-2
#SBATCH --error=/home/awilkins/extrapolation_pix2pix/jobs/logs/err/job.%x.%j.err
#SBATCH --output=/home/awilkins/extrapolation_pix2pix/jobs/logs/out/job.%x.%j.out

CONFIG_FILE=$1

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
echo $CONFIG_FILE

cd /home/awilkins/extrapolation_pix2pix
module load gcc/8.3.0
source setups/setup.sh

python train.py $CONFIG_FILE
