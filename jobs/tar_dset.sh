#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c2
#SBATCH --nodelist=compute-gpu-0-3
#SBATCH --mail-user=alexander.wilkinson.20@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

echo tar_dset.sh
cd /state/partition1/awilkins
tar -czf nd_fd_radi_1-8_trainvalid_vtxaligned_noped.tar.gz nd_fd_radi_1-8_vtxaligned_noped


