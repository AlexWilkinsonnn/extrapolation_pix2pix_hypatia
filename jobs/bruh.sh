#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c2
#SBATCH --nodelist=compute-gpu-0-3
#SBATCH --mail-user=alexander.wilkinson.20@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

echo prepare_dset.sh
cd /home/awilkins/extrapolation_pix2pix
cp output_1-8_radi_numuCC_vtxalignment_noped_morechannels_nd_fd_pairs_trainvalid.tar.gz /state/partition1/awilkins
cd /state/partition1/awilkins
#mv /home/awilkins/depos_X_4492_collection_fsb_nu.tar.gz .
tar -xzf output_1-8_radi_numuCC_vtxalignment_noped_morechannels_nd_fd_pairs_trainvalid.tar.gz 


